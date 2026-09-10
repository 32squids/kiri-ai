"""
Gymnasium environment for Kiri Ai: The Duel.

Wraps state.py's data model and resolution.py's turn resolver as a
single-agent Gymnasium Env, suitable for self-play training: the env
always plays one physical side (the "learner") against a pluggable
`opponent_policy` callable that supplies the other side's action each
turn. Swap the opponent via `set_opponent` (e.g. to a frozen snapshot of
the learner's own policy) to run iterative self-play.

Design notes:

  - Action space: a single Discrete space, fixed at import time, over
    every structurally-valid (card1, card2) ordered pair drawn from the
    *full* Card enum (ALL_TURN_PLANS) -- not just what's legal for any
    one player or state. This keeps the action space's meaning constant
    across the whole game (needed for a policy network's output layer to
    mean the same thing turn to turn), at the cost of most actions being
    illegal in any given state. `action_masks()` exposes which indices
    are currently legal, following sb3-contrib's MaskablePPO convention
    -- use it (or an equivalent invalid-action-masking mechanism); this
    env does not itself prevent an agent from attempting an illegal
    action, it just rejects one with a ValueError if asked to play it.

  - Observation space: a flat Box of one-hot/binary features, built by
    `encode_observation`. Positions are expressed in a *canonical frame*
    relative to whichever side the observation is for: Side Two's board
    coordinates are mirrored (see `_mirror_position`) so that "self"
    always appears to start at BOARD_MIN and move toward BOARD_MAX, the
    same as Side One. Card actions never need this treatment since
    Forward/Charge/attack offsets are already defined relative to a
    player's own forward direction in state.py. The practical effect: a
    single policy network, queried once per side per turn with each
    side's own canonical observation, can play either side without
    needing to know which one it physically is.

  - Reward: `reward_mode="sparse"` (default) pays +1/-1 only when the
    match ends (draws are not reachable -- see GameState.winner);
    `reward_mode="dense"` instead pays +1/-1 for every hit landed/taken
    that turn, which by construction also nets to +-2 by the time a
    match ends 2-0.

  - Episode length: the physical ruleset has no turn limit, and two
    sufficiently passive agents could in principle stall forever. This
    is an RL-training concern, not a rules concern, so it's handled here
    via `max_turns`, surfaced through Gymnasium's `truncated` flag
    (distinct from `terminated`, which means the match actually ended).
"""

from __future__ import annotations

import random
from typing import Callable, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from state import (
    BOARD_MAX,
    BOARD_MIN,
    CARD_PAIRS,
    Card,
    GameState,
    PlayerState,
    Side,
    Stance,
    new_game,
    observation_for,
)
from resolution import Event, HitEvent, TurnPlan, legal_turn_plans, resolve_turn


# --- Fixed action space ----------------------------------------------------
# A stable superset action space over the *entire* Card enum, independent
# of any particular player's hand or game state. Its ordering follows
# Card's definition order in state.py, so a saved policy's action indices
# only stay meaningful as long as that enum's member order doesn't change.

def _same_physical_card(card1: Card, card2: Card) -> bool:
    if card1 == card2:
        return True
    return any(card1 in pair and card2 in pair for pair in CARD_PAIRS)


ALL_TURN_PLANS: Tuple[TurnPlan, ...] = tuple(
    (card1, card2)
    for card1 in Card
    for card2 in Card
    if not _same_physical_card(card1, card2)
)
ACTION_SPACE_SIZE = len(ALL_TURN_PLANS)
TURN_PLAN_TO_ACTION = {plan: index for index, plan in enumerate(ALL_TURN_PLANS)}


# --- Observation encoding ---------------------------------------------------

_NUM_POSITIONS = BOARD_MAX - BOARD_MIN + 1  # one-hot size for a board position
_NUM_STANCES = 2
_NUM_HIT_LEVELS = 3  # 0, 1, or 2 hits taken
_NUM_COOLDOWN_FLAGS = 3  # Heaven / Earth / Neutral Strike
_NUM_OWN_SPECIAL_IDENTITIES = 3  # Counterattack / Zan-tetsu / Kesa, always known to oneself
_NUM_OPPONENT_SPECIAL_CATEGORIES = 4  # unknown + the 3 identities, once revealed

_PER_PLAYER_COMMON_DIMS = _NUM_POSITIONS + _NUM_STANCES + _NUM_HIT_LEVELS + _NUM_COOLDOWN_FLAGS
_OWN_SPECIAL_DIMS = _NUM_OWN_SPECIAL_IDENTITIES + 1  # + a used/unused flag
_OPPONENT_SPECIAL_DIMS = _NUM_OPPONENT_SPECIAL_CATEGORIES
_TURN_FRACTION_DIMS = 1

OBSERVATION_DIM = (
    _PER_PLAYER_COMMON_DIMS + _OWN_SPECIAL_DIMS
    + _PER_PLAYER_COMMON_DIMS + _OPPONENT_SPECIAL_DIMS
    + _TURN_FRACTION_DIMS
)

_SPECIAL_IDENTITY_INDEX = {
    Card.SPECIAL_COUNTERATTACK: 0,
    Card.SPECIAL_ZANTETSU: 1,
    Card.SPECIAL_KESA: 2,
}
_STANDARD_ATTACK_COOLDOWN_ORDER = (Card.HEAVEN_STRIKE, Card.EARTH_STRIKE, Card.NEUTRAL_STRIKE)


def _mirror_position(position: int, perspective: Side) -> int:
    """`position` expressed in the canonical frame for `perspective`: as-is
    for Side One, mirrored around the board's center for Side Two, so
    "self" always appears to start at BOARD_MIN and move toward
    BOARD_MAX regardless of which side is actually observing."""
    if perspective is Side.ONE:
        return position
    return BOARD_MIN + BOARD_MAX - position


def _one_hot(index: Optional[int], size: int) -> List[float]:
    vector = [0.0] * size
    if index is not None:
        vector[index] = 1.0
    return vector


def _encode_player(player: PlayerState, perspective: Side, *, own: bool) -> List[float]:
    mirrored_position = _mirror_position(player.position, perspective)
    features = _one_hot(mirrored_position - BOARD_MIN, _NUM_POSITIONS)
    features += _one_hot(0 if player.stance is Stance.HEAVEN else 1, _NUM_STANCES)
    features += _one_hot(player.hits_taken, _NUM_HIT_LEVELS)
    features += [
        1.0 if card in player.attacks_on_cooldown else 0.0
        for card in _STANDARD_ATTACK_COOLDOWN_ORDER
    ]

    if own:
        # A player always knows their own special's identity, even before
        # playing it -- observation_for only ever masks the opponent's.
        assert player.special_attack is not None, "own special_attack should never be masked"
        features += _one_hot(_SPECIAL_IDENTITY_INDEX[player.special_attack], _NUM_OWN_SPECIAL_IDENTITIES)
        features.append(1.0 if player.special_used else 0.0)
    else:
        # The opponent's special is secret until played; category 0 is
        # "unknown" (player.special_attack is None here, per observation_for).
        if player.special_attack is None:
            features += _one_hot(0, _NUM_OPPONENT_SPECIAL_CATEGORIES)
        else:
            features += _one_hot(_SPECIAL_IDENTITY_INDEX[player.special_attack] + 1, _NUM_OPPONENT_SPECIAL_CATEGORIES)

    return features


def encode_observation(state: GameState, side: Side, max_turns: int) -> np.ndarray:
    """The observation `side` is legitimately allowed to see, as a flat
    float32 vector: their own state, then the opponent's (with the
    opponent's special masked out until revealed -- see
    `state.observation_for`), then how far through the turn cap the
    match is. Board positions are in `side`'s canonical frame (see
    `_mirror_position`)."""
    observed = observation_for(state, side)
    me = observed.player(side)
    opponent = observed.opponent(side)

    features = _encode_player(me, side, own=True)
    features += _encode_player(opponent, side, own=False)
    features.append(min(state.turn_number / max_turns, 1.0))

    return np.array(features, dtype=np.float32)


# --- Reward -----------------------------------------------------------------

def _compute_reward(
    events: List[Event],
    terminated: bool,
    winner: Optional[Side],
    learner_side: Side,
    reward_mode: str,
) -> float:
    if reward_mode == "dense":
        return sum(
            (-1.0 if event.side is learner_side else 1.0)
            for event in events
            if isinstance(event, HitEvent)
        )
    # sparse
    if not terminated:
        return 0.0
    return 1.0 if winner is learner_side else -1.0


# --- Environment -------------------------------------------------------------

OpponentPolicy = Callable[[np.ndarray, np.ndarray], int]


class KiriDuelEnv(gym.Env):
    """
    A single-agent Gymnasium environment for Kiri Ai: The Duel. The env
    always controls one physical side (`self.learner_side`, re-chosen
    each `reset()` when `random_side=True`) and drives the other side
    internally via `opponent_policy`.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        opponent_policy: Optional[OpponentPolicy] = None,
        reward_mode: str = "sparse",
        random_side: bool = True,
        max_turns: int = 100,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        if reward_mode not in ("sparse", "dense"):
            raise ValueError(f"reward_mode must be 'sparse' or 'dense', got {reward_mode!r}")

        self.reward_mode = reward_mode
        self.random_side = random_side
        self.max_turns = max_turns
        self.render_mode = render_mode

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(OBSERVATION_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)

        self._py_random = random.Random()
        self._opponent_policy: OpponentPolicy = opponent_policy or self._default_opponent_policy
        self._state: GameState = new_game(self._py_random)
        self.learner_side: Side = Side.ONE

    def set_opponent(self, policy: Optional[OpponentPolicy]) -> None:
        """Swap the policy driving the non-learner side -- e.g. to a
        frozen snapshot of the learner's own trained policy for
        iterative self-play. `None` restores the default uniform-random
        legal-action policy."""
        self._opponent_policy = policy or self._default_opponent_policy

    def _default_opponent_policy(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        del observation  # unused: a uniform-random policy only needs the mask
        legal_actions = np.flatnonzero(action_mask).tolist()
        return self._py_random.choice(legal_actions)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._py_random.seed(seed)

        self._state = new_game(self._py_random)

        forced_side = (options or {}).get("learner_side")
        if forced_side is not None:
            self.learner_side = forced_side
        elif self.random_side:
            self.learner_side = self._py_random.choice([Side.ONE, Side.TWO])
        else:
            self.learner_side = Side.ONE

        observation = self._observe(self.learner_side)
        info = {"learner_side": self.learner_side}
        return observation, info

    @property
    def state(self) -> GameState:
        """The ground-truth GameState, including information (like the
        opponent's not-yet-revealed special) that `learner_side` isn't
        supposed to see. Read-only introspection for tooling (rendering,
        logging, a debug CLI) -- do not mutate it; game state only ever
        changes through `step`."""
        return self._state

    def step(self, action: int):
        learner_side = self.learner_side
        opponent_side = learner_side.other

        learner_plan = self._plan_for(learner_side, action)

        opponent_observation = self._observe(opponent_side)
        opponent_mask = self._action_mask_for(opponent_side)
        opponent_action = self._opponent_policy(opponent_observation, opponent_mask)
        opponent_plan = self._plan_for(opponent_side, opponent_action)

        plan_one = learner_plan if learner_side is Side.ONE else opponent_plan
        plan_two = opponent_plan if learner_side is Side.ONE else learner_plan

        new_state, reveals = resolve_turn(self._state, plan_one, plan_two)
        self._state = new_state

        terminated = new_state.is_over
        truncated = (not terminated) and new_state.turn_number > self.max_turns
        flat_events = [event for reveal in reveals for event in reveal]
        reward = _compute_reward(flat_events, terminated, new_state.winner, learner_side, self.reward_mode)

        observation = self._observe(learner_side)
        info = {
            "events": flat_events,
            "reveal_events": reveals,
            "winner": new_state.winner if terminated else None,
        }
        return observation, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Boolean mask, `ACTION_SPACE_SIZE` long, over ALL_TURN_PLANS:
        True where that plan is legal for the learner to select this
        turn. Named to match sb3-contrib's MaskablePPO convention."""
        return self._action_mask_for(self.learner_side)

    def render(self):
        if self.render_mode != "ansi":
            return None
        return _render_ansi(self._state, self.learner_side)

    def _observe(self, side: Side) -> np.ndarray:
        return encode_observation(self._state, side, self.max_turns)

    def _action_mask_for(self, side: Side) -> np.ndarray:
        legal = set(legal_turn_plans(self._state.player(side)))
        return np.array([plan in legal for plan in ALL_TURN_PLANS], dtype=bool)

    def _plan_for(self, side: Side, action: int) -> TurnPlan:
        if not (0 <= action < ACTION_SPACE_SIZE):
            raise ValueError(f"action {action} out of range [0, {ACTION_SPACE_SIZE})")
        if not self._action_mask_for(side)[action]:
            raise ValueError(f"action {action} ({ALL_TURN_PLANS[action]}) is not legal for {side} right now")
        return ALL_TURN_PLANS[action]


def _render_ansi(state: GameState, learner_side: Side) -> str:
    cells = []
    for position in range(BOARD_MIN, BOARD_MAX + 1):
        marker = ""
        if state.player_one.position == position:
            marker += "1"
        if state.player_two.position == position:
            marker += "2"
        cells.append(f"[{marker or ' '}]")

    p1, p2 = state.player_one, state.player_two
    return "\n".join([
        "".join(cells),
        f"P1: pos={p1.position} stance={p1.stance.name} hits={p1.hits_taken}",
        f"P2: pos={p2.position} stance={p2.stance.name} hits={p2.hits_taken}",
        f"turn={state.turn_number} learner={learner_side.name}",
    ])
