"""
Game state representation for Kiri Ai: The Duel.

This module defines the data model for a single game of Kiri Ai: The Duel --
a simultaneous-selection, imperfect-information dueling game played on a
5-space board. It intentionally covers ONLY the state: what a game looks
like at a given instant, and the static rules-as-data attached to each
card. Turn resolution (legal action generation, simultaneous reveal,
movement-then-attack priority, hit checking) is deliberately left to a
separate module built on top of this one.

Rules summary this module encodes:
    - The board is a line of 5 spaces, numbered 1 (Player One's start) to
      5 (Player Two's start). Player One's "forward" is toward higher
      numbers; Player Two's "forward" is toward lower numbers. This is
      fixed for the whole match based on starting side.
    - Each player is always in one of two stances: HEAVEN or EARTH.
    - Each player has a fixed 6-card hand for the entire game:
        - one Forward/Backward movement card (the two ends of one
          physical card -- a player commits to an orientation the moment
          they select it for a turn, so it behaves as two distinct
          actions that can never both be played in the same turn)
        - one Stance-Change/Charge movement card (same deal)
        - Heaven Strike, Earth Strike, Neutral Strike (standard attacks)
        - one Special Attack, dealt blind and at random from a shared
          3-card pool (the third card is set aside face down and never enters play)
    - Standard attacks go on a strict one-turn cooldown after use (they
      can't be played again on the very next turn, then return to the
      hand as normal). Movement cards never go on cooldown. Special
      attacks are instead consumed permanently the first time they're
      played -- there is no cooldown because there is no second use.
    - A player loses the moment they've been struck twice.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum, auto
import random
from typing import Dict, FrozenSet, Optional, Tuple


# --- Board ----------------------------------------------------------------

BOARD_MIN = 1  # Player One's starting space.
BOARD_MAX = 5  # Player Two's starting space.


# --- Stances ----------------------------------------------------------------

class Stance(Enum):
    HEAVEN = auto()
    EARTH = auto()

    def opposite(self) -> "Stance":
        return Stance.EARTH if self is Stance.HEAVEN else Stance.HEAVEN


# --- Sides ------------------------------------------------------------------
# A player's side is fixed for the whole match by their starting position,
# and determines which way "forward" points for every movement/attack card.

class Side(Enum):
    ONE = auto()  # Starts at BOARD_MIN; forward means +1.
    TWO = auto()  # Starts at BOARD_MAX; forward means -1.

    @property
    def forward_direction(self) -> int:
        return 1 if self is Side.ONE else -1

    @property
    def start_position(self) -> int:
        return BOARD_MIN if self is Side.ONE else BOARD_MAX

    @property
    def other(self) -> "Side":
        return Side.TWO if self is Side.ONE else Side.ONE


# --- Cards --------------------------------------------------------------

class Card(Enum):
    """
    Every distinct action a player can ever take.

    FORWARD/BACKWARD share one physical card, and STANCE_CHANGE/CHARGE
    share another. They're still modeled as four separate actions here
    because the choice between the two ends of a card is locked in at
    selection time, before reveal. The constraint that a player may only
    play one action per physical card pair per turn (see CARD_PAIRS
    below) is a per-turn *selection* rule, so it's enforced by the
    resolution module's legal-action generator, not by this state model.
    """

    FORWARD = auto()
    BACKWARD = auto()
    STANCE_CHANGE = auto()
    CHARGE = auto()

    HEAVEN_STRIKE = auto()
    EARTH_STRIKE = auto()
    NEUTRAL_STRIKE = auto()

    SPECIAL_COUNTERATTACK = auto()
    SPECIAL_ZANTETSU = auto()
    SPECIAL_KESA = auto()


MOVEMENT_CARDS: FrozenSet[Card] = frozenset({
    Card.FORWARD, Card.BACKWARD, Card.STANCE_CHANGE, Card.CHARGE,
})

# The two physical movement cards, each a pair of mutually exclusive
# actions -- a player may select at most one action from each pair in a
# given turn, since each pair is a single double-sided card.
CARD_PAIRS: Tuple[FrozenSet[Card], FrozenSet[Card]] = (
    frozenset({Card.FORWARD, Card.BACKWARD}),
    frozenset({Card.STANCE_CHANGE, Card.CHARGE}),
)

STANDARD_ATTACKS: FrozenSet[Card] = frozenset({
    Card.HEAVEN_STRIKE, Card.EARTH_STRIKE, Card.NEUTRAL_STRIKE,
})

SPECIAL_ATTACKS: FrozenSet[Card] = frozenset({
    Card.SPECIAL_COUNTERATTACK, Card.SPECIAL_ZANTETSU, Card.SPECIAL_KESA,
})

ATTACK_CARDS: FrozenSet[Card] = STANDARD_ATTACKS | SPECIAL_ATTACKS

# Attacks that sit out exactly one turn after being played. Special
# attacks are deliberately excluded -- they don't cool down, they're
# consumed forever (tracked instead by PlayerState.special_used).
COOLDOWN_ELIGIBLE: FrozenSet[Card] = STANDARD_ATTACKS

# Stance required to play a stance-gated card. A card missing from this
# dict has no stance requirement (Neutral Strike, Counterattack). Playing
# a stance-gated card while in the wrong stance is illegal to even
# select -- unlike illegal movement, which is selectable but a no-op.
REQUIRED_STANCE: Dict[Card, Stance] = {
    Card.HEAVEN_STRIKE: Stance.HEAVEN,
    Card.EARTH_STRIKE: Stance.EARTH,
    Card.SPECIAL_KESA: Stance.HEAVEN,
    Card.SPECIAL_ZANTETSU: Stance.EARTH,
}

# The stance a player swaps into after playing these specials. The
# rules only define this for Zan-tetsu and Kesa; exactly when this
# triggers (e.g. whether it still happens on a miss) is a resolution
# detail, not a state-model concern.
POST_ATTACK_STANCE_SWAP: Dict[Card, Stance] = {
    Card.SPECIAL_ZANTETSU: Stance.HEAVEN,
    Card.SPECIAL_KESA: Stance.EARTH,
}

# Target squares an attack threatens, as offsets from the attacker's own
# position in the attacker's forward direction. Resolution logic turns
# these into real board squares via
#   attacker.position + offset * attacker.side.forward_direction
# Offsets that land outside [BOARD_MIN, BOARD_MAX] simply correspond to a
# square that can never be occupied, so they always miss -- no wraparound
# or clamping is needed.
#
# SPECIAL_COUNTERATTACK has no entry here: it doesn't threaten a square
# of its own, it reacts to whatever the opponent's attack does. See the
# resolution module for how it's handled.
ATTACK_TARGET_OFFSETS: Dict[Card, FrozenSet[int]] = {
    Card.NEUTRAL_STRIKE: frozenset({0}),
    Card.EARTH_STRIKE: frozenset({1}),
    Card.HEAVEN_STRIKE: frozenset({2}),
    Card.SPECIAL_ZANTETSU: frozenset({2, 3}),
    Card.SPECIAL_KESA: frozenset({0, 1}),
}


# --- Per-player state ---------------------------------------------------

@dataclass(frozen=True)
class PlayerState:
    """Everything about one duelist at a single instant."""

    side: Side
    position: int
    stance: Stance
    hits_taken: int = 0  # 0, 1, or 2. 2 means this player has lost.

    # Which of the three special attacks this player was blindly dealt at
    # the start of the game. Ground-truth state always has a real value
    # here; `None` is reserved for a *masked observation* of the
    # opponent's special before it's been revealed by being played (see
    # `observation_for` below).
    special_attack: Optional[Card] = None
    special_used: bool = False

    # Standard attacks currently on cooldown (played last turn, so
    # unusable this turn). Cleared at the start of each turn's selection
    # phase after sitting out exactly one turn. Can hold more than one
    # card, since a player may play two standard attacks in the same turn.
    attacks_on_cooldown: FrozenSet[Card] = frozenset()

    def __post_init__(self) -> None:
        if not (BOARD_MIN <= self.position <= BOARD_MAX):
            raise ValueError(f"position {self.position} is off the board")
        if not (0 <= self.hits_taken <= 2):
            raise ValueError(f"hits_taken {self.hits_taken} out of range")
        if self.special_attack is not None and self.special_attack not in SPECIAL_ATTACKS:
            raise ValueError("special_attack must be one of SPECIAL_ATTACKS or None")
        if not self.attacks_on_cooldown <= COOLDOWN_ELIGIBLE:
            raise ValueError("only standard attacks can be on cooldown")

    @property
    def is_defeated(self) -> bool:
        return self.hits_taken >= 2

    def available_cards(self) -> FrozenSet[Card]:
        """
        The cards this player could pick from this turn, ignoring the
        "one action per physical card pair" and stance-gating rules --
        those depend on how a turn's two cards interact, so they belong
        to the resolution module's legal-action generator, not here.

        Assumes a fully-known state (i.e. call this on your own
        PlayerState, not a masked view of an opponent's).
        """
        cards = set(MOVEMENT_CARDS) | set(STANDARD_ATTACKS)
        cards -= self.attacks_on_cooldown
        if not self.special_used and self.special_attack is not None:
            cards.add(self.special_attack)
        return frozenset(cards)


# --- Full game state ------------------------------------------------------

@dataclass(frozen=True)
class GameState:
    """A complete, fully-observable snapshot of a game in progress."""

    player_one: PlayerState
    player_two: PlayerState
    turn_number: int = 1

    def player(self, side: Side) -> PlayerState:
        return self.player_one if side is Side.ONE else self.player_two

    def opponent(self, side: Side) -> PlayerState:
        return self.player(side.other)

    def with_player(self, side: Side, player_state: PlayerState) -> "GameState":
        """Return a new GameState with one player's state replaced."""
        if side is Side.ONE:
            return replace(self, player_one=player_state)
        return replace(self, player_two=player_state)

    @property
    def is_over(self) -> bool:
        return self.player_one.is_defeated or self.player_two.is_defeated

    @property
    def winner(self) -> Optional[Side]:
        """
        None if the game isn't over. A simultaneous double-defeat is not
        reachable by the rules as specified: a single exchange can never
        land hits both ways (each side's success requires the other's
        failure), and the resolution module halts the instant a player
        reaches two hits, before any later card that turn can be played.
        """
        one_lost = self.player_one.is_defeated
        two_lost = self.player_two.is_defeated
        if one_lost and two_lost:
            return None
        if one_lost:
            return Side.TWO
        if two_lost:
            return Side.ONE
        return None


def new_game(rng: Optional[random.Random] = None) -> GameState:
    """
    Deal a fresh game: both players start at their home space in Heaven
    stance, and the three special attacks are shuffled and blindly dealt
    one each -- the third is set aside and never used.
    """
    rng = rng or random.Random()
    specials = list(SPECIAL_ATTACKS)
    rng.shuffle(specials)
    special_one, special_two, _unused = specials

    player_one = PlayerState(
        side=Side.ONE,
        position=Side.ONE.start_position,
        stance=Stance.HEAVEN,
        special_attack=special_one,
    )
    player_two = PlayerState(
        side=Side.TWO,
        position=Side.TWO.start_position,
        stance=Stance.HEAVEN,
        special_attack=special_two,
    )
    return GameState(player_one=player_one, player_two=player_two)


def observation_for(state: GameState, side: Side) -> GameState:
    """
    The view of `state` that `side` is legitimately allowed to see.

    Every player knows their own special attack, but the opponent's
    special is a secret until the moment it's actually played -- once
    `special_used` flips to True its identity is public (it was just
    revealed on the table). This returns a GameState with the opponent's
    PlayerState.special_attack replaced by None while it's still hidden.

    Consumers (e.g. an RL policy's input encoder) should treat
    `special_attack is None` as "unknown to this viewer", never as "this
    player has no special attack" -- every player always has exactly one.
    """
    opponent = state.opponent(side)
    if not opponent.special_used:
        opponent = replace(opponent, special_attack=None)
    return state.with_player(opponent.side, opponent)
