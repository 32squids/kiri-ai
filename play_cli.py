"""
Interactive human-vs-env CLI for Kiri Ai: The Duel.

A manual play harness for exercising KiriDuelEnv (and, underneath it,
resolution.py and state.py) by hand -- pick moves turn by turn, see
exactly what the resolver did, and catch anything that looks wrong.
The status display respects hidden information the same way a real
match would: the opponent's special attack shows as "hidden" until
they've actually played it (see state.observation_for).

Two ways to choose your plan each turn:
  - Pick a number from the listed legal plans.
  - Type two card names directly, e.g. "FORWARD HEAVEN_STRIKE" -- this
    works even for illegal plans, in which case you'll get the specific
    reason resolution.validate_turn_plan rejected it. Useful for
    checking that something you expect to be illegal actually is.

Usage:
    python play_cli.py [--opponent random|cycle|model] [--model PATH] [--stochastic]
                        [--side one|two|random] [--seed N] [--max-turns N]
                        [--reward-mode sparse|dense]

  --opponent random  Opponent plays a uniform-random legal plan each turn (default).
  --opponent cycle    Opponent deterministically cycles through every entry in
                       ALL_TURN_PLANS (wrapping, skipping whatever's currently
                       illegal) -- so over a session it exercises every action
                       the game state ever allows it, instead of leaving rare
                       combinations to chance.
  --opponent model    Opponent is a trained MaskablePPO model loaded from
                       --model PATH (e.g. runs/latest/best/best_model.zip or
                       runs/latest/final_model.zip, from train.py). Takes its
                       single best action each turn by default; pass
                       --stochastic to instead sample from its policy.
"""

from __future__ import annotations

import argparse
from typing import List, Optional

from env import ALL_TURN_PLANS, KiriDuelEnv, OpponentPolicy, TURN_PLAN_TO_ACTION
from resolution import (
    AttackEvent,
    Event,
    GameEndEvent,
    HitEvent,
    MoveEvent,
    SpecialConsumedEvent,
    StanceChangeEvent,
    TurnPlan,
    validate_turn_plan,
)
from state import BOARD_MAX, BOARD_MIN, Card, GameState, PlayerState, Side, observation_for


class CyclingOpponentPolicy:
    """Deterministically cycles through ALL_TURN_PLANS in a fixed order,
    each call advancing to the next entry (wrapping around) that's
    currently legal. Meant for manual testing: a random opponent might
    never happen to try some rare cooldown/stance-gated combo in a short
    session, this guarantees every combo gets tried as soon as the game
    state makes it available."""

    def __init__(self) -> None:
        self._next_index = 0

    def __call__(self, observation, action_mask) -> int:  # noqa: ANN001 (matches OpponentPolicy's shape)
        del observation
        n = len(action_mask)
        for offset in range(n):
            candidate = (self._next_index + offset) % n
            if action_mask[candidate]:
                self._next_index = (candidate + 1) % n
                return candidate
        raise RuntimeError("opponent has no legal actions -- should never happen")


class ModelOpponentPolicy:
    """Opponent backed by a trained MaskablePPO model, loaded once from
    disk. Imports sb3_contrib lazily so the rest of this CLI (--opponent
    random/cycle) doesn't require the ML dependencies (torch, sb3-contrib)
    to be installed at all."""

    def __init__(self, model_path: str, deterministic: bool = True) -> None:
        from sb3_contrib import MaskablePPO

        self._model = MaskablePPO.load(model_path)
        self._deterministic = deterministic

    def __call__(self, observation, action_mask) -> int:  # noqa: ANN001 (matches OpponentPolicy's shape)
        action, _ = self._model.predict(observation, action_masks=action_mask, deterministic=self._deterministic)
        return int(action)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Human-vs-env CLI for Kiri Ai: The Duel.")
    parser.add_argument(
        "--opponent", choices=["random", "cycle", "model"], default="random",
        help="random: uniform-random legal plan each turn (default). "
             "cycle: deterministically works through every action in ALL_TURN_PLANS. "
             "model: play against a trained MaskablePPO model (requires --model).",
    )
    parser.add_argument("--model", type=str, default=None,
                         help="Path to a trained model .zip, e.g. from train.py (required with --opponent model).")
    parser.add_argument("--stochastic", action="store_true",
                         help="With --opponent model, sample from its policy instead of taking its single best action.")
    parser.add_argument("--side", choices=["one", "two", "random"], default="random",
                         help="Which physical side you play as (default: random each game).")
    parser.add_argument("--seed", type=int, default=None,
                         help="RNG seed, for reproducing a specific deal/side/opponent sequence.")
    parser.add_argument("--max-turns", type=int, default=100,
                         help="Turn cap before the episode truncates (training-only convenience, not a rule).")
    parser.add_argument("--reward-mode", choices=["sparse", "dense"], default="sparse")

    args = parser.parse_args()
    if args.opponent == "model" and args.model is None:
        parser.error("--opponent model requires --model PATH")
    return args


_COOLDOWN_ORDER = (Card.HEAVEN_STRIKE, Card.EARTH_STRIKE, Card.NEUTRAL_STRIKE)


def format_player(label: str, player: PlayerState) -> str:
    cooldowns = [c.name for c in _COOLDOWN_ORDER if c in player.attacks_on_cooldown]
    cooldown_text = ",".join(cooldowns) if cooldowns else "-"
    special = player.special_attack.name if player.special_attack is not None else "hidden"
    used_text = " (used)" if player.special_used else ""
    return (
        f"  {label}: pos={player.position} stance={player.stance.name:<6} hits={player.hits_taken}/2  "
        f"special={special}{used_text}  cooldown={cooldown_text}"
    )


def render_board(state: GameState) -> str:
    cells = []
    for position in range(BOARD_MIN, BOARD_MAX + 1):
        marker = ""
        if state.player_one.position == position:
            marker += "1"
        if state.player_two.position == position:
            marker += "2"
        cells.append(f"[{marker or ' '}]")
    return "".join(cells)


def print_state(env: KiriDuelEnv) -> None:
    state = env.state
    visible = observation_for(state, env.learner_side)  # masks the opponent's not-yet-revealed special
    print()
    print(f"  {render_board(state)}")
    print(f"  Turn {state.turn_number}  |  you are {env.learner_side.name}")
    print(format_player("P1", visible.player_one))
    print(format_player("P2", visible.player_two))


def describe_event(event: Event) -> str:
    if isinstance(event, MoveEvent):
        verb = "moves" if event.moved else "tries to move (blocked -- no-op)"
        return f"{event.side.name} {verb}: {event.card.name} {event.from_position} -> {event.to_position}"
    if isinstance(event, StanceChangeEvent):
        return f"{event.side.name} stance change ({event.card.name}): {event.from_stance.name} -> {event.to_stance.name}"
    if isinstance(event, AttackEvent):
        targets = ",".join(str(t) for t in sorted(event.target_squares)) or "-"
        if event.countered:
            tag = "COUNTERED"
        elif event.hit:
            tag = "HIT"
        elif event.would_have_hit:
            # Would have landed, but the defender's own attack reached back
            # and hit this attacker's square too -- the ruleset's mutual-
            # cancellation clause, not a clean miss.
            tag = "BLOCKED"
        else:
            tag = "miss"
        return f"{event.side.name} attacks with {event.card.name} (targets {{{targets}}}): {tag}"
    if isinstance(event, SpecialConsumedEvent):
        return f"{event.side.name} permanently consumes special: {event.card.name}"
    if isinstance(event, HitEvent):
        return f"{event.side.name} takes a hit ({event.hits_taken_after}/2)"
    if isinstance(event, GameEndEvent):
        return f"GAME OVER -- winner: {event.winner.name if event.winner is not None else 'None'}"
    return str(event)


_PAIRABLE_EVENT_TYPES = (MoveEvent, StanceChangeEvent, AttackEvent)


def format_reveal(events: List[Event]) -> List[str]:
    """
    One reveal's events as display lines. Movement, stance-change, and
    attack events are emitted in ONE/TWO pairs adjacent to each other in
    the underlying event list by construction (see resolution.py), so
    same-type pairs are combined onto one '|'-joined line for at-a-glance
    comparison; anything else (a hit landing, a special being consumed,
    game end) gets its own line, in order.
    """
    lines: List[str] = []
    i = 0
    n = len(events)
    while i < n:
        current = events[i]
        nxt = events[i + 1] if i + 1 < n else None
        if (
            nxt is not None
            and type(current) is type(nxt)
            and isinstance(current, _PAIRABLE_EVENT_TYPES)
            and isinstance(nxt, _PAIRABLE_EVENT_TYPES)
            and current.side is not nxt.side
        ):
            lines.append(f"    {describe_event(current)}    |    {describe_event(nxt)}")
            i += 2
        else:
            lines.append(f"    {describe_event(current)}")
            i += 1
    return lines


def print_turn_result(reveals: List[List[Event]]) -> None:
    print("\n  What happened:")
    for i, reveal in enumerate(reveals, start=1):
        if len(reveals) > 1:
            print(f"  Reveal {i}:")
        for line in format_reveal(reveal):
            print(line)


def parse_card(token: str) -> Optional[Card]:
    key = token.strip().upper().replace("-", "_")
    try:
        return Card[key]
    except KeyError:
        return None


HELP_TEXT = """
Commands:
  <number>              Play the numbered plan from the list below.
  <CARD1> <CARD2>        Play these two cards directly, in that reveal order
                         (e.g. "FORWARD HEAVEN_STRIKE"). Works even if illegal --
                         you'll get the specific reason it was rejected.
  help / h               Show this again.
  quit / q                Quit.

Card names: FORWARD BACKWARD STANCE_CHANGE CHARGE HEAVEN_STRIKE EARTH_STRIKE
            NEUTRAL_STRIKE SPECIAL_COUNTERATTACK SPECIAL_ZANTETSU SPECIAL_KESA
"""


def prompt_for_action(env: KiriDuelEnv) -> Optional[int]:
    mask = env.action_masks()
    legal_indices = [i for i, ok in enumerate(mask) if ok]

    while True:
        print("\n  Your legal plans this turn:")
        for n, idx in enumerate(legal_indices, start=1):
            card1, card2 = ALL_TURN_PLANS[idx]
            print(f"    {n:2d}) {card1.name} -> {card2.name}")

        print_state(env)

        try:
            raw = input("  > ").strip()
        except EOFError:
            return None

        if not raw:
            continue
        lowered = raw.lower()
        if lowered in ("q", "quit"):
            return None
        if lowered in ("h", "help"):
            print(HELP_TEXT)
            continue

        if lowered.isdigit():
            n = int(lowered)
            if 1 <= n <= len(legal_indices):
                return legal_indices[n - 1]
            print(f"  '{n}' isn't in range 1-{len(legal_indices)}.")
            continue

        parts = raw.split()
        if len(parts) == 2:
            card1, card2 = parse_card(parts[0]), parse_card(parts[1])
            if card1 is None or card2 is None:
                bad = parts[0] if card1 is None else parts[1]
                print(f"  '{bad}' isn't a recognized card name. Type 'help' to see the list.")
                continue
            plan: TurnPlan = (card1, card2)
            try:
                validate_turn_plan(env.state.player(env.learner_side), plan)
            except ValueError as exc:
                print(f"  Illegal plan: {exc}")
                continue
            return TURN_PLAN_TO_ACTION[plan]

        print(f"  Didn't understand '{raw}'. Type 'help' for the command list.")


def main() -> None:
    args = parse_args()

    opponent_policy: Optional[OpponentPolicy] = None
    if args.opponent == "cycle":
        opponent_policy = CyclingOpponentPolicy()
    elif args.opponent == "model":
        opponent_policy = ModelOpponentPolicy(args.model, deterministic=not args.stochastic)

    env = KiriDuelEnv(
        opponent_policy=opponent_policy,
        reward_mode=args.reward_mode,
        max_turns=args.max_turns,
    )

    reset_options = None
    if args.side != "random":
        reset_options = {"learner_side": Side.ONE if args.side == "one" else Side.TWO}

    env.reset(seed=args.seed, options=reset_options)

    print("=" * 70)
    print("KIRI AI: THE DUEL -- manual play session")
    opponent_desc = f"model ({args.model}, {'stochastic' if args.stochastic else 'deterministic'})" if args.opponent == "model" else args.opponent
    print(f"Opponent: {opponent_desc}   Reward mode: {args.reward_mode}   Max turns: {args.max_turns}")
    print("The opponent's special attack shows as 'hidden' until they play it.")
    print(HELP_TEXT)
    print("=" * 70)

    terminated = truncated = False
    info: dict = {}
    while not (terminated or truncated):
        action = prompt_for_action(env)
        if action is None:
            print("\nQuitting.")
            return

        _, reward, terminated, truncated, info = env.step(action)
        print_turn_result(info["reveal_events"])
        print(f"\n  reward this turn: {reward:+.2f}")

    print_state(env)
    if terminated:
        winner = info["winner"]
        print("\nYou win!" if winner is env.learner_side else "\nYou lose.")
    else:
        print(f"\nTruncated after {args.max_turns} turns with no winner (training-only cap, not a rules outcome).")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
