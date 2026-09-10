"""
Turn resolution for Kiri Ai: The Duel.

Builds on state.py's data model to implement what happens once both
players' two chosen cards for a turn are revealed. Given a GameState
and each player's already-chosen, already-ordered pair of cards for
the turn, it returns the resulting GameState plus a log of what happened.
It does not enumerate legal actions for a policy to choose from.

Resolution never raises for a rules-legal card that simply fails to do
anything on the board, which results in a no-op, reported as an event
with a false flag, per the ruleset's "illegal movement is selectable but
resolves to nothing" model. `validate_turn_plan` raises only for what the
ruleset treats as illegal to select: an unavailable card (on cooldown, or a
special already spent), playing both ends of one physical card, or a
stance-gated card whose stance requirement isn't met at the relevant
moment (see the ruleset's Card 2 stance-gate timing rule).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, FrozenSet, List, Optional, Tuple, Union

from state import (
    ATTACK_CARDS,
    ATTACK_TARGET_OFFSETS,
    BOARD_MAX,
    BOARD_MIN,
    CARD_PAIRS,
    Card,
    GameState,
    MOVEMENT_CARDS,
    PlayerState,
    POST_ATTACK_STANCE_SWAP,
    REQUIRED_STANCE,
    Side,
    Stance,
    STANDARD_ATTACKS,
)


# A player's committed plan for a turn: exactly two cards, in the order
# they chose to reveal them. Index 0 resolves in Reveal 1, index 1 in
# Reveal 2 (skipped entirely if the game ends after Reveal 1).
TurnPlan = Tuple[Card, Card]


# --- Events -------------------------------------------------------------
# A structured log of what happened during resolution, for replay,
# debugging, and RL reward shaping.

@dataclass(frozen=True)
class MoveEvent:
    side: Side
    card: Card
    from_position: int
    to_position: int
    moved: bool  # False if the card was legally selected but no-op'd on the board (wall, behind-opponent, pass-through, or a simultaneous crossing veto).


@dataclass(frozen=True)
class StanceChangeEvent:
    side: Side
    card: Card
    from_stance: Stance
    to_stance: Stance


@dataclass(frozen=True)
class AttackEvent:
    side: Side  # the attacker
    card: Card
    target_squares: FrozenSet[int]  # empty for Counterattack, which has no targets of its own
    hit: bool  # True iff this attack landed on the defender
    countered: bool  # True iff this attack would have landed but was redirected by the defender's Counterattack instead
    would_have_hit: bool  # True iff the defender's position was in target_squares, regardless of hit -- i.e. distinguishes a clean miss (False) from a mutual-cancellation bounce (True but hit is False and countered is False)


@dataclass(frozen=True)
class SpecialConsumedEvent:
    side: Side
    card: Card


@dataclass(frozen=True)
class HitEvent:
    side: Side  # who got hit
    hits_taken_after: int


@dataclass(frozen=True)
class GameEndEvent:
    winner: Optional[Side]


Event = Union[
    MoveEvent, StanceChangeEvent, AttackEvent, SpecialConsumedEvent, HitEvent, GameEndEvent,
]


# --- Selection-time validation -------------------------------------------

def _slot_of(card: Card) -> FrozenSet[Card]:
    """The physical card `card` belongs to: its pair, if it's one end of a
    double-sided card, else just itself. Two selected cards from the same
    slot means playing both ends of one physical card, which is illegal."""
    for pair in CARD_PAIRS:
        if card in pair:
            return pair
    return frozenset({card})


def stance_after_playing(stance: Stance, card: Card) -> Stance:
    """
    The stance a player will be in immediately after playing `card`, for
    the cards that affect a player's own stance (Stance Change flips it
    unconditionally; Zan-tetsu/Kesa swap to a fixed stance "hit or miss").
    Every other card leaves stance unchanged. This is deterministic and
    depends only on the player's own card, never the opponent's
    simultaneous choice -- which is what makes it safe to use for
    checking Card 2's stance-gate at selection time (see module docstring,
    ruling 1).
    """
    if card in POST_ATTACK_STANCE_SWAP:
        return POST_ATTACK_STANCE_SWAP[card]
    if card is Card.STANCE_CHANGE:
        return stance.opposite()
    return stance


def _plan_violation(player: PlayerState, plan: TurnPlan, available: FrozenSet[Card]) -> Optional[str]:
    """None if `plan` is legal for `player` to select given their
    currently `available` cards, else a human-readable reason it isn't.
    Shared by `validate_turn_plan` (raises) and `legal_turn_plans`
    (filters), so the two can never drift apart."""
    card1, card2 = plan
    if card1 not in available:
        return f"{card1} is not available to {player.side} this turn"
    if card2 not in available:
        return f"{card2} is not available to {player.side} this turn"
    if _slot_of(card1) == _slot_of(card2):
        return (
            f"{card1} and {card2} are two ends of the same physical card; "
            "only one may be played per turn"
        )

    required1 = REQUIRED_STANCE.get(card1)
    if required1 is not None and player.stance is not required1:
        return f"{card1} requires {required1} stance; {player.side} is in {player.stance}"

    stance_at_card2 = stance_after_playing(player.stance, card1)
    required2 = REQUIRED_STANCE.get(card2)
    if required2 is not None and stance_at_card2 is not required2:
        return (
            f"{card2} requires {required2} stance; {player.side} would be in "
            f"{stance_at_card2} after playing {card1}"
        )
    return None


def validate_turn_plan(player: PlayerState, plan: TurnPlan) -> None:
    """
    Raise ValueError if `plan` is not a legal pair of cards for `player`
    to select this turn. Checks only selection-time legality (available
    cards, physical-card-pair exclusivity, stance-gating); it does not
    predict whether the resulting moves will succeed on the board.
    """
    violation = _plan_violation(player, plan, player.available_cards())
    if violation is not None:
        raise ValueError(violation)


def legal_turn_plans(player: PlayerState) -> List[TurnPlan]:
    """
    Every (card1, card2) plan `player` could legally select this turn, in
    reveal order -- i.e. every plan `validate_turn_plan` would accept.

    Order matters: if card1 changes the player's own stance (Stance
    Change, or the unconditional post-attack swap on Zan-tetsu/Kesa),
    the same two cards can be legal in one order and illegal in the
    other, since card2's stance-gate is checked against the stance the
    player will be in *after* card1 (see module docstring, ruling 1). So
    both orderings of every eligible pair are checked independently.
    """
    available = player.available_cards()
    return [
        (card1, card2)
        for card1 in available
        for card2 in available
        if _plan_violation(player, (card1, card2), available) is None
    ]


# --- Movement resolution --------------------------------------------------

_MOVEMENT_TIER: Dict[Card, int] = {
    Card.CHARGE: 0,
    Card.FORWARD: 1,
    Card.BACKWARD: 1,
    Card.STANCE_CHANGE: 2,
}


def _move_delta(card: Card, side: Side) -> int:
    d = side.forward_direction
    if card is Card.FORWARD:
        return d
    if card is Card.BACKWARD:
        return -d
    if card is Card.CHARGE:
        return 2 * d
    return 0


def _path_squares(start: int, destination: int) -> List[int]:
    """Squares strictly between `start` and `destination`, exclusive of
    both endpoints -- what a move must not "pass through"."""
    if destination > start:
        return list(range(start + 1, destination))
    if destination < start:
        return list(range(destination + 1, start))
    return []


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _is_move_legal(
    mover_side: Side, start: int, destination: int, opponent_ref_position: int, *, waive_behind: bool = False
) -> bool:
    """
    Whether a move from `start` to `destination` is legal given the
    opponent's reference position (their position at the moment this
    move is checked against -- pre-step for simultaneous same-tier
    moves, current/actual otherwise). Landing exactly on the opponent is
    a tie and always allowed; ending up past them, or passing through
    their square en route, is not -- unless `waive_behind` is set, in
    which case the "ending up past them" check is skipped (board edges
    and passing through are still always enforced). Used for the
    started-tied case in `_apply_simultaneous_movement`; see there.
    """
    if not (BOARD_MIN <= destination <= BOARD_MAX):
        return False
    d = mover_side.forward_direction
    if not waive_behind and (destination - opponent_ref_position) * d > 0:
        return False
    if opponent_ref_position in _path_squares(start, destination):
        return False
    return True


def _resolve_move(
    mover_side: Side, start: int, full_delta: int, opponent_ref: int, *, waive_behind: bool = False
) -> Tuple[int, bool]:
    """
    Resolve a movement card's destination: try the full distance first,
    and if that's illegal, progressively shorter partial moves in the
    same direction (ruling 5) -- e.g. a Charge that's fully blocked (by
    the opponent or a wall) advances 1 square instead if that's legal,
    rather than not moving at all. Only Charge is affected in practice,
    since Forward/Backward already move exactly 1 square. Board edges,
    pass-through, and the behind-opponent check (unless waived) are
    enforced at every distance tried. Returns (final_position, moved).
    """
    if full_delta == 0:
        return start, False
    step = 1 if full_delta > 0 else -1
    for distance in range(abs(full_delta), 0, -1):
        destination = start + step * distance
        if _is_move_legal(mover_side, start, destination, opponent_ref, waive_behind=waive_behind):
            return destination, True
    return start, False


def _apply_single_movement(state: GameState, side: Side, card: Card, opponent_ref: int) -> Tuple[GameState, List[Event]]:
    player = state.player(side)

    if card is Card.STANCE_CHANGE:
        new_stance = player.stance.opposite()
        state = state.with_player(side, replace(player, stance=new_stance))
        return state, [StanceChangeEvent(side=side, card=card, from_stance=player.stance, to_stance=new_stance)]

    final_position, moved = _resolve_move(side, player.position, _move_delta(card, side), opponent_ref)

    state = state.with_player(side, replace(player, position=final_position))
    event = MoveEvent(side=side, card=card, from_position=player.position, to_position=final_position, moved=moved)
    return state, [event]


def _apply_simultaneous_movement(
    state: GameState, side_a: Side, card_a: Card, side_b: Side, card_b: Card
) -> Tuple[GameState, List[Event]]:
    """Both players played a same-priority-tier movement card this reveal
    (Charge/Charge, Stance Change/Stance Change, or any Forward/Backward
    combination -- see module docstring, ruling 2)."""
    if card_a is Card.STANCE_CHANGE and card_b is Card.STANCE_CHANGE:
        player_a, player_b = state.player(side_a), state.player(side_b)
        state, ev_a = _apply_single_movement(state, side_a, card_a, opponent_ref=player_b.position)
        state, ev_b = _apply_single_movement(state, side_b, card_b, opponent_ref=player_a.position)
        return state, ev_a + ev_b

    player_a, player_b = state.player(side_a), state.player(side_b)
    delta_a = _move_delta(card_a, side_a)
    delta_b = _move_delta(card_b, side_b)
    cand_a = player_a.position + delta_a
    cand_b = player_b.position + delta_b

    if player_a.position == player_b.position:
        # Started this reveal tied on the same square (ruling 4): a mover
        # advancing further in their own forward direction would normally
        # look like ending up "past" the opponent's pre-step position, but
        # if the opponent is simultaneously vacating that same square too,
        # nobody's actually walking through anyone. Board edges and
        # passing through an unrelated square are still always enforced
        # (waive_behind only lifts the tie-specific "behind" check); the
        # waiver for each side is gated on the OTHER side clearing those
        # same edge/pass-through checks, so a move that would've been
        # wall-blocked regardless still blocks the other side's waiver too
        # -- this is what keeps the two sides' legality checks from
        # circularly depending on each other. Ruling 5's partial-move
        # fallback (below) doesn't extend into this tied branch: a Charge
        # that's still blocked here even with the ruling-4 waiver stays a
        # full no-op rather than also trying a 1-square fallback -- doing
        # that would need its own well-defined resolution order between
        # the two sides' interdependent checks, which wasn't worth the
        # added complexity for this narrow, untested intersection.
        legal_a = _is_move_legal(side_a, player_a.position, cand_a, player_b.position)
        legal_b = _is_move_legal(side_b, player_b.position, cand_b, player_a.position)
        clear_a = _is_move_legal(side_a, player_a.position, cand_a, player_b.position, waive_behind=True)
        clear_b = _is_move_legal(side_b, player_b.position, cand_b, player_a.position, waive_behind=True)
        legal_a = legal_a or (clear_a and clear_b)
        legal_b = legal_b or (clear_a and clear_b)
        moved_a, moved_b = legal_a, legal_b
        final_a = cand_a if moved_a else player_a.position
        final_b = cand_b if moved_b else player_b.position

        # A tie means there's no established relative order to "cross" in
        # the sign_before/sign_after sense the non-tied branch below uses
        # -- but the game's one hard global invariant (Side.ONE's position
        # can never exceed Side.TWO's -- they started at opposite ends and
        # can never pass each other) still has to hold on the OUTCOME. If
        # both movers are waived through and each heads off in their own
        # forward direction -- which, for two opposite-facing sides, are
        # genuinely opposite real-board directions -- that's exactly the
        # same "swap through each other" problem the crossing veto exists
        # to prevent, just reached from a tied start instead of adjacent
        # squares. Both moves no-op instead of landing in an impossible
        # crossed state.
        if moved_a and moved_b:
            one_final = final_a if side_a is Side.ONE else final_b
            two_final = final_b if side_a is Side.ONE else final_a
            if one_final > two_final:
                moved_a = moved_b = False
                final_a, final_b = player_a.position, player_b.position
    else:
        # Ruling 5: each mover tries the full distance first, then
        # progressively shorter partial moves (matters only for Charge) --
        # resolved independently against each other's pre-step position,
        # same as the crossing rule's own baseline convention. The
        # crossing veto is then checked against whatever destination each
        # one actually resolved to (full or partial), not just the full
        # candidate -- otherwise two movers who are each individually
        # blocked at full distance could still "swap through" each other
        # via their partial fallbacks (e.g. both Charge while exactly one
        # space apart), which is exactly what the crossing rule exists to
        # prevent, and both moves correctly no-op instead.
        final_a, moved_a = _resolve_move(side_a, player_a.position, delta_a, player_b.position)
        final_b, moved_b = _resolve_move(side_b, player_b.position, delta_b, player_a.position)

        if moved_a and moved_b:
            sign_before = _sign(player_a.position - player_b.position)
            sign_after = _sign(final_a - final_b)
            if sign_before != 0 and sign_after != 0 and sign_before != sign_after:
                moved_a = moved_b = False
                final_a, final_b = player_a.position, player_b.position

    state = state.with_player(side_a, replace(player_a, position=final_a))
    state = state.with_player(side_b, replace(player_b, position=final_b))

    events: List[Event] = [
        MoveEvent(side=side_a, card=card_a, from_position=player_a.position, to_position=final_a, moved=moved_a),
        MoveEvent(side=side_b, card=card_b, from_position=player_b.position, to_position=final_b, moved=moved_b),
    ]
    return state, events


def _resolve_movement_phase(state: GameState, cards: Dict[Side, Card]) -> Tuple[GameState, List[Event]]:
    movers = {side: card for side, card in cards.items() if card in MOVEMENT_CARDS}
    if not movers:
        return state, []

    if len(movers) == 1:
        (side, card), = movers.items()
        return _apply_single_movement(state, side, card, opponent_ref=state.player(side.other).position)

    (side_a, card_a), (side_b, card_b) = movers.items()
    tier_a, tier_b = _MOVEMENT_TIER[card_a], _MOVEMENT_TIER[card_b]

    if tier_a == tier_b:
        return _apply_simultaneous_movement(state, side_a, card_a, side_b, card_b)

    first, first_card, second, second_card = (
        (side_a, card_a, side_b, card_b) if tier_a < tier_b else (side_b, card_b, side_a, card_a)
    )
    state, ev_first = _apply_single_movement(state, first, first_card, opponent_ref=state.player(first.other).position)
    state, ev_second = _apply_single_movement(state, second, second_card, opponent_ref=state.player(second.other).position)
    return state, ev_first + ev_second


# --- Attack resolution -----------------------------------------------------

def _apply_hit(state: GameState, side: Side, events: List[Event]) -> GameState:
    player = state.player(side)
    new_hits = player.hits_taken + 1
    events.append(HitEvent(side=side, hits_taken_after=new_hits))
    return state.with_player(side, replace(player, hits_taken=new_hits))


def _finish_attack(state: GameState, side: Side, card: Card, events: List[Event]) -> GameState:
    """
    Housekeeping that happens whenever an attack card is played,
    regardless of hit, miss, or being countered: a special attack is
    permanently consumed the instant it's played, and Zan-tetsu/Kesa
    swap the player into their fixed post-attack stance "hit or miss".
    """
    if card in {Card.SPECIAL_COUNTERATTACK, Card.SPECIAL_ZANTETSU, Card.SPECIAL_KESA}:
        player = state.player(side)
        state = state.with_player(side, replace(player, special_used=True))
        events.append(SpecialConsumedEvent(side=side, card=card))

    swap_to = POST_ATTACK_STANCE_SWAP.get(card)
    if swap_to is not None:
        player = state.player(side)
        if player.stance is not swap_to:
            events.append(StanceChangeEvent(side=side, card=card, from_stance=player.stance, to_stance=swap_to))
            state = state.with_player(side, replace(player, stance=swap_to))

    return state


def _target_squares(state: GameState, side: Side, card: Card) -> FrozenSet[int]:
    attacker = state.player(side)
    return frozenset(attacker.position + o * side.forward_direction for o in ATTACK_TARGET_OFFSETS[card])


def _resolve_solo_attack(state: GameState, side: Side, card: Card) -> Tuple[GameState, List[Event]]:
    """The opponent didn't attack this reveal (they played a movement
    card, already resolved), so this attacker faces no counter-hit risk
    -- except Counterattack, which has nothing to react to and is simply
    wasted."""
    events: List[Event] = []

    if card is Card.SPECIAL_COUNTERATTACK:
        events.append(AttackEvent(side=side, card=card, target_squares=frozenset(), hit=False, countered=False, would_have_hit=False))
        state = _finish_attack(state, side, card, events)
        return state, events

    targets = _target_squares(state, side, card)
    defender = state.player(side.other)
    hit = defender.position in targets
    events.append(AttackEvent(side=side, card=card, target_squares=targets, hit=hit, countered=False, would_have_hit=hit))

    if hit:
        state = _apply_hit(state, side.other, events)
    state = _finish_attack(state, side, card, events)
    return state, events


def _resolve_mutual_attack(
    state: GameState, side_a: Side, card_a: Card, side_b: Side, card_b: Card
) -> Tuple[GameState, List[Event]]:
    events: List[Event] = []

    counter_side: Optional[Side] = None
    if card_a is Card.SPECIAL_COUNTERATTACK:
        counter_side = side_a
    elif card_b is Card.SPECIAL_COUNTERATTACK:
        counter_side = side_b

    if counter_side is not None:
        attacker_side = counter_side.other
        attacker_card = card_b if counter_side is side_a else card_a

        targets = _target_squares(state, attacker_side, attacker_card)
        defender = state.player(counter_side)
        would_hit = defender.position in targets

        events.append(AttackEvent(side=attacker_side, card=attacker_card, target_squares=targets, hit=False, countered=would_hit, would_have_hit=would_hit))
        events.append(AttackEvent(side=counter_side, card=Card.SPECIAL_COUNTERATTACK, target_squares=frozenset(), hit=would_hit, countered=False, would_have_hit=would_hit))

        if would_hit:
            state = _apply_hit(state, attacker_side, events)

        state = _finish_attack(state, attacker_side, attacker_card, events)
        state = _finish_attack(state, counter_side, Card.SPECIAL_COUNTERATTACK, events)
        return state, events

    targets_a = _target_squares(state, side_a, card_a)
    targets_b = _target_squares(state, side_b, card_b)
    player_a, player_b = state.player(side_a), state.player(side_b)

    b_in_a = player_b.position in targets_a
    a_in_b = player_a.position in targets_b
    hit_a_lands = b_in_a and not a_in_b
    hit_b_lands = a_in_b and not b_in_a

    events.append(AttackEvent(side=side_a, card=card_a, target_squares=targets_a, hit=hit_a_lands, countered=False, would_have_hit=b_in_a))
    events.append(AttackEvent(side=side_b, card=card_b, target_squares=targets_b, hit=hit_b_lands, countered=False, would_have_hit=a_in_b))

    if hit_a_lands:
        state = _apply_hit(state, side_b, events)
    if hit_b_lands:
        state = _apply_hit(state, side_a, events)

    state = _finish_attack(state, side_a, card_a, events)
    state = _finish_attack(state, side_b, card_b, events)
    return state, events


def _resolve_attack_phase(state: GameState, cards: Dict[Side, Card]) -> Tuple[GameState, List[Event]]:
    attackers = {side: card for side, card in cards.items() if card in ATTACK_CARDS}
    if not attackers:
        return state, []

    if len(attackers) == 1:
        (side, card), = attackers.items()
        return _resolve_solo_attack(state, side, card)

    (side_a, card_a), (side_b, card_b) = attackers.items()
    return _resolve_mutual_attack(state, side_a, card_a, side_b, card_b)


# --- Turn resolution ------------------------------------------------------

def _resolve_reveal(state: GameState, cards: Dict[Side, Card]) -> Tuple[GameState, List[Event], bool]:
    """Resolve one reveal: movement (by priority tier) fully resolves
    first, then attacks are checked against the resulting positions.
    Returns the new state, the events produced, and whether the game
    ended (a player reached 2 hits)."""
    state, move_events = _resolve_movement_phase(state, cards)
    state, attack_events = _resolve_attack_phase(state, cards)
    events = move_events + attack_events

    if state.is_over:
        events.append(GameEndEvent(winner=state.winner))
        return state, events, True
    return state, events, False


def _apply_end_of_turn_cooldowns(state: GameState, plan_one: TurnPlan, plan_two: TurnPlan) -> GameState:
    """
    Standard attacks go on a strict 1-turn cooldown after being played.
    This *replaces* (not adds to) each player's cooldown set: a card that
    was on cooldown coming into this turn has now sat out its one
    required turn and is back to normal, whether or not it was played
    again this turn (it wasn't available to select again anyway).
    """
    for side, plan in ((Side.ONE, plan_one), (Side.TWO, plan_two)):
        played_standard = frozenset(c for c in plan if c in STANDARD_ATTACKS)
        player = state.player(side)
        state = state.with_player(side, replace(player, attacks_on_cooldown=played_standard))
    return state


def resolve_turn(state: GameState, plan_one: TurnPlan, plan_two: TurnPlan) -> Tuple[GameState, List[List[Event]]]:
    """
    Resolve one full turn from both players' committed plans.

    `plan_one` and `plan_two` are each a (first_card, second_card) tuple:
    the two cards that player selected for this turn, in the reveal order
    they chose (chosen freely and independently by each player -- see
    ruleset's open house-rule decision on reveal order). Both plans are
    validated against the state as of the start of the turn (see
    `validate_turn_plan`) before anything resolves.

    Card 1 for both sides is revealed and resolved simultaneously; then,
    unless the game already ended, Card 2 is revealed and resolved the
    same way. Returns the resulting state and the events from each
    reveal that actually happened, as a list of 1 or 2 sublists (reveal 2
    is omitted entirely, not returned as an empty list, if the match
    ended after reveal 1) -- a caller that doesn't care which reveal
    produced what can flatten with
    `[event for reveal in reveals for event in reveal]`.
    """
    if state.is_over:
        raise ValueError("cannot resolve a turn: the game is already over")

    validate_turn_plan(state.player_one, plan_one)
    validate_turn_plan(state.player_two, plan_two)

    reveals: List[List[Event]] = []

    state, reveal_events, ended = _resolve_reveal(state, {Side.ONE: plan_one[0], Side.TWO: plan_two[0]})
    reveals.append(reveal_events)

    if not ended:
        state, reveal_events, ended = _resolve_reveal(state, {Side.ONE: plan_one[1], Side.TWO: plan_two[1]})
        reveals.append(reveal_events)

    if ended:
        return state, reveals

    state = _apply_end_of_turn_cooldowns(state, plan_one, plan_two)
    state = replace(state, turn_number=state.turn_number + 1)
    return state, reveals
