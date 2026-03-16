"""Shared feature computation for all AI players.

opportunity_masks()
-------------------
Fast O(G²) scan — mirrors territory.py's exact scoring precedence:

  1. Full square  (1.0 pt): 3 own corners + 1 empty → score at empty,
                             then SKIP triangle checks for this cell
                             (same as territory.py's `continue`).
  2. Triangles   (0.5 pt each): for each of the four possible triangles in
                             a unit square, if exactly 2 of the 3 triangle
                             corners are own and the third is empty → 0.5 pt.

     The four triangles per unit square (corners tl, tr, bl, br):
       UL (tl, tr, bl)  —  closed by a "/" diagonal between tr and bl
       LR (tr, br, bl)  —  closed by a "/" diagonal between tr and bl
       UR (tl, tr, br)  —  closed by a "\" diagonal between tl and br
       LL (tl, bl, br)  —  closed by a "\" diagonal between tl and br

     A triangle scores when 2 of its 3 corners are own and the 3rd is
     empty.  Placing the empty dot auto-creates any missing connection
     (dots connect at Chebyshev distance 1), so "diagonal already exists"
     and "placement creates the diagonal" are both covered.

Scores accumulate across all unit squares a position belongs to, so a
single move can score e.g. 1.0 (square) + 0.5 (triangle) = 1.5 pts.

Boost formula applied by apply_boost():
    multiplier = 1 + (base_boost − 1) × score
  With AI_CLOSE_BOOST=4.0:
    0.5 pt (triangle)    →  2.5×
    1.0 pt (square)      →  4.0×
    1.5 pt (square+tri)  →  5.5×
    2.0 pt (2 squares)   →  7.0×

Boost magnitudes are tunable via settings.py:
  AI_CLOSE_BOOST — base multiplier for own closing moves (default 4.0)
  AI_BLOCK_BOOST — base multiplier for opponent-blocking (default 3.0)
"""

import numpy as np
import settings as S

# The four triangles per unit square, as offset tuples from (cx, cy):
#   tl=(0,0)  tr=(1,0)  bl=(0,1)  br=(1,1)
_TRIANGLES = (
    ((0, 0), (1, 0), (0, 1)),   # UL: tl, tr, bl  (slash /)
    ((1, 0), (1, 1), (0, 1)),   # LR: tr, br, bl  (slash /)
    ((0, 0), (1, 0), (1, 1)),   # UR: tl, tr, br  (backslash \)
    ((0, 0), (0, 1), (1, 1)),   # LL: tl, bl, br  (backslash \)
)


def opportunity_masks(
    board: dict,
    connections: set,
    player: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (own_scores, opp_scores) — flat float32 arrays of length GRID².

    own_scores[i]  = territory pts player gains by placing at position i
    opp_scores[i]  = territory pts opponent would gain (pts blocked)

    Units: 1.0 per full square, 0.5 per triangle.
    A position accumulates scores from all unit squares it participates in.
    Runs in O(G²) — safe to call every turn.
    """
    n   = S.GRID
    opp = 2 if player == 1 else 1

    own_scores = np.zeros(n * n, dtype=np.float32)
    opp_scores = np.zeros(n * n, dtype=np.float32)

    for cx in range(n - 1):
        for cy in range(n - 1):
            tl = (cx,     cy    )
            tr = (cx + 1, cy    )
            bl = (cx,     cy + 1)
            br = (cx + 1, cy + 1)
            sq = (tl, tr, bl, br)

            own_sq   = sum(1 for c in sq if board.get(c) == player)
            opp_sq   = sum(1 for c in sq if board.get(c) == opp)
            empty_sq = [c for c in sq if c not in board]

            # ── Full square checks (mirror territory.py first-check + continue) ─
            scored_own_full = False
            scored_opp_full = False

            if own_sq == 3 and len(empty_sq) == 1:
                gx, gy = empty_sq[0]
                own_scores[gy * n + gx] += 1.0
                scored_own_full = True

            if opp_sq == 3 and len(empty_sq) == 1:
                gx, gy = empty_sq[0]
                opp_scores[gy * n + gx] += 1.0
                scored_opp_full = True

            # Skip triangle checks for this cell if full square already found
            # (mirrors territory.py's `continue` after detecting a full square)
            if scored_own_full and scored_opp_full:
                continue
            if scored_own_full or scored_opp_full:
                # Only skip triangles for the player whose full square fired
                for (da0, db0), (da1, db1), (da2, db2) in _TRIANGLES:
                    c0 = (cx + da0, cy + db0)
                    c1 = (cx + da1, cy + db1)
                    c2 = (cx + da2, cy + db2)
                    tri    = (c0, c1, c2)
                    empty_t = [c for c in tri if c not in board]
                    if len(empty_t) != 1:
                        continue
                    gx, gy = empty_t[0]
                    if not scored_own_full:
                        own_t = sum(1 for c in tri if board.get(c) == player)
                        if own_t == 2:
                            own_scores[gy * n + gx] += 0.5
                    if not scored_opp_full:
                        opp_t = sum(1 for c in tri if board.get(c) == opp)
                        if opp_t == 2:
                            opp_scores[gy * n + gx] += 0.5
                continue

            # ── Triangle checks (no full square for either player in this cell) ─
            for (da0, db0), (da1, db1), (da2, db2) in _TRIANGLES:
                c0 = (cx + da0, cy + db0)
                c1 = (cx + da1, cy + db1)
                c2 = (cx + da2, cy + db2)
                tri    = (c0, c1, c2)
                empty_t = [c for c in tri if c not in board]
                if len(empty_t) != 1:
                    continue
                gx, gy = empty_t[0]

                own_t = sum(1 for c in tri if board.get(c) == player)
                if own_t == 2:
                    own_scores[gy * n + gx] += 0.5

                opp_t = sum(1 for c in tri if board.get(c) == opp)
                if opp_t == 2:
                    opp_scores[gy * n + gx] += 0.5

    return own_scores, opp_scores


def apply_boost(
    probs: np.ndarray,
    own_scores: np.ndarray,
    opp_scores: np.ndarray,
    legal: np.ndarray,
) -> np.ndarray:
    """Scale move probabilities by territory score, then re-normalise.

        multiplier = 1 + (base_boost − 1) × score

    Only legal positions are boosted.  Returns a new normalised float64 array.
    """
    close_boost = getattr(S, 'AI_CLOSE_BOOST', 4.0)
    block_boost = getattr(S, 'AI_BLOCK_BOOST', 3.0)

    p = probs.astype(np.float64)

    own_legal = own_scores.astype(np.float64) * legal
    opp_legal = opp_scores.astype(np.float64) * legal

    if own_legal.any():
        p *= 1.0 + (close_boost - 1.0) * own_legal
    if opp_legal.any():
        p *= 1.0 + (block_boost - 1.0) * opp_legal

    total = p.sum()
    if total > 0:
        p /= total
    else:
        p = legal.astype(np.float64)
        p /= p.sum()

    return p


# ═══════════════════════════════════════════════════════════════════════════════
# Enclosure potential
# ═══════════════════════════════════════════════════════════════════════════════
#
# The core insight: opportunity_masks sees only *immediate* territory (closing a
# square or triangle right now). A player building a ring around 15 squares
# generates ZERO signal there until the very last closing dot. This blinds all
# three AIs to the actual winning strategy.
#
# enclosure_potential() adds the missing strategic view:
#
#   For each connected component of a player's dots, run a flood fill from
#   the outside of the board treating those dots as walls. Grid positions that
#   the flood fill cannot reach are "trapped" — they would become interior if
#   the ring were completed. The count of trapped positions normalised by
#   (GRID-1)² is the component's *enclosure potential*.
#
#   The per-position arrays assign each board position the potential of the
#   component it belongs to (or the max adjacent component if the position is
#   empty). This gives the neural networks a spatial map of where promising
#   rings are being built.
#
# enclosure_scalars() is a lightweight version returning only (own_max, opp_max)
# — suitable for calling inside MCTS rollout leaf evaluation where per-position
# detail is not needed and speed matters.
#
# Input channel assignment:
#   ch 7 : own enclosure potential map  (values in [0, 1])
#   ch 8 : opp enclosure potential map  (values in [0, 1])
#
# Note: component adjacency is derived directly from the board (same-color dots
# at Chebyshev distance 1 auto-connect), so the `connections` argument is not
# required here — the board alone is sufficient.

def _find_components(board: dict, player: int) -> list[set]:
    """Return list of connected-component sets for *player*.

    Two dots are in the same component when they are at Chebyshev distance 1
    (i.e. they would be auto-connected by the game engine).
    """
    p_dots = {pos for pos, v in board.items() if v == player}
    visited: set = set()
    components: list = []
    for start in p_dots:
        if start in visited:
            continue
        comp: set = set()
        stack = [start]
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            comp.add(cur)
            gx, gy = cur
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    if dx == 0 and dy == 0:
                        continue
                    nb = (gx + dx, gy + dy)
                    if nb in p_dots and nb not in visited:
                        stack.append(nb)
        components.append(comp)
    return components


def _component_potential(comp: set, n: int) -> float:
    """Enclosure potential of a single connected component.

    Flood fills from outside the [0, n-1]² dot grid treating the component's
    dot positions as walls. Returns the fraction of interior (trapped)
    positions relative to (n-1)², which is the total number of unit cells.

    O(n²) — safe to call per component per turn.
    """
    if len(comp) < 3:          # need at least 3 dots to form any enclosure
        return 0.0

    outside: set = set()
    stack = [(-1, -1)]
    outside.add((-1, -1))
    while stack:
        gx, gy = stack.pop()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = gx + dx, gy + dy
            pos = (nx, ny)
            if (-1 <= nx <= n and -1 <= ny <= n
                    and pos not in outside
                    and pos not in comp):
                outside.add(pos)
                stack.append(pos)

    interior = sum(
        1 for gx in range(n) for gy in range(n)
        if (gx, gy) not in comp and (gx, gy) not in outside
    )
    total = (n - 1) ** 2 or 1
    return interior / total


def enclosure_potential(
    board: dict,
    connections: set,      # accepted for API consistency; not used
    player: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (own_pot, opp_pot) — float32 arrays of length GRID².

    own_pot[i] = enclosure potential of the component position i belongs to
                 (or the max potential of adjacent components if i is empty).
    opp_pot[i] = same for the opponent.

    Values are in [0, 1]: 1.0 means the component could enclose the entire
    board if completed; 0 means no enclosure is possible from this component.
    """
    n   = S.GRID
    opp = 2 if player == 1 else 1

    own_pot = np.zeros(n * n, dtype=np.float32)
    opp_pot = np.zeros(n * n, dtype=np.float32)

    for p, arr in ((player, own_pot), (opp, opp_pot)):
        p_dots = {pos for pos, v in board.items() if v == p}
        if not p_dots:
            continue

        comps = _find_components(board, p)

        # Map each dot → its component potential
        dot_pot: dict = {}
        for comp in comps:
            pot = _component_potential(comp, n)
            for dot in comp:
                dot_pot[dot] = pot

        # Assign per-position values
        for gx in range(n):
            for gy in range(n):
                idx = gy * n + gx
                pos = (gx, gy)
                if pos in dot_pot:
                    # Own dot: use its component's potential
                    arr[idx] = dot_pot[pos]
                elif pos not in board:
                    # Empty position: max potential of adjacent own components
                    max_p = 0.0
                    for dx in range(-1, 2):
                        for dy in range(-1, 2):
                            if dx == 0 and dy == 0:
                                continue
                            nb = (gx + dx, gy + dy)
                            p_nb = dot_pot.get(nb)
                            if p_nb is not None and p_nb > max_p:
                                max_p = p_nb
                    arr[idx] = max_p

    return own_pot, opp_pot


def enclosure_scalars(board: dict, player: int) -> tuple[float, float]:
    """Fast version returning only (own_max_potential, opp_max_potential).

    Uses the same BFS per component but skips building per-position arrays.
    Suitable for hot paths like MCTS rollout leaf evaluation.
    """
    n   = S.GRID
    opp = 2 if player == 1 else 1

    def _max_pot(p: int) -> float:
        p_dots = {pos for pos, v in board.items() if v == p}
        if not p_dots:
            return 0.0
        best = 0.0
        for comp in _find_components(board, p):
            pot = _component_potential(comp, n)
            if pot > best:
                best = pot
        return best

    return _max_pot(player), _max_pot(opp)



# ── Arc potential (works for partial rings) ───────────────────────────────────
#
# enclosure_potential / enclosure_scalars use a flood-fill-from-outside approach
# that returns 0 for any ring with a gap — which means they are completely blind
# during the entire ring-building phase.  Only when the last gap is closed does
# the signal fire.
#
# arc_potential estimates the same quantity from the component's bounding box
# and a "completion ratio" (actual dots / expected perimeter dots), so it grows
# continuously as the ring is constructed and is non-zero from the moment 4+
# dots form a 2D spread.
#
# Formula per component:
#   completion  = min(1.0, len(comp) / (2 * (span_x + span_y)))
#   interior    = max(0, span_x - 1) * max(0, span_y - 1)
#   potential   = completion * interior / total_cells
#
# This correctly assigns:
#   - 0 to straight lines  (span_y = 0 → interior = 0)
#   - Small values to L-shapes / short arcs
#   - Growing values to ring arcs as they lengthen
#   - Maximum value to a complete ring (completion ≈ 1.0)

def _arc_potential_comp(comp: set, n: int) -> float:
    """Bounding-box estimate of potential enclosed territory for one component.

    Returns a value in TERRITORY CELL UNITS (like own_s), NOT normalised to [0,1].
    A ring likely to enclose 9 squares returns ~8.0; a straight line returns 0.

    Formula:  completion × interior_cells
      completion   = min(1, actual_dots / expected_perimeter_dots)
      interior     = (span_x-1) × (span_y-1)  — bounding-box interior

    This is intentionally in the same units as opportunity_masks (where 1.0 =
    1 territory square) so the two signals compete fairly in PUCT priors.
    A nearly-complete ring enclosing N squares should score ~N, comparable to
    N immediate small-square closes.  Without this the ring is always dominated.
    """
    if len(comp) < 3:
        return 0.0
    xs      = [p[0] for p in comp]
    ys      = [p[1] for p in comp]
    span_x  = max(xs) - min(xs)
    span_y  = max(ys) - min(ys)
    interior = max(0, span_x - 1) * max(0, span_y - 1)
    if interior == 0:
        return 0.0
    expected_perim = 2 * (span_x + span_y)
    completion     = min(1.0, len(comp) / expected_perim)
    return completion * interior


def arc_potential_scalars(board: dict, player: int) -> tuple[float, float]:
    """Fast (scalar) version: return (own_max_arc, opp_max_arc) normalised to [0,1].

    Divides raw arc potential (in territory cell units) by (n-1)² so the result
    fits in [0,1] for use in _eval_fast blend weights.
    """
    n     = S.GRID
    total = (n - 1) ** 2 or 1
    opp   = 2 if player == 1 else 1

    def _max(p: int) -> float:
        raw = max((_arc_potential_comp(c, n) for c in _find_components(board, p)),
                  default=0.0)
        return min(1.0, raw / total)

    return _max(player), _max(opp)


def arc_potential_map(
    board: dict,
    player: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (own_arc, opp_arc) — flat float32 arrays of length GRID².

    own_arc[i] = arc potential of the component position i belongs to
                 (or max potential of adjacent own components if i is empty).
    opp_arc[i] = same for the opponent.

    Replaces enclosure_potential in MCTS priors — works throughout ring-building
    not only after ring completion.
    """
    n   = S.GRID
    opp = 2 if player == 1 else 1
    own_arr = np.zeros(n * n, dtype=np.float32)
    opp_arr = np.zeros(n * n, dtype=np.float32)

    for p, arr in ((player, own_arr), (opp, opp_arr)):
        comps = _find_components(board, p)
        dot_pot: dict = {}
        for comp in comps:
            pot = _arc_potential_comp(comp, n)
            for dot in comp:
                dot_pot[dot] = pot
        for gx in range(n):
            for gy in range(n):
                idx = gy * n + gx
                pos = (gx, gy)
                if pos in dot_pot:
                    arr[idx] = np.float32(dot_pot[pos])
                elif pos not in board:
                    max_p = 0.0
                    for dx in range(-1, 2):
                        for dy in range(-1, 2):
                            if dx == 0 and dy == 0:
                                continue
                            nb = (gx + dx, gy + dy)
                            p_nb = dot_pot.get(nb)
                            if p_nb is not None and p_nb > max_p:
                                max_p = p_nb
                    arr[idx] = np.float32(max_p)

    return own_arr, opp_arr


# ═══════════════════════════════════════════════════════════════════════════════
# Additional strategic features  (channels 9-14)
# ═══════════════════════════════════════════════════════════════════════════════
#
# ch  9 : own  bridge potential — how much merging two own arcs would add
# ch 10 : opp  bridge potential — same for opponent
# ch 11 : disruption map        — placing our dot reduces opp enc potential by X
# ch 12 : fork map              — placing here creates N simultaneous threats
# ch 13 : game phase            — total_dots / max_dots, broadcast to all positions
# ch 14 : centrality            — geometric centre-distance, constant per grid

import math as _math


# ── Centrality (cached per grid size) ────────────────────────────────────────

_CENTRALITY_CACHE: dict = {}


def get_centrality() -> np.ndarray:
    """Precomputed flat float32 centrality map for the current GRID size.

    Value 1.0 at the centre, 0.0 at the farthest corner.  Cached after first
    call so subsequent calls are O(1).
    """
    n = S.GRID
    if n not in _CENTRALITY_CACHE:
        cx = cy = (n - 1) / 2.0
        max_d = cx * _math.sqrt(2) or 1.0
        arr = np.zeros(n * n, dtype=np.float32)
        for gx in range(n):
            for gy in range(n):
                d = _math.sqrt((gx - cx) ** 2 + (gy - cy) ** 2)
                arr[gy * n + gx] = 1.0 - d / max_d
        _CENTRALITY_CACHE[n] = arr
    return _CENTRALITY_CACHE[n]


# ── Component interior finder (used by disruption map) ───────────────────────

def _component_interior(comp: set, n: int) -> set:
    """Return positions inside *comp* (not reachable from outside).

    Uses the same flood-fill as _component_potential but returns the actual
    interior set rather than its normalised count.
    """
    outside: set = set()
    stack = [(-1, -1)]
    outside.add((-1, -1))
    while stack:
        gx, gy = stack.pop()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = gx + dx, gy + dy
            pos = (nx, ny)
            if (-1 <= nx <= n and -1 <= ny <= n
                    and pos not in outside and pos not in comp):
                outside.add(pos)
                stack.append(pos)
    return {
        (gx, gy)
        for gx in range(n) for gy in range(n)
        if (gx, gy) not in comp and (gx, gy) not in outside
    }


# ── Bridge potential ──────────────────────────────────────────────────────────

def bridge_potential(
    board: dict,
    connections: set,
    player: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (own_bridge, opp_bridge) — flat float32 arrays of length GRID².

    own_bridge[i] = extra enclosure potential that placing at position i would
    unlock by bridging two separate own ring arcs into one larger ring.

    Concretely:
        merged_pot − max(individual pots of adjacent components)
    so the value is 0 when the position is only adjacent to one component (no
    bridging benefit beyond what enc_potential already encodes), and positive
    only where two arcs can be joined to form a larger ring.
    """
    n   = S.GRID
    opp = 2 if player == 1 else 1

    own_bridge = np.zeros(n * n, dtype=np.float32)
    opp_bridge = np.zeros(n * n, dtype=np.float32)

    for p, arr in ((player, own_bridge), (opp, opp_bridge)):
        comps = _find_components(board, p)
        if len(comps) < 2:
            continue

        # Use _arc_potential_comp instead of _component_potential so bridge
        # values are non-zero even when arcs aren't yet complete rings.
        # Values are in territory-cell units (not normalised).
        comp_pots = [_arc_potential_comp(c, n) for c in comps]

        for gx in range(n):
            for gy in range(n):
                pos = (gx, gy)
                if pos in board:
                    continue
                # Which components are Chebyshev-adjacent to this position?
                adj_indices = []
                for i, comp in enumerate(comps):
                    for dx in range(-1, 2):
                        for dy in range(-1, 2):
                            if dx == 0 and dy == 0:
                                continue
                            if (gx + dx, gy + dy) in comp:
                                adj_indices.append(i)
                                break
                        else:
                            continue
                        break

                if len(adj_indices) < 2:
                    continue  # no bridging opportunity

                # Merged potential of all adjacent components + this position
                merged: set = {pos}
                for i in adj_indices:
                    merged |= comps[i]
                merged_pot = _arc_potential_comp(merged, n)

                # Bridge benefit = extra potential over best individual arc
                best_individual = max(comp_pots[i] for i in adj_indices)
                benefit = max(0.0, merged_pot - best_individual)
                arr[gy * n + gx] = benefit

    return own_bridge, opp_bridge


# ── Disruption map ────────────────────────────────────────────────────────────

def disruption_map(
    board: dict,
    connections: set,
    player: int,
) -> np.ndarray:
    """Return a flat float32 array of length GRID².

    disruption[i] = fraction of the opponent's maximum enclosure potential
    that placing *player*'s dot at position i would eliminate.

    Implementation: positions that are in the INTERIOR of an opponent's ring
    (i.e. the flood fill from outside cannot reach them) are high-value
    disruption targets — placing there steals territory the opponent was
    building toward.  Value is proportional to the ring's potential so that
    disrupting a large ring scores more than disrupting a tiny one.
    """
    n   = S.GRID
    opp = 2 if player == 1 else 1

    result = np.zeros(n * n, dtype=np.float32)

    opp_comps = _find_components(board, opp)
    if not opp_comps:
        return result

    for comp in opp_comps:
        pot = _component_potential(comp, n)
        if pot == 0.0:
            continue
        interior = _component_interior(comp, n)
        for pos in interior:
            gx, gy = pos
            if pos not in board:
                idx = gy * n + gx
                # Accumulate (a position may be inside multiple rings)
                result[idx] = min(1.0, result[idx] + pot)

    return result


# ── Fork map ─────────────────────────────────────────────────────────────────

def fork_map(board: dict, player: int) -> np.ndarray:
    """Return a flat float32 array of length GRID².

    fork_map[i] = normalised count of *simultaneous future threats* that
    placing at position i would create.

    A future threat is a unit square that, after placing at i, has exactly
    3 own corners and 1 different empty corner — the opponent can only answer
    one fork, guaranteeing us a score.  Value is clipped to [0, 1] by
    dividing by 4 (max possible forks per position).
    """
    n      = S.GRID
    result = np.zeros(n * n, dtype=np.float32)

    for gx in range(n):
        for gy in range(n):
            pos = (gx, gy)
            if pos in board:
                continue
            forks = 0
            # Check each unit square that contains pos as a corner
            for cx in range(gx - 1, gx + 1):
                for cy in range(gy - 1, gy + 1):
                    if not (0 <= cx < n - 1 and 0 <= cy < n - 1):
                        continue
                    corners = [(cx, cy), (cx+1, cy), (cx, cy+1), (cx+1, cy+1)]
                    # Count own dots in the OTHER three corners (not pos)
                    other = [c for c in corners if c != pos]
                    own_other = sum(1 for c in other if board.get(c) == player)
                    empty_other = [c for c in other if c not in board]
                    # After placing pos: square has own_other+1 own, len(empty_other) empty
                    # → creates a future threat when own_other == 2 and len(empty_other) == 1
                    if own_other == 2 and len(empty_other) == 1:
                        forks += 1
            result[gy * n + gx] = min(float(forks), 4.0) / 4.0

    return result



# ── Territory setup map ───────────────────────────────────────────────────────

def close_setup_map(board: dict, player: int) -> np.ndarray:
    """Return a flat float32 array of length GRID².

    close_setup_map[i] = number of unit squares that would have exactly
    3 own corners after placing at position i (creating a next-turn territory
    threat), normalised by 4.

    This fills the gap between:
      - opportunity_masks (scores positions where a square CAN BE CLOSED NOW)
      - enclosure_potential (scores ring-extension, far-future)

    By adding this to PUCT priors, MCTS learns to create territory setups
    instead of only extending ring arcs.  Without it both AIs converge to
    "extend my line, block opponent's line" and never produce unit-square
    territory — even though both are playing rationally from their perspective.
    """
    n   = S.GRID
    opp = 2 if player == 1 else 1
    result = np.zeros(n * n, dtype=np.float32)

    for gx in range(n):
        for gy in range(n):
            pos = (gx, gy)
            if pos in board:
                continue
            count = 0
            for cx in range(max(0, gx - 1), min(n - 1, gx + 1)):
                for cy in range(max(0, gy - 1), min(n - 1, gy + 1)):
                    corners = [(cx,cy),(cx+1,cy),(cx,cy+1),(cx+1,cy+1)]
                    other   = [c for c in corners if c != pos]
                    # After placing at pos: would this square have 3 own + 1 empty?
                    own_cnt = sum(1 for c in other if board.get(c) == player)
                    opp_cnt = sum(1 for c in other if board.get(c) == opp)
                    empty_cnt = sum(1 for c in other if c not in board)
                    if own_cnt == 2 and opp_cnt == 0 and empty_cnt == 1:
                        count += 1
            result[gy * n + gx] = min(float(count), 4.0) / 4.0

    return result


# ── Convenience: all 6 new channels at once ─────────────────────────────────

def strategic_channels(
    board: dict,
    connections: set,
    player: int,
    total_dots: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (own_bridge, opp_bridge, disrupt, fork, phase, centrality).

    All arrays are flat float32 of length GRID².
    Calling this single function avoids redundant component discovery.
    """
    n          = S.GRID
    max_dots   = n * n
    phase_val  = float(total_dots) / max(max_dots, 1)
    phase_arr  = np.full(n * n, phase_val, dtype=np.float32)
    cent       = get_centrality()
    own_br, opp_br = bridge_potential(board, connections, player)
    disrupt    = disruption_map(board, connections, player)
    fork       = fork_map(board, player)
    return own_br, opp_br, disrupt, fork, phase_arr, cent
