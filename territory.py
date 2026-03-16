"""Territory detection via SCALE=6 raster flood fill.

Each player's dots are expanded to 3×3 wall blocks so that connection
junctions are always sealed. Diagonal connections also receive L-corner
fills so that 4-directional flood fill cannot slip through diagonal gaps.

Two independent flood fills (one per player) determine which unit cells
are enclosed. Player 1 wins any cell in overlap when player 2's dots are
trapped inside player 1's walls, and vice versa.
"""

import settings as S
import state

# Grid scaling factor used for rasterisation.
SCALE = 6


def _build_walls(player: int) -> set:
    """Return the set of scaled-grid wall pixels for *player*.

    Every dot becomes a 3×3 block; every connection is rasterised with
    L-corner fills to seal diagonal gaps.
    """
    walls = set()

    sz_inner = SCALE * (S.GRID - 1)
    for gx, gy in state.board:
        if state.board[(gx, gy)] != player:
            continue
        sx, sy = SCALE * gx, SCALE * gy
        for ddx in range(-1, 2):
            for ddy in range(-1, 2):
                wx, wy = sx + ddx, sy + ddy
                # Clamp to the raster interior so the 3x3 expansion never
                # bleeds into the flood-fill border corridor (-1 .. sz+1).
                if 0 <= wx <= sz_inner and 0 <= wy <= sz_inner:
                    walls.add((wx, wy))

    for conn in state.connections:
        a, b = tuple(conn)
        if state.board[a] != player:
            continue
        ax, ay = SCALE * a[0], SCALE * a[1]
        bx, by = SCALE * b[0], SCALE * b[1]
        dx, dy = bx - ax, by - ay
        steps = max(abs(dx), abs(dy))
        prev = (ax, ay)
        for i in range(1, steps + 1):
            cur = (ax + dx * i // steps, ay + dy * i // steps)
            walls.add(cur)
            px, py = prev
            cx2, cy2 = cur
            if px != cx2 and py != cy2:   # diagonal step: seal L-corners
                walls.add((cx2, py))
                walls.add((px, cy2))
            prev = cur

    return walls


def _flood_outside(walls: set, sz: int) -> set:
    """4-directional flood fill from (-1, -1) — returns all reachable pixels."""
    outside: set = set()
    stack = [(-1, -1)]
    outside.add((-1, -1))

    while stack:
        sx, sy = stack.pop()
        for ddx, ddy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = sx + ddx, sy + ddy
            pos = (nx, ny)
            if (
                -1 <= nx <= sz + 1
                and -1 <= ny <= sz + 1
                and pos not in outside
                and pos not in walls
            ):
                outside.add(pos)
                stack.append(pos)

    return outside


def _remove_encircled(winner: int, winner_outside: set, winner_walls: set) -> None:
    """Remove all dots inside the winner's enclosure except the ring itself.

    Kept dots: winner's own dots that are part of the wall (i.e. the ring).
    Removed:   loser's dots and any winner-interior dots.
    """
    to_remove = [
        (gx, gy)
        for gx, gy in state.board
        if (SCALE * gx, SCALE * gy) not in winner_outside
        and not (
            state.board[(gx, gy)] == winner
            and (SCALE * gx, SCALE * gy) in winner_walls
        )
    ]

    for dot in to_remove:
        del state.board[dot]

    dead = frozenset(to_remove)
    for conn in list(state.connections):
        if conn & dead:
            state.connections.discard(conn)


def recompute_territories() -> None:
    """Recompute territories, scores, interior dots and interior connections."""

    if not state.board:
        state.territories = {}
        state.scores = {1: 0.0, 2: 0.0}
        state.interior_dots = set()
        state.interior_conns = set()
        return

    sz = SCALE * (S.GRID - 1)

    walls1 = _build_walls(1)
    outside1 = _flood_outside(walls1, sz)
    walls2 = _build_walls(2)
    outside2 = _flood_outside(walls2, sz)

    # ── Encirclement: remove trapped dots ────────────────────────────────────
    p2_trapped = [
        d for d in state.board
        if state.board[d] == 2
        and (SCALE * d[0], SCALE * d[1]) not in outside1
    ]
    p1_trapped = [
        d for d in state.board
        if state.board[d] == 1
        and (SCALE * d[0], SCALE * d[1]) not in outside2
    ]

    if p2_trapped:
        _remove_encircled(1, outside1, walls1)
        walls1 = _build_walls(1)
        outside1 = _flood_outside(walls1, sz)
        walls2 = _build_walls(2)
        outside2 = _flood_outside(walls2, sz)
    elif p1_trapped:
        _remove_encircled(2, outside2, walls2)
        walls1 = _build_walls(1)
        outside1 = _flood_outside(walls1, sz)
        walls2 = _build_walls(2)
        outside2 = _flood_outside(walls2, sz)

    # ── Overlap ownership (who encircled whom) ───────────────────────────────
    p2_in_enc1 = any(
        (SCALE * gx, SCALE * gy) not in outside1
        for gx, gy in state.board
        if state.board[(gx, gy)] == 2
    )
    p1_in_enc2 = any(
        (SCALE * gx, SCALE * gy) not in outside2
        for gx, gy in state.board
        if state.board[(gx, gy)] == 1
    )

    def owner(sx: int, sy: int) -> int | None:
        in1 = (sx, sy) not in outside1 and (sx, sy) not in walls1
        in2 = (sx, sy) not in outside2 and (sx, sy) not in walls2
        if in1 and in2:
            if p2_in_enc1:
                return 1
            if p1_in_enc2:
                return 2
            return 1        # fallback
        if in1:
            return 1
        if in2:
            return 2
        return None

    # ── Map pixels to unit cells / triangles ─────────────────────────────────
    # Quadrant sample offsets within a SCALE×SCALE cell (never on diagonals):
    #   TOP (3,1)  RIGHT (5,3)  BOTTOM (3,5)  LEFT (1,3)
    new_terr: dict = {}

    for cx in range(S.GRID - 1):
        for cy in range(S.GRID - 1):
            bx_ = SCALE * cx
            by_ = SCALE * cy
            tl = (cx,     cy    )
            tr = (cx + 1, cy    )
            bl = (cx,     cy + 1)
            br = (cx + 1, cy + 1)

            o_top    = owner(bx_ + 3, by_ + 1)
            o_right  = owner(bx_ + 5, by_ + 3)
            o_bottom = owner(bx_ + 3, by_ + 5)
            o_left   = owner(bx_ + 1, by_ + 3)

            # Full square: all four quadrant-samples same owner
            if (
                o_top is not None
                and o_top == o_right == o_bottom == o_left
            ):
                new_terr[(cx, cy)] = (o_top, 'full', [tl, tr, br, bl])
                continue

            # Triangles — only where the splitting diagonal connection exists
            has_slash     = frozenset({tr, bl}) in state.connections  # /
            has_backslash = frozenset({tl, br}) in state.connections  # \

            if has_slash:
                if o_left is not None:
                    new_terr[(cx, cy, 'ul')] = (o_left,  'tri', [tl, tr, bl])
                if o_right is not None:
                    new_terr[(cx, cy, 'lr')] = (o_right, 'tri', [tr, br, bl])

            if has_backslash:
                if o_top is not None:
                    new_terr[(cx, cy, 'ur')] = (o_top,    'tri', [tl, tr, br])
                if o_bottom is not None:
                    new_terr[(cx, cy, 'll')] = (o_bottom, 'tri', [tl, bl, br])

    # ── Scores ────────────────────────────────────────────────────────────────
    new_scores: dict = {1: 0.0, 2: 0.0}
    for key, (own, shape, _) in new_terr.items():
        new_scores[own] += 1.0 if shape == 'full' else 0.5

    state.territories = new_terr
    state.scores = new_scores

    # ── Forbidden positions ───────────────────────────────────────────────────
    # Any grid intersection that lies strictly inside enclosed territory
    # (its scaled pixel is not in outside1 or outside2) cannot receive a new dot.
    new_fp: set = set()
    for gx in range(S.GRID):
        for gy in range(S.GRID):
            if (gx, gy) in state.board:
                continue
            sx, sy = SCALE * gx, SCALE * gy
            if sx < 0 or sy < 0 or sx > sz or sy > sz:
                continue
            # Inside player 1's enclosed area?
            if (sx, sy) not in outside1 and (sx, sy) not in walls1:
                new_fp.add((gx, gy))
                continue
            # Inside player 2's enclosed area?
            if (sx, sy) not in outside2 and (sx, sy) not in walls2:
                new_fp.add((gx, gy))
    state.forbidden_positions = new_fp

    # ── Interior dots ─────────────────────────────────────────────────────────
    # A dot is interior when all adjacent unit cells are full territory of
    # the same owner.
    new_id: set = set()
    for gx in range(S.GRID):
        for gy in range(S.GRID):
            adj = [
                (gx + dcx, gy + dcy)
                for dcx, dcy in ((-1, -1), (0, -1), (-1, 0), (0, 0))
            ]
            in_bounds = [
                (cx, cy) for cx, cy in adj
                if 0 <= cx < S.GRID - 1 and 0 <= cy < S.GRID - 1
            ]
            if not in_bounds:
                continue
            owners: set = set()
            ok = True
            for cx, cy in in_bounds:
                t = new_terr.get((cx, cy))
                if t is None or t[1] != 'full':
                    ok = False
                    break
                owners.add(t[0])
            if ok and len(owners) == 1:
                new_id.add((gx, gy))
    state.interior_dots = new_id

    # ── Interior connections ──────────────────────────────────────────────────
    def full_owner(cx: int, cy: int) -> int | None:
        if not (0 <= cx < S.GRID - 1 and 0 <= cy < S.GRID - 1):
            return None
        t = new_terr.get((cx, cy))
        return t[0] if t and t[1] == 'full' else None

    new_ic: set = set()
    for conn in state.connections:
        a, b = tuple(conn)
        dx_ = b[0] - a[0]
        dy_ = b[1] - a[1]

        if abs(dx_) + abs(dy_) == 1:     # orthogonal connection
            mx, my = min(a[0], b[0]), min(a[1], b[1])
            if dy_ == 0:
                c1, c2 = (mx, my - 1), (mx, my)
            else:
                c1, c2 = (mx - 1, my), (mx, my)
            v1 = 0 <= c1[0] < S.GRID - 1 and 0 <= c1[1] < S.GRID - 1
            v2 = 0 <= c2[0] < S.GRID - 1 and 0 <= c2[1] < S.GRID - 1
            o1 = full_owner(*c1)
            o2 = full_owner(*c2)
            if (
                (v1 and v2 and o1 is not None and o1 == o2)
                or (v1 and not v2 and o1 is not None)
                or (v2 and not v1 and o2 is not None)
            ):
                new_ic.add(conn)
        else:                             # diagonal connection
            cx, cy = min(a[0], b[0]), min(a[1], b[1])
            t = new_terr.get((cx, cy))
            if t and t[1] == 'full':
                new_ic.add(conn)

    state.interior_conns = new_ic
