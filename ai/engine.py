"""Exact bitboard game engine for AI search.

`BitGame` plays Dot Grid under the REAL rules — flood-fill territory, triangle
half-points, encirclement capture, forbidden interiors, both win conditions —
at a few microseconds per move, ~100× faster than the raster engine in
territory.py / SimGame.  The search players simulate with it, so they see
large rings and captures exactly instead of the old unit-square approximation.

Representation
--------------
Points and cells share one index space:  idx = gy * W + gx  with  W = GRID + 1
(one padding column so horizontal shifts never wrap rows).  A cell is indexed
by its top-left point.  A position is a handful of Python ints used as
bitboards: d1 / d2 (dots per player) and forb (forbidden points).

Territory model (equivalent to the SCALE=6 raster flood fill)
--------------------------------------------------------------
territory.py samples four pixels per cell — its TOP / RIGHT / BOTTOM / LEFT
quadrants.  For one player's walls those quadrants connect like this:

  * T–R and L–B inside a cell unless the "/" diagonal (tr–bl) is walled
  * T–L and R–B inside a cell unless the "\\" diagonal (tl–br) is walled
  * T of a cell – B of the cell above unless the shared edge is walled
    (same for L/R); border cells connect to the outside through open edges

A segment is walled when both its endpoints hold that player's dots (adjacent
same-colour dots are always connected).  Flood-filling this quadrant graph
from the outside with bit-parallel shifts gives exactly the raster result:

  * quadrant not reached      → enclosed by that player
  * point (non-own) enclosed  ⇔ the T quadrant of the cell below-right of it
                                 is enclosed; border points never are

`tests/test_engine.py` cross-checks every move of thousands of random games
against SimGame.

Skipping recomputation
----------------------
Territory, captures and forbidden points only change when the mover closes a
cycle.  A 512-entry table over the mover's dots in the 3×3 neighbourhood
tells whether a move can close a non-local ring (flood fill needed) or only
adds half-point triangles beside the new dot (score patched in O(1)).
"""

import settings as S


class _Geo:
    """Per-grid-size masks and lookup tables."""

    def __init__(self, n: int):
        self.n = n
        self.W = W = n + 1
        self.size = n * W                      # bit-length of a board int
        self.points = [gy * W + gx for gy in range(n) for gx in range(n)]
        self.xy = {gy * W + gx: (gx, gy) for gy in range(n) for gx in range(n)}

        self.FULL = sum(1 << i for i in self.points)
        cells = [cy * W + cx for cy in range(n - 1) for cx in range(n - 1)]
        self.CELLS = sum(1 << c for c in cells)
        self.TOP = sum(1 << cx for cx in range(n - 1))
        self.BOT = sum(1 << ((n - 2) * W + cx) for cx in range(n - 1))
        self.LEFT = sum(1 << (cy * W) for cy in range(n - 1))
        self.RIGHT = sum(1 << (cy * W + n - 2) for cy in range(n - 1))
        self.BORDER = sum(
            1 << (gy * W + gx) for gy in range(n) for gx in range(n)
            if gx in (0, n - 1) or gy in (0, n - 1))

        # 8-neighbour mask per point index (list indexed by idx; padding = 0)
        self.NB = [0] * self.size
        for gy in range(n):
            for gx in range(n):
                m = 0
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if (dx or dy) and 0 <= gx + dx < n and 0 <= gy + dy < n:
                            m |= 1 << ((gy + dy) * W + gx + dx)
                self.NB[gy * W + gx] = m

        self.threshold = S.WIN_PCT * (n - 1) ** 2

        # LOCAL[key] classifies a move by the mover's dots among the 8
        # neighbours (key = 3 bits per row, see BitGame.play):
        #   -1  → neighbours form 2+ locally disconnected groups: the move
        #         may close a big ring → full recompute needed
        #   k≥0 → at most one group: the only new territory is k half-point
        #         triangles next to the new dot (no capture, no forbidden
        #         change), so the score is patched without a flood fill.
        offs = [(-1, -1), (0, -1), (1, -1), (-1, 0), None, (1, 0),
                (-1, 1), (0, 1), (1, 1)]
        self.LOCAL = [0] * 512
        for key in range(512):
            own = {offs[b] for b in range(9) if b != 4 and key >> b & 1}
            if key >> 4 & 1:
                continue                      # centre bit never set
            groups = 0
            seen = set()
            for start in own:
                if start in seen:
                    continue
                groups += 1
                stack = [start]
                while stack:
                    cx, cy = stack.pop()
                    if (cx, cy) in seen:
                        continue
                    seen.add((cx, cy))
                    stack.extend(o for o in own if o not in seen
                                 and abs(o[0] - cx) <= 1 and abs(o[1] - cy) <= 1)
            if groups >= 2:
                self.LOCAL[key] = -1
                continue
            halves = 0
            for sx in (-1, 1):
                for sy in (-1, 1):
                    if (((sx, 0) in own) + ((0, sy) in own)
                            + ((sx, sy) in own)) >= 2:
                        halves += 1
            self.LOCAL[key] = halves


_GEO: dict = {}


def geo(n: int = None) -> _Geo:
    """Cached geometry for grid size *n* (default: current S.GRID)."""
    n = n or S.GRID
    g = _GEO.get(n)
    if g is None or g.threshold != S.WIN_PCT * (n - 1) ** 2:
        g = _GEO[n] = _Geo(n)
    return g


def bits(mask: int) -> list:
    """Indices of the set bits of *mask*, ascending."""
    out = []
    while mask:
        low = mask & -mask
        out.append(low.bit_length() - 1)
        mask ^= low
    return out


class BitGame:
    """One game position.  Mutable; `copy()` is a handful of int copies."""

    __slots__ = ('g', 'd1', 'd2', 'forb', 's1', 's2', 'player',
                 'done', 'winner', 'dirty')

    def __init__(self, g: _Geo = None):
        self.g = g or geo()
        self.d1 = 0
        self.d2 = 0
        self.forb = 0
        self.s1 = 0.0
        self.s2 = 0.0
        self.player = 1
        self.done = False
        self.winner = None      # 1 / 2, or None (not over, or draw)
        self.dirty = False      # trapped dots pending → recompute next move

    # ── Construction / conversion ─────────────────────────────────────────────

    @classmethod
    def from_board(cls, board: dict, player: int, scores: dict = None,
                   forbidden=None) -> 'BitGame':
        """Build from the dict-based representation used by the rest of the
        game.  With *scores* given, scores/forbidden are trusted as the real
        engine computed them; otherwise they are derived here."""
        g = geo()
        W = g.W
        bg = cls(g)
        for (gx, gy), p in board.items():
            if p == 1:
                bg.d1 |= 1 << (gy * W + gx)
            else:
                bg.d2 |= 1 << (gy * W + gx)
        bg.player = player
        if scores is None:
            bg._recompute()
        else:
            bg.s1 = float(scores.get(1, 0.0))
            bg.s2 = float(scores.get(2, 0.0))
            for gx, gy in (forbidden or ()):
                bg.forb |= 1 << (gy * W + gx)
            # Trapped dots the real engine has not removed yet (it resolves
            # one side per recompute) must be resolved on our first move too.
            out1 = bg._flood(bg.d1)[0] | g.BORDER
            out2 = bg._flood(bg.d2)[0] | g.BORDER
            bg.dirty = bool((bg.d2 & ~out1) or (bg.d1 & ~out2))
        bg._check_end()
        return bg

    def copy(self) -> 'BitGame':
        c = BitGame.__new__(BitGame)
        c.g = self.g
        c.d1 = self.d1
        c.d2 = self.d2
        c.forb = self.forb
        c.s1 = self.s1
        c.s2 = self.s2
        c.player = self.player
        c.done = self.done
        c.winner = self.winner
        c.dirty = self.dirty
        return c

    def board(self) -> dict:
        xy = self.g.xy
        out = {xy[i]: 1 for i in bits(self.d1)}
        out.update({xy[i]: 2 for i in bits(self.d2)})
        return out

    def forbidden(self) -> set:
        xy = self.g.xy
        return {xy[i] for i in bits(self.forb)}

    def connections(self) -> set:
        """Connection set in the dict-engine format (frozenset pairs): every
        pair of same-colour dots at Chebyshev distance 1."""
        xy, W = self.g.xy, self.g.W
        out = set()
        for d in (self.d1, self.d2):
            for shift in (1, W - 1, W, W + 1):
                for i in bits(d & (d >> shift)):
                    out.add(frozenset((xy[i], xy[i + shift])))
        return out

    # ── Moves ─────────────────────────────────────────────────────────────────

    def legal(self) -> int:
        """Bitboard of playable points."""
        return self.g.FULL & ~(self.d1 | self.d2 | self.forb)

    def play(self, i: int) -> None:
        """Place the current player's dot at point index *i* (must be legal)."""
        g = self.g
        W = g.W
        p = self.player
        if p == 1:
            own = self.d1
            self.d1 = own | (1 << i)
            self.player = 2
        else:
            own = self.d2
            self.d2 = own | (1 << i)
            self.player = 1
        x = (own << (W + 1)) >> i          # 3×3 window around i, row stride W
        k = g.LOCAL[(x & 7) | ((x >> W & 7) << 3) | ((x >> (W + W) & 7) << 6)]
        if self.dirty or k < 0:
            self._recompute()
        elif k:
            if p == 1:
                self.s1 += 0.5 * k
            else:
                self.s2 += 0.5 * k
        if k:
            if self.s1 >= g.threshold:
                self.done = True
                self.winner = 1
                return
            if self.s2 >= g.threshold:
                self.done = True
                self.winner = 2
                return
        if not g.FULL & ~(self.d1 | self.d2 | self.forb):
            self._end_by_score()

    def gain(self, i: int) -> float:
        """Exact swing for the player to move from playing *i*:
        (own score gained) + (opponent score destroyed).  Non-mutating."""
        if self.player == 1:
            if not self.dirty and (self.d1 & self.g.NB[i]).bit_count() < 2:
                return 0.0
            c = self.copy()
            c.play(i)
            return (c.s1 - self.s1) + (self.s2 - c.s2)
        if not self.dirty and (self.d2 & self.g.NB[i]).bit_count() < 2:
            return 0.0
        c = self.copy()
        c.play(i)
        return (c.s2 - self.s2) + (self.s1 - c.s1)

    def enclosed(self, player: int) -> tuple:
        """(T, R, B, L) bitboards of the quadrants enclosed by *player*'s
        walls (before overlap resolution)."""
        C = self.g.CELLS
        return tuple(C & ~q for q in
                     self._flood(self.d1 if player == 1 else self.d2))

    # ── Internals ─────────────────────────────────────────────────────────────

    def _check_end(self) -> None:
        g = self.g
        if self.s1 >= g.threshold:
            self.done, self.winner = True, 1
        elif self.s2 >= g.threshold:
            self.done, self.winner = True, 2
        elif not g.FULL & ~(self.d1 | self.d2 | self.forb):
            self._end_by_score()

    def _end_by_score(self) -> None:
        self.done = True
        self.winner = (None if self.s1 == self.s2
                       else (1 if self.s1 > self.s2 else 2))

    def _flood(self, d: int) -> tuple:
        """Flood the quadrant graph from outside the board against the walls
        of dot set *d*.  Returns (T, R, B, L) bitboards of REACHED quadrants."""
        g = self.g
        W = g.W
        C = g.CELLS
        eh = d & (d >> 1)                  # walled horizontal segments
        ev = d & (d >> W)                  # walled vertical segments
        nbs = ~(d & (d >> (W + 1)))        # cells without a "\" wall
        nsl = ~((d >> 1) & (d >> W))       # cells without a "/" wall
        ot = C & ~eh                       # cells whose top edge is open
        ob = C & ~(eh >> W)
        ol = C & ~ev
        orr = C & ~(ev >> 1)
        T = g.TOP & ot
        B = g.BOT & ob
        L = g.LEFT & ol
        R = g.RIGHT & orr
        prev = -1
        while True:
            T |= (R & nsl) | (L & nbs) | ((B << W) & ot)
            R |= (T & nsl) | (B & nbs) | ((L >> 1) & orr)
            B |= (L & nsl) | (R & nbs) | ((T >> W) & ob)
            L |= (B & nsl) | (T & nbs) | ((R << 1) & ol)
            cur = T + R + B + L
            if cur == prev:
                return T, R, B, L
            prev = cur

    def _recompute(self) -> None:
        """Mirror of SimGame._recompute / territory.recompute_territories."""
        g = self.g
        W = g.W
        C = g.CELLS
        d1, d2 = self.d1, self.d2

        f1 = self._flood(d1)
        f2 = self._flood(d2)
        out1 = f1[0] | g.BORDER            # points NOT enclosed by player 1
        out2 = f2[0] | g.BORDER

        # Encirclement — one side per recompute, player 1 first (as the
        # real engine does).
        if d2 & ~out1:
            d2 = self.d2 = d2 & out1
            f2 = self._flood(d2)
            out2 = f2[0] | g.BORDER
        elif d1 & ~out2:
            d1 = self.d1 = d1 & out2
            f1 = self._flood(d1)
            out1 = f1[0] | g.BORDER

        p1_in_enc2 = d1 & ~out2
        self.dirty = bool(p1_in_enc2)      # resolved by the next recompute

        # Quadrant ownership.  Overlap goes to player 1 unless player-1 dots
        # sit inside player 2's enclosure.
        q1 = 0
        q2 = 0
        full1 = full2 = C
        own1 = []
        own2 = []
        for a, b in zip(f1, f2):
            in1 = C & ~a
            in2 = C & ~b
            if p1_in_enc2:
                o2 = in2
                o1 = in1 & ~in2
            else:
                o1 = in1
                o2 = in2 & ~in1
            own1.append(o1)
            own2.append(o2)
            full1 &= o1
            full2 &= o2
            q1 += o1.bit_count()
            q2 += o2.bit_count()

        # Cells crossed by both a "/" and a "\" connection score each enclosed
        # quadrant as its own half-point triangle (unless uniformly owned).
        sl = ((d1 >> 1) & (d1 >> W)) | ((d2 >> 1) & (d2 >> W))
        bs = (d1 & (d1 >> (W + 1))) | (d2 & (d2 >> (W + 1)))
        mixed = sl & bs & C & ~full1 & ~full2
        if mixed:
            for o1, o2 in zip(own1, own2):
                q1 += (o1 & mixed).bit_count()
                q2 += (o2 & mixed).bit_count()

        self.s1 = q1 * 0.25
        self.s2 = q2 * 0.25
        self.forb = g.FULL & ~(d1 | d2) & ~(out1 & out2)
