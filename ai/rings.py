"""Ring-race analysis: how many more dots does a player need to enclose how
much area?

The decisive strategy in Dot Grid is a big ring: a closed chain of dots that
encloses half the board wins on the spot and captures everything inside.
The local signals in ai/features.py (unit squares, bounding boxes) cannot see
such a plan.  This module measures it exactly for *star-shaped* rings:

    profile[k] = the largest area the player can enclose around a centre
                 with at most k additional dots, if left undisturbed.

Method — dynamic programming around a centre
---------------------------------------------
A ring that is star-shaped around a centre c visits its dots in strictly
increasing angle, so sorting the points by angle turns "all rings" into paths
of a DAG whose edges are king steps u→v with cross(u−c, v−c) > 0.  A dot
costs 0 on an own dot, 1 on an empty playable point, and opponent dots /
forbidden points are impassable.  A forward DP over (node, cost) maximising
area gives the profile; a backward DP gives, for every point, the best ring
THROUGH it — i.e. which moves advance a ring, and which opponent points are
worth taking.

Area is accumulated as the line integral ∮ x dy, one term per edge: a step
between rows y and y+1 contributes ± the area lying to its left in that row
band.  For a closed ring the terms sum to exactly the game's score for the
enclosure (full cells plus half-cell triangles).  Territory the player
ALREADY owns is left out of the band areas, so "area" always means newly
gained score and `moves_to(threshold − score)` is the true distance to a win.

Rings may cross an opponent's diagonal link (the rules allow it), and dents
around blockers are handled as long as the ring stays star-shaped around the
centre.  Several centres are analysed so rings in a board half or quadrant are
found too.
"""

import math

import numpy as np

import settings as S
from ai.engine import geo, bits

NEG = -1e9
NEVER = 999         # "no such ring" distance


def k_max(n: int) -> int:
    """Cost horizon in dots for grid size n.  A winning octagon needs about
    2·n dots (19 on 10×10), so 2.6·n leaves room for detours."""
    return max(12, int(2.6 * n))


class _CentreGeo:
    """Static angular DAG for one (grid size, centre)."""

    def __init__(self, n: int, cx: float, cy: float):
        W = n + 1
        nodes = [(gx, gy) for gy in range(n) for gx in range(n)
                 if (gx - cx, gy - cy) != (0.0, 0.0)]
        nodes.sort(key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
        self.order = [gy * W + gx for gx, gy in nodes]       # bit index per rank
        self.rank_of = {i: r for r, i in enumerate(self.order)}
        rank = {p: r for r, p in enumerate(nodes)}
        N = len(nodes)

        # Edge weight = sign · V[slot], where V is the per-position table of
        # band areas built by _band_areas() (see module docstring).
        cells_w = n                        # x = 0..n-1 per cell row
        def slot(u, v):
            (ux, uy), (vx, vy) = u, v
            if uy == vy:
                return 0, 0.0              # horizontal step: no area
            y = min(uy, vy)
            sign = 1.0 if vy > uy else -1.0
            if ux == vx:
                kind, x = 0, ux            # vertical step at column x
            elif (vx - ux) * (vy - uy) > 0:
                kind, x = 1, min(ux, vx)   # "\" diagonal of cell (x, y)
            else:
                kind, x = 2, min(ux, vx)   # "/" diagonal of cell (x, y)
            return 1 + (kind * (n - 1) + y) * cells_w + x, sign

        edges = [[] for _ in range(N)]     # non-wrapping edges into rank r
        wrap = {}                          # start rank → [(pred rank, slot, sign)]
        for u, ru in rank.items():
            ux, uy = u
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    v = (ux + dx, uy + dy)
                    if (dx or dy) and v in rank:
                        cross = ((ux - cx) * (v[1] - cy)
                                 - (uy - cy) * (v[0] - cx))
                        if cross <= 1e-9:
                            continue
                        rv = rank[v]
                        sl, sg = slot(u, v)
                        if rv > ru:
                            edges[rv].append((ru, sl, sg))
                        else:
                            wrap.setdefault(rv, []).append((ru, sl, sg))

        self.n = n
        self.N = N
        self.K = k_max(n)
        self.starts = sorted(wrap)                         # start ranks
        self.pred_idx = [np.array([u for u, _, _ in e], dtype=np.intp)
                         for e in edges]
        # Flat (slot, sign) arrays over all pred edges, sliced per rank
        self.pred_slice = []
        slots, signs = [], []
        for e in edges:
            self.pred_slice.append((len(slots), len(slots) + len(e)))
            slots += [sl for _, sl, _ in e]
            signs += [sg for _, _, sg in e]
        self.pred_slot = np.array(slots, dtype=np.intp)
        self.pred_sign = np.array(signs, dtype=np.float32)

        self.wrap_idx = [np.array([u for u, _, _ in wrap[s]], dtype=np.intp)
                         for s in self.starts]
        self.wrap_slot = [np.array([sl for _, sl, _ in wrap[s]], dtype=np.intp)
                          for s in self.starts]
        self.wrap_sign = [np.array([sg for _, _, sg in wrap[s]],
                                   dtype=np.float32) for s in self.starts]

        # Successor lists for the backward pass
        succs = [[] for _ in range(N)]
        for rv, e in enumerate(edges):
            for ru, sl, sg in e:
                succs[ru].append((rv, sl, sg))
        self.succ_idx = [np.array([v for v, _, _ in sc], dtype=np.intp)
                         for sc in succs]
        self.succ_slice = []
        slots, signs = [], []
        for sc in succs:
            self.succ_slice.append((len(slots), len(slots) + len(sc)))
            slots += [sl for _, sl, _ in sc]
            signs += [sg for _, _, sg in sc]
        self.succ_slot = np.array(slots, dtype=np.intp)
        self.succ_sign = np.array(signs, dtype=np.float32)
        # wrap edges leaving rank r: [(start slot j, area slot, sign)]
        self.wrap_out = [[] for _ in range(N)]
        for j, s in enumerate(self.starts):
            for ru, sl, sg in wrap[s]:
                self.wrap_out[ru].append((j, sl, sg))

        # Band-area table for a player with no territory (the common case)
        self.V0 = _band_areas(n, None)


def _band_areas(n: int, owned) -> np.ndarray:
    """Table V of not-yet-owned area left of each possible edge.

    Slot layout (matches _CentreGeo.slot): 0 is a constant 0, then three
    (n-1)×n blocks indexed [y][x] for cell row y:
      kind 0  vertical step at column x      → area of cells 0..x-1
      kind 1  "\" diagonal of cell (x, y)    → that + the cell's L,B quadrants
      kind 2  "/" diagonal of cell (x, y)     → that + the cell's T,L quadrants
    *owned* is None or the player's enclosed quadrants (T, R, B, L bitboards).
    """
    W = n + 1
    V = np.zeros(1 + 3 * (n - 1) * n, dtype=np.float32)
    for y in range(n - 1):
        acc = 0.0
        for x in range(n):
            base = 1 + y * n + x
            V[base] = acc
            if x == n - 1:
                break
            if owned is None:
                lb = tl = 0.5
                cell = 1.0
            else:
                c = y * W + x
                t, r, b, l = (1 - (q >> c & 1) for q in owned)
                lb = 0.25 * (l + b)
                tl = 0.25 * (t + l)
                cell = 0.25 * (t + r + b + l)
            V[base + (n - 1) * n] = acc + lb
            V[base + 2 * (n - 1) * n] = acc + tl
            acc += cell
    return V


_CENTRES: dict = {}


def centres(n: int = None) -> list:
    """Analysis centres for grid size *n*: the board centre first (the only
    place a winning ring can be centred), then the centres of the four board
    halves and of the four quadrants, so smaller rings anywhere are seen."""
    n = n or S.GRID
    cs = _CENTRES.get(n)
    if cs is None:
        m = (n - 1) / 2.0
        q = (n - 1) / 4.0
        # Offsets of 0.02/0.03 keep every centre off the grid lines so no
        # king step is exactly radial (zero cross product).
        pts = [(m, m),
               (q, m), (m + q, m), (m, q), (m, m + q),
               (q, q), (m + q, q), (q, m + q), (m + q, m + q)]
        cs = _CENTRES[n] = [_CentreGeo(n, x + 0.02, y + 0.03) for x, y in pts]
    return cs


class RingInfo:
    """Result of analysing one player's rings around one centre."""

    __slots__ = ('profile', 'F', 'G', 'cg', 'cost')

    def moves_to(self, area: float) -> int:
        """Fewest additional dots to enclose at least *area* (NEVER if no
        ring within the cost horizon does)."""
        hit = np.nonzero(self.profile >= area - 1e-6)[0]
        return int(hit[0]) if len(hit) else NEVER


def band_areas(n: int, owned) -> np.ndarray:
    """Area table for a player whose enclosed quadrants are *owned* — pass
    the result as `V` to several analyse() calls on the same position."""
    return _band_areas(n, owned)


def analyse(own: int, blocked: int, cg: _CentreGeo,
            backward: bool = False, owned=None, V=None) -> RingInfo:
    """Forward (and optionally backward) ring DP for one player.

    own      bitboard of the player's dots (cost 0)
    blocked  bitboard of impassable points (opponent dots, forbidden points)
    owned    the player's enclosed quadrants (T, R, B, L) or None if it has
             no territory — already-owned area is not counted again
    V        precomputed band_areas() table (overrides *owned*)
    """
    N = cg.N
    S_ = len(cg.starts)
    K1 = cg.K + 1
    order = cg.order
    cost = [(-1 if blocked >> i & 1 else (0 if own >> i & 1 else 1))
            for i in order]
    if V is None:
        V = cg.V0 if owned is None else _band_areas(cg.n, owned)
    pred_w = (cg.pred_sign * V[cg.pred_slot]).reshape(-1, 1, 1)

    F = np.full((N, S_, K1), NEG, dtype=np.float32)
    for j, s in enumerate(cg.starts):
        if cost[s] >= 0:
            F[s, j, cost[s]] = 0.0
    pred_idx, pred_slice = cg.pred_idx, cg.pred_slice
    for r in range(N):
        c = cost[r]
        a, b = pred_slice[r]
        if c < 0 or a == b:
            continue
        cand = (F[pred_idx[r]] + pred_w[a:b]).max(axis=0)
        if c:
            np.maximum(F[r, :, 1:], cand[:, :-1], out=F[r, :, 1:])
        else:
            np.maximum(F[r], cand, out=F[r])

    tot = np.full((S_, K1), NEG, dtype=np.float32)
    for j in range(S_):
        w = (cg.wrap_sign[j] * V[cg.wrap_slot[j]]).reshape(-1, 1)
        tot[j] = (F[cg.wrap_idx[j], j] + w).max(axis=0)
    profile = np.maximum.accumulate(np.maximum(tot.max(axis=0), 0.0))

    info = RingInfo()
    info.profile = profile
    info.F = F
    info.cg = cg
    info.cost = cost
    info.G = None
    if backward:
        # G[r, j, k] = best area from rank r to closing at start j, spending
        # k on the nodes AFTER r.  H is G shifted by the node's own cost
        # (what a predecessor sees when stepping into the node).
        G = np.full((N, S_, K1), NEG, dtype=np.float32)
        H = np.full((N, S_, K1), NEG, dtype=np.float32)
        succ_w = (cg.succ_sign * V[cg.succ_slot]).reshape(-1, 1, 1)
        succ_idx, succ_slice = cg.succ_idx, cg.succ_slice
        start_ok = [cost[s] >= 0 for s in cg.starts]
        for r in range(N - 1, -1, -1):
            c = cost[r]
            if c < 0:
                continue
            g = G[r]
            a, b = succ_slice[r]
            if a != b:
                g[:] = (H[succ_idx[r]] + succ_w[a:b]).max(axis=0)
            for j, sl, sg in cg.wrap_out[r]:
                if start_ok[j]:
                    w = sg * V[sl]
                    if w > g[j, 0]:
                        g[j, 0] = w
            if c:
                H[r, :, 1:] = g[:, :-1]
            else:
                H[r] = g
        info.G = G
    return info


def through_table(info: RingInfo) -> np.ndarray:
    """best[r, k] = largest area of a ring through rank r costing k dots in
    total (the point's own cost included).  Needs a backward pass."""
    F, G = info.F, info.G
    K1 = F.shape[2]
    best = np.full((F.shape[0], K1), NEG, dtype=np.float32)
    for k2 in range(K1):
        g = G[:, :, k2:k2 + 1]
        if not (g > NEG / 2).any():
            continue
        np.maximum(best[:, k2:], (F[:, :, :K1 - k2] + g).max(axis=1),
                   out=best[:, k2:])
    return best


def through_costs(best: np.ndarray, area: float) -> np.ndarray:
    """Per rank: fewest dots for a ring of at least *area* through that
    point (NEVER where impossible)."""
    ok = best >= area - 1e-6
    return np.where(ok.any(axis=1), ok.argmax(axis=1), NEVER)


def discounted(profile: np.ndarray, gamma: float) -> float:
    """max_k area[k]·gamma^k — value of the best ring, nearer ones first."""
    k = np.arange(profile.shape[-1], dtype=np.float32)
    return (np.maximum(profile, 0.0) * gamma ** k).max(axis=-1)
