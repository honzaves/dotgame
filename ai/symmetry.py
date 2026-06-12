"""D4 (dihedral) board symmetries.

The game is fully invariant under the 8 transforms of the square: 4 rotations
× optional mirror.  Two users:

  * MCTSPlayer experience keys — canonical_board_key() maps all 8 variants of
    a position to one key, multiplying warm-start hit rate.
  * NeuralMCTSPlayer training — transform_planes()/transform_flat() augment
    (state, visit-dist) pairs with a random transform per sampled minibatch
    item (value targets are invariant).

Conventions: spatial arrays are indexed [.., gy, gx] (row-major, idx = gy*n+gx)
— same as encode_state().  transform_planes and transform_flat apply the SAME
permutation for a given op index, so encodings, legal masks and policy targets
stay aligned.

Correspondence between point_transforms and array ops
------------------------------------------------------
For op k (no flip):  one CCW quarter-turn maps (gx, gy) → (n-1-gy, gx).
  np.rot90 on axes=(1,2) with k=1 (CCW) maps array element [gy][gx] to
  [gx][n-1-gy], i.e. new_gx=gy, new_gy=n-1-gx — which is NOT the same
  as the point formula.  The point formula and the array ops therefore do
  NOT correspond index-for-index; they differ by a transposition.  What IS
  guaranteed (and what the tests require) is:
    1. Each point_transform is a bijection of the n×n grid (permutation test).
    2. The 8 point transforms form the D4 group (invariance test).
    3. transform_planes and transform_flat produce the same flat result
       (they share one implementation, so this holds by construction).
  The canonical key only needs D4 closure, not per-index match to the array
  ops — both guarantees are met.
"""

import numpy as np

import settings as S

N_TRANSFORMS = 8

# op → (number of np.rot90 quarter-turns, mirror first?)
_OPS = [(k, f) for f in (False, True) for k in range(4)]


def transform_planes(planes: np.ndarray, op: int) -> np.ndarray:
    """Transform a (C, G, G) array.  Returns a contiguous copy."""
    k, flip = _OPS[op]
    out = planes[:, :, ::-1] if flip else planes
    return np.ascontiguousarray(np.rot90(out, k, axes=(1, 2)))


def transform_flat(vec: np.ndarray, op: int) -> np.ndarray:
    """Transform a flat (G²,) row-major vector with the same permutation."""
    n = S.GRID
    return transform_planes(vec.reshape(1, n, n), op).reshape(-1)


def point_transforms(n: int):
    """The 8 (gx, gy) → (gx', gy') maps forming the D4 group.

    One CCW quarter-turn: (gx, gy) → (n-1-gy, gx).
    Mirror (before rotation): gx → n-1-gx.

    These form the dihedral group D4 and are thus closed under composition
    and inversion, which is all canonical_board_key requires.

    Note: these point maps do NOT correspond index-for-index to
    transform_planes ops — they differ by a transposition.  See module
    docstring for details.
    """
    def rot(k, gx, gy):
        for _ in range(k):
            gx, gy = n - 1 - gy, gx      # one CCW quarter turn of indices
        return gx, gy

    out = []
    for k, flip in _OPS:
        def t(gx, gy, k=k, flip=flip):
            if flip:
                gx = n - 1 - gx
            return rot(k, gx, gy)
        out.append(t)
    return out


def canonical_board_key(board: dict) -> str:
    """Lexicographically smallest str(sorted((gx, gy, p))) over all 8 transforms.

    Same string format as the previous MCTS _board_key, so the experience
    JSON schema is unchanged — old non-canonical keys simply stop matching
    and age out via the AI_MAX_EXPERIENCE cap.
    """
    n = S.GRID
    best = None
    for t in point_transforms(n):
        s = str(sorted((*t(gx, gy), p) for (gx, gy), p in board.items()))
        if best is None or s < best:
            best = s
    return best
