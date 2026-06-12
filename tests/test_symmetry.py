"""Tests for ai/symmetry.py — D4 board transforms."""
import numpy as np

import settings as S
from ai.symmetry import (N_TRANSFORMS, point_transforms,
                         transform_planes, transform_flat,
                         canonical_board_key)


class TestPointAndPlaneConsistency:
    def test_planes_and_flat_agree(self):
        n = S.GRID
        planes = np.zeros((1, n, n), dtype=np.float32)
        planes[0, 2, 5] = 1.0                       # [gy=2][gx=5]
        flat = np.zeros(n * n, dtype=np.float32)
        flat[2 * n + 5] = 1.0
        for op in range(N_TRANSFORMS):
            a = transform_planes(planes, op).reshape(-1)
            b = transform_flat(flat, op)
            assert np.array_equal(a, b), f"op {op} disagrees"

    def test_each_transform_is_a_permutation(self):
        n = S.GRID
        for op in range(N_TRANSFORMS):
            t = point_transforms(n)[op]
            seen = {t(gx, gy) for gx in range(n) for gy in range(n)}
            assert len(seen) == n * n


class TestCanonicalKey:
    def test_invariant_under_rotation(self):
        n = S.GRID
        board = {(1, 2): 1, (3, 4): 2, (0, 0): 1}
        rot90 = {(n - 1 - gy, gx): p for (gx, gy), p in board.items()}
        assert canonical_board_key(board) == canonical_board_key(rot90)

    def test_distinguishes_different_positions(self):
        assert (canonical_board_key({(1, 2): 1})
                != canonical_board_key({(1, 2): 2}))


class TestD4Closure:
    def test_canonical_key_invariant_under_all_eight_point_maps(self):
        from ai.symmetry import point_transforms, canonical_board_key
        n = S.GRID
        rng = np.random.default_rng(7)
        for _ in range(5):
            cells = rng.choice(n * n, size=12, replace=False)
            board = {(int(c % n), int(c // n)): int(1 + (c % 2)) for c in cells}
            base = canonical_board_key(board)
            for t in point_transforms(n):
                tb = {t(gx, gy): p for (gx, gy), p in board.items()}
                assert canonical_board_key(tb) == base
