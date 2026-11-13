"""Tests for ai/strategy.py — ring-race move scoring."""
import math

import numpy as np
import pytest

import settings as S
from ai import strategy
from ai.engine import BitGame


def _border(n):
    return [(x, y) for y in range(n) for x in range(n)
            if x in (0, n - 1) or y in (0, n - 1)]


@pytest.fixture
def small(monkeypatch):
    monkeypatch.setattr(S, 'GRID', 5)      # 16 cells, threshold 8
    return 5


class TestRaceValue:
    def test_no_rings_is_neutral(self):
        assert strategy.race_value(strategy.INF_W, strategy.INF_W, True) == 0.0

    def test_side_to_move_wins_ties(self):
        assert strategy.race_value(5, 5, me_to_move=True) > 0
        assert strategy.race_value(5, 5, me_to_move=False) < 0

    def test_monotonic_in_both_distances(self):
        assert strategy.race_value(4, 8, False) > strategy.race_value(6, 8, False)
        assert strategy.race_value(6, 9, False) > strategy.race_value(6, 7, False)

    def test_lead_counts_more_near_closure(self):
        assert (strategy.race_value(2, 4, False)
                > strategy.race_value(12, 14, False) > 0)


class TestAnalysis:
    def test_empty_board_is_symmetric(self):
        pa = strategy.analyse_position(BitGame())
        assert pa.W[1] == pa.W[2] < strategy.INF_W
        assert pa.U[1] == pa.U[2] > 0

    def test_one_gap_ring_is_one_move_from_winning(self, small):
        ring = _border(5)
        board = {p: 1 for p in ring if p != (2, 0)}
        pa = strategy.analyse_position(BitGame.from_board(board, 1))
        assert pa.W[1] == 1
        assert pa.tc[1][0 * 6 + 2] == 1        # the gap is on the ring

    def test_forward_only_analysis_agrees_on_distance(self):
        bg = BitGame.from_board({(4, 3): 1, (5, 3): 1, (6, 4): 2}, 1)
        full = strategy.analyse_position(bg)
        quick = strategy.analyse_position(bg, detail=False)
        assert full.W == quick.W
        assert quick.tc[1] == {}


class TestMoveScores:
    def test_winning_move_scores_infinite(self, small):
        board = {p: 1 for p in _border(5) if p != (2, 0)}
        bg = BitGame.from_board(board, 1)
        scores = strategy.move_scores(bg)
        assert scores[0 * bg.g.W + 2] == math.inf

    def test_blocks_the_only_closing_point(self, small):
        # P1's border ring lacks (2,0); the detour through (2,1) is already
        # taken by P2, so (2,0) is the single point that stops the win.
        board = {p: 1 for p in _border(5) if p != (2, 0)}
        board[(2, 1)] = 2
        bg = BitGame.from_board(board, 2)
        scores = strategy.move_scores(bg)
        best = max(scores, key=scores.get)
        assert bg.g.xy[best] == (2, 0)

    def test_scores_cover_exactly_the_legal_moves(self):
        bg = BitGame.from_board({(4, 4): 1, (5, 5): 2}, 1)
        scores = strategy.move_scores(bg)
        assert set(scores) == {i for i in bg.g.points if bg.legal() >> i & 1}

    def test_fast_mode_runs_and_ranks_ring_points(self):
        bg = BitGame.from_board({(4, 3): 1, (5, 3): 1, (6, 4): 1, (0, 0): 2,
                                 (9, 9): 2, (0, 9): 2}, 1)
        pa = strategy.analyse_position(bg, n_centres=1)
        scores = strategy.move_scores(bg, pa, exact_blocks=False)
        assert all(math.isfinite(v) for v in scores.values())


class TestTacticalGains:
    def test_reports_capture_for_both_sides(self):
        # Diamond around (2,2) lacks (1,2): P1 closing gains 2, and P2
        # playing there gains nothing.
        board = {(2, 1): 1, (3, 2): 1, (2, 3): 1, (9, 9): 2, (9, 7): 2, (9, 5): 2}
        bg = BitGame.from_board(board, 1)
        own, opp = strategy.tactical_gains(bg)
        i = 2 * bg.g.W + 1
        assert own[i] == 2.0 and i not in opp


class TestGuidanceOffsets:
    def test_shape_range_and_illegal_zero(self):
        board = {(4, 3): 1, (5, 3): 1, (6, 4): 2}
        off = strategy.guidance_offsets(board, 1, {1: 0.0, 2: 0.0}, set())
        n = S.GRID
        assert off.shape == (n * n,) and off.dtype == np.float32
        assert off.max() == 0.0 and off.min() >= -S.AI_GUIDE_CLIP
        for gx, gy in board:
            assert off[gy * n + gx] == 0.0

    def test_winning_move_is_the_maximum(self, small):
        board = {p: 1 for p in _border(5) if p != (2, 0)}
        off = strategy.guidance_offsets(board, 1, None, None)
        legal = [gy * 5 + gx for gy in range(5) for gx in range(5)
                 if (gx, gy) not in board]
        assert max(legal, key=lambda f: off[f]) == 0 * 5 + 2
        assert off[0 * 5 + 2] == 0.0
