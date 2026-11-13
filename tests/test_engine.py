"""Tests for ai/engine.py — the exact bitboard engine.

The core guarantee is equivalence with the reference rules (SimGame, itself
a copy of territory.py): every move of randomly played games must produce
the same board, scores, forbidden set and game-over state.
"""
import random

import pytest

import settings as S
from ai.engine import BitGame, geo, bits
from ai.trainer import SimGame


def _play(bg, moves):
    W = bg.g.W
    for gx, gy in moves:
        bg.play(gy * W + gx)


def _same(bg, sim):
    return (bg.board() == sim.board
            and bg.forbidden() == sim.forbidden
            and bg.s1 == sim.scores[1] and bg.s2 == sim.scores[2]
            and bg.done == sim.game_over and bg.winner == sim.winner
            and bg.player == sim.current
            and bg.connections() == sim.connections)


class TestMatchesReferenceEngine:
    @pytest.mark.parametrize("grid,games,bias", [
        (4, 12, 0.0), (5, 10, 0.7), (6, 8, 0.9), (10, 3, 0.0), (10, 3, 0.8),
    ])
    def test_random_games_identical(self, monkeypatch, grid, games, bias):
        monkeypatch.setattr(S, 'GRID', grid)
        rnd = random.Random(grid * 1000 + int(bias * 10))
        for _ in range(games):
            sim, bg = SimGame(), BitGame(geo(grid))
            W = bg.g.W
            while not sim.game_over:
                legal = sim.legal_moves()
                if bias and sim.board and rnd.random() < bias:
                    # Prefer moves next to own dots → more rings and captures
                    own = [p for p, v in sim.board.items() if v == sim.current]
                    near = [m for m in legal
                            if any(abs(m[0] - o[0]) <= 1 and abs(m[1] - o[1]) <= 1
                                   for o in own)]
                    legal = near or legal
                m = rnd.choice(legal)
                sim.place(*m)
                bg.play(m[1] * W + m[0])
                assert _same(bg, sim), f"diverged after {m}"

    def test_from_board_reproduces_mid_game_state(self):
        rnd = random.Random(7)
        sim = SimGame()
        for _ in range(45):
            if sim.game_over:
                break
            sim.place(*rnd.choice(sim.legal_moves()))
        bg = BitGame.from_board(sim.board, sim.current, sim.scores, sim.forbidden)
        assert _same(bg, sim)
        # ...and deriving scores/forbidden from the board alone agrees too
        bg2 = BitGame.from_board(sim.board, sim.current)
        assert (bg2.s1, bg2.s2, bg2.forbidden()) == (bg.s1, bg.s2, bg.forbidden())


class TestScoring:
    def test_triangle_scores_half(self):
        bg = BitGame()
        # P1: (0,0) (1,0) (1,1); P2 plays far away in between
        _play(bg, [(0, 0), (9, 9), (1, 0), (9, 8), (1, 1)])
        assert bg.s1 == 0.5 and bg.s2 == 0.0

    def test_unit_square_scores_one(self):
        bg = BitGame()
        _play(bg, [(0, 0), (9, 9), (1, 0), (9, 7), (0, 1), (9, 5), (1, 1)])
        assert bg.s1 == 1.0

    def test_ring_scores_area_and_forbids_interior(self):
        bg = BitGame()
        ring = [(2, 1), (3, 2), (2, 3), (1, 2)]          # diamond around (2,2)
        far = [(9, 9), (9, 7), (9, 5), (9, 3)]
        for a, b in zip(ring, far):
            _play(bg, [a, b])
        assert bg.s1 == 2.0
        assert bg.forbidden() == {(2, 2)}
        assert not bg.legal() >> (2 * bg.g.W + 2) & 1

    def test_capture_removes_enclosed_dot(self):
        bg = BitGame()
        # P2 dot at (2,2) gets surrounded by the P1 diamond
        _play(bg, [(2, 1), (2, 2), (3, 2), (9, 9), (2, 3), (9, 7), (1, 2)])
        assert (2, 2) not in bg.board()
        assert bg.s1 == 2.0 and bg.forbidden() == {(2, 2)}

    def test_gain_is_exact_and_non_mutating(self):
        bg = BitGame()
        _play(bg, [(2, 1), (2, 2), (3, 2), (9, 9), (2, 3), (9, 7)])
        before = (bg.d1, bg.d2, bg.s1, bg.player)
        assert bg.gain(2 * bg.g.W + 1) == 2.0          # closes the diamond
        assert bg.gain(5 * bg.g.W + 5) == 0.0
        assert (bg.d1, bg.d2, bg.s1, bg.player) == before


class TestGameEnd:
    def test_threshold_win(self, monkeypatch):
        monkeypatch.setattr(S, 'GRID', 4)       # 9 cells, threshold 4.5
        ring = [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2), (3, 3),
                (2, 3), (1, 3), (0, 3), (0, 2), (0, 1)]
        board = {p: 1 for p in ring[:-1]}
        board[(1, 1)] = 2                       # will be captured
        bg = BitGame.from_board(board, 1)
        assert not bg.done
        gx, gy = ring[-1]
        bg.play(gy * bg.g.W + gx)
        assert bg.done and bg.winner == 1 and bg.s1 == 9.0
        assert (1, 1) not in bg.board()

    def test_full_board_ends_by_score(self, monkeypatch):
        monkeypatch.setattr(S, 'GRID', 4)
        rnd = random.Random(3)
        bg = BitGame(geo(4))
        while not bg.done:
            bg.play(rnd.choice(bits(bg.legal())))
        assert not bg.legal() or max(bg.s1, bg.s2) >= bg.g.threshold
        if max(bg.s1, bg.s2) < bg.g.threshold:
            assert bg.winner == (None if bg.s1 == bg.s2
                                 else (1 if bg.s1 > bg.s2 else 2))

    def test_copy_is_independent(self):
        bg = BitGame()
        c = bg.copy()
        c.play(0)
        assert bg.d1 == 0 and c.d1 == 1 and bg.player == 1 and c.player == 2


class TestBits:
    def test_bits_lists_set_indices(self):
        assert bits(0b101001) == [0, 3, 5]
        assert bits(0) == []
