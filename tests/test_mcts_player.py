"""Tests for ai/mcts.py — MCTSPlayer wrapper (book, persistence, API)."""
import json

import pytest

import settings as S
from ai.mcts import MCTSPlayer
from ai.symmetry import canonical_board_key


@pytest.fixture
def exp_path(tmp_path, monkeypatch):
    """Point the experience file at a temp location and keep turns short."""
    monkeypatch.setattr(S, 'AI_EXPERIENCE_BASE', str(tmp_path / 'exp'))
    monkeypatch.setattr(S, 'AI_THINK_MS', 40)
    return tmp_path / f'exp_{S.GRID}x{S.GRID}.json'


class TestChooseMove:
    def test_returns_legal_move_and_records_position(self, exp_path):
        p = MCTSPlayer(1)
        board = {(4, 4): 2}
        move = p.choose_move(board, set(), 1, {1: 0.0, 2: 0.0}, set())
        assert move not in board
        assert 0 <= move[0] < S.GRID and 0 <= move[1] < S.GRID
        after = dict(board); after[move] = 1
        assert p._game_moves == [canonical_board_key(after)]

    def test_respects_forbidden(self, exp_path):
        p = MCTSPlayer(1)
        forbidden = {(x, y) for x in range(S.GRID) for y in range(S.GRID)
                     if (x, y) != (3, 3)}
        assert p.choose_move({}, set(), 1, {1: 0.0, 2: 0.0}, forbidden) == (3, 3)

    def test_profiles_set_weights(self, exp_path):
        opp, strat = MCTSPlayer(1, profile=1), MCTSPlayer(2, profile=2)
        w1, w2 = opp._searcher.cfg.weights, strat._searcher.cfg.weights
        assert w1.gain > w2.gain and w2.race > w1.race
        default = MCTSPlayer(1)._searcher.cfg.weights
        assert default.race == S.STRAT_RACE_W


class TestExperience:
    def test_outcome_stored_from_movers_perspective(self, exp_path):
        p = MCTSPlayer(1)
        p._game_moves = ['mine']
        p._obs_moves = ['theirs']
        p.record_outcome(1, final_scores={1: 30.0, 2: 10.0})
        assert p.experience['mine'] == {'wins': 0.75, 'visits': 1}
        assert p.experience['theirs'] == {'wins': 0.25, 'visits': 1}
        assert p._game_moves == [] and p._obs_moves == []

    def test_draw_counts_half(self, exp_path):
        p = MCTSPlayer(1)
        p._game_moves = ['k']
        p.record_outcome(0)
        assert p.experience['k']['wins'] == 0.5

    def test_save_load_roundtrip(self, exp_path):
        p = MCTSPlayer(1)
        p._game_moves = ['k']
        p.record_outcome(1)
        assert json.loads(exp_path.read_text())['version'] == 2
        assert MCTSPlayer(2).experience == p.experience

    def test_old_format_file_is_ignored(self, exp_path):
        exp_path.write_text(json.dumps({'some-key': {'wins': 1.0, 'visits': 2}}))
        assert MCTSPlayer(1).experience == {}

    def test_book_seeds_remembered_positions(self, exp_path):
        p = MCTSPlayer(1)
        key = canonical_board_key({(4, 4): 1})
        p.experience[key] = {'wins': 4.0, 'visits': 5}
        from ai.engine import BitGame
        bg = BitGame()
        seed = p._book(bg)
        i = 4 * bg.g.W + 4
        assert seed[i] == (5, 0.8)
        # all 8 symmetric images of the position share the entry
        assert len(seed) == len({(4, 4), (5, 4), (4, 5), (5, 5)})

    def test_book_only_in_the_opening(self, exp_path, monkeypatch):
        monkeypatch.setattr(S, 'MCTS_BOOK_MAX_DOTS', 1)
        p = MCTSPlayer(1)
        p.experience['x'] = {'wins': 1.0, 'visits': 3}
        from ai.engine import BitGame
        bg = BitGame.from_board({(0, 0): 1, (9, 9): 2}, 1)
        assert p._book(bg) == {}
