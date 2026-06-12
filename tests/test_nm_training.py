"""Tests for NeuralMCTSPlayer training (buffer, value targets, augmentation)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

import settings as S
from ai.neural_mcts_player import NeuralMCTSPlayer


def _fake_episode(player, steps=3):
    n = S.GRID
    for i in range(steps):
        enc  = np.random.default_rng(i).normal(size=(15, n, n)).astype(np.float32)
        mask = np.ones(n * n, dtype=bool)
        dist = np.zeros(n * n, dtype=np.float32); dist[i] = 1.0
        player._episode.append((enc, mask, dist, 0.0))


class TestValueTargets:
    def test_pure_z_when_weight_is_one(self, monkeypatch):
        monkeypatch.setattr(S, 'NM_Z_WEIGHT', 1.0)
        p = NeuralMCTSPlayer(1); p.save = lambda: None
        targets = p._value_targets([0.2, -0.1, 0.4], None, terminal=0.8)
        assert np.allclose(targets, 0.8)

    def test_clipped_to_unit_interval(self):
        p = NeuralMCTSPlayer(1); p.save = lambda: None
        targets = p._value_targets([0.9, 0.9], [5.0, 5.0], terminal=1.0)
        assert (targets <= 1.0).all() and (targets >= -1.0).all()


class TestBufferTraining:
    def test_record_outcome_fills_buffer_and_stays_finite(self):
        p = NeuralMCTSPlayer(1); p.save = lambda: None
        _fake_episode(p, steps=3)
        p.record_outcome(1, intermediate_rewards=[0.0, 0.1, 0.0],
                         final_scores={1: 10.0, 2: 2.0})
        assert len(p._buffer) == 3
        assert p._episode == []
        assert all(torch.isfinite(q).all() for q in p._net.parameters())

    def test_buffer_capped(self, monkeypatch):
        monkeypatch.setattr(S, 'NM_BUFFER_MAX', 4)
        p = NeuralMCTSPlayer(1); p.save = lambda: None
        for _ in range(3):
            _fake_episode(p, steps=2)
            p.record_outcome(1, final_scores={1: 5.0, 2: 1.0})
        assert len(p._buffer) == 4


class TestTreeReuse:
    def _node(self, move, parent, who):
        from ai.neural_mcts_player import _NMNode
        return _NMNode(move=move, parent=parent, children=None,
                       player_who_moved=who)

    def test_promotes_matching_grandchild(self):
        p = NeuralMCTSPlayer(1); p.save = lambda: None
        # Hand-built saved subtree: our chosen move (2,2), opponent replies
        chosen = self._node((2, 2), None, 1)
        chosen.children = [self._node((3, 3), chosen, 2),
                           self._node((4, 4), chosen, 2)]
        chosen.children[0].visits = 7
        p._saved_root  = chosen
        p._saved_board = {(2, 2): 1}
        new_root = p._try_reuse_tree({(2, 2): 1, (3, 3): 2})
        assert new_root is chosen.children[0]
        assert new_root.visits == 7
        assert p._saved_root is None        # state cleared after reuse

    def test_discards_on_removed_dots(self):
        p = NeuralMCTSPlayer(1); p.save = lambda: None
        p._saved_root  = self._node((2, 2), None, 1)
        p._saved_board = {(2, 2): 1, (5, 5): 2}
        assert p._try_reuse_tree({(2, 2): 1, (0, 0): 2}) is None

    def test_choose_move_saves_subtree(self):
        S.AI_THINK_MS = 30
        p = NeuralMCTSPlayer(1); p.save = lambda: None
        p.choose_move({}, set(), 1, {1: 0.0, 2: 0.0}, set())
        assert p._saved_root is not None
        assert p._saved_board is not None


class TestVirtualLoss:
    def test_apply_and_revert_restores_counts(self):
        from ai.neural_mcts_player import _NMNode, NeuralMCTSPlayer
        root  = _NMNode(None, None, [], player_who_moved=2)
        child = _NMNode((1, 1), root, None, player_who_moved=1)
        root.children = [child]
        root.visits, root.total_value   = 5, 2.5
        child.visits, child.total_value = 3, 1.0
        NeuralMCTSPlayer._apply_virtual(child, ai_player=1, sign=+1)
        assert (child.visits, root.visits) == (4, 6)
        NeuralMCTSPlayer._apply_virtual(child, ai_player=1, sign=-1)
        assert (child.visits, child.total_value) == (3, 1.0)
        assert (root.visits, root.total_value)   == (5, 2.5)


class TestBatchedSearch:
    def test_choose_move_returns_legal_and_counts_consistent(self):
        S.AI_THINK_MS = 60
        p = NeuralMCTSPlayer(1); p.save = lambda: None
        board = {(4, 4): 2}
        move = p.choose_move(board, set(), 1, {1: 0.0, 2: 0.0}, set())
        assert move not in board
        root = p._saved_root        # chosen child saved by tree reuse
        assert root.visits >= 1
        assert root.visits >= sum(c.visits for c in (root.children or []))
        def no_negative(n):
            assert n.visits >= 0
            for c in (n.children or []):
                no_negative(c)
        no_negative(root)
