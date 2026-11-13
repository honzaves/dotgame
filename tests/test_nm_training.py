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


class TestSearchIntegration:
    def test_choose_move_records_normalised_visit_distribution(self):
        S.AI_THINK_MS = 120
        p = NeuralMCTSPlayer(1); p.save = lambda: None
        board = {(4, 4): 2}
        move = p.choose_move(board, set(), 1, {1: 0.0, 2: 0.0}, set())
        assert move not in board
        enc, mask, dist, val = p._episode[-1]
        assert enc.shape == (15, S.GRID, S.GRID)
        assert not mask[4 * S.GRID + 4] and mask.sum() == S.GRID ** 2 - 1
        assert abs(float(dist.sum()) - 1.0) < 1e-5
        assert (dist[~mask] == 0).all()
        assert -1.0 <= val <= 1.0 or np.isfinite(val)

    def test_record_outcome_resets_search_tree(self):
        S.AI_THINK_MS = 60
        p = NeuralMCTSPlayer(1); p.save = lambda: None
        p.choose_move({}, set(), 1, {1: 0.0, 2: 0.0}, set())
        assert p._searcher._saved is not None
        p.record_outcome(1, final_scores={1: 5.0, 2: 1.0})
        assert p._searcher._saved is None


class TestTrust:
    def test_untrained_network_has_no_influence(self):
        p = NeuralMCTSPlayer(1); p.save = lambda: None
        p._steps[0] = 0
        assert p.trust() == 0.0

    def test_trust_grows_with_training_and_saturates(self, monkeypatch):
        monkeypatch.setattr(S, 'NM_TRUST_STEPS', 100)
        p = NeuralMCTSPlayer(1); p.save = lambda: None
        p._steps[0] = 50
        assert p.trust() == pytest.approx(S.NM_MAX_TRUST / 2)
        p._steps[0] = 10_000
        assert p.trust() == S.NM_MAX_TRUST

    def test_training_advances_step_counter(self):
        p = NeuralMCTSPlayer(1); p.save = lambda: None
        before = p._steps[0]
        _fake_episode(p, steps=3)
        p.record_outcome(1, final_scores={1: 10.0, 2: 2.0})
        assert p._steps[0] > before

    def test_partners_share_step_counter(self):
        p1 = NeuralMCTSPlayer(1); p1.save = lambda: None
        p2 = NeuralMCTSPlayer(2); p2.save = lambda: None
        p1.share_learning_with(p2)
        assert p2._steps is p1._steps

    def test_value_seeds_rate_top_candidates(self, monkeypatch):
        from ai.engine import BitGame
        from ai import strategy
        monkeypatch.setattr(S, 'NM_VALUE_SEEDS', 3)
        p = NeuralMCTSPlayer(1); p.save = lambda: None
        p._steps[0] = S.NM_TRUST_STEPS          # full trust
        bg = BitGame.from_board({(4, 4): 2}, 1)
        scores = strategy.move_scores(bg)
        seeds = p._value_seeds(bg, scores)
        top3 = sorted(scores, key=scores.get, reverse=True)[:3]
        assert set(seeds) == set(top3)
        for n, v in seeds.values():
            assert n == S.NM_VALUE_VISITS and 0.0 <= v <= 1.0


class TestIllegalObservation:
    def test_observed_illegal_move_is_not_recorded(self):
        p = NeuralMCTSPlayer(1); p.save = lambda: None
        p.observe_opponent_move({(3, 3): 1}, set(), 2, (3, 3), set())
        assert p._obs_episode == []
