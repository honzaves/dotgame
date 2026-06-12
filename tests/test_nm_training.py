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
