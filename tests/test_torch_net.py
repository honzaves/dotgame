"""Tests for the shared residual CNN (pytorch_player._build_net)."""
import pytest

torch = pytest.importorskip("torch")

import numpy as np
import settings as S
from ai.pytorch_player import _build_net


class TestGroupNorm:
    def test_no_batchnorm_modules(self):
        net = _build_net(S.GRID, n_blocks=2, channels=32)
        assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in net.modules())
        assert any(isinstance(m, torch.nn.GroupNorm) for m in net.modules())

    def test_no_buffers_mutated_by_inference(self):
        # GroupNorm keeps no running statistics → the net has no buffers
        # that inference could silently mutate.
        net = _build_net(S.GRID, n_blocks=2, channels=32)
        assert list(net.buffers()) == []

    def test_forward_shapes(self):
        net = _build_net(S.GRID, n_blocks=2, channels=32)
        x = torch.zeros(3, 15, S.GRID, S.GRID)
        logits, values = net(x)
        assert logits.shape == (3, S.GRID * S.GRID)
        assert values.shape == (3,)


class TestPTBoostConsistency:
    def test_stored_logprob_matches_sampling_distribution(self):
        import numpy as np
        from ai.pytorch_player import PyTorchPlayer
        p = PyTorchPlayer(player_id=1)
        p.save = lambda: None                      # don't touch disk
        board = {(0, 0): 1, (1, 0): 1, (0, 1): 1}  # 3-corner square → boost fires
        np.random.seed(0)
        move = p.choose_move(board, set(), 1, {1: 0.0, 2: 0.0}, set())
        enc, mask, action, lp_old, _, boost = p._traj[-1]
        assert move == (action % S.GRID, action // S.GRID)
        # Recompute the boosted policy and compare the stored log-prob
        x = torch.tensor(enc, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, _ = p._net(x)
        from ai.pytorch_player import _bound
        logits = _bound(logits[0]) + torch.tensor(boost)
        logits[~torch.tensor(mask)] = float('-inf')
        log_p = torch.log_softmax(logits, dim=0)
        assert abs(float(log_p[action]) - lp_old) < 1e-5


class TestPTTrainingStaysFinite:
    def _fake_traj(self, p, steps=3):
        import numpy as np
        n = S.GRID
        for i in range(steps):
            enc   = np.random.default_rng(i).normal(
                size=(15, n, n)).astype(np.float32)
            mask  = np.zeros(n * n, dtype=bool); mask[: n * n // 2] = True
            boost = np.zeros(n * n, dtype=np.float32)
            p._traj.append((enc, mask, i, -2.0, 0.1, boost))

    def test_record_outcome_keeps_parameters_finite(self):
        import numpy as np
        from ai.pytorch_player import PyTorchPlayer
        p = PyTorchPlayer(player_id=1)
        p.save = lambda: None
        self._fake_traj(p, steps=3)
        p.record_outcome(1, intermediate_rewards=[0.0, 0.1, 0.0],
                         final_scores={1: 10.0, 2: 2.0})
        assert p._traj == []
        assert all(torch.isfinite(q).all() for q in p._net.parameters())

    def test_observed_moves_training_keeps_parameters_finite(self):
        import numpy as np
        n = S.GRID
        from ai.pytorch_player import PyTorchPlayer
        p = PyTorchPlayer(player_id=1)
        p.save = lambda: None
        for i in range(3):
            enc   = np.random.default_rng(10 + i).normal(
                size=(15, n, n)).astype(np.float32)
            mask  = np.zeros(n * n, dtype=bool); mask[: n * n // 2] = True
            boost = np.zeros(n * n, dtype=np.float32)
            p._obs_traj.append((enc, mask, i, -2.0, 0.1, boost))
        p.record_outcome(2, final_scores={1: 2.0, 2: 10.0})
        assert p._obs_traj == []
        assert all(torch.isfinite(q).all() for q in p._net.parameters())
