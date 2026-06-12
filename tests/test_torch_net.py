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
        logits = logits[0] + torch.tensor(boost)
        logits[~torch.tensor(mask)] = float('-inf')
        log_p = torch.log_softmax(logits, dim=0)
        assert abs(float(log_p[action]) - lp_old) < 1e-5
