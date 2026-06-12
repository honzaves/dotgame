"""Tests for the shared residual CNN (pytorch_player._build_net)."""
import pytest

torch = pytest.importorskip("torch")

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
