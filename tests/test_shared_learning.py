"""Tests for share_learning_with() — shared self-play learning."""
import numpy as np
import pytest

import settings as S
from ai.mcts import MCTSPlayer
from ai.nn_player import NNPlayer


class TestMCTSSharing:
    def test_experience_dict_is_shared(self):
        p1 = MCTSPlayer(1); p1.save = lambda: None
        p2 = MCTSPlayer(2); p2.save = lambda: None
        p1.share_learning_with(p2)
        assert p2.experience is p1.experience

    def test_observe_disabled_when_shared(self):
        p1 = MCTSPlayer(1); p1.save = lambda: None
        p2 = MCTSPlayer(2); p2.save = lambda: None
        p1.share_learning_with(p2)
        p1.observe_opponent_move({}, set(), 2, (3, 3))
        p2.observe_opponent_move({}, set(), 1, (4, 4))
        assert p1._obs_moves == [] and p2._obs_moves == []


class TestNNSharing:
    def test_network_is_shared(self):
        p1 = NNPlayer(1); p1.save = lambda: None
        p2 = NNPlayer(2); p2.save = lambda: None
        p1.share_learning_with(p2)
        assert p2._net is p1._net

    def test_observe_disabled_when_shared(self):
        p1 = NNPlayer(1); p1.save = lambda: None
        p2 = NNPlayer(2); p2.save = lambda: None
        p1.share_learning_with(p2)
        p2.observe_opponent_move({(0, 0): 1}, set(), 1, (0, 0), set())
        assert p2._obs_traj == []


torch = pytest.importorskip("torch")
from ai.pytorch_player import PyTorchPlayer
from ai.neural_mcts_player import NeuralMCTSPlayer


class TestTorchSharing:
    def test_pt_net_and_opt_shared(self):
        p1 = PyTorchPlayer(1); p1.save = lambda: None
        p2 = PyTorchPlayer(2); p2.save = lambda: None
        p1.share_learning_with(p2)
        assert p2._net is p1._net and p2._opt is p1._opt

    def test_nm_net_and_buffer_shared(self):
        p1 = NeuralMCTSPlayer(1); p1.save = lambda: None
        p2 = NeuralMCTSPlayer(2); p2.save = lambda: None
        p1.share_learning_with(p2)
        assert p2._net is p1._net and p2._opt is p1._opt
