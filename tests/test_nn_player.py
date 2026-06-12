"""Tests for ai/nn_player.py — pure-numpy actor-critic."""
import numpy as np
import pytest

import settings as S
from ai.nn_player import NNPlayer, _ACNetwork


def _fresh_player():
    p = NNPlayer.__new__(NNPlayer)          # skip load() — fresh random net
    p.player_id = 1
    p._hidden = [16]
    p._net = _ACNetwork(15 * S.GRID * S.GRID, [16], S.GRID * S.GRID)
    p._traj, p._obs_traj = [], []
    p.save = lambda: None
    p._learning_shared = False              # added in a later task; harmless before
    return p


class TestMaskedPolicyGradient:
    def test_update_does_not_push_probability_to_illegal_moves(self):
        net = _ACNetwork(15 * S.GRID * S.GRID, [16], S.GRID * S.GRID)
        x = np.random.default_rng(1).normal(size=15 * S.GRID * S.GRID)
        legal = np.zeros(S.GRID * S.GRID, dtype=bool)
        legal[:5] = True
        probs, val, acts = net.policy(x, legal)
        for _ in range(20):
            net.update(acts, action=2, advantage=1.0, value_target=0.5,
                       entropy_bonus=0.01, lr=0.01,
                       legal_mask=legal, boost_log=None)
        probs2, _, _ = net.policy(x, legal)
        assert probs2[~legal].sum() < 1e-6
        assert probs2[2] > probs[2]          # advantage pushed the action up

    def test_update_gradient_is_zero_at_illegal_columns(self):
        net = _ACNetwork(15 * S.GRID * S.GRID, [16], S.GRID * S.GRID)
        x = np.random.default_rng(3).normal(size=15 * S.GRID * S.GRID)
        legal = np.zeros(S.GRID * S.GRID, dtype=bool)
        legal[:5] = True
        _, _, acts = net.policy(x, legal)
        w_before = net._pol_w.w.copy()
        b_before = net._pol_b.w.copy()
        net.update(acts, action=1, advantage=1.0, value_target=0.5,
                   entropy_bonus=0.01, lr=0.01,
                   legal_mask=legal, boost_log=None)
        # Policy-head columns of ILLEGAL positions must not move at all:
        # their gradient is exactly zeroed, so Adam (zero-initialised moments)
        # produces a zero step for them.
        assert np.allclose(net._pol_w.w[:, ~legal], w_before[:, ~legal])
        assert np.allclose(net._pol_b.w[~legal], b_before[~legal])
        # Legal columns DO move (sanity that the update isn't a no-op).
        assert not np.allclose(net._pol_w.w[:, legal], w_before[:, legal])


class TestTrajectoryCarriesMaskAndBoost:
    def test_choose_move_stores_six_tuple(self):
        p = _fresh_player()
        p.choose_move({(0, 0): 1, (1, 0): 1, (0, 1): 1}, set(), 1,
                      {1: 0.0, 2: 0.0}, set())
        entry = p._traj[-1]
        assert len(entry) == 6
        enc, action, acts, val, boost, legal = entry
        assert boost.shape == (S.GRID * S.GRID,)
        assert legal.dtype == bool

    def test_record_outcome_runs_clean(self):
        p = _fresh_player()
        for _ in range(3):
            p.choose_move({(0, 0): 1}, set(), 1, {1: 0.0, 2: 0.0}, set())
        p.record_outcome(1, intermediate_rewards=[0.1, 0.0, 0.2],
                         final_scores={1: 5.0, 2: 1.0})
        assert p._traj == []
