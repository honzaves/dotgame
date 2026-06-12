"""Tests for ai/replay_trainer.py — replay learning must actually learn."""
import settings as S
from ai.base_player import BasePlayer
from ai.replay_trainer import ReplayTrainer


class _StubPlayer(BasePlayer):
    def __init__(self, pid):
        self.player_id = pid
        self.observes = 0
        self.outcomes = []
        self._learning_shared = False

    def choose_move(self, board, connections, player, scores, forbidden=None):
        return (0, 0)

    def observe_opponent_move(self, board, connections, player_who_moved,
                              move, forbidden=None):
        if self._learning_shared:
            return
        self.observes += 1

    def record_outcome(self, winner, intermediate_rewards=None,
                       final_scores=None):
        self.outcomes.append(winner)

    def save(self):
        pass

    def load(self):
        pass


def _fake_game(n_moves=6):
    moves = []
    for i in range(n_moves):
        moves.append([1 + (i % 2), i % S.GRID, i // S.GRID])
    return {'moves': moves, 'winner': 1,
            'final_scores': {1: 4.0, 2: 1.0}, 'grid': S.GRID}


class TestReplayDrivesLearning:
    def test_both_players_observe_every_move(self):
        p1, p2 = _StubPlayer(1), _StubPlayer(2)
        t = ReplayTrainer(p1, p2, repeats=1)
        t._replay_one(_fake_game(6))
        assert p1.observes == 6 and p2.observes == 6
        assert p1.outcomes == [1] and p2.outcomes == [1]

    def test_shared_players_would_learn_nothing(self):
        # Documents WHY dotgame.py must not link replay players via
        # share_learning_with: the observe gate silences the only signal.
        p1, p2 = _StubPlayer(1), _StubPlayer(2)
        p1._learning_shared = p2._learning_shared = True
        t = ReplayTrainer(p1, p2, repeats=1)
        t._replay_one(_fake_game(6))
        assert p1.observes == 0 and p2.observes == 0
