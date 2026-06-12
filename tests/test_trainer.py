"""Tests for ai/trainer.py — the offline self-play trainer."""
import settings as S
from ai.base_player import BasePlayer
from ai.trainer import Trainer


class _StubPlayer(BasePlayer):
    """Deterministic player that records every observation it receives."""

    def __init__(self, pid):
        self.player_id = pid
        self.observed = []      # (board_snapshot, move) per observe call
        self.chosen = 0

    def choose_move(self, board, connections, player, scores, forbidden=None):
        self.chosen += 1
        forbidden = forbidden or set()
        for gx in range(S.GRID):
            for gy in range(S.GRID):
                if (gx, gy) not in board and (gx, gy) not in forbidden:
                    return (gx, gy)
        return (0, 0)

    def observe_opponent_move(self, board, connections, player_who_moved,
                              move, forbidden=None):
        self.observed.append((dict(board), move))

    def record_outcome(self, winner, intermediate_rewards=None,
                       final_scores=None):
        pass

    def save(self):
        pass

    def load(self):
        pass


class TestObserveContract:
    def _run_one_game(self):
        p1, p2 = _StubPlayer(1), _StubPlayer(2)
        t = Trainer(p1, p2, rounds=1)
        t._run()        # synchronous — no thread
        return p1, p2

    def test_observe_called_exactly_once_per_opponent_move(self):
        p1, p2 = self._run_one_game()
        # p1 observes every p2 move exactly once, and vice versa
        assert len(p1.observed) == p2.chosen
        assert len(p2.observed) == p1.chosen

    def test_observe_receives_pre_move_board(self):
        p1, p2 = self._run_one_game()
        for board, move in p1.observed + p2.observed:
            assert move not in board, (
                "observe_opponent_move must receive the board BEFORE the move")
