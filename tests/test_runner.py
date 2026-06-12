"""Tests for ai/runner.py — AIRunner wrapper."""
import inspect

import ai.runner
from ai.runner import AIRunner
from ai.base_player import BasePlayer


class _Recorder(BasePlayer):
    def __init__(self):
        self.player_id = 1
        self.calls = []

    def choose_move(self, board, connections, player, scores, forbidden=None):
        return (0, 0)

    def observe_opponent_move(self, board, connections, player_who_moved,
                              move, forbidden=None):
        self.calls.append((board, player_who_moved, move, forbidden))


class TestObserveForwarding:
    def test_single_definition(self):
        src = inspect.getsource(ai.runner)
        assert src.count("def observe_opponent_move") == 1

    def test_forwards_arguments(self):
        p = _Recorder()
        r = AIRunner(p)
        r.observe_opponent_move({(0, 0): 1}, set(), 1, (0, 0), {(5, 5)})
        assert p.calls == [({(0, 0): 1}, 1, (0, 0), {(5, 5)})]
