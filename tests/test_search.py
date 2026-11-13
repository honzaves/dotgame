"""Tests for ai/search.py — the exact-rules tree search."""
import pytest

import settings as S
from ai.engine import BitGame, bits
from ai.search import Searcher, SearchConfig


def _border(n):
    return [(x, y) for y in range(n) for x in range(n)
            if x in (0, n - 1) or y in (0, n - 1)]


@pytest.fixture
def small(monkeypatch):
    monkeypatch.setattr(S, 'GRID', 5)
    return 5


class TestSearch:
    def test_returns_legal_move_and_counts_visits(self):
        bg = BitGame.from_board({(4, 4): 2}, 1)
        res = Searcher().search(bg, 0.25)
        assert bg.legal() >> res.move & 1
        assert res.sims > 0
        assert sum(res.visits.values()) == res.sims
        assert 0.0 <= res.value <= 1.0

    def test_plays_immediate_win_without_searching(self, small):
        board = {p: 1 for p in _border(5) if p != (2, 0)}
        bg = BitGame.from_board(board, 1)
        res = Searcher().search(bg, 0.2)
        assert bg.g.xy[res.move] == (2, 0)
        assert res.sims == 0

    def test_blocks_the_only_closing_point(self, small):
        board = {p: 1 for p in _border(5) if p != (2, 0)}
        board[(2, 1)] = 2
        bg = BitGame.from_board(board, 2)
        res = Searcher().search(bg, 0.3)
        assert bg.g.xy[res.move] == (2, 0)

    def test_single_legal_move(self, small):
        full = {(x, y): 1 + (x + y) % 2 for x in range(5) for y in range(5)}
        del full[(4, 4)]
        bg = BitGame.from_board(full, 1, {1: 0.0, 2: 0.0}, set())
        res = Searcher().search(bg, 0.1)
        assert bg.g.xy[res.move] == (4, 4)

    def test_short_think_uses_fast_path(self):
        bg = BitGame.from_board({(4, 4): 2, (5, 5): 1}, 1)
        res = Searcher().search(bg, 0.01)
        assert bg.legal() >> res.move & 1

    def test_unknown_config_option_rejected(self):
        with pytest.raises(TypeError):
            SearchConfig(no_such_option=1)


class TestTreeReuse:
    def _after(self, bg, *moves):
        c = bg.copy()
        for m in moves:
            c.play(m)
        return c

    def test_subtree_reused_after_expected_reply(self):
        s = Searcher()
        bg = BitGame()
        res = s.search(bg, 0.3)
        node, after = s._saved
        assert node.children, "chosen move should have been expanded"
        reply = max(node.children, key=lambda c: c.N)
        inherited = reply.N
        nxt = self._after(after, reply.move)
        root = s._reuse(nxt)
        assert root is reply and root.parent is None and root.N == inherited

    def test_discarded_when_position_does_not_follow(self):
        s = Searcher()
        bg = BitGame()
        s.search(bg, 0.15)
        other = BitGame.from_board({(0, 0): 1, (9, 9): 2, (5, 5): 2}, 1)
        assert s._reuse(other) is None
        assert s._saved is None

    def test_reset_forgets_subtree(self):
        s = Searcher()
        s.search(BitGame(), 0.1)
        s.reset()
        assert s._saved is None


class TestHooks:
    def test_prior_mix_and_seed_visits_are_applied(self):
        bg = BitGame.from_board({(4, 4): 2}, 1)
        target = bits(bg.legal())[0]            # a corner — heuristically poor
        s = Searcher(SearchConfig(noise_frac=0.0))
        res = s.search(bg, 0.25,
                       prior_mix={target: 1.0}, mix_weight=0.9,
                       seed_visits={target: (500.0, 1.0)})
        # Heavily seeded + heavily mixed prior → the search follows it
        assert res.move == target
        # seeded (virtual) visits are not reported as simulations
        assert sum(res.visits.values()) == res.sims

    def test_seed_fn_receives_root_scores(self):
        bg = BitGame.from_board({(4, 4): 2}, 1)
        seen = {}

        def seed_fn(b, scores):
            seen['n'] = len(scores)
            return {}

        Searcher().search(bg, 0.1, seed_fn=seed_fn)
        assert seen['n'] == len(bits(bg.legal()))
