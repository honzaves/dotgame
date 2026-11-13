"""Tests for ai/parallel.py — worker processes for root-parallel search."""
import pytest

import settings as S
from ai import parallel
from ai.engine import BitGame
from ai.search import Searcher


@pytest.fixture
def workers(monkeypatch):
    monkeypatch.setattr(S, 'SEARCH_WORKERS', 2)
    monkeypatch.setattr(parallel, '_POOL', None)
    yield
    if parallel._POOL:
        parallel._POOL.close()


class TestWorkerCount:
    def test_zero_disables_pool(self, monkeypatch):
        monkeypatch.setattr(S, 'SEARCH_WORKERS', 0)
        assert parallel.worker_count() == 0 and parallel.pool() is None

    def test_automatic_leaves_cores_free(self, monkeypatch):
        monkeypatch.setattr(S, 'SEARCH_WORKERS', None)
        assert 0 <= parallel.worker_count() <= 8


class TestParallelSearch:
    def test_workers_add_simulations(self, workers):
        bg = BitGame.from_board({(4, 4): 2}, 1)
        solo_pool, S.SEARCH_WORKERS = S.SEARCH_WORKERS, 0
        solo = Searcher().search(bg, 0.5)
        S.SEARCH_WORKERS = solo_pool
        Searcher().search(bg, 0.5)             # warm-up: starts the workers
        res = Searcher().search(bg, 0.5)
        assert bg.legal() >> res.move & 1
        assert sum(res.visits.values()) == res.sims
        assert res.sims > 1.5 * solo.sims

    def test_dead_pool_falls_back_to_single_process(self, workers):
        bg = BitGame.from_board({(4, 4): 2}, 1)
        pool = parallel.pool()
        for w in pool.workers:
            w.proc.kill()
        res = Searcher().search(bg, 0.4)
        assert bg.legal() >> res.move & 1 and res.sims > 0
        assert parallel.pool() is None         # pool gave up, game goes on

    def test_short_searches_stay_single_process(self, workers):
        bg = BitGame.from_board({(4, 4): 2}, 1)
        Searcher().search(bg, 0.05)
        assert parallel._POOL is None          # never even started
