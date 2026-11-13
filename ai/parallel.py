"""Worker processes for root-parallel search.

Python threads cannot run simulations in parallel, so the Searcher farms
them out: each worker process receives the root position with its priors,
runs its own independent tree for the thinking time and reports the visit
statistics of the root moves; the Searcher adds them to its own.  Search
strength grows steeply with the number of simulations, so this is the
cheapest strength there is on a multi-core machine.

Workers are plain subprocesses (`python -m ai.parallel`) talking pickle over
stdin/stdout — no multiprocessing module, so the game's main module (pygame,
torch) is never re-imported in a worker.  They exit when their stdin closes,
i.e. when the game exits.  Every failure path (worker crash, timeout, pipe
error) shuts the pool down and the Searcher silently continues on its own.
"""

import atexit
import os
import pickle
import queue
import struct
import subprocess
import sys
import threading
import time

import settings as S


def _send(stream, obj) -> None:
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(struct.pack('<I', len(data)))
    stream.write(data)
    stream.flush()


def _recv(stream):
    head = stream.read(4)
    if len(head) < 4:
        raise EOFError
    (size,) = struct.unpack('<I', head)
    data = stream.read(size)
    if len(data) < size:
        raise EOFError
    return pickle.loads(data)


class _Worker:

    def __init__(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.proc = subprocess.Popen(
            [sys.executable, '-m', 'ai.parallel'], cwd=root,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL)
        self.results: queue.Queue = queue.Queue()
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self) -> None:
        try:
            while True:
                self.results.put(_recv(self.proc.stdout))
        except Exception:
            self.results.put(None)             # worker died

    def close(self) -> None:
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.kill()
        except Exception:
            pass


class Pool:
    """A fixed set of search workers.  Use via pool()."""

    def __init__(self, n: int):
        self.workers = [_Worker() for _ in range(n)]
        self.lock = threading.Lock()           # one search at a time
        self.dead = False
        atexit.register(self.close)

    def submit(self, job: dict) -> bool:
        """Send *job* to every worker.  False if the pool is unusable."""
        try:
            for w in self.workers:
                _send(w.proc.stdin, job)
            return True
        except Exception:
            self.close()
            return False

    def collect(self, timeout: float) -> list:
        """One result per worker (those that answered in time)."""
        out = []
        deadline = time.time() + timeout
        for w in self.workers:
            try:
                r = w.results.get(timeout=max(0.0, deadline - time.time()))
            except queue.Empty:
                r = None
            if r is None:                      # crash or timeout: give up on
                self.close()                   # the pool, keep what we have
                break
            out.append(r)
        return out

    def close(self) -> None:
        self.dead = True
        for w in self.workers:
            w.close()
        self.workers = []


_POOL = None


def pool():
    """The shared worker pool, or None when parallel search is off
    (SEARCH_WORKERS = 0) or has failed.  Created on first use."""
    global _POOL
    n = worker_count()
    if n <= 0:
        return None
    if _POOL is None:
        try:
            _POOL = Pool(n)
        except Exception:
            _POOL = False
    if _POOL is False or _POOL.dead:
        return None
    return _POOL


def worker_count() -> int:
    """Resolve S.SEARCH_WORKERS (None = automatic: leave two cores free for
    the UI and the main search, at most 8 workers)."""
    n = S.SEARCH_WORKERS
    if n is None:
        n = max(0, min(8, (os.cpu_count() or 1) - 2))
    return n


# ── Worker process ────────────────────────────────────────────────────────────

def _serve() -> None:
    from ai.engine import BitGame, geo
    from ai.search import Searcher, SearchConfig, _Node

    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    while True:
        try:
            job = _recv(stdin)
        except EOFError:
            return
        S.GRID, S.WIN_PCT = job['grid'], job['win_pct']
        bg = BitGame(geo(S.GRID))
        (bg.d1, bg.d2, bg.forb, bg.s1, bg.s2, bg.player, bg.dirty) = job['state']
        me = bg.player
        searcher = Searcher(SearchConfig(c_puct=job['c_puct'],
                                         prior_temp=job['prior_temp']))
        root = _Node(None, None, 3 - me)
        root.children = [_Node(i, root, me, p) for i, p in job['priors'].items()]
        sims = searcher._simulate(root, bg, job['maps'],
                                  time.time() + job['think_s'])
        _send(stdout, (sims, {c.move: (c.N, c.V) for c in root.children}))


if __name__ == '__main__':
    _serve()
