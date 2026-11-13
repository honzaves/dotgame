#!/usr/bin/env python3
"""Headless matches between AI players — a strength regression check.

Plays full games under the real rules (SimGame) and prints the score.  Use
it after touching the engine, the ring analysis, the move scoring or the
search to see whether the AIs got stronger or weaker.

Players
  mcts, mcts1, mcts2   MCTSPlayer (default / Opportunist / Strategist profile)
  nn, pt, nm           the three network players (current weight files)
  greedy               plays the best strategy.move_scores() move, no search
  ringbot              scripted "human-like" opponent: builds its cheapest
                       winning ring and ignores what the other side does —
                       the plan that used to beat every AI
  random               uniformly random legal moves

Nothing is written to disk: saves are disabled and the MCTS experience file
is redirected to a temporary directory.

Usage:  $PY scripts/arena.py mcts ringbot --games 10 --think 500
"""
import argparse
import math
import os
import random
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import settings as S                         # noqa: E402
from ai import strategy                      # noqa: E402
from ai.engine import BitGame, bits          # noqa: E402
from ai.trainer import SimGame               # noqa: E402


class _Scripted:
    """Minimal BasePlayer-compatible bot."""

    def observe_opponent_move(self, *a, **k):
        pass

    def record_outcome(self, *a, **k):
        pass


class RandomBot(_Scripted):
    def choose_move(self, board, connections, player, scores, forbidden=None):
        bg = BitGame.from_board(board, player, scores, forbidden)
        return bg.g.xy[random.choice(bits(bg.legal()))]


class GreedyBot(_Scripted):
    """Best heuristic move, ties broken at random."""

    def choose_move(self, board, connections, player, scores, forbidden=None):
        bg = BitGame.from_board(board, player, scores, forbidden)
        sc = strategy.move_scores(bg)
        best = max(sc.values())
        return bg.g.xy[random.choice(
            [i for i, v in sc.items() if v >= best - 1e-9])]


class RingBot(_Scripted):
    """Builds its cheapest winning ring; takes immediate gains; never
    defends."""

    def choose_move(self, board, connections, player, scores, forbidden=None):
        bg = BitGame.from_board(board, player, scores, forbidden)
        pa = strategy.analyse_position(bg)
        own_gain, _ = strategy.tactical_gains(bg)
        tc, tu = pa.tc[player], pa.tu[player]

        def key(i):
            g = own_gain.get(i, 0.0)
            return (g == math.inf, -tc.get(i, strategy.INF_W), g,
                    tu.get(i, 0.0), random.random())

        return bg.g.xy[max(bits(bg.legal()), key=key)]


def make_player(kind: str, player_id: int):
    if kind == 'random':
        return RandomBot()
    if kind == 'greedy':
        return GreedyBot()
    if kind == 'ringbot':
        return RingBot()
    if kind in ('mcts', 'mcts1', 'mcts2'):
        from ai.mcts import MCTSPlayer
        p = MCTSPlayer(player_id, profile=int(kind[4:] or 0))
    elif kind == 'nn':
        from ai.nn_player import NNPlayer
        p = NNPlayer(player_id)
    elif kind == 'pt':
        from ai.pytorch_player import PyTorchPlayer
        p = PyTorchPlayer(player_id)
    elif kind == 'nm':
        from ai.neural_mcts_player import NeuralMCTSPlayer
        p = NeuralMCTSPlayer(player_id)
    else:
        raise SystemExit(f"unknown player {kind!r}")
    p.save = lambda: None                    # never touch the weight files
    return p


def play_game(p1, p2):
    """One game; returns (winner or None, final scores, number of moves)."""
    game = SimGame()
    players = {1: p1, 2: p2}
    moves = 0
    while not game.game_over:
        cp = game.current
        board, conns = dict(game.board), set(game.connections)
        forb = set(game.forbidden)
        move = players[cp].choose_move(board, conns, cp, dict(game.scores), forb)
        if not game.place(*move):
            raise RuntimeError(f"player {cp} returned illegal move {move}")
        players[3 - cp].observe_opponent_move(board, conns, cp, move, forb)
        moves += 1
    return game.winner, dict(game.scores), moves


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('a', help="first player (see module docstring)")
    ap.add_argument('b', help="second player")
    ap.add_argument('--games', type=int, default=6)
    ap.add_argument('--think', type=int, default=S.AI_THINK_DEFAULT,
                    help="thinking time per move in ms")
    ap.add_argument('--seed', type=int, default=None)
    args = ap.parse_args()

    S.AI_THINK_MS = args.think
    S.AI_EXPERIENCE_BASE = os.path.join(tempfile.mkdtemp(), 'arena_experience')
    if args.seed is not None:
        random.seed(args.seed)

    wins = {'a': 0, 'b': 0, 'draw': 0}
    margin = 0.0
    t0 = time.time()
    for g in range(args.games):
        a_is = 1 if g % 2 == 0 else 2        # alternate colours
        pa, pb = make_player(args.a, a_is), make_player(args.b, 3 - a_is)
        winner, scores, moves = play_game(*((pa, pb) if a_is == 1 else (pb, pa)))
        for p in (pa, pb):
            p.record_outcome(winner or 0, final_scores=scores)
        who = 'draw' if winner is None else ('a' if winner == a_is else 'b')
        wins[who] += 1
        margin += scores[a_is] - scores[3 - a_is]
        print(f"game {g + 1}: {args.a} as P{a_is}  "
              f"{scores[a_is]:.1f} – {scores[3 - a_is]:.1f}  "
              f"({moves} moves)  → {who}", flush=True)

    print(f"\n{args.a} {wins['a']}  –  {args.b} {wins['b']}  "
          f"(draws {wins['draw']}),  average margin for {args.a}: "
          f"{margin / max(args.games, 1):+.1f},  {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
