#!/usr/bin/env python3
"""Headless training smoke test.

Runs short self-play Trainer sessions for every AI backend and asserts that
all learned parameters remain finite afterwards.  This catches training-
pipeline corruption (e.g. the post-move observe bug that NaN'd the PyTorch
weights) without opening the pygame UI.

Saves are monkeypatched to no-ops — the real weight/experience files on disk
are never written.

Usage:  $PY scripts/validate_training.py
Exit code 0 = all backends healthy.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import settings as S

S.AI_THINK_MS = 50          # keep MCTS / NM turns fast

from ai.trainer import Trainer            # noqa: E402
from ai.mcts import MCTSPlayer            # noqa: E402
from ai.nn_player import NNPlayer         # noqa: E402


def _silence_saves(*players):
    for p in players:
        p.save = lambda: None


def _run_pair(name, p1, p2, rounds, check_finite):
    _silence_saves(p1, p2)
    if type(p1) is type(p2) and hasattr(p1, 'share_learning_with'):
        p1.share_learning_with(p2)
    t = Trainer(p1, p2, rounds)
    t._run()                              # synchronous — no thread
    ok = check_finite()
    print(f"{name:14s} rounds={rounds}  wins={t.wins}  "
          f"finite={'OK' if ok else 'FAIL'}")
    return ok


def main():
    results = []

    nn1, nn2 = NNPlayer(1), NNPlayer(2)

    def nn_finite():
        net = nn1._net
        arrs = ([p.w for p in net._trunk_w] + [p.w for p in net._trunk_b]
                + [net._pol_w.w, net._pol_b.w, net._val_w.w, net._val_b.w])
        return all(np.isfinite(a).all() for a in arrs)

    results.append(_run_pair("NN   vs NN", nn1, nn2, 5, nn_finite))

    m1, m2 = MCTSPlayer(1, profile=1), MCTSPlayer(2, profile=2)

    def mcts_finite():
        return all(np.isfinite(e['wins']) for e in m1.experience.values())

    results.append(_run_pair("MCTS vs MCTS", m1, m2, 2, mcts_finite))

    try:
        import torch
        from ai.pytorch_player import PyTorchPlayer
        from ai.neural_mcts_player import NeuralMCTSPlayer

        pt1, pt2 = PyTorchPlayer(1), PyTorchPlayer(2)

        def pt_finite():
            return all(torch.isfinite(p).all()
                       for p in pt1._net.parameters())

        results.append(_run_pair("PT   vs PT", pt1, pt2, 5, pt_finite))

        nm1, nm2 = NeuralMCTSPlayer(1), NeuralMCTSPlayer(2)

        def nm_finite():
            return all(torch.isfinite(p).all()
                       for p in nm1._net.parameters())

        results.append(_run_pair("NM   vs NM", nm1, nm2, 2, nm_finite))
    except ImportError:
        print("torch not installed — skipping PT / NM checks")

    if not all(results):
        print("FAILED: at least one backend produced non-finite parameters.")
        sys.exit(1)
    print("All training smoke tests passed.")


if __name__ == "__main__":
    main()
