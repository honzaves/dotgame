# AI Player Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the verified training-corruption bugs in the dotgame AI players and implement the full set of agreed improvements: GroupNorm, boost-as-logits PPO fix, shared self-play learning, rollout throughput, Dirichlet floor, D4 symmetry, MCTS experience canonicalisation, NM replay buffer + z-blend value targets, NM tree reuse, NM leaf batching, and a headless validation script.

**Architecture:** All changes live in `ai/` plus small wiring edits in `dotgame.py` and new constants in `settings.py`. A new `ai/symmetry.py` module provides D4 board transforms shared by MCTS experience keys and Neural-MCTS training augmentation. No UI changes.

**Tech Stack:** Python 3.10+, numpy, pygame, PyTorch (optional backends), pytest.

---

## Environment notes (read first)

- **Interpreter:** the project `.venv` is empty. The only interpreter on this machine with pygame + numpy + pytest + torch is:
  ```
  PY=/Library/Frameworks/Python.framework/Versions/3.13/bin/python3
  ```
  Run all tests as `$PY -m pytest tests/ -q`. Define `PY` in your shell at the start of every session.
- **Baseline:** 176 tests pass before this plan starts. They must still pass after every task.
- **Branch:** create a working branch first (Task 0). The working tree has unrelated dirt (`.idea/`, `.DS_Store`, modified `replay_buffer_10x10.json`) — leave it alone, never `git add -A`.
- **Commits:** end every commit message with:
  ```
  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
  ```
- **Verified facts this plan relies on** (re-verify if the code has moved):
  - `ai/trainer.py` calls `observe_opponent_move` twice per move (~lines 299–312); the second call passes the post-move board and is the bug that NaN'd `pt_weights_10x10.pt` (332,680 of 332,689 params are NaN — verified).
  - `ai/runner.py` defines `observe_opponent_move` twice (~lines 88–99 and ~105–115); the second definition shadows the first.
  - Root `runner.py`, root `trainer.py`, and `ai/dotgame.py` are stale copies; nothing imports them (`grep` verified). The live entry point is root `dotgame.py`.
  - `nm_weights_10x10.pt` is finite but trained with BatchNorm — invalidated by the GroupNorm switch (Task 4); user approved deleting it.

---

### Task 0: Branch setup

**Files:** none (git only)

- [ ] **Step 1: Create the working branch**

```bash
cd /Users/jan/Projects/dotgame
git checkout -b ai-improvements
```

- [ ] **Step 2: Verify baseline tests pass**

Run: `$PY -m pytest tests/ -q`
Expected: `176 passed`

---

### Task 1: Delete stale duplicates and poisoned weights

**Files:**
- Delete: `runner.py` (repo root), `trainer.py` (repo root), `ai/dotgame.py`, `nm_weights_10x10.pt` (tracked)
- Delete: `pt_weights_10x10.pt` (untracked)
- Modify: `CLAUDE.md` (remove the stale-duplicates blockquote and pitfall line)

- [ ] **Step 1: Confirm nothing imports the doomed modules**

Run: `grep -rn "^import runner\|^import trainer\|from runner import\|from trainer import" --include="*.py" . | grep -v ".venv\|__pycache__"`
Expected: no output.

- [ ] **Step 2: Delete the files**

```bash
git rm runner.py trainer.py ai/dotgame.py nm_weights_10x10.pt
rm -f pt_weights_10x10.pt
```

(`pt_weights_10x10.pt` is untracked, hence plain `rm`. Do **not** delete `nn_weights_10x10.npz` or `ai_experience_10x10.json`.)

- [ ] **Step 3: Update CLAUDE.md**

In `CLAUDE.md`, remove the blockquote that starts `> **Stale duplicates:**` (in "Module responsibilities (ai/ detail)") and remove the Common-pitfalls bullet that starts `- Root-level `runner.py` / `trainer.py` and `ai/dotgame.py` are **stale duplicates**`.

- [ ] **Step 4: Run tests**

Run: `$PY -m pytest tests/ -q`
Expected: `176 passed`

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md
git commit -m "chore: remove stale duplicate modules and NaN-poisoned/invalidated weight files"
```

---

### Task 2: Fix the double `observe_opponent_move` call in `ai/trainer.py`

This is the bug that destroyed the PyTorch weights: the second call passes the **post-move** board, so the action's position is occupied → masked → `log_prob_old = -inf` → PPO ratio `exp(-inf - (-inf)) = NaN` → NaN gradients.

**Files:**
- Modify: `ai/trainer.py` (~lines 299–312, inside `Trainer._run`)
- Test: `tests/test_trainer.py` (new file)

- [ ] **Step 1: Write the failing test**

Create `tests/test_trainer.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `$PY -m pytest tests/test_trainer.py -q`
Expected: both tests FAIL — observe count is 2× the move count, and half the observed boards contain the move.

- [ ] **Step 3: Fix `ai/trainer.py`**

In `Trainer._run`, find this block and delete the second call (keep the first):

```python
                # Symmetric learning: let the OTHER player observe this move
                # from their own perspective.  Doubles training data and
                # teaches each player to recognise strong/weak opponent play.
                players[3 - cp].observe_opponent_move(
                    board_snap, conn_snap, cp, move, forbidden_snap)

                # Let the OTHER player observe the resulting board position.
                # They learn: "this board (created by the opponent's move) had
                # this eventual outcome from my perspective."
                opp = players[3 - cp]
                if hasattr(opp, 'observe_opponent_move'):
                    opp.observe_opponent_move(
                        dict(game.board), set(game.connections),
                        cp, move, set(game.forbidden))
```

becomes:

```python
                # Symmetric learning: let the OTHER player observe this move
                # from their own perspective, using the PRE-move snapshots.
                # (Passing the post-move board corrupts training: the action's
                # position would be occupied → masked → log_prob = -inf.)
                players[3 - cp].observe_opponent_move(
                    board_snap, conn_snap, cp, move, forbidden_snap)
```

- [ ] **Step 4: Run tests**

Run: `$PY -m pytest tests/test_trainer.py tests/ -q`
Expected: all pass (176 + 2 new).

- [ ] **Step 5: Commit**

```bash
git add ai/trainer.py tests/test_trainer.py
git commit -m "fix: trainer called observe_opponent_move twice, once with post-move board (NaN'd PT weights)"
```

---

### Task 3: Remove duplicate `observe_opponent_move` in `ai/runner.py`

**Files:**
- Modify: `ai/runner.py` (two definitions, ~lines 88–99 and ~105–115)
- Test: `tests/test_runner.py` (new file)

- [ ] **Step 1: Write the test**

Create `tests/test_runner.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `$PY -m pytest tests/test_runner.py -q`
Expected: `test_single_definition` FAILS (count is 2).

- [ ] **Step 3: Fix `ai/runner.py`**

Delete the **first** definition (the one with the `hasattr` guard, between `start_thinking` and `on_game_end`) and keep the second, direct-forwarding one (after `on_game_end`). `BasePlayer` defines a no-op default, so the guard is dead weight. After the edit there is exactly one definition:

```python
    def observe_opponent_move(self, board: dict, connections: set,
                              player_who_moved: int, move: tuple,
                              forbidden: set | None = None) -> None:
        """Forward an opponent's move to the player for symmetric learning.

        Called synchronously from the main loop after every move made by a
        player OTHER than the one this runner wraps.  Fast enough to run on
        the main thread (dict hash for MCTS, one forward pass for NN types).
        """
        self._player.observe_opponent_move(
            board, connections, player_who_moved, move, forbidden)
```

- [ ] **Step 4: Sync the `BasePlayer.record_outcome` signature**

Every caller (`AIRunner.on_game_end`, `Trainer._run`, `ReplayTrainer._replay_one`) passes
`final_scores`, and every concrete player accepts it, but the abstract base in
`ai/base_player.py` is missing the parameter. Replace:

```python
    def record_outcome(self, winner: int, intermediate_rewards=None) -> None:
        """Called once the game ends.  Use to back-propagate the result."""
```

with:

```python
    def record_outcome(self, winner: int, intermediate_rewards=None,
                       final_scores: dict | None = None) -> None:
        """Called once the game ends.  Use to back-propagate the result.

        winner: 1/2, 0 (draw/unknown from Trainer) or None (draw, live game).
        final_scores: {1: float, 2: float} territory scores at game end.
        """
```

- [ ] **Step 5: Run tests**

Run: `$PY -m pytest tests/ -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add ai/runner.py ai/base_player.py tests/test_runner.py
git commit -m "fix: remove shadowed duplicate observe_opponent_move in AIRunner; sync BasePlayer.record_outcome signature"
```

---

### Task 4: Switch BatchNorm → GroupNorm in the CNN backends

`PyTorchPlayer` and `NeuralMCTSPlayer` never call `net.eval()`: every inference forward normalises with single-sample batch statistics **and** mutates BN running stats (NM does this thousands of times per move). GroupNorm has no running stats and no train/eval mode, eliminating the bug class. Old BN checkpoints are rejected via a new `'norm': 'gn'` entry in the arch dict (the weight files were already deleted in Task 1).

**Files:**
- Modify: `ai/pytorch_player.py` (`_build_net`, `PyTorchPlayer.__init__`)
- Modify: `ai/neural_mcts_player.py` (`NeuralMCTSPlayer.__init__` arch dict)
- Test: `tests/test_torch_net.py` (new file)

- [ ] **Step 1: Write the failing test**

Create `tests/test_torch_net.py`:

```python
"""Tests for the shared residual CNN (pytorch_player._build_net)."""
import pytest

torch = pytest.importorskip("torch")

import settings as S
from ai.pytorch_player import _build_net


class TestGroupNorm:
    def test_no_batchnorm_modules(self):
        net = _build_net(S.GRID, n_blocks=2, channels=32)
        assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in net.modules())
        assert any(isinstance(m, torch.nn.GroupNorm) for m in net.modules())

    def test_no_buffers_mutated_by_inference(self):
        # GroupNorm keeps no running statistics → the net has no buffers
        # that inference could silently mutate.
        net = _build_net(S.GRID, n_blocks=2, channels=32)
        assert list(net.buffers()) == []

    def test_forward_shapes(self):
        net = _build_net(S.GRID, n_blocks=2, channels=32)
        x = torch.zeros(3, 15, S.GRID, S.GRID)
        logits, values = net(x)
        assert logits.shape == (3, S.GRID * S.GRID)
        assert values.shape == (3,)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `$PY -m pytest tests/test_torch_net.py -q`
Expected: `test_no_batchnorm_modules` and `test_no_buffers_mutated_by_inference` FAIL.

- [ ] **Step 3: Replace BatchNorm with GroupNorm in `ai/pytorch_player.py`**

Inside `_build_net`, add a norm factory and use it everywhere `nn.BatchNorm2d` appears (2× in `ResBlock.__init__`, 1× in the stem):

```python
def _build_net(grid: int, n_blocks: int, channels: int = 64):
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch not available")

    def _norm(ch):
        # GroupNorm: no running stats, identical behaviour in train/eval —
        # safe for batch-size-1 inference (BatchNorm was silently using
        # batch statistics AND mutating running stats on every forward).
        return nn.GroupNorm(8 if ch % 8 == 0 else 1, ch)

    class ResBlock(nn.Module):
        def __init__(self, ch):
            super().__init__()
            self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
            self.bn1   = _norm(ch)
            self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
            self.bn2   = _norm(ch)
        ...
```

and in the stem:

```python
            self.stem = nn.Sequential(
                nn.Conv2d(15, channels, 3, padding=1, bias=False),
                _norm(channels),
                nn.ReLU(),
            )
```

(`forward` methods are unchanged; keep the `bn1`/`bn2` attribute names to minimise the diff.)

- [ ] **Step 4: Tag the arch dicts so old BN checkpoints are rejected**

In `PyTorchPlayer.__init__`:

```python
        self._arch     = {**arch, 'in_channels': 15, 'norm': 'gn'}
```

In `NeuralMCTSPlayer.__init__`:

```python
        self._arch     = {**arch, 'in_channels': 15, 'norm': 'gn'}
```

(Both `load()` methods already compare `ck.get('arch') == self._arch`, so any old checkpoint without `'norm': 'gn'` is silently rejected.)

- [ ] **Step 5: Run tests**

Run: `$PY -m pytest tests/ -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add ai/pytorch_player.py ai/neural_mcts_player.py tests/test_torch_net.py
git commit -m "fix: replace BatchNorm with GroupNorm (batch-1 inference was using batch stats and polluting running stats)"
```

---

### Task 5: Fold the tactical boost into the policy logits (PPO/PG correctness)

Today NN/PT **sample** actions from the `apply_boost`-modified distribution but store the **unboosted** log-prob, so the PPO ratio (PT) / REINFORCE gradient (NN) is computed against the wrong behaviour policy. Mathematically `apply_boost(softmax(z)) == softmax(z + log m)` where `m` is the boost multiplier — so the fix is to add `log m` to the logits *before* the softmax, both at action time and at training time. The offsets are stored in the trajectory.

**Files:**
- Modify: `ai/features.py` (new `boost_log_offsets`)
- Modify: `ai/pytorch_player.py` (`choose_move`, `record_outcome`, `observe_opponent_move`)
- Test: `tests/test_features.py` (append a class)

This task changes only `features.py` + `pytorch_player.py`; the NN player follows in Task 6. Keep `apply_boost` itself — existing tests use it.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_features.py`:

```python
class TestBoostLogOffsets:
    def test_equivalent_to_apply_boost(self):
        import numpy as np
        from ai.features import apply_boost, boost_log_offsets
        n = S.GRID
        rng = np.random.default_rng(42)
        logits = rng.normal(size=n * n)
        own = np.zeros(n * n, dtype=np.float32); own[5]  = 1.0; own[17] = 0.5
        opp = np.zeros(n * n, dtype=np.float32); opp[5]  = 0.5; opp[30] = 1.0
        legal = np.ones(n * n, dtype=bool); legal[2] = False

        masked = np.where(legal, logits, -1e9)
        probs  = np.exp(masked - masked.max()); probs /= probs.sum()
        expected = apply_boost(probs, own, opp, legal)

        off = boost_log_offsets(own, opp, legal)
        boosted = np.where(legal, logits + off, -1e9)
        got = np.exp(boosted - boosted.max()); got /= got.sum()

        assert np.allclose(got, expected, atol=1e-9)

    def test_zero_at_illegal_positions(self):
        import numpy as np
        from ai.features import boost_log_offsets
        n = S.GRID
        own = np.ones(n * n, dtype=np.float32)
        opp = np.ones(n * n, dtype=np.float32)
        legal = np.zeros(n * n, dtype=bool)
        assert (boost_log_offsets(own, opp, legal) == 0.0).all()
```

(`tests/test_features.py` already imports `settings as S` and numpy-adjacent helpers at the top; if `S` is not imported there, add `import settings as S`.)

- [ ] **Step 2: Run the test to verify it fails**

Run: `$PY -m pytest tests/test_features.py -q -k Boost`
Expected: ImportError — `boost_log_offsets` does not exist.

- [ ] **Step 3: Add `boost_log_offsets` to `ai/features.py`** (directly after `apply_boost`)

```python
def boost_log_offsets(
    own_scores: np.ndarray,
    opp_scores: np.ndarray,
    legal: np.ndarray,
) -> np.ndarray:
    """Log-domain equivalent of apply_boost().

    softmax(logits + boost_log_offsets(...)) over the legal set equals
    apply_boost(softmax(logits), ...).  Folding the boost into the logits
    keeps the sampling distribution and the trainable policy identical, so
    PPO ratios / policy gradients are computed against the true behaviour
    policy.  Returns float32 (G²,) with 0.0 at illegal positions.
    """
    close_boost = getattr(S, 'AI_CLOSE_BOOST', 4.0)
    block_boost = getattr(S, 'AI_BLOCK_BOOST', 3.0)
    mult = ((1.0 + (close_boost - 1.0) * own_scores.astype(np.float64))
            * (1.0 + (block_boost - 1.0) * opp_scores.astype(np.float64)))
    return np.where(legal, np.log(mult), 0.0).astype(np.float32)
```

- [ ] **Step 4: Run the feature tests**

Run: `$PY -m pytest tests/test_features.py -q`
Expected: all pass.

- [ ] **Step 5: Use it in `PyTorchPlayer.choose_move`**

Replace the body of the `with torch.no_grad():` block (and the post-hoc `apply_boost` call) so the boost is applied as a logit offset before sampling, and the offsets are stored as a 6th trajectory element. Change the import line at the top of the file from `apply_boost` to `boost_log_offsets`:

```python
from ai.features import (opportunity_masks, boost_log_offsets,
                          arc_potential_map, strategic_channels)
```

New `choose_move` core (after `enc_np = encode_state(...)`):

```python
        own_scores, opp_scores = opportunity_masks(board, connections, player)
        boost_np = boost_log_offsets(own_scores, opp_scores, mask_np)

        with torch.no_grad():
            x    = torch.tensor(enc_np, dtype=torch.float32,
                                device=self._device).unsqueeze(0)
            mask = torch.tensor(mask_np, dtype=torch.bool,
                                device=self._device)

            logits, val = self._net(x)
            logits = logits[0] + torch.tensor(boost_np, device=self._device)
            logits[~mask] = float('-inf')
            log_p  = F.log_softmax(logits, dim=0)

            p_np = log_p.exp().cpu().numpy().astype(np.float64)
            p_np = np.where(np.isfinite(p_np), np.clip(p_np, 0.0, None), 0.0)
            total = p_np.sum()
            if total <= 0.0:
                p_np = mask_np.astype(np.float64); p_np /= p_np.sum()
            else:
                p_np /= total

            action       = int(np.random.choice(n * n, p=p_np))
            log_prob_old = float(log_p[action].item())
            value_old    = float(val[0].item())

        self._traj.append((enc_np, mask_np, action, log_prob_old,
                           value_old, boost_np))
        return (action % n, action // n)
```

- [ ] **Step 6: Apply the stored offsets at training time**

In `record_outcome`, the own-move branch: after `lp_old_t`, add

```python
            boosts_t  = torch.tensor(
                np.stack([t[5] for t in self._traj]),
                dtype=torch.float32, device=self._device)
```

and inside the minibatch loop, change

```python
                    logits, vals_pred = self._net(b_states)

                    logits = logits.clone()
                    logits[~b_masks] = float('-inf')
```

to

```python
                    logits, vals_pred = self._net(b_states)

                    logits = logits + boosts_t[batch]
                    logits[~b_masks] = float('-inf')
```

(the `.clone()` is no longer needed — `logits + boosts` already creates a new tensor).

- [ ] **Step 7: Same treatment for the observation path**

In `observe_opponent_move`, compute the offsets from the mover's perspective and store them:

```python
        enc = encode_state(board, connections, forbidden, player_who_moved)
        action = gy * n + gx
        own_scores, opp_scores = opportunity_masks(
            board, connections, player_who_moved)
        boost_np = boost_log_offsets(own_scores, opp_scores, mask_np)

        with torch.no_grad():
            x      = torch.tensor(enc, dtype=torch.float32,
                                  device=self._device).unsqueeze(0)
            mask_t = torch.tensor(mask_np, dtype=torch.bool,
                                  device=self._device)
            logits, val = self._net(x)
            logits = logits[0] + torch.tensor(boost_np, device=self._device)
            logits[~mask_t] = float('-inf')
            log_p  = F.log_softmax(logits, dim=0)
            log_prob_old = float(log_p[action].item())
            value_old    = float(val[0].item())

        self._obs_traj.append((enc, mask_np, action, log_prob_old,
                               value_old, boost_np))
```

And in the obs-training branch of `record_outcome`, mirror Step 6: build `obs_boosts_t` from `t[5]` and use `logits = logits + obs_boosts_t[batch]` instead of `logits.clone()`.

- [ ] **Step 8: Add a consistency test**

Append to `tests/test_torch_net.py`:

```python
class TestPTBoostConsistency:
    def test_stored_logprob_matches_sampling_distribution(self):
        import numpy as np
        from ai.pytorch_player import PyTorchPlayer
        p = PyTorchPlayer(player_id=1)
        p.save = lambda: None                      # don't touch disk
        board = {(0, 0): 1, (1, 0): 1, (0, 1): 1}  # 3-corner square → boost fires
        np.random.seed(0)
        move = p.choose_move(board, set(), 1, {1: 0.0, 2: 0.0}, set())
        enc, mask, action, lp_old, _, boost = p._traj[-1]
        assert move == (action % S.GRID, action // S.GRID)
        # Recompute the boosted policy and compare the stored log-prob
        x = torch.tensor(enc, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, _ = p._net(x)
        logits = logits[0] + torch.tensor(boost)
        logits[~torch.tensor(mask)] = float('-inf')
        log_p = torch.log_softmax(logits, dim=0)
        assert abs(float(log_p[action]) - lp_old) < 1e-5
```

- [ ] **Step 9: Run tests, then commit**

Run: `$PY -m pytest tests/ -q`
Expected: all pass.

```bash
git add ai/features.py ai/pytorch_player.py tests/test_features.py tests/test_torch_net.py
git commit -m "fix: fold tactical boost into PT policy logits so PPO trains against the true behaviour policy"
```

---

### Task 6: NN player — boost into logits, masked gradients, trunk backprop fix

Three fixes in `ai/nn_player.py`:
1. Same boost-as-logits change as Task 5 (the NN's REINFORCE gradient assumed actions were sampled from the unboosted policy).
2. `_ACNetwork.update` computes its policy/entropy gradients on the **unmasked** softmax — pass the legal mask + boost through the trajectory and use them.
3. The trunk backprop uses **post-update** layer weights (`w_pre` is captured then never used in the layer loop).

**Files:**
- Modify: `ai/nn_player.py` (`_ACNetwork.policy`, `_ACNetwork.update`, `NNPlayer.choose_move`, `record_outcome`, `observe_opponent_move`)
- Test: `tests/test_nn_player.py` (new file)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_nn_player.py`:

```python
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
    p._learning_shared = False              # added in Task 7; harmless before
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
```

- [ ] **Step 2: Run to verify failure**

Run: `$PY -m pytest tests/test_nn_player.py -q`
Expected: FAIL — `update()` has no `legal_mask`/`boost_log` params; traj entries are 4-tuples.

- [ ] **Step 3: Implement the changes in `ai/nn_player.py`**

Change the import to add `boost_log_offsets`:

```python
from ai.features import (opportunity_masks, boost_log_offsets,
                          arc_potential_map, strategic_channels)
```

`_ACNetwork.policy` — accept an optional boost vector:

```python
    def policy(self, x: np.ndarray, legal_mask: np.ndarray, boost=None):
        logits, value, acts = self.forward(x)
        masked = logits.copy()
        if boost is not None:
            masked += boost
        masked[~legal_mask] = -1e9
        probs = _softmax(masked)
        return probs, value, acts
```

`_ACNetwork.update` — masked, boosted policy gradient + correct trunk backprop. Full replacement:

```python
    def update(self, acts: list, action: int, advantage: float,
               value_target: float, entropy_bonus: float, lr: float,
               legal_mask=None, boost_log=None) -> None:
        """Adam step for one (state, action, advantage, value_target) tuple.

        legal_mask / boost_log reproduce the exact distribution the action
        was sampled from — without them the gradient pushes probability mass
        toward illegal positions and mismatches the behaviour policy.
        """
        self._t += 1
        t  = self._t
        h  = acts[-1]   # trunk output

        # ── Policy head (masked + boosted softmax) ───────────────────────────
        logits = h @ self._pol_w.w + self._pol_b.w
        if boost_log is not None:
            logits = logits + boost_log
        if legal_mask is not None:
            logits = np.where(legal_mask, logits, -1e9)
        probs = _softmax(logits)

        # Policy gradient: ∇log π(a) · advantage  + entropy bonus.
        # boost_log is a constant offset, so d(logits)/d(params) is unchanged
        # and the (onehot − π) form stays exact.
        d_pol = -probs.copy()
        d_pol[action] += 1.0
        d_pol *= advantage
        log_probs  = np.log(np.clip(probs, 1e-10, None))
        entropy    = -float((probs * log_probs).sum())
        d_entropy  = -(log_probs - entropy)
        d_pol     += _ENTROPY_COEF * d_entropy
        if legal_mask is not None:
            d_pol[~legal_mask] = 0.0      # no gradient through dead logits

        # Save pre-update weights for correct trunk backprop
        w_pol_pre = self._pol_w.w.copy()
        w_val_pre = self._val_w.w.copy()

        self._pol_w.step(np.outer(h, d_pol), lr, t)
        self._pol_b.step(d_pol, lr, t)
        d_trunk_pol = d_pol @ w_pol_pre.T

        # ── Value head (MSE loss) ─────────────────────────────────────────────
        v_pred    = float(np.tanh(h @ self._val_w.w + self._val_b.w)[0])
        d_val_raw = (value_target - v_pred) * (1 - v_pred ** 2)   # tanh deriv
        d_val_vec = np.array([d_val_raw])

        self._val_w.step(S.NN_VALUE_COEF * np.outer(h, d_val_vec), lr, t)
        self._val_b.step(S.NN_VALUE_COEF * d_val_vec, lr, t)
        d_trunk_val = (d_val_vec @ w_val_pre.T).ravel()

        # ── Shared trunk (backprop through PRE-update weights) ────────────────
        delta = d_trunk_pol + d_trunk_val
        for i in reversed(range(len(self._trunk_w))):
            inp   = acts[i]
            w_pre = self._trunk_w[i].w.copy()
            self._trunk_w[i].step(np.outer(inp, delta), lr, t)
            self._trunk_b[i].step(delta, lr, t)
            if i > 0:
                delta = (delta @ w_pre.T) * (acts[i] > 0).astype(float)
```

`NNPlayer.choose_move` — compute boost first, drop the post-hoc `apply_boost`, store a 6-tuple:

```python
        enc = encode_state(board, connections, forbidden, player)
        own_scores, opp_scores = opportunity_masks(board, connections, player)
        boost = boost_log_offsets(own_scores, opp_scores, legal_flat)

        probs, val, acts = self._net.policy(enc, legal_flat, boost)

        probs = np.where(np.isfinite(probs), probs, 0.0)
        probs = np.clip(probs, 0.0, None)
        total = probs.sum()
        if total <= 0.0:
            probs = legal_flat.astype(np.float64); probs /= probs.sum()
        else:
            probs = probs / total

        action = int(np.random.choice(n * n, p=probs))
        self._traj.append((enc, action, acts, val, boost, legal_flat))
        return (action % n, action // n)
```

`NNPlayer.record_outcome` — both update loops unpack 6-tuples and forward the extras. Own-move loop:

```python
            vals  = [t[3] for t in self._traj]
            ...
            for t, (enc, action, acts, val, boost, legal) in enumerate(self._traj):
                self._net.update(
                    acts          = acts,
                    action        = action,
                    advantage     = float(advantages[t]),
                    value_target  = float(returns[t]),
                    entropy_bonus = _ENTROPY_COEF,
                    lr            = lr,
                    legal_mask    = legal,
                    boost_log     = boost,
                )
```

(the GAE code reading `t[3]` for values is unchanged — value stays at index 3). Obs loop identically: `obs_vals = [t[3] for ...]` unchanged, and the update call gains `legal_mask=legal, boost_log=boost`.

`NNPlayer.observe_opponent_move` — compute and store boost + mask too:

```python
        enc    = encode_state(board, connections, forbidden, player_who_moved)
        action = gy * n + gx
        own_scores, opp_scores = opportunity_masks(
            board, connections, player_who_moved)
        boost = boost_log_offsets(own_scores, opp_scores, legal_mask)

        probs, val, acts = self._net.policy(enc, legal_mask, boost)
        self._obs_traj.append((enc, action, acts, val, boost, legal_mask))
```

Also fix the stale module/`encode_state` docstrings while in the file: the module docstring's "5-CHANNEL INPUT" header and `encode_state`'s "length 7 × GRID × GRID" line should say **15** channels.

- [ ] **Step 4: Run tests**

Run: `$PY -m pytest tests/test_nn_player.py tests/ -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add ai/nn_player.py tests/test_nn_player.py
git commit -m "fix: NN player trains on the masked+boosted policy it samples from; trunk backprop uses pre-update weights"
```

---

### Task 7: Shared learning for self-play pairs

When two players of the same type self-play, each trains its own copy and `p2.save()` overwrites `p1`'s file — half the learning is discarded. Add `share_learning_with(other)` to all four players: the pair shares one network/optimizer (or experience dict), and **both** sides stop recording opponent observations (with one net, the partner's own-move training already covers those positions).

**Files:**
- Modify: `ai/mcts.py`, `ai/nn_player.py`, `ai/pytorch_player.py`, `ai/neural_mcts_player.py`
- Modify: `dotgame.py` (`_make_runners` + the two `_make_train_player` closures)
- Test: `tests/test_shared_learning.py` (new file)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_shared_learning.py`:

```python
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
```

(NM's `_buffer` sharing is added in Task 11 — extend `test_nm_net_and_buffer_shared` with `assert p2._buffer is p1._buffer` there.)

- [ ] **Step 2: Run to verify failure**

Run: `$PY -m pytest tests/test_shared_learning.py -q`
Expected: AttributeError — `share_learning_with` does not exist.

- [ ] **Step 3: Implement in each player**

Every player's `__init__` gains `self._learning_shared = False`, every `observe_opponent_move` gains a first line `if self._learning_shared: return`, and each class gets a `share_learning_with` method:

`ai/mcts.py` (`MCTSPlayer`):

```python
    def share_learning_with(self, other: 'MCTSPlayer') -> None:
        """Share the experience dict with *other* (self-play pairs).

        Disables opponent-observation recording on both sides: with one
        shared dict, the partner's own-move recording already covers those
        positions (recording both would double-count every visit)."""
        other.experience = self.experience
        self._learning_shared = other._learning_shared = True
```

`ai/nn_player.py` (`NNPlayer`):

```python
    def share_learning_with(self, other: 'NNPlayer') -> None:
        """Make *other* train and act with this player's network."""
        other._net    = self._net
        other._hidden = self._hidden
        self._learning_shared = other._learning_shared = True
```

`ai/pytorch_player.py` (`PyTorchPlayer`):

```python
    def share_learning_with(self, other: 'PyTorchPlayer') -> None:
        """Make *other* train and act with this player's network/optimizer."""
        other._net  = self._net
        other._opt  = self._opt
        other._arch = self._arch
        self._learning_shared = other._learning_shared = True
```

`ai/neural_mcts_player.py` (`NeuralMCTSPlayer`) — same as PT (plus `other._buffer = self._buffer` once Task 11 lands).

- [ ] **Step 4: Wire it up in `dotgame.py`**

In `_make_runners`, after building the dict, link same-type pairs:

```python
def _make_runners(mode: Mode, profile: int = 0) -> dict:
    p1_type, p2_type = _MODE_PLAYERS.get(mode, ('mcts', 'mcts'))
    if mode == Mode.MCTS_VS_MCTS:
        p1_profile, p2_profile = 1, 2
    else:
        p1_profile = profile if p1_type == 'mcts' else 0
        p2_profile = profile if p2_type == 'mcts' else 0
    r1 = _make_runner(1, p1_type, p1_profile)
    r2 = _make_runner(2, p2_type, p2_profile)
    # Same-type self-play: share one network / experience store so the two
    # sides don't clobber each other's learning at save time.
    if p1_type == p2_type and p1_type != 'human':
        r1._player.share_learning_with(r2._player)
    return {1: r1, 2: r2}
```

In `main()`, the two identical `_make_train_player` closures (START and REPLAY branches): after `tp1`/`tp2` (resp. `rp1`/`rp2`) are constructed successfully, add:

```python
                                    if resolved_opp == ai_label:
                                        rp1.share_learning_with(rp2)
```

(and `tp1.share_learning_with(tp2)` in the START branch). Place it inside the `else:` arm, before the Trainer/ReplayTrainer is created.

- [ ] **Step 5: Run tests**

Run: `$PY -m pytest tests/ -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add ai/mcts.py ai/nn_player.py ai/pytorch_player.py ai/neural_mcts_player.py dotgame.py tests/test_shared_learning.py
git commit -m "feat: share one network/experience store between same-type self-play pairs"
```

---

### Task 8: MCTS rollout throughput — one `opportunity_masks` call per step, lazy forks

`_rollout` currently computes `closing_moves`, `blocking_moves` (each its own `opportunity_masks` scan) **and** `fork_moves` eagerly on every rollout step. Compute the masks once and only look for forks when there is no closing move.

**Files:**
- Modify: `ai/mcts.py` (`FastBoard` + `MCTSPlayer._rollout`)
- Test: `tests/test_fastboard.py` (append a class)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_fastboard.py`:

```python
class TestTacticalMoves:
    def test_matches_closing_and_blocking(self):
        board = {(0, 0): 1, (1, 0): 1, (0, 1): 1,      # P1 3-corner square
                 (5, 5): 2, (6, 5): 2, (5, 6): 2}      # P2 3-corner square
        fb = fb_with(board)
        closing, blocking = fb.tactical_moves(1)
        assert closing == fb.closing_moves(1)
        assert blocking == fb.blocking_moves(1)
        assert (1, 1) in closing
        assert (6, 6) in blocking

    def test_single_opportunity_masks_call(self, monkeypatch):
        import ai.features as F
        calls = {'n': 0}
        real = F.opportunity_masks
        def counting(*a, **kw):
            calls['n'] += 1
            return real(*a, **kw)
        monkeypatch.setattr(F, 'opportunity_masks', counting)
        fb = fb_with({(0, 0): 1, (1, 0): 1, (0, 1): 1})
        fb.tactical_moves(1)
        assert calls['n'] == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `$PY -m pytest tests/test_fastboard.py -q -k Tactical`
Expected: AttributeError — `tactical_moves` does not exist.

- [ ] **Step 3: Implement `tactical_moves` and make the old methods thin wrappers**

In `FastBoard`, replace the bodies of `closing_moves` and `blocking_moves` and add:

```python
    def tactical_moves(self, player):
        """Return (closing, blocking) move lists from ONE opportunity_masks
        call (each list sorted by score, best first).  closing_moves() and
        blocking_moves() each previously ran their own O(G²) scan; rollouts
        call both every step, so sharing the scan ≈ halves the per-step cost."""
        from ai.features import opportunity_masks
        own_scores, opp_scores = opportunity_masks(
            self.board, self.connections, player)
        n = S.GRID

        def _collect(scores):
            moves = [
                (int(scores[idx]), idx % n, idx // n)
                for idx in np.where(scores)[0]
                if (idx % n, idx // n) not in self.board
                and (idx % n, idx // n) not in self.forbidden
            ]
            moves.sort(reverse=True)
            return [(gx, gy) for _, gx, gy in moves]

        return _collect(own_scores), _collect(opp_scores)

    def closing_moves(self, player):
        """Return legal moves sorted by squares closed (most first)."""
        return self.tactical_moves(player)[0]

    def blocking_moves(self, player):
        """Return legal moves sorted by opponent squares blocked (most first)."""
        return self.tactical_moves(player)[1]
```

(The `(int(scores[idx]), …)` sort key matches the existing methods exactly — keep it.)

- [ ] **Step 4: Make `_rollout` lazy**

In `MCTSPlayer._rollout`, replace

```python
            closing  = sim.closing_moves(p)
            blocking = sim.blocking_moves(p)
            forks    = sim.fork_moves(p)

            if forks and not closing:
                move = random.choice(forks)
            elif closing:
                move = closing[0]
            elif blocking:
                move = blocking[0]
            else:
```

with

```python
            closing, blocking = sim.tactical_moves(p)

            if closing:
                move = closing[0]
            else:
                forks = sim.fork_moves(p)
                if forks:
                    move = random.choice(forks)
                elif blocking:
                    move = blocking[0]
                else:
```

(then re-indent the existing `setup`/`adj`/`random` block one level deeper under that final `else:`). The priority order is unchanged: closing → forks → blocking → setup → adjacent → random; `fork_moves` (O(G²·4)) now only runs when there is no closing move.

- [ ] **Step 5: Run tests, commit**

Run: `$PY -m pytest tests/ -q`
Expected: all pass.

```bash
git add ai/mcts.py tests/test_fastboard.py
git commit -m "perf: single opportunity_masks scan per rollout step, lazy fork detection"
```

---

### Task 9: Dirichlet noise — floor the adaptive alpha

`max(0.01, alpha * 10 / len(legal))` produces alpha ≈ 0.03–0.06 on an open 10×10 board — so spiky that nearly all noise mass lands on one arbitrary move. Floor it at 0.1 via a shared helper.

**Files:**
- Modify: `ai/features.py` (new helper), `ai/mcts.py` (~line 560), `ai/neural_mcts_player.py` (~line 205)
- Test: `tests/test_features.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_features.py`:

```python
class TestAdaptiveDirichletAlpha:
    def test_floor_at_point_one(self):
        from ai.features import adaptive_dirichlet_alpha
        assert adaptive_dirichlet_alpha(0.3, 100) == pytest.approx(0.1)

    def test_scales_inversely_with_branching(self):
        from ai.features import adaptive_dirichlet_alpha
        assert adaptive_dirichlet_alpha(0.3, 10) == pytest.approx(0.3)
        assert adaptive_dirichlet_alpha(0.3, 5)  == pytest.approx(0.6)
```

(add `import pytest` at the top of the file if missing).

- [ ] **Step 2: Run to verify failure**

Run: `$PY -m pytest tests/test_features.py -q -k Dirichlet`
Expected: ImportError.

- [ ] **Step 3: Implement**

In `ai/features.py` (near `get_centrality`):

```python
def adaptive_dirichlet_alpha(base_alpha: float, n_legal: int) -> float:
    """AlphaZero-style root-noise concentration: alpha ∝ 1/branching factor.

    Floored at 0.1 — below that the Dirichlet sample is so spiky that
    essentially all noise mass lands on a single arbitrary move, which makes
    the noise a coin-flip override instead of gentle diversification.
    """
    return max(0.1, base_alpha * 10.0 / max(n_legal, 1))
```

In `ai/mcts.py` `choose_move`, replace

```python
            adaptive_alpha = max(0.01, self._noise_alpha * 10.0 / len(legal))
```

with

```python
            from ai.features import adaptive_dirichlet_alpha
            adaptive_alpha = adaptive_dirichlet_alpha(self._noise_alpha, len(legal))
```

(or add it to the existing `from ai.features import (...)` block at the top — preferred).

In `ai/neural_mcts_player.py` `choose_move`, replace

```python
            alpha = max(0.01, S.NM_NOISE_ALPHA * 10.0 / len(legal))
```

with

```python
            alpha = adaptive_dirichlet_alpha(S.NM_NOISE_ALPHA, len(legal))
```

and add `adaptive_dirichlet_alpha` to the file's `from ai.features import ...` line.

- [ ] **Step 4: Run tests, commit**

Run: `$PY -m pytest tests/ -q`
Expected: all pass.

```bash
git add ai/features.py ai/mcts.py ai/neural_mcts_player.py tests/test_features.py
git commit -m "fix: floor adaptive Dirichlet alpha at 0.1 to avoid single-spike root noise"
```

---

### Task 10: D4 symmetry module + canonical MCTS experience keys

The game is invariant under the 8 dihedral board transforms. This task creates `ai/symmetry.py` (also used by Task 11) and canonicalises MCTS experience keys so a position and its rotations/mirrors share one experience entry (~8× warm-start hit rate). Existing `ai_experience_10x10.json` entries mostly stop matching; they get evicted naturally by the `AI_MAX_EXPERIENCE` cap — no migration needed.

**Files:**
- Create: `ai/symmetry.py`
- Modify: `ai/mcts.py` (`_board_key`, `observe_opponent_move`)
- Test: `tests/test_symmetry.py` (new file)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_symmetry.py`:

```python
"""Tests for ai/symmetry.py — D4 board transforms."""
import numpy as np

import settings as S
from ai.symmetry import (N_TRANSFORMS, point_transforms,
                         transform_planes, transform_flat,
                         canonical_board_key)


class TestPointAndPlaneConsistency:
    def test_planes_and_flat_agree(self):
        n = S.GRID
        planes = np.zeros((1, n, n), dtype=np.float32)
        planes[0, 2, 5] = 1.0                       # [gy=2][gx=5]
        flat = np.zeros(n * n, dtype=np.float32)
        flat[2 * n + 5] = 1.0
        for op in range(N_TRANSFORMS):
            a = transform_planes(planes, op).reshape(-1)
            b = transform_flat(flat, op)
            assert np.array_equal(a, b), f"op {op} disagrees"

    def test_each_transform_is_a_permutation(self):
        n = S.GRID
        for op in range(N_TRANSFORMS):
            t = point_transforms(n)[op]
            seen = {t(gx, gy) for gx in range(n) for gy in range(n)}
            assert len(seen) == n * n


class TestCanonicalKey:
    def test_invariant_under_rotation(self):
        n = S.GRID
        board = {(1, 2): 1, (3, 4): 2, (0, 0): 1}
        rot90 = {(n - 1 - gy, gx): p for (gx, gy), p in board.items()}
        assert canonical_board_key(board) == canonical_board_key(rot90)

    def test_distinguishes_different_positions(self):
        assert (canonical_board_key({(1, 2): 1})
                != canonical_board_key({(1, 2): 2}))
```

- [ ] **Step 2: Run to verify failure**

Run: `$PY -m pytest tests/test_symmetry.py -q`
Expected: ImportError — module doesn't exist.

- [ ] **Step 3: Create `ai/symmetry.py`**

```python
"""D4 (dihedral) board symmetries.

The game is fully invariant under the 8 transforms of the square: 4 rotations
× optional mirror.  Two users:

  * MCTSPlayer experience keys — canonical_board_key() maps all 8 variants of
    a position to one key, multiplying warm-start hit rate.
  * NeuralMCTSPlayer training — transform_planes()/transform_flat() augment
    (state, visit-dist) pairs with a random transform per sampled minibatch
    item (value targets are invariant).

Conventions: spatial arrays are indexed [.., gy, gx] (row-major, idx = gy*n+gx)
— same as encode_state().  transform_planes and transform_flat apply the SAME
permutation for a given op index, so encodings, legal masks and policy targets
stay aligned.
"""

import numpy as np

import settings as S

N_TRANSFORMS = 8

# op → (number of np.rot90 quarter-turns, mirror first?)
_OPS = [(k, f) for f in (False, True) for k in range(4)]


def transform_planes(planes: np.ndarray, op: int) -> np.ndarray:
    """Transform a (C, G, G) array.  Returns a contiguous copy."""
    k, flip = _OPS[op]
    out = planes[:, :, ::-1] if flip else planes
    return np.ascontiguousarray(np.rot90(out, k, axes=(1, 2)))


def transform_flat(vec: np.ndarray, op: int) -> np.ndarray:
    """Transform a flat (G²,) row-major vector with the same permutation."""
    n = S.GRID
    return transform_planes(vec.reshape(1, n, n), op).reshape(-1)


def point_transforms(n: int):
    """The 8 (gx, gy) → (gx', gy') maps matching transform_planes op order.

    Derived from the array ops: flip reverses the x axis (gx → n-1-gx);
    np.rot90 on axes (gy, gx) maps (gx, gy) → (gy', gx') per quarter turn.
    """
    def rot(k, gx, gy):
        for _ in range(k):
            gx, gy = n - 1 - gy, gx      # one CCW quarter turn of indices
        return gx, gy

    out = []
    for k, flip in _OPS:
        def t(gx, gy, k=k, flip=flip):
            if flip:
                gx = n - 1 - gx
            return rot(k, gx, gy)
        out.append(t)
    return out


def canonical_board_key(board: dict) -> str:
    """Lexicographically smallest str(sorted((gx, gy, p))) over all 8 transforms.

    Same string format as the previous MCTS _board_key, so the experience
    JSON schema is unchanged — old non-canonical keys simply stop matching
    and age out via the AI_MAX_EXPERIENCE cap.
    """
    n = S.GRID
    best = None
    for t in point_transforms(n):
        s = str(sorted((*t(gx, gy), p) for (gx, gy), p in board.items()))
        if best is None or s < best:
            best = s
    return best
```

**Note on `point_transforms`:** the `test_planes_and_flat_agree` test pins plane/flat consistency, and `TestCanonicalKey.test_invariant_under_rotation` pins key invariance. If the rotation direction in `rot()` disagrees with `np.rot90`, the canonical-key tests still pass (any permutation group works for canonicalisation) but verify the consistency test passes before moving on — it is the source of truth.

- [ ] **Step 4: Run the symmetry tests**

Run: `$PY -m pytest tests/test_symmetry.py -q`
Expected: all pass. If `test_planes_and_flat_agree` fails, fix `_OPS`/`rot` until it passes — do not weaken the test.

- [ ] **Step 5: Canonicalise the MCTS experience keys**

In `ai/mcts.py`, add `from ai.symmetry import canonical_board_key` to the imports, then:

```python
    @staticmethod
    def _board_key(fb: FastBoard) -> str:
        return canonical_board_key(fb.board)
```

and in `observe_opponent_move`, replace

```python
        # Build the board key directly (same format as _board_key)
        key = str(sorted((bx, by, p) for (bx, by), p in new_board.items()))
```

with

```python
        key = canonical_board_key(new_board)
```

- [ ] **Step 6: Run all tests, commit**

Run: `$PY -m pytest tests/ -q`
Expected: all pass.

```bash
git add ai/symmetry.py ai/mcts.py tests/test_symmetry.py
git commit -m "feat: D4 symmetry module; canonical MCTS experience keys (~8x warm-start hit rate)"
```

---

### Task 11: Neural MCTS — replay buffer, z-blended value targets, symmetry augmentation

Today NM takes one Adam step per game on that game's positions, then discards them, and its value target is a pure GAE bootstrap. Replace with the AlphaZero recipe: append `(enc, mask, visit_dist, value_target)` to an in-memory buffer, blend the final outcome `z` into the value target (`NM_Z_WEIGHT`), and train `NM_TRAIN_STEPS` minibatches per game with a random D4 transform per sampled item.

> **Why augmentation is NM-only:** NM's policy loss is supervised cross-entropy against the
> MCTS visit distribution, which transforms exactly under D4. PT's PPO loss (and NN's
> REINFORCE) depend on `log_prob_old` recorded in the *original* orientation — transforming
> the state but keeping the stored log-prob breaks the importance-sampling ratio. Do **not**
> extend this augmentation to `pytorch_player.py` / `nn_player.py`.

**Files:**
- Modify: `settings.py` (5 new constants)
- Modify: `ai/neural_mcts_player.py` (`__init__`, `record_outcome` → `_value_targets` + `_train_from_buffer`, `share_learning_with`)
- Test: `tests/test_nm_training.py` (new file), `tests/test_shared_learning.py` (extend)

- [ ] **Step 1: Add the settings constants**

In `settings.py`, after `NM_DEAD_SPOT_PENALTY`:

```python
NM_BUFFER_MAX       = 8000             # max (state, π, v) samples kept in memory
NM_BATCH_SIZE       = 256              # minibatch size per training step
NM_TRAIN_STEPS      = 4                # minibatches sampled per finished game
NM_Z_WEIGHT         = 0.5              # blend of final outcome z into value target
NM_AUGMENT          = True             # random D4 transform per sampled item
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_nm_training.py`:

```python
"""Tests for NeuralMCTSPlayer training (buffer, value targets, augmentation)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

import settings as S
from ai.neural_mcts_player import NeuralMCTSPlayer


def _fake_episode(player, steps=3):
    n = S.GRID
    for i in range(steps):
        enc  = np.random.default_rng(i).normal(size=(15, n, n)).astype(np.float32)
        mask = np.ones(n * n, dtype=bool)
        dist = np.zeros(n * n, dtype=np.float32); dist[i] = 1.0
        player._episode.append((enc, mask, dist, 0.0))


class TestValueTargets:
    def test_pure_z_when_weight_is_one(self, monkeypatch):
        monkeypatch.setattr(S, 'NM_Z_WEIGHT', 1.0)
        p = NeuralMCTSPlayer(1); p.save = lambda: None
        targets = p._value_targets([0.2, -0.1, 0.4], None, terminal=0.8)
        assert np.allclose(targets, 0.8)

    def test_clipped_to_unit_interval(self):
        p = NeuralMCTSPlayer(1); p.save = lambda: None
        targets = p._value_targets([0.9, 0.9], [5.0, 5.0], terminal=1.0)
        assert (targets <= 1.0).all() and (targets >= -1.0).all()


class TestBufferTraining:
    def test_record_outcome_fills_buffer_and_stays_finite(self):
        p = NeuralMCTSPlayer(1); p.save = lambda: None
        _fake_episode(p, steps=3)
        p.record_outcome(1, intermediate_rewards=[0.0, 0.1, 0.0],
                         final_scores={1: 10.0, 2: 2.0})
        assert len(p._buffer) == 3
        assert p._episode == []
        assert all(torch.isfinite(q).all() for q in p._net.parameters())

    def test_buffer_capped(self, monkeypatch):
        monkeypatch.setattr(S, 'NM_BUFFER_MAX', 4)
        p = NeuralMCTSPlayer(1); p.save = lambda: None
        for _ in range(3):
            _fake_episode(p, steps=2)
            p.record_outcome(1, final_scores={1: 5.0, 2: 1.0})
        assert len(p._buffer) == 4
```

- [ ] **Step 3: Run to verify failure**

Run: `$PY -m pytest tests/test_nm_training.py -q`
Expected: AttributeError — `_value_targets` / `_buffer` do not exist.

- [ ] **Step 4: Implement in `ai/neural_mcts_player.py`**

Add to the imports:

```python
from ai.symmetry import transform_planes, transform_flat, N_TRANSFORMS
```

In `__init__`, after `self._obs_episode`:

```python
        # Replay buffer of (enc, legal_mask, visit_dist, value_target) — the
        # AlphaZero training store.  In-memory only; cleared on restart.
        self._buffer: list[tuple] = []
        self._learning_shared = False
```

(`_learning_shared` already exists if Task 7 ran first — keep one assignment.)

Replace `record_outcome` entirely:

```python
    def record_outcome(self, winner: int,
                       intermediate_rewards=None,
                       final_scores: dict | None = None) -> None:
        if not self._episode and not self._obs_episode:
            return

        # ── Terminal value in [-1, 1] ─────────────────────────────────────────
        if final_scores and winner != 0:
            total_fields = max((S.GRID - 1) ** 2, 1)
            own_s  = final_scores.get(self.player_id, 0.0)
            opp_s  = final_scores.get(3 - self.player_id, 0.0)
            terminal = float(max(-1.0, min(1.0, (own_s - opp_s) / total_fields)))
        else:
            terminal = (1.0 if winner == self.player_id else
                       -1.0 if winner != 0               else 0.0)

        # ── Append episodes to the replay buffer ─────────────────────────────
        if self._episode:
            targets = self._value_targets(
                [t[3] for t in self._episode], intermediate_rewards, terminal)
            for (enc, mask, dist, _), tgt in zip(self._episode, targets):
                self._buffer.append((enc, mask, dist, float(tgt)))
            self._episode.clear()

        if self._obs_episode:
            # Opponent's perspective: inverted terminal, no shaping rewards.
            obs_targets = self._value_targets(
                [t[3] for t in self._obs_episode], None, -terminal)
            for (enc, mask, dist, _), tgt in zip(self._obs_episode, obs_targets):
                self._buffer.append((enc, mask, dist, float(tgt)))
            self._obs_episode.clear()

        if len(self._buffer) > S.NM_BUFFER_MAX:
            del self._buffer[:len(self._buffer) - S.NM_BUFFER_MAX]

        self._train_from_buffer()
        self.save()

    def _value_targets(self, vals: list, shape, terminal: float) -> np.ndarray:
        """Per-step value targets: GAE returns blended with the final outcome.

        Pure GAE bootstraps through the network's own (initially wrong) value
        estimates; pure z (AlphaZero) ignores shaping.  The NM_Z_WEIGHT blend
        gives early ring-building moves direct credit for the final margin
        while keeping the dense shaping signal.
        """
        T = len(vals)
        shape = list(shape) if (shape and len(shape) == T) else [0.0] * T
        γ, λ  = S.NM_DISCOUNT, S.NM_GAE_LAMBDA
        vnext = vals[1:] + [terminal]
        adv   = np.zeros(T, dtype=np.float32)
        gae   = 0.0
        for t in reversed(range(T)):
            delta  = shape[t] + γ * vnext[t] - vals[t]
            gae    = delta + γ * λ * gae
            adv[t] = gae
        returns = adv + np.array(vals, dtype=np.float32)
        w = S.NM_Z_WEIGHT
        return np.clip((1.0 - w) * returns + w * float(terminal), -1.0, 1.0)

    def _train_from_buffer(self) -> None:
        """Sample NM_TRAIN_STEPS minibatches from the replay buffer and take
        one Adam step each.  Each sampled item gets a random D4 transform
        (policy target and mask transformed identically; value is invariant)."""
        if not self._buffer:
            return
        n_steps = (S.NM_TRAIN_STEPS
                   if len(self._buffer) >= S.NM_BATCH_SIZE else 1)
        for _ in range(n_steps):
            k   = min(S.NM_BATCH_SIZE, len(self._buffer))
            idx = np.random.choice(len(self._buffer), size=k, replace=False)
            encs, masks, dists, tgts = [], [], [], []
            for i in idx:
                enc, mask, dist, tgt = self._buffer[i]
                if S.NM_AUGMENT:
                    op   = int(np.random.randint(N_TRANSFORMS))
                    enc  = transform_planes(enc, op)
                    mask = transform_flat(
                        mask.astype(np.float32), op) > 0.5
                    dist = transform_flat(dist, op)
                encs.append(enc); masks.append(mask)
                dists.append(dist); tgts.append(tgt)

            states_t = torch.tensor(np.stack(encs), dtype=torch.float32,
                                    device=self._device)
            masks_t  = torch.tensor(np.stack(masks), dtype=torch.bool,
                                    device=self._device)
            pi_t     = torch.tensor(np.stack(dists), dtype=torch.float32,
                                    device=self._device)
            ret_t    = torch.tensor(np.array(tgts, dtype=np.float32),
                                    device=self._device)

            logits, vals_pred = self._net(states_t)
            logits_m = logits.clone()
            logits_m[~masks_t] = float('-inf')
            log_probs = F.log_softmax(logits_m, dim=1)
            probs     = log_probs.exp()

            # 0 * -inf = nan — zero illegal log-probs before the dot product.
            safe_lp = log_probs.clone()
            safe_lp[~masks_t] = 0.0
            pol_loss = -(pi_t * safe_lp).sum(dim=1).mean()
            val_loss = F.mse_loss(vals_pred, ret_t)
            legal_lp = torch.where(masks_t, log_probs,
                                   torch.zeros_like(log_probs))
            entropy  = -(probs * legal_lp).sum(dim=1).mean()

            loss = (S.NM_POLICY_COEF   * pol_loss
                    + S.NM_VALUE_COEF  * val_loss
                    - S.NM_ENTROPY_COEF * entropy)

            self._opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self._net.parameters(), S.PT_MAX_GRAD_NORM)
            self._opt.step()
```

(The old own-episode and obs-episode gradient blocks are deleted — `_train_from_buffer` replaces both.)

In `share_learning_with` (from Task 7), add buffer sharing:

```python
        other._buffer = self._buffer
```

and extend `tests/test_shared_learning.py::test_nm_net_and_buffer_shared` with `assert p2._buffer is p1._buffer`.

- [ ] **Step 5: Run tests, commit**

Run: `$PY -m pytest tests/ -q`
Expected: all pass.

```bash
git add settings.py ai/neural_mcts_player.py tests/test_nm_training.py tests/test_shared_learning.py
git commit -m "feat: NM replay buffer with z-blended value targets and D4 augmentation"
```

---

### Task 12: Neural MCTS — tree reuse between turns

NM rebuilds the tree from scratch every turn even though network calls dominate its cost. Reuse the chosen child's subtree like `MCTSPlayer` does, refreshing the surviving children's priors from the fresh root evaluation (so this turn's tactical boost + Dirichlet noise still apply).

**Files:**
- Modify: `ai/neural_mcts_player.py` (`__init__`, `choose_move`, `record_outcome`, new `_try_reuse_tree`/`_discard_tree`)
- Test: `tests/test_nm_training.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_nm_training.py`:

```python
class TestTreeReuse:
    def _node(self, move, parent, who):
        from ai.neural_mcts_player import _NMNode
        return _NMNode(move=move, parent=parent, children=None,
                       player_who_moved=who)

    def test_promotes_matching_grandchild(self):
        p = NeuralMCTSPlayer(1); p.save = lambda: None
        # Hand-built saved subtree: our chosen move (2,2), opponent replies
        chosen = self._node((2, 2), None, 1)
        chosen.children = [self._node((3, 3), chosen, 2),
                           self._node((4, 4), chosen, 2)]
        chosen.children[0].visits = 7
        p._saved_root  = chosen
        p._saved_board = {(2, 2): 1}
        new_root = p._try_reuse_tree({(2, 2): 1, (3, 3): 2})
        assert new_root is chosen.children[0]
        assert new_root.visits == 7
        assert p._saved_root is None        # state cleared after reuse

    def test_discards_on_removed_dots(self):
        p = NeuralMCTSPlayer(1); p.save = lambda: None
        p._saved_root  = self._node((2, 2), None, 1)
        p._saved_board = {(2, 2): 1, (5, 5): 2}
        assert p._try_reuse_tree({(2, 2): 1, (0, 0): 2}) is None

    def test_choose_move_saves_subtree(self):
        S.AI_THINK_MS = 30
        p = NeuralMCTSPlayer(1); p.save = lambda: None
        p.choose_move({}, set(), 1, {1: 0.0, 2: 0.0}, set())
        assert p._saved_root is not None
        assert p._saved_board is not None
```

- [ ] **Step 2: Run to verify failure**

Run: `$PY -m pytest tests/test_nm_training.py -q -k Reuse`
Expected: AttributeError — `_try_reuse_tree` / `_saved_root` do not exist.

- [ ] **Step 3: Implement**

In `__init__`, after the buffer:

```python
        # Tree reuse: subtree of the move we chose last turn + the board we
        # expected after playing it (used to diff out the opponent's reply).
        self._saved_root:  '_NMNode | None' = None
        self._saved_board: dict | None      = None
```

New methods (place near `_select`):

```python
    def _discard_tree(self) -> None:
        self._saved_root  = None
        self._saved_board = None

    def _try_reuse_tree(self, current_board: dict) -> '_NMNode | None':
        """Promote the grandchild matching the opponent's actual reply.

        Returns the new root (with inherited visit counts) or None.  Discards
        on encirclement (dots removed) or any unexpected board diff — same
        policy as MCTSPlayer._try_reuse_tree.
        """
        if self._saved_root is None or self._saved_board is None:
            return None
        prev = self._saved_board
        if any(pos not in current_board for pos in prev):
            self._discard_tree()
            return None
        new_dots = [pos for pos in current_board if pos not in prev]
        if len(new_dots) != 1:
            self._discard_tree()
            return None
        opp_move = new_dots[0]
        node = self._saved_root
        self._discard_tree()
        if node.children:
            for child in node.children:
                if child.move == opp_move:
                    child.parent = None
                    return child
        return None
```

In `choose_move`, replace the root construction block

```python
        root = _NMNode(move=None, parent=None, children=[],
                       player_who_moved=3 - player, prior=0.0)
        for gx, gy in legal:
            child = _NMNode(
                move=(gx, gy), parent=root, children=None,
                player_who_moved=player,
                prior=float(root_priors[gy * n + gx])
            )
            root.children.append(child)
```

with

```python
        reused = self._try_reuse_tree(board)
        if reused is not None and reused.children:
            # Inherit the searched subtree; refresh priors from THIS turn's
            # root evaluation (tactical boost + fresh Dirichlet noise) and
            # add children for any move that wasn't expanded last turn.
            root = reused
            root.move = None
            legal_set = set(legal)
            root.children = [c for c in root.children if c.move in legal_set]
            have = {c.move for c in root.children}
            for c in root.children:
                gx, gy = c.move
                c.prior = float(root_priors[gy * n + gx])
            for gx, gy in legal:
                if (gx, gy) not in have:
                    root.children.append(_NMNode(
                        move=(gx, gy), parent=root, children=None,
                        player_who_moved=player,
                        prior=float(root_priors[gy * n + gx])))
        else:
            root = _NMNode(move=None, parent=None, children=[],
                           player_who_moved=3 - player, prior=0.0)
            for gx, gy in legal:
                root.children.append(_NMNode(
                    move=(gx, gy), parent=root, children=None,
                    player_who_moved=player,
                    prior=float(root_priors[gy * n + gx])))
```

And just before `return best.move` at the end:

```python
        # Save the chosen subtree for next turn's reuse
        best.parent = None
        self._saved_root = best
        fb2 = fb.copy(); fb2.play(*best.move)
        self._saved_board = dict(fb2.board)
        return best.move
```

Also update the `len(legal) == 1` early-return path to call `self._discard_tree()` before returning, and add `self._discard_tree()` at the top of `record_outcome` (the game ended; the subtree is stale).

Finally update the file's module docstring / class docstring claim "no tree reuse — stateless between turns" to describe the new behaviour.

- [ ] **Step 4: Run tests, commit**

Run: `$PY -m pytest tests/ -q`
Expected: all pass.

```bash
git add ai/neural_mcts_player.py tests/test_nm_training.py
git commit -m "feat: NM tree reuse across turns with refreshed root priors"
```

---

### Task 13: Neural MCTS — batched leaf evaluation with virtual loss

Each tree expansion currently makes a batch-size-1 CNN call. Collect up to `NM_LEAF_BATCH` leaves per iteration (virtual loss steers selection away from already-pending leaves), evaluate them in one forward pass, then expand and backprop.

**Files:**
- Modify: `settings.py` (`NM_LEAF_BATCH`)
- Modify: `ai/neural_mcts_player.py` (`_NMNode.__slots__`, simulation loop, new `_apply_virtual` / `_expand_with`, `_expand_node` becomes a thin wrapper)
- Test: `tests/test_nm_training.py` (append)

- [ ] **Step 1: Add the setting**

In `settings.py`, after `NM_AUGMENT`:

```python
NM_LEAF_BATCH       = 8                # leaves evaluated per batched network call
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_nm_training.py`:

```python
class TestVirtualLoss:
    def test_apply_and_revert_restores_counts(self):
        from ai.neural_mcts_player import _NMNode, NeuralMCTSPlayer
        root  = _NMNode(None, None, [], player_who_moved=2)
        child = _NMNode((1, 1), root, None, player_who_moved=1)
        root.children = [child]
        root.visits, root.total_value   = 5, 2.5
        child.visits, child.total_value = 3, 1.0
        NeuralMCTSPlayer._apply_virtual(child, ai_player=1, sign=+1)
        assert (child.visits, root.visits) == (4, 6)
        NeuralMCTSPlayer._apply_virtual(child, ai_player=1, sign=-1)
        assert (child.visits, child.total_value) == (3, 1.0)
        assert (root.visits, root.total_value)   == (5, 2.5)


class TestBatchedSearch:
    def test_choose_move_returns_legal_and_counts_consistent(self):
        S.AI_THINK_MS = 60
        p = NeuralMCTSPlayer(1); p.save = lambda: None
        board = {(4, 4): 2}
        move = p.choose_move(board, set(), 1, {1: 0.0, 2: 0.0}, set())
        assert move not in board
        root = p._saved_root        # chosen child saved by tree reuse
        assert root.visits >= 1
        assert root.visits >= sum(c.visits for c in (root.children or []))
        def no_negative(n):
            assert n.visits >= 0
            for c in (n.children or []):
                no_negative(c)
        no_negative(root)
```

- [ ] **Step 3: Run to verify failure**

Run: `$PY -m pytest tests/test_nm_training.py -q -k Virtual`
Expected: AttributeError — `_apply_virtual` does not exist.

- [ ] **Step 4: Implement**

`_NMNode` gains a `pending` flag:

```python
    __slots__ = ('move', 'parent', 'children', 'visits', 'total_value',
                 'player_who_moved', 'prior', 'pending')

    def __init__(self, move, parent, children, player_who_moved, prior=0.0):
        ...
        self.prior            = prior
        self.pending          = False    # awaiting batched network evaluation
```

New static method on `NeuralMCTSPlayer`:

```python
    @staticmethod
    def _apply_virtual(node: '_NMNode', ai_player: int, sign: int) -> None:
        """Apply (+1) or revert (-1) a virtual loss along the path to root.

        Pending leaves are temporarily scored as a loss for ai_player so the
        next selection in the same batch explores a different leaf."""
        cur = node
        while cur is not None:
            cur.visits += sign
            if cur.player_who_moved != ai_player:
                cur.total_value += sign      # a win for the opponent
            cur = cur.parent
```

Split `_expand_node` so expansion can consume a precomputed network row:

```python
    def _expand_with(self, node: '_NMNode', fb, logits_row, value: float) -> float:
        """Populate node.children from a precomputed network output row.
        Returns the value in [-1, 1] from fb.player's perspective."""
        legal = fb.legal_moves()
        if not legal:
            node.children = []
            return 0.0
        n    = S.GRID
        mask = torch.zeros(n * n, dtype=torch.bool, device=self._device)
        for gx, gy in legal:
            mask[gy * n + gx] = True
        lg = logits_row.clone()
        lg[~mask] = float('-inf')
        priors = F.softmax(lg, dim=0).cpu().numpy()

        node_phase = len(fb.board) / max(n * n, 1)
        priors = self._tactical_priors(
            priors, fb.board, fb.connections, fb.player, mask.cpu().numpy(),
            suppress_dead=False, game_phase=node_phase)

        node.children = [
            _NMNode(move=(gx, gy), parent=node, children=None,
                    player_who_moved=fb.player,
                    prior=float(priors[gy * n + gx]))
            for gx, gy in legal
        ]
        return value

    def _expand_node(self, node: '_NMNode', fb) -> float:
        """Single-leaf expansion (kept for clarity/compat): one network call."""
        enc = _encode(fb.board, fb.connections, fb.forbidden, fb.player)
        with torch.no_grad():
            x = torch.tensor(enc, dtype=torch.float32,
                             device=self._device).unsqueeze(0)
            lg, v = self._net(x)
        return self._expand_with(node, fb, lg[0], float(v[0].item()))
```

Replace the simulation loop in `choose_move`:

```python
        deadline   = time.time() + S.AI_THINK_MS / 1000.0
        batch_size = getattr(S, 'NM_LEAF_BATCH', 8)

        while time.time() < deadline:
            pending = []                 # (node, sim_fb) awaiting evaluation
            for _ in range(batch_size):
                node, sim_fb = self._select(root, fb.copy())

                if sim_fb.done:
                    s1, s2 = sim_fb.scores.get(1, 0.0), sim_fb.scores.get(2, 0.0)
                    total  = (s1 + s2) or 1.0
                    val01  = (s1 if player == 1 else s2) / total
                    self._backprop(node, val01, player)
                    continue
                if node.children is not None:
                    # Expanded non-terminal with nothing to descend — neutral.
                    self._backprop(node, 0.5, player)
                    continue
                if node.pending:
                    # Virtual loss couldn't steer away — batch is saturated.
                    break
                node.pending = True
                self._apply_virtual(node, player, +1)
                pending.append((node, sim_fb))

            if not pending:
                continue

            encs = np.stack([
                _encode(fb_.board, fb_.connections, fb_.forbidden, fb_.player)
                for _, fb_ in pending])
            with torch.no_grad():
                x     = torch.tensor(encs, dtype=torch.float32,
                                     device=self._device)
                lg, v = self._net(x)

            for i, (node, fb_) in enumerate(pending):
                self._apply_virtual(node, player, -1)
                node.pending = False
                raw_v = self._expand_with(node, fb_, lg[i], float(v[i].item()))
                if fb_.player == player:
                    val01 = (raw_v + 1.0) / 2.0
                else:
                    val01 = (1.0 - raw_v) / 2.0
                self._backprop(node, val01, player)
```

- [ ] **Step 5: Run tests, commit**

Run: `$PY -m pytest tests/ -q`
Expected: all pass.

```bash
git add settings.py ai/neural_mcts_player.py tests/test_nm_training.py
git commit -m "perf: batched NM leaf evaluation with virtual loss (NM_LEAF_BATCH)"
```

---

### Task 14: Headless training validation script

A smoke test that runs short Trainer sessions for every backend and asserts all weights stay finite — this is the check that would have caught the NaN poisoning immediately. Saves are silenced so the script never touches the real weight/experience files.

**Files:**
- Create: `scripts/validate_training.py`

- [ ] **Step 1: Create the script**

Create `scripts/validate_training.py`:

```python
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
```

- [ ] **Step 2: Run it**

Run: `$PY scripts/validate_training.py`
Expected output: four `finite=OK` lines and `All training smoke tests passed.`, exit code 0. The MCTS and NM pairs take the longest (~50 ms × ~60–100 moves × 2 rounds each — a couple of minutes total). If any line says `FAIL`, stop and debug that backend before continuing.

- [ ] **Step 3: Commit**

```bash
git add scripts/validate_training.py
git commit -m "test: headless training smoke test asserting finite weights for all backends"
```

---

### Task 15: Documentation sync

Update the docs to match the new code, then run the full suite one last time.

**Files:**
- Modify: `CLAUDE.md`, `README.md`

- [ ] **Step 1: Update CLAUDE.md**

Make these edits (each refers to an existing section):

1. **ai/ module table:** add `| ai/symmetry.py | D4 board transforms (canonical experience keys, NM training augmentation) |`.
2. **MCTS `record_outcome` / experience:** note that experience keys are now **canonical under D4 symmetry** (`ai/symmetry.canonical_board_key`); old non-canonical entries age out via `AI_MAX_EXPERIENCE`.
3. **PyTorch player section:** the residual blocks/stem use **GroupNorm** (no running stats; arch dict carries `'norm': 'gn'` and old BatchNorm checkpoints are rejected). Trajectory tuples are 6-long: `(enc, mask, action, log_prob_old, value_old, boost_log)`; the tactical boost is folded into the logits via `features.boost_log_offsets` both at action time and in PPO training — never re-add a post-hoc probability boost, it breaks the PPO ratio.
4. **NN player section:** same boost-as-logits scheme; `_ACNetwork.update` takes `legal_mask` and `boost_log` and backprops through pre-update trunk weights.
5. **Neural MCTS section:** replace the "no tree reuse" paragraph (it now reuses the chosen subtree with refreshed priors), document the replay buffer (`NM_BUFFER_MAX/NM_BATCH_SIZE/NM_TRAIN_STEPS`), z-blended value targets (`NM_Z_WEIGHT`), D4 augmentation (`NM_AUGMENT`), and batched leaf evaluation with virtual loss (`NM_LEAF_BATCH`, `_NMNode.pending`).
6. **Shared learning:** new subsection under AIRunner or Trainer: same-type self-play pairs are linked via `share_learning_with()` (one net / experience dict, observation recording disabled on both sides); wired in `_make_runners` and both train-screen player builders.
7. **Common pitfalls:** add `- `_NMNode.pending` must always be reset to False and virtual losses reverted before real backprop — a leaked virtual loss permanently biases that subtree.` and update the "no tree reuse" pitfall line for NM.
8. **Project files table:** add `scripts/validate_training.py`.

- [ ] **Step 2: Update README.md**

In the configuration table, add rows (match the table's existing format) for:
`NM_BUFFER_MAX` (8000, NM replay-buffer size), `NM_BATCH_SIZE` (256), `NM_TRAIN_STEPS` (4), `NM_Z_WEIGHT` (0.5, outcome blend in NM value targets), `NM_AUGMENT` (True, D4 augmentation), `NM_LEAF_BATCH` (8, batched leaf evaluation).

- [ ] **Step 3: Final verification**

Run: `$PY -m pytest tests/ -q` — expected: all pass (~200 tests).
Run: `$PY scripts/validate_training.py` — expected: `All training smoke tests passed.`
Run: `$PY -c "import dotgame"` — expected: no import errors.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs: sync CLAUDE.md and README with AI training pipeline changes"
```

---

## Post-plan notes (manual, not part of execution)

- **Retraining:** PT and NM weight files were deleted; both backends start fresh. The recommended bootstrap (per CLAUDE.md) is training NM/PT **against MCTS** first via the in-game TRAIN screen, then self-play. `nn_weights_10x10.npz` was kept but was trained under the observe bug — consider deleting it for a clean slate.
- **`ai_experience_10x10.json`** keeps working; old keys just stop matching the canonical form and get evicted over time.
- Merging: `git checkout main && git merge ai-improvements` once you're satisfied (or open a PR).