# CLAUDE.md — Developer & AI Context

This file is for AI coding assistants and contributors. It documents architecture, invariants,
pitfalls, and conventions for the Dot Grid codebase.

---

## Module responsibilities

| Module | Owns | Must not |
|---|---|---|
| `settings.py` | All constants, MCTS profiles, file paths | Import any other project module |
| `state.py` | Board dict, connections, scores, snapshots, `reset()`, `place()`, `check_win()` | Call pygame |
| `territory.py` | Flood-fill engine, encirclement, `forbidden_positions` | Modify viewport or draw |
| `viewport.py` | `cell_size`, `offset_x/y`, coordinate math | Touch game state |
| `draw.py` | Every pygame Surface operation | Modify game state |
| `game_mode.py` | `Mode` enum, player name resolution | Import pygame |
| `dotgame.py` | Event loop, AI scheduling, screen routing, training wiring | Contain game logic |
| `ai/` | AI player implementations | Call pygame directly |

---

## Module responsibilities (ai/ detail)

| Module | Owns |
|---|---|
| `ai/base_player.py` | Abstract `BasePlayer` interface |
| `ai/features.py` | All board encoding and strategic signal computation |
| `ai/mcts.py` | Pure MCTS (per-instance profiles, PUCT priors, 6-tier rollouts, Dirichlet noise) |
| `ai/nn_player.py` | Actor-critic NN, pure numpy, 15-channel input |
| `ai/pytorch_player.py` | Residual CNN + PPO + GAE, 15-channel spatial input |
| `ai/neural_mcts_player.py` | AlphaZero-style: CNN guides MCTS tree (policy=prior, value=leaf eval) |
| `ai/runner.py` | Background thread wrapper |
| `ai/trainer.py` | Offline self-play trainer (territory + arc shaping rewards) |
| `ai/paths.py` | Grid-stamped experience file path helper |

---

## State module (`state.py`)

### Module-level globals

```python
board, connections, territories, interior_dots, interior_conns
forbidden_positions: set       # grid coords blocked from placement
scores, current_player, last_move, total_moves, game_over, winner_player
snapshots: list                # full state dict per move (index 0 = empty board)
snapshot_index: int            # -1 = live view
TOTAL_FIELDS = (GRID-1)**2
```

### Draw condition

`winner_player` is `None` when both scores are equal at end-of-board. All consumers must handle
`None`:

```python
# check_win() when board is full:
winner_player = None if scores[1] == scores[2] else (1 if scores[1] > scores[2] else 2)

# draw_win_screen():
if p is None:
    # render "DRAW / Equal score!"
```

### Critical: `reset()` global declaration

`reset()` must declare **all** module-level variables in its `global` statement, including
`forbidden_positions`. If any variable is missing, Python creates a local instead of modifying
the module-level one, leaving stale state from the previous game.

### Key functions

- `place(gx, gy)` — checks `forbidden_positions`, places dot, builds connections, calls
  `recompute_territories()`, then `check_win()`, appends snapshot.
- `check_win()` — triggers on score threshold **or** `placeable <= 0`. Sets
  `winner_player = None` on a draw.
- `_capture()` / `_restore()` — must stay in sync; both include `forbidden_positions`. Any new
  state field must be added to both.

---

## Territory engine (`territory.py`)

### Algorithm

- **SCALE = 6** raster: each grid intersection maps to a 6×6 pixel block.
- Each dot rasterised as a **3×3 wall block** clamped to `[0, sz]`.
- Diagonal connections receive **L-corner fills** so 4-directional flood fill can't leak through.
- Two independent flood fills from `(-1, -1)` find all pixels reachable per player.
- Enclosed = not in `outside` and not a wall pixel.

### Critical: clamp the 3×3 expansion

Must be clamped to `[0, sz]`. Writing to pixel `-1` blocks the flood fill seed, falsely scoring
the entire board.

### Encirclement sequence

1. Build walls and flood fills.
2. Detect trapped opponent dots.
3. `_remove_encircled()` — remove trapped dots except the encircler's ring.
4. **Rebuild walls and flood fills** — removing dots changes geometry.
5. Compute territory normally.

Step 4 is mandatory.

---

## Feature module (`ai/features.py`)

All board encoding and strategic signal computation. Shared by all three AI players.

### Signal overview

| Function | Returns | Notes |
|---|---|---|
| `opportunity_masks(board, connections, player)` | (own, opp) float32[G²] | Mirrors territory.py scoring: full square = 1.0, triangle = 0.5 |
| `apply_boost(probs, own_scores, opp_scores, legal)` | probs modified in place | Post-policy multiplier boost |
| `_find_components(board, player)` | list[set] | Connected components via BFS |
| `_component_potential(comp, n)` | float | Flood-fill interior fraction — 0 for any open ring |
| `enclosure_potential(board, connections, player)` | (own, opp) float32[G²] | Per-position flood-fill potential (only fires on complete rings) |
| `enclosure_scalars(board, player)` | (own_max, opp_max) | Scalar version of above |
| `_arc_potential_comp(comp, n)` | float (territory-cell units) | Bounding-box × completion; fires for partial rings |
| `arc_potential_scalars(board, player)` | (own_max, opp_max) in [0,1] | Normalised arc potential scalars for `_eval_fast` and trainer shaping |
| `arc_potential_map(board, player)` | (own, opp) float32[G²] in territory-cell units | Used in MCTS priors |
| `bridge_potential(board, connections, player)` | (own, opp) float32[G²] | Extra potential from bridging two arcs; uses `_arc_potential_comp` |
| `disruption_map(board, connections, player)` | float32[G²] | Value of placing inside opponent-claimed interior |
| `fork_map(board, player)` | float32[G²] | Positions creating 2+ simultaneous threats |
| `close_setup_map(board, player)` | float32[G²] | Positions that create a 3-own-corner unit square |
| `get_centrality()` | float32[G²], cached | 1.0 at centre, 0.0 at corner |
| `strategic_channels(board, connections, player, total_dots)` | 6-tuple | Convenience: all strategic channels at once |

### `_arc_potential_comp` units

Returns **territory-cell units** (not normalised to [0,1]). A ring enclosing 9 cells at 90%
completion returns ~8.1. Divide by `(n-1)²` before use in blend weights (which
`arc_potential_scalars` does automatically). In MCTS priors the raw value is divided inline:
`float(own_arc[idx]) / total_cells * weight`.

### Why `enclosure_potential` vs `arc_potential_map`

`enclosure_potential` (flood-fill) returns 0 for any ring with a single gap — completely blind
during ring construction. `arc_potential_map` (bounding-box × completion) fires continuously from
the first arc placed. MCTS priors and `_eval_fast` use `arc_potential_*`. The neural network
input channels (ch 7–8) still use `enclosure_potential` because the NN can learn to infer ring
progress from adjacent channels; replacing them would invalidate saved weights.

`arc_potential_scalars` is also called in `Trainer._run()` to compute the arc-potential-delta
shaping reward — this is the primary reward signal that teaches NN/PT ring-building strategy.

---

## MCTS player (`ai/mcts.py`)

### Per-instance profiles

`MCTSPlayer.__init__` now accepts a `profile: int` parameter (default 0 = global defaults):

```python
MCTSPlayer(player_id=1, profile=1)   # Opportunist
MCTSPlayer(player_id=2, profile=2)   # Strategist
```

Profiles are defined in `settings.MCTS_PROFILES` as dicts with keys:
`close_w, block_w, arc_w, arc_opp_w, bridge_w, bridge_opp_w, setup_w, adj_w, centrality_w,
disrupt_w, noise_alpha, noise_frac, exploration, sel_terr_w`.

Profile 0 (default) reads from the global `S.MCTS_*` constants — used for human vs MCTS and
any other single-MCTS mode. Profiles 1 and 2 are applied in `dotgame._make_runners()` only when
`mode == Mode.MCTS_VS_MCTS`.

All per-instance weights are stored as `self._*` attributes on the player object and are
immutable after construction.

### FastBoard

Lightweight board simulator. Key methods:

- `legal_moves()` — positions not in board and not in forbidden.
- `closing_moves(p)` — positions completing a unit square for player p.
- `blocking_moves(p)` — positions where opponent could close next turn.
- `adjacent_own(p)` — empty positions Chebyshev-adjacent to own dots.
- `setup_moves(p)` — positions creating a 3-own-corner square (next-turn threat).
- `fork_moves(p)` — positions creating 2+ simultaneous threats.
- `play(gx, gy)` — places dot, builds connections, calls `_update_scores_fast()` or
  `_update_territory_real()`, then `_check_terminal()`.
- `_check_terminal()` — O(1) in fast mode (`len(board) >= n²`); O(G²) scan only in real mode
  where forbidden positions are tracked.

### PUCT priors (per `choose_move` call)

Eight signals combined using per-instance weights:

```python
raw = (own_s[idx]     * self._close_w       # immediate close
     + opp_s[idx]     * self._block_w       # block opponent close
     + own_setup[idx] * self._setup_w       # create 3-corner setup
     + arc_own        * self._arc_w         # extend ring arc
     + arc_opp        * self._arc_opp_w     # block opponent ring arc
     + br_own         * self._bridge_w      # bridge own arcs
     + br_opp         * self._bridge_opp_w  # block opponent bridge
     + own_adj        * self._adj_w         # cluster growth
     + centrality[idx]* self._centrality_w  # baseline (breaks empty-board tie)
     + disrupt[idx]   * self._disrupt_w)    # disrupt opponent interior
```

Default profile weights (profile 0): close=4.0, block=3.0, arc=5.0, arc_opp=4.0, bridge=3.0,
bridge_opp=2.5, setup=2.5, adj=1.5, centrality=0.5, disrupt=0.3.

Arc and bridge values are in territory-cell units; divide by `total_cells` before applying
weights (done inline in `choose_move`).

### Dirichlet noise

After computing `raw_priors`, noise is injected using per-instance alpha/frac:

```python
adaptive_alpha = max(0.01, self._noise_alpha * 10.0 / len(legal))
eta = np.random.dirichlet([adaptive_alpha] * len(legal))
raw_priors[pos] = (1 - self._noise_frac) * val + self._noise_frac * eta[i] * max_score
```

Adaptive alpha ensures spikiness is consistent: with 10 legal moves it stays at `noise_alpha`;
with 80 moves it drops proportionally, producing 1-2 large spikes rather than near-uniform noise.

### UCB exploration and final selection

`_Node.best_child(exploration)` now accepts the exploration constant explicitly.
`_select()` passes `self._exploration` to every `best_child()` call so the profile's exploration
constant is applied throughout the entire search.

Final move selection blends win rate with the prior score using `self._sel_terr_w`:

```python
combined = win_rate + self._sel_terr_w * score
```

### `_eval_fast`

Four-signal blend for non-terminal rollout leaves (must be cheap — called ~500×/sec):

```python
own_arc, opp_arc = arc_potential_scalars(fb.board, ai_player)   # normalised [0,1]
enc_pot = 0.5 + 0.5 * clip(own_arc - opp_arc, -1, 1)
return 0.15*dot_ratio + 0.10*conn_ratio + 0.25*terr_pot + 0.50*enc_pot
```

Arc potential weight is 50% — intentionally dominant. `bridge_potential` is excluded from
`_eval_fast` — it's O(G²·components) per call and would destroy rollout throughput.

### Rollout policy (6 tiers)

```
1. fork_moves     — 2+ simultaneous threats (if no immediate close)
2. closing_moves  — complete a unit square now
3. blocking_moves — stop opponent closing
4. setup_moves    — create a 3-own-corner position
5. adjacent_own   — extend own cluster (50% bias)
6. random legal
```

The `setup_moves` tier is critical for MCTS vs MCTS play. Without it, rollouts consist only of
line-extension moves, territory never forms in simulations, win rates converge to 0.5 everywhere,
and priors dominate all decisions permanently.

### `_check_terminal` — performance invariant

Must remain O(1) in fast mode. The `len(board) >= n²` check is the fast path. The O(G²) scan
only runs when `S.MCTS_REAL_ROLLOUT = True`. Any future change that adds forbidden tracking to
fast mode must also add the O(G²) scan.

### `record_outcome`

Accepts `winner` (int, 0 = draw/unknown), `intermediate_rewards` (ignored), and `final_scores`
(ignored — MCTS uses tree statistics). Both parameters exist for interface compatibility with
`AIRunner`.

---

## Neural Network player (`ai/nn_player.py`)

### Architecture

- Input: `15 × G²` float32 vector (flat)
- Trunk: `[15·G² → hidden[0] → hidden[1] → …]` with ReLU — sizes from `S.nn_hidden_sizes()`
- Policy head: `[hidden[-1] → G²]` logits → masked softmax
- Value head: `[hidden[-1] → 1]` → tanh

### Training: `record_outcome(winner, intermediate_rewards, final_scores)`

- Terminal value uses margin: `(own_score − opp_score) / (GRID-1)²` clipped to [-1,1].
  Falls back to ±1 if `final_scores` is None.
- `intermediate_rewards` now contains both territory-delta and arc-potential-delta shaping
  (combined by `Trainer`).
- GAE advantages (λ=0.95), entropy bonus (0.01), value coef (0.5).
- Adam (β₁=0.9, β₂=0.999).

### Weight compatibility

Saved metadata keys: `meta_grid`, `meta_hidden`, `meta_channels` (must equal 15). Mismatches
silently rejected on load.

---

## PyTorch player (`ai/pytorch_player.py`)

### Architecture

```
Input  (15, G, G)
Stem   Conv2d(15→64, 3×3) → BN → ReLU
Tower  N × ResBlock(64ch):  Conv→BN→ReLU→Conv→BN  +  skip  → ReLU
Policy Conv(64→2, 1×1) → flatten → Linear(2G²→G²)
Value  Conv(64→1, 1×1) → flatten → Linear(G²→64) → ReLU → Linear(64→1)
```

Architecture dict includes `'in_channels': 15` for load compatibility. Old weights without it
or with `in_channels != 15` are silently rejected.

### PPO training loop

1. GAE advantages: `δ_t = r_t + γ·V(s_{t+1}) − V(s_t)`, `A_t = Σ(γλ)^k·δ_{t+k}`
2. Terminal value = margin = `(own_score − opp_score) / (GRID-1)²`
3. `intermediate_rewards` now includes arc-potential-delta shaping from `Trainer`
4. For `PT_PPO_EPOCHS` passes, minibatches of `PT_PPO_MINIBATCH`:
   - `ratio = exp(log_π − log_π_old)`
   - `L_clip = −min(ratio·A, clip(ratio, 1−ε, 1+ε)·A)`
   - `L_total = L_clip + PT_VALUE_COEF·L_value − PT_ENTROPY_COEF·L_ent`
5. Gradient clip to `PT_MAX_GRAD_NORM`, Adam step.

---

## `AIRunner` (`ai/runner.py`)

- `on_game_end(winner, final_scores=None)` — forwards both to `record_outcome`.
- `winner` may be `None` (draw); pass through without modification — player implementations
  handle `None` as a non-win outcome.
- Snapshots `forbidden_positions` at `start_thinking()` — never reads live state from the thread.
- `except Exception` in thread catches all failures; logs to stderr, returns random legal fallback.
- `finally: self._thinking = False` always runs.

---

## `Trainer` (`ai/trainer.py`)

`SimGame` is a fully self-contained game — zero shared state with the live game.

### Draw handling in SimGame

`SimGame._check_win()` now sets `winner = None` when both scores are equal at end-of-board,
matching `state.check_win()`. `Trainer._run()` passes `winner or 0` (i.e. `0` for draws) to
`record_outcome` — consistent with the int-based interface on all player types.

### Shaping reward (per move)

```python
# Territory delta
terr_delta = game.scores[cp] - scores_before[cp]

# Arc potential delta — rewards ring-building progress each step
arc_before, _ = arc_potential_scalars(board_before, cp)
arc_after,  _ = arc_potential_scalars(board_after,  cp)
arc_delta = max(0.0, arc_after - arc_before)   # only positive increments rewarded

shaping = terr_delta * S.REWARD_TERRITORY + arc_delta * S.REWARD_ARC_SHAPING
```

The arc shaping reward fixes the NN/PT "small-grab bias": without it the network sees a nonzero
reward only when a ring closes, so it learns to prefer small immediate captures. With it, every
step extending a ring arc is rewarded proportionally to the increase in ring-building potential.

`arc_potential_scalars` is called twice per move (before and after) — this is O(G²·components)
but acceptable at training speed (no real-time constraint). Do not add this to `_eval_fast`.

---

## Draw module (`draw.py`)

### Screen routing overview

```
picking_mode    → draw_mode_picker       → returns (mode_rects, name_rects, train_r, settings_r, arch_r)
settings_screen → draw_settings_screen   → returns (back_r, prev_think_r, next_think_r, deep_terr_r)
arch_select     → draw_arch_select       → returns (back_r, mcts_r, nn_r, pt_r, nm_r)
arch_overlay    → draw_arch_overlay      → returns (close_r, up_r, down_r, content_h)
training_mode   → draw_train_config      → returns 11-tuple (see below)
training        → draw_train_progress    → (no return value)
paused          → draw_pause_menu        → returns (resume_r, new_game_r, quit_r)
game over       → draw_win_screen        → (no return value)
                  draw_replay_controls   → returns (prev_r, next_r)
always          → draw_board             → (no return value)
always          → draw_panel             → returns (reset_r, zoom_rects, stop_r, None, None)
```

### `draw_panel` return value

Returns a **5-tuple**: `(reset_rect, zoom_rects, stop_rect, None, None)`. The last two slots are
always `None` — the AI architecture button and Deep Terr toggle have moved to the AI Info and
Settings screens respectively. Always unpack as 5-tuple.

### `draw_train_config` return value

Returns an **11-tuple**:

```python
(back_r, start_r,
 prev_ai, next_ai, prev_opp, next_opp,
 prev_rnd, next_rnd, prev_think, next_think,
 deep_terr_r)   # ← new: toggle MCTS_REAL_ROLLOUT during training
```

Always unpack as 11 values. The `deep_terr_r` button toggles `S.MCTS_REAL_ROLLOUT` directly.

### `draw_arch_select` return value

Returns a **5-tuple**: `(back_r, mcts_r, nn_r, pt_r, nm_r)`. Always unpack as 5 values.

```python
back_r, mcts_r, nn_r, pt_r, nm_r = arch_sel_rects
# clicking nm_r sets arch_type = "NM" and opens arch_overlay
```

```python
draw_arch_overlay(surf, fonts, win_w, win_h, ai_type, scroll_y=0)
# ai_type: "MCTS" | "NN" | "PT"
# Returns: (close_r, up_r, down_r, content_h)
# up_r / down_r are None when content fits without scrolling
```

The overlay renders to an off-screen surface and blits a `scroll_y`-offset slice. Scroll state
is owned by `dotgame.py` as `arch_scroll: int`. Mouse wheel, `↑`/`↓` keys, and the `▲`/`▼`
buttons all step by `_SCROLL_STEP = 40` pixels. `arch_content_h` (returned by the draw call)
is used to clamp `arch_scroll` to `max(0, content_h - visible_h)`.

### Text safety

All label text passes through `_fit_label(font, text, max_w)` before rendering. This truncates
with `…` if the rendered width exceeds `max_w`. All `_button()` calls apply this automatically.

---

## `dotgame.py` — event loop conventions

### Screen state flags

```python
picking_mode    # main menu visible
settings_screen # settings overlay visible
arch_select     # AI info select screen visible
arch_overlay    # arch diagram visible
arch_type       # "MCTS" | "NN" | "PT"
arch_scroll     # int, scroll offset in pixels
arch_content_h  # int, total diagram height (for clamping)
training_mode   # train config overlay visible (not running)
training        # training actively running
paused          # pause menu visible
```

`any_overlay = (picking_mode or training_mode or training or paused or
                settings_screen or arch_select or arch_overlay)`

AI trigger and move application are both guarded by `not any_overlay`.

### Panel 5-tuple unpack

```python
(reset_rect, zoom_rects, stop_rect, _, _) = draw_panel(...)
```

The last two slots are always `None` — do not attempt to use them.

### Train config 11-tuple unpack

```python
(back_r, start_r,
 pra, nxa, pro, nxo,
 prr, nxr, prt, nxt,
 train_deep_r) = train_cfg_rects
```

### Arch select 5-tuple unpack

```python
back_r, mcts_r, nn_r, pt_r, nm_r = arch_sel_rects
```

### MCTS profile assignment

```python
p1_profile = 1 if (mode == Mode.MCTS_VS_MCTS and p1_type == 'mcts') else 0
p2_profile = 2 if (mode == Mode.MCTS_VS_MCTS and p2_type == 'mcts') else 0
```

Only `MCTS_VS_MCTS` mode receives profiles 1 and 2. Every other MCTS use (human vs MCTS, etc.)
uses profile 0 (global defaults), which behaves identically to the previous single-profile design.

### `on_game_end` — all three call sites pass `final_scores`

```python
r.on_game_end(state.winner_player, final_scores=dict(state.scores))
```

`state.winner_player` may be `None` (draw). `AIRunner` passes it through unchanged.

### Stop button draw condition

```python
s1, s2 = state.scores[1], state.scores[2]
state.winner_player = (None if s1 == s2 else (1 if s1 > s2 else 2))
```

---

## Neural MCTS player (`ai/neural_mcts_player.py`)

### Design

AlphaZero-style search. Reuses `pytorch_player._build_net` and `encode_state` verbatim —
no architecture code duplication. Separate weight file (`nm_weights_GxG.pt`) so PyTorch
weights are never affected.

### `_NMNode` vs `_Node` (mcts.py)

`_NMNode` has no `untried` list. AlphaZero expands all children at once on first visit rather
than one at a time. `children = None` means "never expanded"; `children = []` means "expanded
but terminal". `total_value` accumulates the sum of backpropagated values; Q = total_value/visits.

### PUCT formula

```
Q + c_puct · P(s,a) · √N(s) / (1 + N(s,a))
```

`N(s)` = parent visits, `N(s,a)` = child visits, `P(s,a)` = network policy prior.
`c_puct` is `S.NM_C_PUCT` (default 1.5). Higher values → more exploration.

### `choose_move` flow

1. Single network call at root → `root_priors` (policy) + `root_val` (stored for training)
2. Tactical boost applied to `root_priors` as safety net for untrained network
3. Dirichlet noise injected (adaptive alpha, same formula as `MCTSPlayer`)
4. Root's children pre-populated with those priors (no tree reuse — stateless between turns)
5. MCTS loop: `_select` → if leaf: `_expand_node` (network call) → `_backprop`; no rollouts
6. Visit distribution stored in `self._episode`
7. Most-visited child chosen (always `argmax(visits)`)

### No tree reuse

`MCTSPlayer` reuses the subtree across turns because its nodes are cheap to reconstruct.
`NeuralMCTSPlayer` does not reuse the tree — each turn starts fresh. The network call at
the root and at each new leaf is the dominant cost, not tree construction.

### `record_outcome` — single gradient update per game

```python
L = NM_POLICY_COEF  · CE(π_mcts, π_net)    # cross-entropy: replicate visit distribution
  + NM_VALUE_COEF   · MSE(v_pred, returns)  # value: predict GAE returns
  - NM_ENTROPY_COEF · H(π_net)              # entropy: keep exploration alive
```

One Adam step per game — no PPO, no minibatch shuffling. The visit distribution is a stable
external supervised target; PPO's stability mechanisms are not needed here.

GAE is identical to `PyTorchPlayer` (same γ, λ, shaping rewards). Intermediate rewards contain
both territory-delta and arc-potential-delta shaping from `Trainer`.

### `_backprop` sign convention

Values are always in `[0, 1]` from `ai_player`'s perspective.
- Node where `ai_player` moved → `total_value += val01`
- Node where opponent moved → `total_value += (1 − val01)`

This means Q(s,a) is always the average win probability for `ai_player`, regardless of whose
turn it is at that node.

### Settings keys

```python
NM_EXPERIENCE_BASE  = "../nm_weights"
NM_LEARNING_RATE    = 1e-3
NM_DISCOUNT         = 0.99
NM_GAE_LAMBDA       = 0.95
NM_VALUE_COEF       = 1.0
NM_POLICY_COEF      = 1.0
NM_ENTROPY_COEF     = 0.01
NM_C_PUCT           = 1.5
NM_NOISE_ALPHA      = 0.3
NM_NOISE_FRAC       = 0.25
NM_OVERRIDE         = None   # {'blocks': 4, 'channels': 64}
```

`nm_arch()` uses the same `_PT_PRESETS` table as `pt_arch()`. Both NM and PT use the same
architecture at any given grid size, but their weights are always separate files.

### Weight compatibility

Saved keys: `grid`, `arch` (includes `in_channels: 15`). Mismatches silently rejected on load,
same as `PyTorchPlayer`.

---

## Common pitfalls

- `draw_arch_select` returns **5 values** — always unpack as `(back_r, mcts_r, nn_r, pt_r, nm_r)`.
- `NeuralMCTSPlayer` does **not** reuse the tree between turns — every `choose_move` builds fresh.
- `_NMNode.children = None` means unexpanded; `= []` means expanded terminal. Never confuse these.
- `NM_OVERRIDE` uses the same dict format as `PT_OVERRIDE`: `{'blocks': N, 'channels': C}`.
- Neural MCTS weight files (`nm_weights_GxG.pt`) are separate from PyTorch (`pt_weights_GxG.pt`).
- A freshly created `NeuralMCTSPlayer` with random weights is weaker than pure MCTS — the random
  priors mislead the search. Always bootstrap against MCTS before self-play.
- `_NM_AVAILABLE` mirrors `_PT_AVAILABLE` — both require PyTorch. The import guard in `dotgame.py`
  uses a separate try/except so a future split of NM to a different backend remains easy.
- **Never** let 3×3 dot blocks write to pixel `-1` — blocks flood fill seed.
- **Never** use `elif` chains that swallow the human board-click handler.
- **Always** add new state fields to `_capture()`, `_restore()`, and `reset()` — all three.
- **Always** guard AI scheduling with `not any_overlay` (the full flag union, not just
  `not picking_mode`).
- AI `choose_move` runs in a thread — never write to `state.*` from inside it.
- `draw_panel` returns **5 values** — always unpack as `(a, b, c, _, _)`.
- `draw_train_config` returns **11 values** — always include `train_deep_r` in the unpack.
- `winner_player` can be `None` (draw) — all rendering and record_outcome paths must handle it.
- `MCTSPlayer.record_outcome` accepts `final_scores` for interface compatibility but ignores it.
- `MCTSPlayer(profile=0)` uses global `S.MCTS_*` defaults; profiles 1/2 override all weights.
- `_check_terminal` must remain O(1) in fast mode — any change that adds forbidden tracking to
  fast mode must also add the O(G²) scan path.
- `_arc_potential_comp` returns values in **territory-cell units**, not [0,1]. Divide by
  `(n-1)²` before use in blend weights or comparisons with `own_s`.
- `bridge_potential` uses `_arc_potential_comp` internally (not `_component_potential`) so it
  returns non-zero values for partial rings. Keep it this way.
- `arc_potential_scalars` is called in the trainer loop on every move — this is intentional and
  acceptable at training speed. Do **not** add it to `_eval_fast` (too slow).

---

## Project files

| File | Purpose |
|---|---|
| `pyproject.toml` | Project metadata, entry point, optional `[torch]` dependency group |
| `requirements.txt` | Plain pip requirements (pygame + numpy; torch commented out) |
| `CLAUDE.md` | This file — AI/contributor context |
| `README.md` | User-facing docs |
| `HOW_THE_AI_WORKS.md` | Plain-English AI guide |

---

## Adding a new game mode

1. Add `Mode` variant to enum in `game_mode.py`.
2. Add label to `MODE_LABELS`.
3. Add type-label tuple to `_MODE_TYPE_LABELS`.
4. Update `_MODE_PLAYERS` and `_make_runners()` in `dotgame.py`.
5. Update `human_controls()` in `game_mode.py` if needed.
6. If it's a new MCTS vs MCTS variant, assign profiles in `_make_runners()`.

## Adding a new AI backend

1. Subclass `BasePlayer` in a new file under `ai/`.
2. Implement `choose_move(board, connections, player, scores, forbidden=None)`.
3. Implement `record_outcome(winner, intermediate_rewards=None, final_scores=None)`.
   Handle `winner=None` (draw) and `winner=0` (draw/unknown from Trainer).
4. Implement `save()` and `load()` using `experience_path()` from `ai/paths.py`.
5. Add to `_make_runner()` / `_make_runners()` in `dotgame.py` with an `_AVAILABLE` guard if
   the backend requires optional dependencies (see `_PT_AVAILABLE`, `_NM_AVAILABLE` pattern).
6. Add `Mode` variants to `game_mode.py` (enum, `MODE_LABELS`, `_MODE_TYPE_LABELS`,
   `human_controls`).
7. Add to `_MODE_PLAYERS` in `dotgame.py`.
8. Add labels to `_TRAIN_AI_OPTIONS` and `_TRAIN_OPP_OPTIONS` in `draw.py`.
9. Add to `_ARCH_DESCRIPTIONS` in `draw.py` and update `draw_arch_select` return tuple.
10. Update the arch_select unpack in `dotgame.py`.

## Adding a new input channel

1. Add computation to `ai/features.py` (new function or extend `strategic_channels()`).
2. Add to `encode_state()` in both `nn_player.py` and `pytorch_player.py`.
3. Update input sizes: `15 * n * n` → new count in `nn_player.py`; `Conv2d(15, …)` → new count
   in `pytorch_player.py`.
4. Update `meta_channels` in `nn_player.py` save/load.
5. Update `'in_channels'` in the PT arch dict.
6. Update the channel legend in `draw_arch_overlay` in `draw.py`.
7. Update the channel table in `README.md` and `CLAUDE.md`.
8. Delete old weight files — they will be rejected automatically, but deleting avoids confusion.

## Adding a new settings constant

1. Add to `settings.py` with a comment.
2. Add to the configuration table in `README.md`.
3. Reference it by name (never by value) in all consumers — allows central tuning.
4. If it affects MCTS: check whether it should be a global default (profile 0) or a per-profile
   override. Per-profile overrides belong in `MCTS_PROFILES`, not in top-level constants.
