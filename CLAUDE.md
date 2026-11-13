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
| `ai/engine.py` | `BitGame` — exact bitboard game engine (territory, captures, forbidden, win checks) used by all search |
| `ai/rings.py` | Ring-race analysis: per-centre DP giving "max area enclosable with k more dots" and per-point ring costs |
| `ai/strategy.py` | `analyse_position`, `move_scores` (exact tactics + ring race), `guidance_offsets` for the policy nets |
| `ai/search.py` | `Searcher` — PUCT tree + full-game playouts on `BitGame`; tree reuse; prior/seed hooks |
| `ai/parallel.py` | Worker subprocesses for root-parallel search (pickle over stdin/stdout, self-disabling on failure) |
| `ai/features.py` | Board encoding channels for the neural nets (and legacy heuristic signals) |
| `ai/mcts.py` | `MCTSPlayer` — thin wrapper: `Searcher` + profiles + opening-book experience |
| `ai/nn_player.py` | Actor-critic NN, pure numpy, 15-channel input, guided by `guidance_offsets` |
| `ai/pytorch_player.py` | Residual CNN + PPO + GAE, 15-channel spatial input, guided by `guidance_offsets` |
| `ai/neural_mcts_player.py` | `Searcher` + CNN: net steers root priors/values by a training-based trust; trains on visit distributions |
| `ai/runner.py` | Background thread wrapper, `observe_opponent_move` forwarding |
| `ai/trainer.py` | Offline self-play trainer (territory + arc shaping rewards), `SimGame` |
| `ai/replay_trainer.py` | Replays recorded human games through the players' learning pipelines |
| `ai/game_recorder.py` | Persists completed human games to `replay_buffer_GxG.json` (capped at `S.REPLAY_BUFFER_MAX`) |
| `ai/symmetry.py` | D4 board transforms (canonical experience keys, NM training augmentation) |
| `ai/paths.py` | Grid-stamped experience file path helper |

Dependency direction inside `ai/`: `engine` ← `rings` ← `strategy` ← `search` ← players.
None of these four import pygame, `state` or `territory`.

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

## Exact engine (`ai/engine.py`)

`BitGame` is the game the AIs simulate with.  It implements the REAL rules — the same results as
`territory.py` / `SimGame` on every move — at a few microseconds per move.

### Representation

- Points and cells share one index space: `idx = gy * W + gx`, `W = GRID + 1` (one padding column
  so horizontal shifts never wrap).  A cell is indexed by its top-left point.
- State = Python ints used as bitboards: `d1`, `d2` (dots), `forb` (forbidden points), plus
  `s1`, `s2`, `player`, `done`, `winner`, `dirty`.  `copy()` is ten attribute copies.
- NOTE the two index conventions: engine **bit index** `gy*W+gx` vs the neural nets' **flat
  index** `gy*n+gx`.  Convert explicitly (`g.xy[idx]` gives `(gx, gy)`).

### Territory model — the quadrant graph

`territory.py` samples four pixels per cell (TOP/RIGHT/BOTTOM/LEFT quadrants).  For one player's
walls: T–R and L–B connect unless the `/` diagonal is walled; T–L and R–B unless the `\` diagonal
is walled; T of a cell connects to B of the cell above unless the shared edge is walled; border
cells connect to the outside.  A segment is walled iff both endpoints hold that player's dots.
`_flood()` floods this graph from outside with bit-parallel shifts; unreached quadrants are
enclosed.  A non-own point is enclosed ⇔ the T quadrant of the cell below-right of it is
enclosed (border points never are) — that gives captures and forbidden points.

### Rules mirrored exactly (do not "simplify")

- Encirclement resolves **one side per recompute, player 1 first**; leftover trapped player-1
  dots set `dirty`, forcing a recompute on the next move (as the real engine recomputes every move).
- Overlap ownership goes to player 1 unless player-1 dots sit inside player 2's enclosure.
- Score = 0.25 per owned quadrant, **except** cells crossed by both a `/` and a `\` connection
  that are not uniformly owned: there each owned quadrant counts 0.5 (the real engine scores
  four separate triangles in such a cell).
- Game end: threshold reached, or no legal point left (winner by score, `None` on a tie).

### Skipping the flood fill

`play()` looks up the mover's dots in the 3×3 neighbourhood in `geo().LOCAL` (512 entries):
`-1` = neighbours form 2+ locally disconnected groups → a big ring may close → full
`_recompute()`; `k ≥ 0` = only `k` new half-point triangles next to the dot → patch the score.
Any change to scoring rules must keep this table and `_recompute()` consistent.

### The equivalence test is the contract

`tests/test_engine.py::TestMatchesReferenceEngine` plays random games on several grid sizes and
compares board, scores, forbidden set, connections and game-over state with `SimGame` after
**every** move.  If you touch `territory.py`, `SimGame` or `engine.py`, that test must stay green.

---

## Ring-race analysis (`ai/rings.py`)

The winning plan in this game is a big ring (a closed chain enclosing ≥ 50% wins instantly and
captures everything inside).  `rings.analyse()` measures, for one player and one centre:

    profile[k] = largest NEW area enclosable around the centre with ≤ k more dots

- Rings are **star-shaped around a centre**: sorting points by angle makes every such ring a path
  in a DAG (king steps with positive cross product).  Own dots cost 0, empty legal points 1,
  opponent dots and forbidden points are impassable.  Rings may cross an opponent's diagonal link.
- Area is the line integral ∮x dy accumulated per edge from a band-area table (`_band_areas`);
  quadrants the player already owns are excluded, so areas are **new score** and
  `moves_to(threshold − score)` is the true distance to a win.
- Forward DP → `profile`; with `backward=True` also `G`, and `through_table()` gives the best
  ring **through each point** (used for "which moves advance my ring / hurt theirs").
- `centres(n)`: board centre, the four half-board centres and the four quadrant centres.  Offsets of 0.02/0.03 keep centres
  off grid lines (no exactly-radial steps).  The cost horizon is `k_max(n) = 2.6·n` dots;
  `NEVER` (999) means no ring within the horizon.
- Cost: ~0.4 ms forward, ~1.4 ms with backward + through-table (10×10, one centre).  It is a
  Python loop over ~100 nodes of small numpy ops — do not call it per playout step.

---

## Strategy (`ai/strategy.py`)

- `analyse_position(bg, detail=True, n_centres=None)` → `PositionAnalysis` with, per player:
  `W` (dots to a winning ring), `U` (discounted ring potential, `RING_DISCOUNT^k`), `tc`/`tu`
  (per-point cost/potential of the best ring through the point), plus per-centre data.
- `tactical_gains(bg)` → exact one-move swings for both sides (own score gained + opponent score
  destroyed; `math.inf` = immediately winning move).
- `move_scores(bg, pa, w, exact_blocks=True)` → `{idx: score}` for the side to move:

  ```python
  score = w.race      * race_value(W_me', W_opp', me_to_move=False)   # in (-1, 1)
        + w.potential * (U_me' - U_opp') / cells
        + w.gain      * own_gain / cells
        - w.threat    * opponent's best remaining one-move swing / cells
        + tiny tie-breaks (own neighbours, centrality)
  ```

  `W_me'` comes from the through-point maps (exact, free).  `W_opp'`/`U_opp'` after taking a
  point are **re-analysed with the point blocked** (`_blocked_outlook`, lazy per centre) — this
  is what makes the AI ignore futile blocks that a ring simply steps around.  The threat term
  uses the opponent's best swing *other than* the candidate point, so blocking one of two
  equal threats earns nothing.
- `race_value`: the side to move wins ties; steepness grows as either ring nears closure.
- `fast_mode(think_s, n)`: below `SEARCH_FAST_BELOW_S · (n/10)³` the players use one centre and
  `exact_blocks=False`.
- `guidance_offsets(board, player, scores, forbidden, fast=None)` → float32[G²] log-offsets
  `(score − best) / AI_GUIDE_TEMP` clipped to `[-AI_GUIDE_CLIP, 0]`, 0 at illegal points, in
  **flat** index order.  Used by NN/PT in place of the old `boost_log_offsets`.

Weights live in `settings.STRAT_*`; `strategy.Weights` carries per-profile overrides.

---

## Search (`ai/search.py`)

`Searcher.search(bg, think_s, prior_mix=None, mix_weight=0.5, seed_visits=None, seed_fn=None)`
returns a `SearchResult` (`move`, real `visits` per root move, `value`, root `scores`, `sims`).

1. **Root analysis**: `analyse_position` + `move_scores`.  A `+inf` score (immediate win) is
   returned at once, without simulations.
2. **Root children** (`_set_root_children`): prior = softmax(scores / `SEARCH_PRIOR_TEMP`),
   optionally mixed with a network policy (`prior_mix`) and Dirichlet noise.  On a reused
   subtree the priors are refreshed and missing moves added; visit counts carry over.
3. **Seeds**: `seed_visits` / `seed_fn` add virtual visits with a value in true outcome scale
   (opening book, value network).  `node.virtual` tracks how many of `node.N` are virtual so
   the reported visit counts are real simulations only.
4. **Selection**: PUCT `Q + c·P·√N_parent/(1+N)`, first-play urgency slightly below the parent's
   mean.  `node.V` is always summed from the perspective of `node.mover`.
5. **Expansion**: priors from the root's score maps (one map per player; the opponent's is
   computed at the root with `exact_blocks=False`).  Stale deeper in the tree, but cheap.
6. **Playout** (`_playout`): a full game on `BitGame` with uniformly random moves (one shuffled
   pass over the points).  Result (`_result`): 0.9/0.1 win/loss ± 0.1·margin, 0.5 for a draw.
7. **Decision**: most-visited root child.  Its subtree and the position after the move are
   saved; `_reuse()` promotes the matching grandchild next turn only if replaying that reply
   reproduces the new position exactly (captures included).

`SearchConfig` holds the knobs (`c_puct`, `prior_temp`, `noise_alpha`, `noise_frac`, `weights`);
unknown keyword → `TypeError`.

### Parallel search (`ai/parallel.py`)

Strength grows steeply with simulations (1200 ms beat 300 ms 20–0), and threads can't run them
in parallel, so `search()` also sends the root (position, priors, score maps, time left) to
worker **subprocesses** (`python -m ai.parallel`).  Each runs `Searcher._simulate` on its own
tree and returns `(sims, {move: (N, V)})` for the root moves; the Searcher adds these to its own
root statistics before choosing ("root parallelisation").  The saved subtree for reuse is the
main process's own.

- `S.SEARCH_WORKERS`: `None` = automatic (cores − 2, max 8), `0` = off.  Searches with less than
  `SEARCH_PARALLEL_MIN_S` left stay single-process (training at short think times never spawns
  workers).
- Plain `subprocess` + length-prefixed pickle — deliberately NOT `multiprocessing`, whose spawn
  start method would re-import the main module (pygame, torch) in every worker.
- Workers exit when their stdin closes (parent gone) and are killed at exit.  Any failure
  (crash, pipe error, no answer within 1 s) closes the pool for good and the search carries on
  alone — parallelism must never be able to break a game.
- One search at a time (`pool.lock`); a second concurrent search just runs single-process.
- Workers set `S.GRID` / `S.WIN_PCT` from the job; any other setting a worker's simulation
  depends on must be shipped in the job dict too.
- `tests/conftest.py` sets `SEARCH_WORKERS = 0`; only `tests/test_parallel.py` enables workers.

**Tried and rejected (measured in `scripts/arena.py`-style matches, 20–30 games each at 400 ms):**
seeding the heuristic score as virtual visits (18–6 against), biasing playouts toward ring
points (12–7 against), exact re-analysis of busy tree nodes (13–5 against), a sharper prior
temperature of 0.08 (12–6 against).  The heuristic scores are relative, not calibrated win
probabilities — they belong in the priors, not in Q.  Re-test before reintroducing any of these.

**Pitfall — `+inf` scores:** the opponent's score map can contain `inf` (a winning reply
exists).  `_priors()` caps it; never feed raw scores to `math.exp`.

---

## Feature module (`ai/features.py`)

Board-encoding channels for the neural networks (`encode_state` in `nn_player.py` /
`pytorch_player.py`) and the trainer's arc shaping reward.  **The search no longer uses these
heuristics** — MCTS priors, playouts and tactics come from `engine`/`rings`/`strategy`.
`apply_boost` / `boost_log_offsets` are legacy (kept with their tests); the players use
`strategy.guidance_offsets` instead.

### Signal overview

| Function | Returns | Notes |
|---|---|---|
| `opportunity_masks(board, connections, player)` | (own, opp) float32[G²] | Mirrors territory.py scoring: full square = 1.0, triangle = 0.5 |
| `apply_boost(probs, own_scores, opp_scores, legal)` | new normalised array | Legacy multiplier boost — unused by the players |
| `boost_log_offsets(own_scores, opp_scores, legal)` | float32[G²] | Legacy log-domain form of `apply_boost`; superseded by `strategy.guidance_offsets` |
| `adaptive_dirichlet_alpha(base_alpha, n_legal)` | float | Root-noise alpha ∝ 1/branching, floored at 0.1; used by `Searcher` |
| `_find_components(board, player)` | list[set] | Connected components via BFS |
| `_component_potential(comp, n)` | float | Flood-fill interior fraction — 0 for any open ring |
| `enclosure_potential(board, connections, player)` | (own, opp) float32[G²] | Per-position flood-fill potential (only fires on complete rings) |
| `enclosure_scalars(board, player)` | (own_max, opp_max) | Scalar version of above |
| `_arc_potential_comp(comp, n)` | float (territory-cell units) | Bounding-box × completion; fires for partial rings |
| `arc_potential_scalars(board, player)` | (own_max, opp_max) in [0,1] | Normalised arc potential scalars for the trainer's shaping reward |
| `arc_potential_map(board, player)` | (own, opp) float32[G²] in territory-cell units | NN input channels 7–8 (normalised) |
| `bridge_potential(board, connections, player)` | (own, opp) float32[G²] | Extra potential from bridging two arcs; uses `_arc_potential_comp` |
| `disruption_map(board, connections, player)` | float32[G²] | Value of placing inside opponent-claimed interior |
| `fork_map(board, player)` | float32[G²] | Positions creating 2+ simultaneous threats |
| `close_setup_map(board, player)` | float32[G²] | Positions that create a 3-own-corner unit square |
| `get_centrality()` | float32[G²], cached | 1.0 at centre, 0.0 at corner |
| `strategic_channels(board, connections, player, total_dots)` | 6-tuple | Convenience: all strategic channels at once |

### `_arc_potential_comp` units

Returns **territory-cell units** (not normalised to [0,1]). A ring enclosing 9 cells at 90%
completion returns ~8.1. `arc_potential_scalars` divides by `(n-1)²`; the NN encoders normalise
by `(n-1)²` before writing channels 7–8.

### Why `enclosure_potential` vs `arc_potential_map`

`enclosure_potential` (flood-fill) returns 0 for any ring with a single gap. `arc_potential_map`
(bounding-box × completion) fires continuously from the first arc placed and feeds NN channels
7–8; the switch is guarded by an encoding-version stamp in the weight files (`meta_enc_version` /
`enc_version` == 2). `enclosure_potential` / `_component_potential` remain in use by
`disruption_map` and `enclosure_scalars`.

`arc_potential_scalars` is called in `Trainer._run()` for the arc-potential-delta shaping reward.
These are coarse bounding-box heuristics; for anything decision-relevant use `ai/rings.py`.

---

## MCTS player (`ai/mcts.py`)

`MCTSPlayer` is a thin wrapper around `Searcher` (see "Search" above): it converts the dict
board to a `BitGame`, runs the search for `S.AI_THINK_MS`, and maintains the opening-book
experience.  It holds no search logic of its own.

### Per-instance profiles

```python
MCTSPlayer(player_id=1, profile=1)   # Opportunist
MCTSPlayer(player_id=2, profile=2)   # Strategist
```

Profiles are dicts in `settings.MCTS_PROFILES` with optional keys
`race_w, potential_w, gain_w, threat_w` (→ `strategy.Weights`), `exploration` (→ `c_puct`),
`noise_alpha, noise_frac`.  Missing keys fall back to `STRAT_*`, `SEARCH_C_PUCT` and
`MCTS_NOISE_*`.  Profile 0 = all defaults.  `Mode.MCTS_VS_MCTS` always gets profiles 1 and 2 in
`dotgame._make_runners()`; every other MCTS instance uses the profile picked on the Settings
screen (`game_mcts_profile`, default 0).  Labels live in `S.MCTS_PROFILE_LABELS`.

### `record_outcome` and experience (opening book)

The experience outcome is a [0,1] **margin value** (`own_score / total_score`) when
`final_scores` is given, else 1 / 0 / 0.5.  `intermediate_rewards` is accepted and ignored.

Every stored value is from the perspective of **the player who moved into the position**:
positions after our own moves (`_game_moves`) get `outcome`, positions after observed opponent
moves (`_obs_moves`) get `1 − outcome`.  This holds in shared and non-shared mode alike.

Keys are **canonical under D4 symmetry** (`ai/symmetry.canonical_board_key`).  The file format
is `{"version": 2, "positions": {...}}`; files without that version stamp (the old format mixed
perspectives) are ignored on load and overwritten on the next save.

Use: `_book(bg)` looks up the position after each legal root move — only while the board has at
most `MCTS_BOOK_MAX_DOTS` dots — and seeds entries with ≥ 2 visits as up to
`MCTS_BOOK_MAX_VISITS` virtual visits at the remembered mean (`seed_visits` of `Searcher`).

In same-type self-play the two players share ONE experience dict via `share_learning_with()`.

### `observe_opponent_move`

`BasePlayer` defines a no-op `observe_opponent_move(board, connections, player_who_moved, move,
forbidden)` hook; **all four players override it** (symmetric learning). The contract: board /
connections / forbidden are snapshots taken **before** the move. Implementations:

- `MCTSPlayer` — records the resulting board key for the opening book.
- `NNPlayer` / `PyTorchPlayer` — forward pass from the mover's perspective, store
  (state, action, log_prob, value) in `_obs_traj`; trained at `record_outcome` with the
  **inverted** terminal value (PT uses `PT_PPO_EPOCHS // 2` epochs for observed moves).
- `NeuralMCTSPlayer` — one-hot policy target on the played move, stored in `_obs_episode`.

All three network players **drop an observed move that is not legal** in the given position
(its log-prob would be −inf and poison training — see pitfalls).

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

### Guidance-as-logits and masked gradients

Trajectory tuples are **6-long**: `(enc, action, acts, val, boost, legal_mask)`. `boost` is the
`strategy.guidance_offsets()` array — exact move scores as additive log-offsets — folded into
the logits BEFORE the masked softmax, both when sampling (`_ACNetwork.policy(x, legal_mask,
boost)`) and when training (`update(..., legal_mask=, boost_log=)`), so the REINFORCE gradient
matches the true behaviour policy.  An untrained net therefore plays close to the best
heuristic move; training shapes the logits relative to it.  The raw logits are first squashed
with `_bound()` = `C·tanh(x/C)`, `C = AI_NET_LOGIT_BOUND`, so the net can re-rank reasonable
moves but never outvote the guidance on a blunder; `update()` applies the matching chain-rule
factor `1 − tanh²`.  `NNPlayer.load()` discards a **collapsed** network (raw logit spread on the
empty board > 10·C — it would sit in the flat part of the tanh and never recover).  `update()` computes policy/entropy
gradients on the masked+offset softmax, zeroes `d_pol` at illegal positions, and backprops the
trunk through **pre-update weight copies** (`w_pre = layer.w.copy()` before each Adam step —
using post-update weights was a bug).

### Weight compatibility

Saved metadata keys: `meta_grid`, `meta_hidden`, `meta_channels` (must equal 15),
`meta_enc_version` (must equal 2 — arc-potential channels 7–8). Mismatches silently rejected
on load, and so is any file containing non-finite values.

---

## PyTorch player (`ai/pytorch_player.py`)

### Architecture

```
Input  (15, G, G)
Stem   Conv2d(15→64, 3×3) → GN → ReLU
Tower  N × ResBlock(64ch):  Conv→GN→ReLU→Conv→GN  +  skip  → ReLU
Policy Conv(64→2, 1×1) → flatten → Linear(2G²→G²)
Value  Conv(64→1, 1×1) → flatten → Linear(G²→64) → ReLU → Linear(64→1)
```

Normalisation is **GroupNorm** (8 groups; 1 group if channels aren't divisible by 8), not
BatchNorm: GN has no running stats and behaves identically in train/eval mode, so batch-size-1
inference is safe. (BatchNorm silently normalised with single-sample batch statistics AND
mutated its running stats on every inference forward.) The arch dict carries `'norm': 'gn'`
alongside `'in_channels': 15`, and the checkpoint carries `'enc_version': 2` — old BatchNorm
or enclosure-encoding weights are silently rejected on load.

### PPO training loop

1. GAE advantages: `δ_t = r_t + γ·V(s_{t+1}) − V(s_t)`, `A_t = Σ(γλ)^k·δ_{t+k}`
2. Terminal value = margin = `(own_score − opp_score) / (GRID-1)²`
3. `intermediate_rewards` now includes arc-potential-delta shaping from `Trainer`
4. For `PT_PPO_EPOCHS` passes, minibatches of `PT_PPO_MINIBATCH`:
   - `ratio = exp(log_π − log_π_old)`
   - `L_clip = −min(ratio·A, clip(ratio, 1−ε, 1+ε)·A)`
   - `L_total = L_clip + PT_VALUE_COEF·L_value − PT_ENTROPY_COEF·L_ent`
5. Gradient clip to `PT_MAX_GRAD_NORM`, Adam step.

### Guidance-as-logits (PPO correctness)

Trajectory tuples are **6-long**: `(enc, mask, action, log_prob_old, value_old, boost_log)`.
`boost_log` is the `strategy.guidance_offsets()` array, added to the **bounded** logits
(`_bound()` = `C·tanh(x/C)`, `C = AI_NET_LOGIT_BOUND` — same reason as in the NN player) at BOTH
action time and training time, with the stored offsets replayed in the PPO loops. Never apply guidance
post-hoc to the probabilities: the action would be sampled from a distribution the PPO ratio
doesn't see, biasing every update. Entropy is intentionally measured on the guided policy (the
distribution actions are actually sampled from). The same scheme applies to
`observe_opponent_move` / `_obs_traj` (offsets computed from the mover's perspective, with
`fast=True` because observers run on the UI thread).

### NaN safety net

`record_outcome` snapshots the state dict before training; if any parameter is non-finite
afterwards it restores the snapshot and clears the Adam state instead of saving a dead network.
`load()` rejects checkpoints with non-finite weights (a poisoned file used to make the player
silently fall back to uniformly random moves).  `NeuralMCTSPlayer` has the same two guards.

### Training masks: `-1e9`, never `float('-inf')`

The two PPO loops mask illegal logits with `logits.masked_fill(~mask, -1e9)`. Using
`float('-inf')` makes the entropy term compute `prob × log_prob = 0 × (−inf) = NaN` in the
**forward** pass, which NaNs every parameter after one optimizer step — this was the root
cause of the historical weight-file poisoning. `-1e9` produces a bitwise-identical distribution
at legal positions (masked probs underflow to exactly 0) while keeping all tensors finite.
Inference paths (`choose_move`, `observe_opponent_move`) still use `-inf` — safe because no
entropy is computed there. `tests/test_torch_net.py::TestPTTrainingStaysFinite` pins this.

---

## `AIRunner` (`ai/runner.py`)

- `on_game_end(winner, final_scores=None)` — forwards both to `record_outcome`.
- `winner` may be `None` (draw); pass through without modification — player implementations
  handle `None` as a non-win outcome.
- `observe_opponent_move(...)` — forwards to the player synchronously (cheap: dict hash for
  MCTS, one forward pass for NN types). Called from the main loop with **pre-move** snapshots.
- Snapshots `forbidden_positions` at `start_thinking()` — never reads live state from the thread.
- `except Exception` in thread catches all failures; logs to stderr, returns random legal fallback.
- `finally: self._thinking = False` always runs.

---

## Shared learning (`share_learning_with`)

When two players of the **same type** self-play, both used to train independent copies and the
second `save()` overwrote the first's file — half the learning was discarded. All four players
now implement `share_learning_with(other)`:

- `MCTSPlayer` — `other.experience = self.experience` (one dict, both write to it).
- `NNPlayer` — shares `_net` (+ `_hidden`).
- `PyTorchPlayer` / `NeuralMCTSPlayer` — share `_net` + `_opt` (+ `_arch`); NM also shares
  `_buffer`.

The method sets `_learning_shared = True` on **both** sides, and every concrete
`observe_opponent_move` starts with `if self._learning_shared: return` — with one shared
net/dict, the partner's own-move training already covers those positions; recording both would
double-count. Wired in `dotgame._make_runners` (same-type, non-human pairs) and in the START
train-screen branch (when `resolved_opp == ai_label`). The REPLAY branch deliberately does NOT
link the pair — replay learning flows entirely through observe_opponent_move, which the sharing
gate would silence.

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

shaping = clip(terr_delta * S.REWARD_TERRITORY
               + arc_delta * S.REWARD_ARC_SHAPING, -1.0, 1.0)
```

The arc shaping reward fixes the NN/PT "small-grab bias": without it the network sees a nonzero
reward only when a ring closes, so it learns to prefer small immediate captures. With it, every
step extending a ring arc is rewarded proportionally to the increase in ring-building potential.

`arc_potential_scalars` is called twice per move (before and after) — this is O(G²·components)
but acceptable at training speed (no real-time constraint).

### Symmetric observation

After each move, `Trainer._run()` calls the **other** player's `observe_opponent_move` with the
**pre-move** snapshots (`board_snap`, `conn_snap`, `forbidden_snap`) and the move — same
contract as the live game loop.  If a player returns an illegal move, a random legal one is
played and `move` is reassigned, so shaping and observation refer to the move actually played.

## Human-game replay (`ai/game_recorder.py`, `ai/replay_trainer.py`)

When any human plays, `dotgame.py` accumulates `current_game_moves` (`[player, gx, gy]` per
move) and calls `record_game()` at game end. Games are stored in `replay_buffer_GxG.json`,
capped at `S.REPLAY_BUFFER_MAX` (oldest evicted), grid-stamped like weight files.

`ReplayTrainer` re-simulates each saved game with `SimGame` and feeds **every legal** move
(entries that `SimGame` rejects are skipped — old recordings could contain rejected clicks) to both
players via `observe_opponent_move(board_before, …)` — the player's own moves become imitation
targets, the opponent's moves are the usual symmetric signal — then calls
`record_outcome(winner, final_scores=…)` to trigger the gradient update. Same
`start/cancel/is_running/on_progress/on_done` interface as `Trainer`, so the training-progress
UI works unchanged. Launched from the train-config screen's REPLAY button.

---

## Draw module (`draw.py`)

### Screen routing overview

```
picking_mode    → draw_mode_picker       → returns (mode_rects, name_rects, train_r, settings_r, arch_r)
settings_screen → draw_settings_screen   → returns 6-tuple (back_r, prev_think_r, next_think_r,
                                                            None, prev_prof_r, next_prof_r)
arch_select     → draw_arch_select       → returns (back_r, mcts_r, nn_r, pt_r, nm_r)
arch_overlay    → draw_arch_overlay      → returns (close_r, up_r, down_r, content_h)
training_mode   → draw_train_config      → returns 16-tuple (see below)
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

Returns a **16-tuple**:

```python
(back_r, start_r, replay_r,
 prev_ai, next_ai, prev_opp, next_opp,
 prev_rnd, next_rnd, prev_think, next_think,
 deep_terr_r,                    # always None (former Deep Terr toggle)
 prev_ai_prof, next_ai_prof,     # None when AI is not MCTS
 prev_opp_prof, next_opp_prof)   # None when opponent is not MCTS
```

Always unpack as 16 values; slot 12 (`deep_terr_r`) and slot 4 of the settings 6-tuple are
always `None` — the Deep Terr toggle was removed when the search moved to the exact engine, the
slots stay so the tuple shapes are stable.  Never call `.collidepoint` on them.  `replay_r` starts a `ReplayTrainer` over the recorded human games
(rounds value = repeat count); the profile pickers select `MCTS_PROFILES` entries per player and
only render when that player resolves to MCTS.

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

### Train config 16-tuple unpack

```python
(back_r, start_r, replay_r,
 pra, nxa, pro, nxo,
 prr, nxr, prt, nxt,
 train_deep_r,
 pr_aip, nx_aip,
 pr_opp_p, nx_opp_p) = train_cfg_rects
```

### Settings screen 6-tuple unpack

```python
back_r, prev_t, next_t, deep_r, prev_prof, next_prof = settings_rects
```

`prev_prof` / `next_prof` step `game_mcts_profile` (0–2), which is passed to `_make_runners()`
for single-MCTS game modes.

### Arch select 5-tuple unpack

```python
back_r, mcts_r, nn_r, pt_r, nm_r = arch_sel_rects
```

### MCTS profile assignment

```python
if mode == Mode.MCTS_VS_MCTS:
    p1_profile, p2_profile = 1, 2          # always Opportunist vs Strategist
else:
    p1_profile = profile if p1_type == 'mcts' else 0   # user-selected game_mcts_profile
    p2_profile = profile if p2_type == 'mcts' else 0
```

`MCTS_VS_MCTS` always gets profiles 1 and 2. Every other MCTS use receives the profile chosen
on the Settings screen (`game_mcts_profile`, default 0 = global defaults).

### `on_game_end` — all three call sites pass `final_scores`

```python
r.on_game_end(state.winner_player, final_scores=dict(state.scores))
```

`state.winner_player` may be `None` (draw). `AIRunner` passes it through unchanged.

### Move application — observe + record pattern

Both move paths (AI pending-move and human click) follow the same sequence: snapshot
`board/connections/forbidden_positions` **before** `state.place()`, then — **only if `place()`
returned True** — append `[player, gx, gy]` to `current_game_moves` and call the *other*
runner's `observe_opponent_move(pre_move_snapshots, mover, move)` (only if that side is not
human).  A rejected click (occupied / forbidden point) must never be recorded or observed: it
used to end up in the replay buffer and in the PT player's trajectory with log-prob −inf, which
NaN'd the weights.
At game end, if any player is human, `record_game(current_game_moves, winner or 0, scores)`
saves the game to the replay buffer; `current_game_moves` is then cleared (also on new game).

### Stop button draw condition

```python
s1, s2 = state.scores[1], state.scores[2]
state.winner_player = (None if s1 == s2 else (1 if s1 > s2 else 2))
```

---

## Neural MCTS player (`ai/neural_mcts_player.py`)

### Design

`Searcher` (the same exact-rules search as `MCTSPlayer`) plus a residual CNN that learns from
it.  Reuses `pytorch_player._build_net` and `encode_state` verbatim.  Separate weight file
(`nm_weights_GxG.pt`) so PyTorch weights are never affected.  There is no NM-specific tree code
any more (`_NMNode`, virtual loss, `_tactical_priors` are gone).

### `choose_move` flow

1. One network call at the root → masked-softmax policy + `root_val` (stored for training).
2. `trust = NM_MAX_TRUST · min(1, steps / NM_TRUST_STEPS)` where `steps` counts optimizer steps
   taken so far (persisted in the checkpoint as `'steps'`; `self._steps` is a one-element list so
   self-play partners share it).  An untrained net has trust 0 and the player IS the MCTS player.
3. `Searcher.search(bg, think, prior_mix=policy, mix_weight=trust, seed_fn=_value_seeds)` —
   with trust 0 both network hooks are skipped.
4. `_value_seeds`: the value head rates the positions after the `NM_VALUE_SEEDS` best heuristic
   moves in ONE batched forward; each becomes `NM_VALUE_VISITS · trust/NM_MAX_TRUST` virtual
   visits at `(1 − v)/2` (v is from the opponent's perspective — they are to move).
5. The search's **real** visit counts (virtual visits excluded) → normalised visit distribution
   → `self._episode` together with the encoding, legal mask and `root_val`.

Index convention: the search works in engine bit indices, the episode/policy arrays in flat
`gy*n+gx` indices — `choose_move` converts via its `flat` map.

### `record_outcome` — replay buffer training

```python
L = NM_POLICY_COEF  · CE(π_search, π_net)   # cross-entropy: replicate visit distribution
  + NM_VALUE_COEF   · MSE(v_pred, targets)  # value: blended GAE/z targets
  - NM_ENTROPY_COEF · H(π_net)              # entropy: keep exploration alive
```

Both episodes — own moves (with shaping rewards) and `_obs_episode` (observed opponent moves:
one-hot policy targets, **inverted** terminal, no shaping) — are converted to per-step value
targets by `_value_targets`: GAE returns blended with the final outcome `z` as
`(1−NM_Z_WEIGHT)·returns + NM_Z_WEIGHT·z`, clipped to [-1, 1].  The `(enc, mask, visit_dist,
target)` tuples land in an in-memory replay `_buffer` (capped at `NM_BUFFER_MAX`, oldest
evicted; cleared on restart; shared between self-play partners).  `_train_from_buffer()` samples
`NM_TRAIN_STEPS` minibatches of `NM_BATCH_SIZE` without replacement (one pass over the whole
buffer while it's still smaller than a batch), takes one Adam step each and increments the step
counter.  A non-finite result rolls back to the pre-training snapshot.

When `NM_AUGMENT` is on, each sampled item gets a random D4 transform — `transform_planes` for
the encoding, `transform_flat` for mask and visit distribution, all with the SAME op, value
target unchanged.  **Do not extend augmentation to PT/NN**: their stored log-probs are
orientation-bound, so transforming the state breaks the PPO/REINFORCE importance ratio.

Entropy is computed over **legal positions only** (`torch.where(masks, log_probs, 0)`).

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
NM_NOISE_FRAC       = 0.15
NM_MAX_TRUST        = 0.5    # max weight of the policy head in the root priors
NM_TRUST_STEPS      = 600    # training steps until full trust
NM_VALUE_SEEDS      = 8      # top candidates rated by the value head
NM_VALUE_VISITS     = 6      # virtual visits per rated move at full trust
NM_BUFFER_MAX       = 8000   # max (state, π, v) samples kept in memory
NM_BATCH_SIZE       = 256    # minibatch size per training step
NM_TRAIN_STEPS      = 4      # minibatches sampled per finished game
NM_Z_WEIGHT         = 0.5    # blend of final outcome z into value target
NM_AUGMENT          = True   # random D4 transform per sampled item
NM_OVERRIDE         = None   # {'blocks': 4, 'channels': 64}
```

`nm_arch()` uses the same `_PT_PRESETS` table as `pt_arch()`.

### Weight compatibility

Saved keys: `grid`, `arch` (includes `in_channels: 15` and `norm: 'gn'`), `steps`. Mismatches
and non-finite weights are silently rejected on load.  A checkpoint without `steps` loads with
trust 0 (weights kept, influence earned by further training).

---

## Common pitfalls

- **The AIs simulate with `ai/engine.BitGame`, not `territory.py`.** Any rule change must be made
  in `territory.py`, `SimGame` (trainer.py) AND `engine.py`; `tests/test_engine.py` compares
  them move by move.
- Engine **bit index** (`gy*W+gx`, `W=GRID+1`) ≠ network **flat index** (`gy*n+gx`). Mixing them
  silently shifts every move by a row offset.
- `move_scores()` can return `math.inf` (immediate win). Cap before `exp`/arithmetic
  (`Searcher._priors` does); `Searcher.search` returns such a move without simulating.
- `rings.analyse()` is ~0.4–1.4 ms — fine per candidate move at the root, far too slow per
  playout step. Playouts must stay table/bit-op only.
- `Searcher` visit counts: `node.N` includes virtual (seeded) visits; subtract `node.virtual`
  for anything reported or trained on (`SearchResult.visits` already does).
- A saved search subtree must not survive into a new game — every player's `record_outcome`
  calls `Searcher.reset()`.
- `draw_arch_select` returns **5 values** — always unpack as `(back_r, mcts_r, nn_r, pt_r, nm_r)`.
- Never mask TRAINING logits with `float('-inf')` in the PT player — the entropy term computes
  `0 × -inf = NaN` in the forward pass and NaNs every parameter. Use `masked_fill(~mask, -1e9)`
  (see the PPO loops). NM guards the same hazard with `safe_lp` zeroing.
- Never record or observe a move that `place()` rejected (see "Move application"): an illegal
  action has log-prob −inf → PPO ratio = inf → NaN weights. The network players also drop
  illegal observed moves themselves, and roll back non-finite updates.
- D4 symmetry augmentation is **NM-only**. PT/NN stored log-probs are orientation-bound;
  transforming their states breaks the PPO/REINFORCE ratio.
- `NM_OVERRIDE` uses the same dict format as `PT_OVERRIDE`: `{'blocks': N, 'channels': C}`.
- Neural MCTS weight files (`nm_weights_GxG.pt`) are separate from PyTorch (`pt_weights_GxG.pt`).
- A freshly created `NeuralMCTSPlayer` plays exactly like `MCTSPlayer` (trust 0); the network
  only gains influence through training steps.
- `_NM_AVAILABLE` mirrors `_PT_AVAILABLE` — both require PyTorch. The import guard in `dotgame.py`
  uses a separate try/except so a future split of NM to a different backend remains easy.
- **Never** let 3×3 dot blocks write to pixel `-1` — blocks flood fill seed.
- **Never** use `elif` chains that swallow the human board-click handler.
- **Always** add new state fields to `_capture()`, `_restore()`, and `reset()` — all three.
- **Always** guard AI scheduling with `not any_overlay` (the full flag union, not just
  `not picking_mode`).
- AI `choose_move` runs in a thread — never write to `state.*` from inside it.
- `observe_opponent_move` runs on the **main thread** — keep it cheap (the policy nets pass
  `fast=True` to `guidance_offsets` there).
- `draw_panel` returns **5 values** — always unpack as `(a, b, c, _, _)`.
- `draw_train_config` returns **16 values**; `draw_settings_screen` returns **6 values**; their
  `deep_terr_r` slots are always `None`.
- `winner_player` can be `None` (draw) — all rendering and record_outcome paths must handle it.
- `observe_opponent_move` must always receive **pre-move** snapshots (board/connections/
  forbidden as they were before the move). Passing the post-move board corrupts training:
  the action position is occupied → masked → `log_prob = -inf` → PPO ratio blows up.
- `S.AI_THINK_MS` is the live thinking time (set by the UI; default `AI_THINK_DEFAULT`). Below
  `SEARCH_FAST_BELOW_S` the players switch to the cheap analysis — training at the 10 ms preset
  runs mostly on move scores, not on playouts.
- `_arc_potential_comp` returns values in **territory-cell units**, not [0,1]; the NN channels
  normalise by `(n-1)²` first.
- `arc_potential_scalars` is called in the trainer loop on every move — intentional and
  acceptable at training speed.

---

## Project files

| File | Purpose |
|---|---|
| `pyproject.toml` | Project metadata, entry point, optional `[torch]` dependency group |
| `requirements.txt` | Plain pip requirements (pygame + numpy; torch commented out) |
| `CLAUDE.md` | This file — AI/contributor context |
| `README.md` | User-facing docs |
| `HOW_THE_AI_WORKS.md` | Plain-English AI guide |
| `scripts/validate_training.py` | Headless training smoke test — asserts finite weights for all four backends |
| `scripts/arena.py` | Headless matches between AI players and scripted opponents (strength regression check) |

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
2. Implement `choose_move(board, connections, player, scores, forbidden=None)`. For anything
   that looks ahead, build a `BitGame.from_board(...)` and use `Searcher` / `strategy` rather
   than writing another simulator.
3. Implement `record_outcome(winner, intermediate_rewards=None, final_scores=None)`.
   Handle `winner=None` (draw) and `winner=0` (draw/unknown from Trainer).
   Optionally override `observe_opponent_move(board, connections, player_who_moved, move,
   forbidden)` for symmetric learning — arguments are pre-move snapshots; default is a no-op.
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
4. If it affects move scoring or the search: check whether it should be a global default
   (`STRAT_*` / `SEARCH_*`) or a per-profile override. Per-profile overrides belong in
   `MCTS_PROFILES` (and must be read in `MCTSPlayer.__init__`), not in top-level constants.
