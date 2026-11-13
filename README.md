# Dot Grid

A two-player territory game played on a grid of dots. Connect your dots, enclose territory, and
outsmart your opponent — or train an AI to do it for you.

---

## Requirements

Python 3.10 or later required.

**Quick install:**

```bash
pip install pygame numpy          # required
pip install torch                 # optional — enables PyTorch AI modes
```

**Or install from the project file (includes optional groups):**

```bash
pip install -e .                  # required deps only
pip install -e ".[torch]"         # required + PyTorch AI modes
```

A plain `requirements.txt` is also provided for environments that prefer it.

---

## Running the game

```bash
python dotgame.py
```

---

## How to play

Players take turns placing a dot on any free grid intersection. When two dots of the same colour
are placed adjacent (including diagonally), they are automatically connected.

### Claiming territory

- A **closed loop** of connections encloses territory.
- Each fully enclosed **1×1 square** scores **1 point**.
- Each enclosed **triangle** (half a square, from a diagonal connection) scores **0.5 points**.

### Encirclement

When a player's loop completely encircles the opponent's dots, the trapped dots and any territory
they held are removed. The encircling player claims the entire area. Only the ring itself remains.

### Winning

The game ends when a player reaches **50% of all fields**, or no more legal moves remain (all
intersections are occupied or forbidden). The player with the most territory wins. If both players
have equal territory when the board fills up, the game ends in a **draw**.

---

## Controls

| Input | Action |
|---|---|
| Left-click intersection | Place dot |
| Scroll wheel / `+` / `-` | Zoom in / out |
| Right-click drag | Pan |
| `R` | New game (opens mode picker) |
| `Esc` | Pause menu |
| `←` / `→` | Step through moves after game ends |

---

## Screens and menus

### Main menu (mode picker)

Opens at startup and whenever you press `R` or choose New Game. Select a game mode on the left.
On the right:

- **Player name fields** — click a field and type to rename either player.
- **⚙ SETTINGS** — opens the Settings screen (think time, MCTS profile).
- **◉ AI INFO** — opens the AI Info screen (architecture diagrams).
- **TRAIN AI →** — opens the training configuration screen.

### Settings screen

- **AI think time** — use `◀` / `▶` to step through presets (10–5000 ms). Applies to every AI
  type: longer thinking means more simulated games per move.
- **MCTS personality profile** — Default, Opportunist or Strategist for single-MCTS games.

### AI Info screen

Pick MCTS, Neural Network, or PyTorch CNN to see a scrollable architecture diagram with a plain-
English description. Use the mouse wheel, `↑`/`↓` keys, or the `▲`/`▼` buttons to scroll.

---

## Game modes

| Mode | Player 1 | Player 2 |
|---|---|---|
| Human vs Human | Human | Human |
| Human vs MCTS | Human | MCTS |
| Human vs Neural Net | Human | Neural Net |
| Human vs PyTorch Net | Human | PyTorch Net |
| Human vs Neural MCTS | Human | Neural MCTS |
| MCTS vs MCTS | MCTS (Opportunist) | MCTS (Strategist) |
| MCTS vs Neural Net | MCTS | Neural Net |
| MCTS vs PyTorch Net | MCTS | PyTorch Net |
| MCTS vs Neural MCTS | MCTS | Neural MCTS |
| Neural Net vs Neural Net | Neural Net | Neural Net |
| Neural Net vs PyTorch Net | Neural Net | PyTorch Net |
| Neural Net vs Neural MCTS | Neural Net | Neural MCTS |
| PyTorch Net vs PyTorch Net | PyTorch Net | PyTorch Net |
| PyTorch Net vs Neural MCTS | PyTorch Net | Neural MCTS |
| Neural MCTS vs Neural MCTS | Neural MCTS | Neural MCTS |

In **MCTS vs MCTS**, the two players use deliberately different strategy profiles (see AI players
below) so they play meaningfully different styles rather than mirroring each other.

---

## AI players

All four AIs share one foundation (new in this version):

- **Exact engine** (`ai/engine.py`) — a bitboard implementation of the real rules (flood-fill
  territory, half-point triangles, encirclement captures, forbidden points, both win
  conditions) that is fast enough to simulate thousands of full games per second. The AIs see a
  40-cell ring closing exactly as they see a unit square closing.
- **Ring-race analysis** (`ai/rings.py`) — the decisive plan in Dot Grid is a big ring: enclose
  half the board and you win on the spot. The analysis computes, for each player, how many more
  dots are needed to close a winning ring, and which points those rings pass through.
- **Move scoring** (`ai/strategy.py`) — every legal move is scored by its exact one-move result
  (territory gained, opponent territory destroyed, opponent threats left standing) plus the ring
  race after the move. Taking a point only counts as a block if the opponent's ring really gets
  longer — a ring that simply steps around the dot is not delayed.

### MCTS (Monte Carlo Tree Search)

Runs a PUCT tree search within the configured time budget (`ai/search.py`). The move scores above
decide which moves the tree explores first; every simulation is then a **full game played to the
end with the exact rules**, and the results decide. The most-visited move is played and its
subtree is reused next turn.

**Parallel search:** on a multi-core machine the simulations also run in background worker
processes (`ai/parallel.py`, `SEARCH_WORKERS`), multiplying the number of games played per move.
The workers start on the first AI move and exit with the game.

**MCTS vs MCTS personality profiles:** when two MCTS instances face each other they receive
distinct profiles from `settings.py`:

| Profile | Style | Race weight | Potential weight | Gain weight | Noise frac | Exploration |
|---------|-------|-------------|------------------|-------------|------------|-------------|
| 1 — Opportunist | Values immediate territory swings | 0.8 | 1.0 | 3.5 | 0.25 | 1.3 |
| 2 — Strategist | Values the ring race and ring potential | 1.2 | 2.2 | 1.5 | 0.10 | 1.7 |

**Opening book:** outcomes of finished games are stored per position (shared across the 8 board
symmetries) and seed the search in the first few moves.

- Experience file: `ai_experience_<GRID>x<GRID>.json` (format version 2 — older files are ignored)
- **1000 ms (default)** provides a strong opponent; lower values are faster but weaker.
- No training required.

### Neural Network (pure numpy)

An **actor-critic** network trained with Adam, GAE, entropy regularisation, and two shaping
rewards: a territory-delta reward and an **arc-potential-delta reward** that gives credit for ring-
building progress on every step — not just when a ring completes. This prevents the network from
over-learning small immediate captures at the expense of strategic large-ring play.

- **Input:** 15 × GRID² flat vector — see channel table below
- **Trunk:** hidden layers auto-scaled by board size, ReLU activations
- **Policy head:** GRID² logits → masked softmax
- **Value head:** scalar estimate → tanh output in [-1, 1]
- **Guidance:** the exact move scores are added to the logits (`AI_GUIDE_TEMP`), at play time and
  at training time alike — an untrained network already plays close to the best heuristic move,
  and training refines it from there
- **Training:** GAE advantages (λ=0.95), entropy bonus (0.01), territory-delta shaping
  (`REWARD_TERRITORY = 0.05`), arc-potential-delta shaping (`REWARD_ARC_SHAPING = 0.10`),
  margin-based terminal value `(own_score − opp_score) / total_fields`
- Weights file: `nn_weights_<GRID>x<GRID>.npz`

### PyTorch Net (requires `pip install torch`)

A **residual CNN actor-critic** trained with **PPO + GAE** and the same dual shaping rewards as
the Neural Network.

- **Input:** (15, GRID, GRID) — same 15 channels in spatial layout
- **Stem:** Conv2d(15→64, 3×3) → GroupNorm → ReLU
- **Tower:** N × ResBlock(64ch) — Conv→GN→ReLU→Conv→GN + skip → ReLU
- **Policy head:** Conv(64→2, 1×1) → flatten → Linear(2G²→G²)
- **Value head:** Conv(64→1, 1×1) → flatten → Linear(G²→64) → ReLU → Linear(64→1)
- **Training:** PPO clipped surrogate, GAE (λ=0.95), 4 epochs per game, gradient clipping,
  territory-delta and arc-potential-delta shaping, margin-based terminal value
- **Guidance:** same exact move scores added to the logits as the Neural Network
- **Safety:** weight files containing NaN are rejected on load, and an update that produces
  non-finite weights is rolled back
- Weights file: `pt_weights_<GRID>x<GRID>.pt`

All AI types use **separate experience files per grid size** — changing `GRID` never corrupts
existing weights.

### Neural MCTS (requires `pip install torch`)

The MCTS search plus a residual CNN that learns from it.

- **Search:** identical to the MCTS player (exact engine, ring-race priors, full-game playouts)
- **Network at the root:** the policy head is blended into the root priors, and the value head
  rates the positions after the `NM_VALUE_SEEDS` best candidate moves
- **Trust:** both contributions are scaled by a trust factor that rises from 0 to `NM_MAX_TRUST`
  over the first `NM_TRUST_STEPS` training steps — an untrained network cannot mislead the search
  (the player then plays exactly like MCTS), a trained one increasingly steers it
- **Architecture:** identical residual CNN to PyTorch Net (same presets, separate weight file)
- **Training signal:** two targets per position
  - *Policy target:* the search's visit distribution
  - *Value target:* GAE returns (territory + arc shaping) blended with the final outcome `z` by
    `NM_Z_WEIGHT`
- **Replay buffer:** finished games land in an in-memory buffer (`NM_BUFFER_MAX` samples);
  each game triggers `NM_TRAIN_STEPS` minibatches of `NM_BATCH_SIZE`, every sampled item
  getting a random board rotation/mirror (`NM_AUGMENT` — 8× sample efficiency)
- **Loss:** `NM_POLICY_COEF · CE(π_search, π_net) + NM_VALUE_COEF · MSE(v, target) − entropy`
- Weights file: `nm_weights_<GRID>x<GRID>.pt`

---

## Input channels (Neural Network, PyTorch Net, and Neural MCTS)

All three neural AI players share the same 15-channel board encoding:

| Ch | Description | Range |
|----|-------------|-------|
| 0 | Own dots | {0,1} |
| 1 | Opponent dots | {0,1} |
| 2 | Forbidden positions | {0,1} |
| 3 | Own connection mask | {0,1} |
| 4 | Opponent connection mask | {0,1} |
| 5 | Own closing score ÷ 4 | [0,1] |
| 6 | Opponent threat score ÷ 4 | [0,1] |
| 7 | Own enclosure potential (flood-fill, complete rings only) | [0,1] |
| 8 | Opponent enclosure potential | [0,1] |
| 9 | Own bridge potential | [0,1] |
| 10 | Opponent bridge potential | [0,1] |
| 11 | Disruption map | [0,1] |
| 12 | Fork map | [0,1] |
| 13 | Game phase (broadcast) | [0,1] |
| 14 | Centrality (constant) | [0,1] |

---

## Architecture presets

NN, PyTorch Net, and Neural MCTS automatically select an architecture matched to the board size:

| Grid size | NN hidden layers | PT/NM res-blocks | PT/NM channels | PT/NM receptive field |
|---|---|---|---|---|
| G ≤ 7 | `[128, 64]` | 2 | 32 | 11×11 |
| G ≤ 12 | `[512, 256]` | 4 | 64 | 19×19 |
| G ≤ 20 | `[768, 384]` | 6 | 64 | 27×27 |
| G > 20 | `[1024, 512]` | 8 | 128 | 35×35 |

Override via `settings.py`:

```python
NN_HIDDEN_OVERRIDE = [256, 128]
PT_OVERRIDE        = {'blocks': 6, 'channels': 128}
NM_OVERRIDE        = {'blocks': 6, 'channels': 128}
```

**Changing GRID or overrides invalidates existing NN/PT/NM weights.** All neural players
automatically reject checkpoints from a different architecture and start fresh. MCTS experience
is unaffected.

---

## Training AI offline

The **TRAIN AI →** button in the main menu opens the training configuration screen, which runs
self-play games at full speed in a background thread.

| Setting | Options | Notes |
|---|---|---|
| AI type | MCTS / Neural Net / PyTorch Net / Neural MCTS | Which AI to train |
| Opponent | Self / MCTS / Neural Net / PyTorch Net / Neural MCTS | Who it plays against |
| Think time (ms) | 10 – 5000 | Per-move budget during training |
| Rounds | 10 – 5000 | Number of games to simulate |

### Recommended training workflow

MCTS needs no training — it is strong out of the box. Training matters for the three network
players.

**1. (Optional) Let MCTS build an opening book.**
`MCTS vs MCTS` at **500 ms, 50–100 rounds** fills the experience file with opening results.

**2. Train Neural Net / PyTorch Net against MCTS.**
Opponent `MCTS`, **100–500 ms, 200+ rounds**. Both networks start from the exact move scores, so
they play sensibly from the first game; training against a strong teacher teaches them where
the heuristic is wrong. Below 150 ms the AIs switch to a cheaper analysis — fine for bulk
training, weaker for play.

**3. Train Neural MCTS by self-play or against MCTS.**
**200–500 ms, 150+ rounds**. Every game adds `NM_TRAIN_STEPS` training steps; after
`NM_TRUST_STEPS` (600 by default, i.e. ~150 games) the network has full influence on the search.

**4. Keep grid size fixed while training.**
Weight and experience files are tied to `GRID`.

**Checking strength:** `python scripts/arena.py mcts ringbot --games 10 --think 500` plays
headless matches between any two AIs or scripted opponents (`ringbot` builds one big ring and
never defends — the plan that used to beat every AI).

---

## Configuration (`settings.py`)

| Setting | Default | Description |
|---|---|---|
| `GRID` | 10 | Intersections per side |
| `WIN_PCT` | 0.50 | Territory fraction needed to win |
| `AI_THINK_PRESETS` | `[10,100,500,…,5000]` | Available think-time values (ms) |
| `AI_THINK_DEFAULT` | 1000 | Think time at startup |
| `AI_MAX_EXPERIENCE` | 20 000 | Max MCTS position entries before pruning |
| `MCTS_NOISE_ALPHA` | 0.3 | Dirichlet noise concentration (default profile) |
| `MCTS_NOISE_FRAC` | 0.10 | Noise fraction of prior (default profile; 0 = off) |
| `MCTS_PROFILES` | see code | Per-instance weight profiles for MCTS vs MCTS |
| `MCTS_BOOK_MAX_DOTS` | 12 | Opening book is consulted while the board has at most this many dots |
| `MCTS_BOOK_MAX_VISITS` | 12 | Max virtual visits a remembered position seeds |
| `RING_DISCOUNT` | 0.92 | Value of a ring decays by this factor per dot still missing |
| `RACE_STEEPNESS` | 1.2 | How sharply a lead in the ring race counts |
| `STRAT_RACE_W` | 1.0 | Move-score weight: ring-race outlook |
| `STRAT_POTENTIAL_W` | 1.5 | Move-score weight: ring potential difference |
| `STRAT_GAIN_W` | 2.0 | Move-score weight: exact territory swing of the move |
| `STRAT_THREAT_W` | 2.0 | Move-score weight: opponent's best remaining one-move swing |
| `SEARCH_C_PUCT` | 1.5 | PUCT exploration constant |
| `SEARCH_PRIOR_TEMP` | 0.25 | Softmax temperature turning move scores into priors |
| `SEARCH_WORKERS` | `None` | Worker processes for parallel search: `None` = automatic (cores − 2, max 8), `0` = off |
| `SEARCH_PARALLEL_MIN_S` | 0.20 | Searches shorter than this stay single-process |
| `SEARCH_FAST_BELOW_S` | 0.15 | Think times below this use the cheap analysis |
| `AI_GUIDE_TEMP` | 0.015 | NN/PT: temperature of the move-score guidance added to the logits |
| `AI_GUIDE_CLIP` | 12.0 | NN/PT: maximum suppression of a move by the guidance |
| `AI_NET_LOGIT_BOUND` | 3.0 | NN/PT: the network's own logits are squashed to ± this before guidance is added |
| `AI_CLOSE_BOOST` | 4.0 | Legacy boost (unused by the players) |
| `AI_BLOCK_BOOST` | 3.0 | Legacy boost (unused by the players) |
| `AI_EXPERIENCE_BASE` | `../ai_experience` | MCTS file base path |
| `NN_EXPERIENCE_BASE` | `../nn_weights` | NN weights base path |
| `NN_HIDDEN_OVERRIDE` | `None` | Force specific NN hidden sizes |
| `NN_LEARNING_RATE` | 0.001 | NN Adam learning rate |
| `NN_DISCOUNT` | 0.99 | Reward discount γ |
| `NN_VALUE_COEF` | 0.5 | Value loss weight |
| `NN_ENTROPY_COEF` | 0.01 | Entropy bonus weight |
| `NN_GAE_LAMBDA` | 0.95 | GAE λ |
| `PT_EXPERIENCE_BASE` | `../pt_weights` | PyTorch weights base path |
| `PT_OVERRIDE` | `None` | Force specific PT architecture |
| `PT_LEARNING_RATE` | 3e-4 | Adam learning rate |
| `PT_GAE_LAMBDA` | 0.95 | GAE λ |
| `PT_PPO_CLIP` | 0.2 | PPO clipping ε |
| `PT_PPO_EPOCHS` | 4 | Optimisation passes per game |
| `PT_VALUE_COEF` | 0.5 | Value loss weight |
| `PT_ENTROPY_COEF` | 0.01 | Entropy bonus weight |
| `PT_MAX_GRAD_NORM` | 0.5 | Gradient clip norm |
| `REWARD_TERRITORY` | 0.05 | Per-move territory-delta shaping reward scale |
| `REWARD_ARC_SHAPING` | 0.10 | Per-move arc-potential-delta shaping reward scale |
| `NM_EXPERIENCE_BASE` | `../nm_weights` | Neural MCTS weights base path |
| `NM_OVERRIDE` | `None` | Force specific NM architecture (same format as PT_OVERRIDE) |
| `NM_LEARNING_RATE` | 1e-3 | Adam learning rate |
| `NM_DISCOUNT` | 0.99 | Reward discount γ |
| `NM_GAE_LAMBDA` | 0.95 | GAE λ |
| `NM_VALUE_COEF` | 1.0 | MSE value loss weight |
| `NM_POLICY_COEF` | 1.0 | Cross-entropy policy loss weight |
| `NM_ENTROPY_COEF` | 0.01 | Entropy bonus weight |
| `NM_C_PUCT` | 1.5 | PUCT exploration constant |
| `NM_NOISE_ALPHA` | 0.3 | Dirichlet noise concentration at root |
| `NM_NOISE_FRAC` | 0.15 | Fraction of root prior that comes from noise |
| `NM_MAX_TRUST` | 0.5 | Max weight of the policy head in the root priors |
| `NM_TRUST_STEPS` | 600 | Training steps until the network gets full trust |
| `NM_VALUE_SEEDS` | 8 | Top candidate moves rated by the value head |
| `NM_VALUE_VISITS` | 6 | Virtual visits per rated move at full trust |
| `NM_BUFFER_MAX` | 8000 | Replay-buffer size (training samples kept in memory) |
| `NM_BATCH_SIZE` | 256 | Minibatch size per training step |
| `NM_TRAIN_STEPS` | 4 | Minibatches sampled per finished game |
| `NM_Z_WEIGHT` | 0.5 | Final-outcome blend in NM value targets |
| `NM_AUGMENT` | `True` | Random D4 board transform per sampled training item |
| `PLAYER_NAMES` | `{1:"Player I",…}` | Default display names |

---

## Project structure

```
dotgame.py          Entry point and main event loop
settings.py         All configuration constants (including MCTS_PROFILES)
state.py            Board, connections, scores, move logic, replay snapshots
territory.py        Territory detection (SCALE=6 raster flood fill, encirclement)
viewport.py         Camera: zoom, pan, grid↔screen coordinate conversion
draw.py             All rendering (board, panel, mode picker, settings, AI info overlays)
game_mode.py        Mode enum and player name resolution

ai/
  base_player.py        Abstract BasePlayer interface
  engine.py             Exact bitboard game engine used by all AI simulation
  rings.py              Ring-race analysis (dots needed for a winning ring)
  strategy.py           Move scoring: exact tactics + ring race; guidance for the networks
  search.py             PUCT tree search with full-game playouts
  parallel.py           Worker processes that run search simulations on other cores
  features.py           Board-encoding channels for the neural networks
  mcts.py               MCTS player (search + profiles + opening book)
  nn_player.py          Actor-critic policy network (pure numpy, 15-channel input)
  pytorch_player.py     Residual CNN + PPO + GAE (requires torch, 15-channel input)
  neural_mcts_player.py Search steered by a residual CNN that learns from it (requires torch)
  runner.py             Background thread wrapper — keeps pygame loop responsive
  trainer.py            Offline self-play trainer (territory + arc shaping rewards)
  paths.py              Grid-stamped experience file path helper
```
