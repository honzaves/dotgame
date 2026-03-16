# Dot Grid

A two-player territory game played on a grid of dots. Connect your dots, enclose territory, and
outsmart your opponent — or train an AI to do it for you.

---

## Requirements

```bash
pip install pygame numpy          # required
pip install torch                 # optional — enables PyTorch AI modes
```

Python 3.10 or later required.

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
- **⚙ SETTINGS** — opens the Settings screen (think time, Deep Terr toggle).
- **◉ AI INFO** — opens the AI Info screen (architecture diagrams).
- **TRAIN AI →** — opens the training configuration screen.

### Settings screen

- **MCTS think time** — use `◀` / `▶` to step through presets (10–5000 ms). Applies to all
  MCTS play and MCTS self-play training.
- **DEEP TERR toggle** — switches MCTS rollout scoring between fast lightweight mode
  (~500 rollouts/sec) and full flood-fill territory detection (~5 rollouts/sec, more accurate).
  Only worth enabling at 2000 ms+ think time.

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

### MCTS (Monte Carlo Tree Search)

Runs tree search within the configured time budget. Uses PUCT priors built from eight strategic
signals: immediate close, opponent block, territory setup, arc extension, arc block, arc bridge,
cluster growth, and centrality. Rollouts follow a six-tier biased policy (fork → close → block →
setup → adjacent → random). Leaf positions are evaluated with a four-signal blend that weights
arc potential at 50%.

**Arc potential** uses a bounding-box × completion-ratio formula that grows continuously as a ring
is built, unlike older flood-fill approaches that return zero until ring completion. This gives
MCTS genuine ring-building instincts from the very first arc placed.

**Dirichlet noise** is injected into root priors each turn (adaptive alpha, 25% weight by default)
so games are varied and non-deterministic.

**MCTS vs MCTS personality profiles:** when two MCTS instances face each other they receive
distinct profiles from `settings.py`:

| Profile | Style | Close weight | Arc weight | Noise frac | Exploration |
|---------|-------|-------------|------------|------------|-------------|
| 1 — Opportunist | Grabs immediate territory, plays noisily | 5.5 | 2.5 | 0.35 | 1.20 |
| 2 — Strategist | Builds large rings, explores patiently | 2.5 | 7.0 | 0.18 | 1.65 |

Experience (`board_key → {wins, visits}`) is saved between games to warm-start future searches.

- Experience file: `ai_experience_<GRID>x<GRID>.json`
- **1000 ms (default)** provides a strong opponent; lower values are faster but weaker.
- Deep Terr mode is now in the **Settings screen**, not the side panel.

### Neural Network (pure numpy)

An **actor-critic** network trained with Adam, GAE, entropy regularisation, and two shaping
rewards: a territory-delta reward and an **arc-potential-delta reward** that gives credit for ring-
building progress on every step — not just when a ring completes. This prevents the network from
over-learning small immediate captures at the expense of strategic large-ring play.

- **Input:** 15 × GRID² flat vector — see channel table below
- **Trunk:** hidden layers auto-scaled by board size, ReLU activations
- **Policy head:** GRID² logits → masked softmax
- **Value head:** scalar estimate → tanh output in [-1, 1]
- **Training:** GAE advantages (λ=0.95), entropy bonus (0.01), territory-delta shaping
  (`REWARD_TERRITORY = 0.05`), arc-potential-delta shaping (`REWARD_ARC_SHAPING = 0.10`),
  margin-based terminal value `(own_score − opp_score) / total_fields`
- Weights file: `nn_weights_<GRID>x<GRID>.npz`

### PyTorch Net (requires `pip install torch`)

A **residual CNN actor-critic** trained with **PPO + GAE** and the same dual shaping rewards as
the Neural Network.

- **Input:** (15, GRID, GRID) — same 15 channels in spatial layout
- **Stem:** Conv2d(15→64, 3×3) → BatchNorm → ReLU
- **Tower:** N × ResBlock(64ch) — Conv→BN→ReLU→Conv→BN + skip → ReLU
- **Policy head:** Conv(64→2, 1×1) → flatten → Linear(2G²→G²)
- **Value head:** Conv(64→1, 1×1) → flatten → Linear(G²→64) → ReLU → Linear(64→1)
- **Training:** PPO clipped surrogate, GAE (λ=0.95), 4 epochs per game, gradient clipping,
  territory-delta and arc-potential-delta shaping, margin-based terminal value
- Weights file: `pt_weights_<GRID>x<GRID>.pt`

All AI types use **separate experience files per grid size** — changing `GRID` never corrupts
existing weights.

### Neural MCTS — AlphaZero-style (requires `pip install torch`)

The strongest AI. Combines MCTS tree search with a trained residual CNN, fixing the core weakness
of each: MCTS gets a learned prior instead of hand-crafted signals, and the CNN gets genuine
lookahead instead of reacting to the current board alone.

- **Input:** (15, GRID, GRID) — same 15 channels as PyTorch Net, in spatial layout
- **Architecture:** identical residual CNN to PyTorch Net (same presets, separate weight file)
- **Search:** AlphaZero PUCT — `Q + c_puct · P · √N_parent / (1 + N_child)`
  - Policy head output → PUCT prior for each child (replaces hand-crafted `raw_priors`)
  - Value head output → leaf evaluation (replaces random rollouts entirely)
  - No rollouts at all: one network call per new leaf, then backpropagate the value
- **Training signal:** two targets per game
  - *Policy target:* MCTS visit distribution `π_mcts` — more accurate than the raw network prior
  - *Value target:* GAE returns from territory-delta + arc-potential-delta shaping (same as PT)
- **Loss:** `NM_POLICY_COEF · CE(π_mcts, π_net) + NM_VALUE_COEF · MSE(v, return) − entropy`
- No PPO needed — visit distributions are a stable supervised signal, not a self-bootstrapped estimate
- **Self-improvement loop:** stronger network → better priors + leaf values → better search → better training data → stronger network
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
| DEEP TERR | ON / OFF | Full territory in MCTS rollouts (slow but accurate) |

### Recommended training workflow

**1. Warm up MCTS first.**
Train `MCTS vs MCTS` at **500 ms, 50–100 rounds**. Builds a solid opening book and improves
position win-rate estimates. No neural-network training required for a noticeably better MCTS.

**2. Train Neural Net against MCTS.**
Select `Neural Net`, opponent `MCTS`, **100–500 ms, 200+ rounds**. Playing against a competent
teacher gives much better signal than pure self-play from random initialisation. The arc-potential
shaping reward means ring-building behaviour now emerges earlier in training.

**3. Train PyTorch Net against MCTS or itself.**
Select `PyTorch Net`, opponent `MCTS`, **200 ms, 100–500 rounds** to get started. PPO is stable
enough for `Self` play too, but MCTS accelerates early learning.

**4. Bootstrap Neural MCTS against MCTS, then switch to self-play.**
Select `Neural MCTS`, opponent `MCTS`, **200–500 ms, 100–200 rounds**. The MCTS opponent provides
a solid teacher during the early phase when the network is essentially random. Once Neural MCTS
starts winning consistently against MCTS, switch to `Self` play to enter the self-improvement
loop — each game's visit distributions become training targets that push the network to replicate
what the tree found, which then makes the tree even better.

**5. Keep grid size fixed while training.**
Experience files are tied to `GRID`. Changing it in `settings.py` causes old NN/PT/NM weights to
be ignored. MCTS experience is grid-stamped and always compatible.

---

## Configuration (`settings.py`)

| Setting | Default | Description |
|---|---|---|
| `GRID` | 10 | Intersections per side |
| `WIN_PCT` | 0.50 | Territory fraction needed to win |
| `AI_THINK_PRESETS` | `[10,100,500,…,5000]` | Available think-time values (ms) |
| `AI_THINK_DEFAULT` | 1000 | Think time at startup |
| `AI_MAX_EXPERIENCE` | 20 000 | Max MCTS position entries before pruning |
| `MCTS_REAL_ROLLOUT` | `False` | Full flood-fill per simulated move (slow) |
| `MCTS_PRIOR_WEIGHT` | 2.0 | PUCT prior weight in UCB formula |
| `MCTS_SELECTION_TERRITORY_WEIGHT` | 0.3 | Territory blend in final move selection (default profile) |
| `MCTS_NOISE_ALPHA` | 0.3 | Dirichlet noise concentration (default profile) |
| `MCTS_NOISE_FRAC` | 0.25 | Noise fraction of prior (default profile; 0 = off) |
| `MCTS_PROFILES` | see code | Per-instance weight profiles for MCTS vs MCTS |
| `AI_CLOSE_BOOST` | 4.0 | Post-policy multiplier for territory-closing moves |
| `AI_BLOCK_BOOST` | 3.0 | Post-policy multiplier for opponent-blocking moves |
| `AI_FORK_BOOST` | 2.5 | Boost for fork moves |
| `AI_BRIDGE_BOOST` | 2.0 | Boost for arc-bridging moves |
| `AI_DISRUPT_BOOST` | 1.8 | Boost for disruption moves |
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
| `NM_NOISE_FRAC` | 0.25 | Fraction of root prior that comes from noise |
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
  features.py           All board encoding and strategic signal computation
  mcts.py               MCTS (per-instance profiles, PUCT priors, 6-tier rollouts, Dirichlet noise)
  nn_player.py          Actor-critic policy network (pure numpy, 15-channel input)
  pytorch_player.py     Residual CNN + PPO + GAE (requires torch, 15-channel input)
  neural_mcts_player.py AlphaZero-style MCTS guided by residual CNN (requires torch)
  runner.py             Background thread wrapper — keeps pygame loop responsive
  trainer.py            Offline self-play trainer (territory + arc shaping rewards)
  paths.py              Grid-stamped experience file path helper
```
