# ── Dot Grid — Settings ───────────────────────────────────────────────────────
# Edit this file to change game behaviour.

# Grid
GRID                = 10

# Window
WIN_W               = 1100
WIN_H               = 800
PANEL_W             = 220

# Zoom
CELL_INIT           = 14
CELL_MIN            = 4
CELL_MAX            = 80
ZOOM_STEP           = 1.15

# Frame rate
FPS                 = 60

# Win condition: fraction of (GRID-1)^2 cells needed to win.
# Game also ends when all grid intersections are occupied or all
# remaining positions are forbidden (inside claimed territory).
WIN_PCT             = 0.50

# MCTS AI
AI_THINK_PRESETS    = [10, 100, 500, 1000, 1500, 2000, 3000, 5000]
AI_THINK_DEFAULT    = 1000      # must be one of the presets above
AI_THINK_MS         = AI_THINK_DEFAULT   # current thinking time (set from the UI at runtime)
AI_EXPERIENCE_BASE  = "../ai_experience"       # grid size appended at runtime
AI_MAX_EXPERIENCE   = 20_000
REPLAY_BUFFER_MAX   = 20    # max human games stored for replay training

# Neural-network AI (pure numpy)
NN_EXPERIENCE_BASE  = "../nn_weights"          # grid size appended at runtime
NN_HIDDEN_SIZES     = [512, 256]   # hidden layer widths
NN_LEARNING_RATE    = 0.001
NN_DISCOUNT         = 0.99         # reward discount factor

# ── Neural-network architecture presets (auto-selected by grid size) ──────────
#
# Each preset is chosen so the model capacity matches the board complexity:
#   NN hidden sizes scale with input dimension (5·G²)
#   PT residual blocks scale so the receptive field (3+4·N) covers the board
#
# Override by setting NN_HIDDEN_OVERRIDE or PT_OVERRIDE to a dict, e.g.:
#   NN_HIDDEN_OVERRIDE = [256, 128]
#   PT_OVERRIDE = {'blocks': 6, 'channels': 128}
#
_NN_PRESETS = {
    7:  [128, 64],
    12: [512, 256],
    20: [768, 384],
}   # G > 20 → [1024, 512]

_PT_PRESETS = {
    7:  {'blocks': 2, 'channels': 32},
    12: {'blocks': 4, 'channels': 64},
    20: {'blocks': 6, 'channels': 64},
}   # G > 20 → {'blocks': 8, 'channels': 128}

def _pick(presets, fallback):
    for threshold in sorted(presets):
        if GRID <= threshold:
            return presets[threshold]
    return fallback

NN_HIDDEN_OVERRIDE  = None          # set to e.g. [256,128] to force a specific size
PT_OVERRIDE         = None          # set to e.g. {'blocks':6,'channels':128} to override

def nn_hidden_sizes():
    if NN_HIDDEN_OVERRIDE is not None:
        return list(NN_HIDDEN_OVERRIDE)
    return _pick(_NN_PRESETS, [1024, 512])

def pt_arch():
    if PT_OVERRIDE is not None:
        return dict(PT_OVERRIDE)
    return _pick(_PT_PRESETS, {'blocks': 8, 'channels': 128})

# Neural-network actor-critic hyperparameters
NN_VALUE_COEF       = 0.5          # weight of value loss vs policy loss
NN_ENTROPY_COEF     = 0.01         # entropy bonus to discourage premature determinism
NN_GAE_LAMBDA       = 0.95         # GAE λ for advantage estimation

# PyTorch AI  (requires: pip install torch)
PT_EXPERIENCE_BASE  = "../pt_weights"          # grid size appended at runtime
PT_LEARNING_RATE    = 3e-4
PT_DISCOUNT         = 0.99         # reward discount factor (γ)
PT_GAE_LAMBDA       = 0.95         # GAE λ
PT_PPO_CLIP         = 0.2          # PPO clipping epsilon
PT_PPO_EPOCHS       = 4            # optimisation passes per game
PT_PPO_MINIBATCH    = 32           # minibatch size
PT_VALUE_COEF       = 0.5          # value loss weight
PT_ENTROPY_COEF     = 0.01         # entropy bonus weight
PT_MAX_GRAD_NORM    = 0.5          # gradient clip norm

# Reward shaping (shared by NN and PT trainers)
REWARD_TERRITORY    = 0.05         # reward per territory point gained in a step
REWARD_ARC_SHAPING  = 0.10         # reward per unit increase in arc_potential (ring progress)

# ── MCTS player personality profiles ─────────────────────────────────────────
# Used to differentiate MCTS vs MCTS games so the two players pursue distinct
# strategies rather than mirroring each other.
#
# Profile 1 — "Opportunist": values immediate territory swings higher
# Profile 2 — "Strategist":  values the ring race and ring potential higher
#
# Each profile overrides the move-scoring weights (STRAT_* further below),
# the PUCT exploration constant and the root-noise parameters.  Missing keys
# fall back to the global defaults.
MCTS_PROFILES = {
    1: dict(race_w=0.8, potential_w=1.0, gain_w=3.5, threat_w=3.0,
            noise_alpha=0.50, noise_frac=0.25, exploration=1.3),
    2: dict(race_w=1.2, potential_w=2.2, gain_w=1.5, threat_w=2.0,
            noise_alpha=0.30, noise_frac=0.10, exploration=1.7),
}
# Profile 0 = global defaults (for single MCTS)
MCTS_DEFAULT_PROFILE  = 0
MCTS_PROFILE_LABELS   = {0: "Default", 1: "Opportunist", 2: "Strategist"}

# Dirichlet exploration noise mixed into the MCTS root priors each turn —
# keeps self-play games from repeating.  The playouts are random anyway, so
# a little goes a long way.
# alpha: concentration parameter — lower = spikier noise
# frac:  fraction of the prior that comes from noise (0 = off)
MCTS_NOISE_ALPHA  = 0.3
MCTS_NOISE_FRAC   = 0.10

# MCTS opening book (experience file): positions with at most this many dots
# are looked up, and remembered results seed at most this many virtual visits.
MCTS_BOOK_MAX_DOTS   = 12
MCTS_BOOK_MAX_VISITS = 12

# Legacy multiplier boosts (ai/features.apply_boost / boost_log_offsets).
# The players now use strategy.guidance_offsets() instead.
AI_CLOSE_BOOST      = 4.0          # multiplier for moves that close a unit square
AI_BLOCK_BOOST      = 3.0          # multiplier for moves that block opponent closing

# Policy guidance for the policy-network players (NN, PyTorch): the exact
# move scores of ai/strategy.py are added to the network's logits as
# (score − best) / AI_GUIDE_TEMP.  Lower temperature = play closer to the
# best heuristic move; the clip bounds how far a move can be suppressed so
# training can still reach it.
AI_GUIDE_TEMP       = 0.015
AI_GUIDE_CLIP       = 12.0
# The network's own logits are squashed to ±AI_NET_LOGIT_BOUND (C·tanh(x/C))
# before the guidance is added, so a badly trained network can re-rank
# reasonable moves but can never drag a blunder (guidance ≈ −CLIP) to the top.
AI_NET_LOGIT_BOUND  = 3.0

# Player display names (shown in the UI)
PLAYER_NAMES        = {1: "Player I", 2: "Player II"}

# Colours  (R, G, B)
C_BG        = ( 14,  15,  18)
C_PANEL     = ( 22,  24,  32)
C_BORDER    = ( 42,  44,  58)
C_GRID      = ( 80,  88, 120)
C_GRID5     = (130, 140, 185)
C_P1        = ( 46, 204, 113)   # green
C_P2        = (231,  76,  60)   # red
C_GOLD      = (201, 168,  76)
C_TEXT      = (200, 194, 188)
C_DIM       = ( 90,  88,  85)

# ── Neural MCTS (requires: pip install torch) ────────────────────────────────
#
# Runs the same exact-rules tree search as the MCTS player and trains the
# residual CNN (same architecture as PyTorchPlayer) on its results: policy
# head ← visit distributions, value head ← outcomes.  The network steers the
# root of the search in proportion to how much it has been trained.
# Uses separate weight files so PyTorchPlayer weights are never affected.
NM_EXPERIENCE_BASE  = "../nm_weights"          # grid-stamped weight file base path

NM_LEARNING_RATE    = 1e-3             # Adam learning rate
NM_DISCOUNT         = 0.99             # reward discount γ
NM_GAE_LAMBDA       = 0.95             # GAE λ for advantage estimation
NM_VALUE_COEF       = 1.0              # MSE value loss weight
NM_POLICY_COEF      = 1.0              # cross-entropy policy loss weight
NM_ENTROPY_COEF     = 0.01             # entropy bonus (exploration during training)

NM_C_PUCT           = 1.5              # PUCT exploration constant (higher = more diverse)
NM_NOISE_ALPHA      = 0.3              # Dirichlet noise concentration at root
NM_NOISE_FRAC       = 0.15             # fraction of root prior that comes from noise

NM_MAX_TRUST        = 0.5              # max weight of the policy head in the root priors
NM_TRUST_STEPS      = 600              # training steps until the network gets full trust
NM_VALUE_SEEDS      = 8                # top candidate moves rated by the value head
NM_VALUE_VISITS     = 6                # virtual visits per rated move at full trust

NM_BUFFER_MAX       = 8000             # max (state, π, v) samples kept in memory
NM_BATCH_SIZE       = 256              # minibatch size per training step
NM_TRAIN_STEPS      = 4                # minibatches sampled per finished game
NM_Z_WEIGHT         = 0.5             # blend of final outcome z into value target
NM_AUGMENT          = True             # random D4 transform per sampled item

NM_OVERRIDE         = None             # override architecture: {'blocks':4,'channels':64}

def nm_arch():
    """Same preset logic as pt_arch() — NM and PT share architecture presets."""
    if NM_OVERRIDE is not None:
        return dict(NM_OVERRIDE)
    return _pick(_PT_PRESETS, {'blocks': 8, 'channels': 128})

# ── Strategic evaluation (ai/strategy.py — shared by all AI players) ─────────
#
# The AIs measure the "ring race": how many more dots each player needs to
# close a ring that wins the game (ai/rings.py), and score moves by how they
# change it, plus exact one-move tactics from the bitboard engine.
RING_DISCOUNT       = 0.92    # value of a ring decays by this factor per dot still missing
RACE_STEEPNESS      = 1.2     # how sharply a lead in the ring race counts (grows near closure)
STRAT_RACE_W        = 1.0     # weight: ring-race outlook after the move, in (-1, 1)
STRAT_POTENTIAL_W   = 1.5     # weight: ring potential difference (fraction of the board)
STRAT_GAIN_W        = 2.0     # weight: exact territory swing of the move (fraction of the board)
STRAT_THREAT_W      = 2.0     # weight: opponent's best remaining one-move swing

# ── Tree search (ai/search.py — the engine of the MCTS and Neural MCTS players) ──
SEARCH_C_PUCT            = 1.5    # PUCT exploration constant
SEARCH_PRIOR_TEMP        = 0.25   # softmax temperature turning move scores into priors
SEARCH_WORKERS           = None   # worker processes for parallel search: None = automatic (cores − 2, max 8), 0 = off
SEARCH_PARALLEL_MIN_S    = 0.20   # searches with less time left than this stay single-process
SEARCH_FAST_BELOW_S      = 0.15   # thinking times below this (on 10×10; scaled by (GRID/10)³) use the cheap analysis
