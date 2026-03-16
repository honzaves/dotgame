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
# Profile 1 — "Opportunist": prefers immediate territory grabs, explores widely
# Profile 2 — "Strategist":  prefers building large rings, more patient
#
# Each profile overrides the raw_priors weights and noise params in mcts.py.
#
# arc_w / bridge_w are in territory-cell units (NOT normalised by board size).
# Calibration guide:
#   arc_w = 0.22  →  9-cell ring scores 1.98 ≈ 0.5 × single close (neutral)
#   arc_w = 0.08  →  9-cell ring scores 0.72 ≪ close_w=5.5   (Opportunist: ignore rings)
#   arc_w = 0.55  →  9-cell ring scores 4.95 > close_w=2.5   (Strategist: prefer rings)
#                    16-cell ring scores 8.80 ≫ close_w=2.5   (Strategist: dominate)
MCTS_PROFILES = {
    1: dict(close_w=5.5, block_w=4.5, arc_w=0.08, bridge_w=0.05, setup_w=1.8,
            adj_w=1.2, centrality_w=0.4, disrupt_w=0.2,
            arc_opp_w=0.20, bridge_opp_w=0.12,
            noise_alpha=0.50, noise_frac=0.35, exploration=1.20,
            sel_terr_w=0.20,
            eval_terr_w=0.40, eval_enc_w=0.25),   # rollout: care about immediate terr
    2: dict(close_w=2.5, block_w=2.0, arc_w=0.55, bridge_w=0.42, setup_w=3.5,
            adj_w=2.0, centrality_w=0.6, disrupt_w=0.5,
            arc_opp_w=0.60, bridge_opp_w=0.45,
            noise_alpha=0.12, noise_frac=0.18, exploration=1.65,
            sel_terr_w=0.45,
            eval_terr_w=0.10, eval_enc_w=0.70),   # rollout: care about ring potential
}
# Profile 0 = use global S.MCTS_* settings unchanged (default for single MCTS)
MCTS_DEFAULT_PROFILE  = 0
MCTS_PROFILE_LABELS   = {0: "Default", 1: "Opportunist", 2: "Strategist"}

# MCTS algorithm tuning
MCTS_PRIOR_WEIGHT                = 2.0   # PUCT prior weight (higher = more territory-biased exploration)
MCTS_SELECTION_TERRITORY_WEIGHT  = 0.3   # territory weight in final move selection

# Dirichlet exploration noise injected into root priors each turn.
# Prevents deterministic/mirror play between two MCTS instances.
# alpha: concentration parameter — lower = spikier noise (more diverse)
#   0.3 is the AlphaZero value for Go; good range for Dot Grid is 0.15–0.5
# frac: fraction of prior that comes from noise vs computed signal
#   0.25 means 75% prior + 25% noise — enough to diversify without drowning signal
MCTS_NOISE_ALPHA  = 0.3   # Dirichlet α — controls noise spike shape
MCTS_NOISE_FRAC   = 0.25  # weight of noise vs computed prior (0 = off, 1 = pure noise)

# Move-selection boosts (applied post-policy to all three AI types)
AI_CLOSE_BOOST      = 4.0          # multiplier for moves that close a unit square
AI_BLOCK_BOOST      = 3.0          # multiplier for moves that block opponent closing

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

# MCTS rollout territory detection
# False = fast lightweight square-counting (default)
# True  = full flood-fill territory engine (accurate but ~10× slower per move)
MCTS_REAL_ROLLOUT   = False

# Additional move-boost signals (applied post-policy alongside AI_CLOSE/BLOCK_BOOST)
AI_FORK_BOOST       = 2.5   # bonus for moves that create 2+ simultaneous threats
AI_BRIDGE_BOOST     = 2.0   # bonus for moves that bridge two own ring arcs
AI_DISRUPT_BOOST    = 1.8   # bonus for moves that reduce opponent enclosure potential

# ── Neural MCTS (AlphaZero-style — requires: pip install torch) ───────────────
#
# Uses the same residual CNN architecture as PyTorchPlayer but guides a MCTS
# tree with it: policy head → PUCT priors, value head → leaf evaluation.
# Trains on MCTS visit distributions (more stable than raw policy targets).
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
NM_NOISE_FRAC       = 0.25             # fraction of root prior that comes from noise
NM_DEAD_SPOT_PENALTY = 5.0             # divide prior of positions with zero territorial potential

NM_OVERRIDE         = None             # override architecture: {'blocks':4,'channels':64}

def nm_arch():
    """Same preset logic as pt_arch() — NM and PT share architecture presets."""
    if NM_OVERRIDE is not None:
        return dict(NM_OVERRIDE)
    return _pick(_PT_PRESETS, {'blocks': 8, 'channels': 128})
