# How the AI Players Work — Plain English Guide

This document explains the three different computer opponents in Dot Grid. Each one learns and
plays the game in a completely different way. You don't need any technical knowledge to understand
this — think of it like explaining how three different kinds of chess players think.

---

## The Game, Briefly

Before explaining the AIs, it helps to recall what they're trying to do. In Dot Grid, two players
take turns placing dots on a grid. When your dots connect and surround an area — either a full
square or a triangle — you score points. The first player to claim enough territory wins. A key
twist: if you completely surround your opponent's dots, those dots are removed from the board and
the entire enclosed area becomes yours. If the board fills up with equal scores it's a draw.

---

## 1. MCTS — The Planner

**Full name:** Monte Carlo Tree Search

### The core idea

Imagine you're deciding whether to move to a new city. You could reason it through logically, or
you could just imagine thousands of different futures — what if I take that job? what if I don't?
— and count how many turned out well. MCTS does exactly the latter. Instead of figuring out *why*
a move is good, it runs thousands of imaginary futures forward and counts the wins.

### How it works, step by step

**Step 1 — Building a tree.**
Every time MCTS thinks about a move, it constructs a tree. At the top is the current board
position. Each branch is a possible move, each branch of that branch is a possible response, and
so on. The tree is a map of all possible futures.

**Step 2 — Choosing which branches to explore (PUCT priors).**
With 80+ possible moves, MCTS can't explore everything. It gives each candidate move a *prior
score* before any exploration — a first impression of how promising that move looks. These priors
are built from eight signals, ranked roughly by certainty:

| Signal | What it rewards |
|--------|-----------------|
| Immediate close | Completing a unit square or triangle right now |
| Opponent block | Stopping the opponent completing one |
| Territory setup | Creating a 3-own-corner position — a next-turn threat |
| Arc extension | Extending a ring arc in progress (partial-ring aware, see below) |
| Arc block | Stopping the opponent extending their ring |
| Arc bridge | Closing a gap between two of your own arcs to form a larger ring |
| Cluster growth | Placing adjacent to your existing dots to grow 2-dimensionally |
| Centrality | A small baseline so central positions are explored first on an empty board |

Dirichlet noise is added to these priors each turn so that different games explore different
territory (see "Personality profiles and non-determinism" below).

**Step 3 — Running quick simulations.**
Once the tree points to an unexplored position, the AI plays out the rest of the game quickly
using a six-tier priority:

1. **Fork** — creates two simultaneous threats the opponent can only answer one of
2. **Close** — scores territory right now
3. **Block** — stops the opponent scoring
4. **Setup** — creates a 3-own-corner position for next turn
5. **Adjacent** — extends your own dot cluster (50% bias)
6. **Random** — last resort

**Step 4 — Evaluating positions mid-rollout.**
Most simulations don't finish the full game. The AI evaluates the position using four signals:

| Signal | Weight | Why |
|--------|--------|-----|
| Dot ratio | 15% | Basic board presence |
| Connection ratio | 10% | Local structure |
| Territory potential | 25% | Immediate scoring opportunities on each side |
| Arc potential | 50% | How large a ring each side is building (partial-ring aware) |

The arc potential signal is the most important — it ensures MCTS values a nearly-complete ring
around 15 squares far more than 15 scattered dots.

**Step 5 — Updating the tree.**
Results ripple back up the tree. Over hundreds of simulations, the tree learns which moves lead
to wins.

**Step 6 — Choosing the final move.**
After the time budget runs out, the AI picks the move scoring best on win rate combined with the
same eight prior signals. A move visited 500 times at 60% beats one visited 50 times at 70% —
the second hasn't been tested enough to trust.

**Step 7 — Remembering for next time.**
MCTS saves its tree between moves (reusing thinking from earlier in the game) and saves position
win-rates to a file between games to warm-start future searches.

### What "partial-ring aware" means

Earlier versions of the ring signal used a flood-fill that leaked through any gap in the ring,
returning zero until the very last gap was closed. This made MCTS completely blind to ring-building
during the entire construction phase — every step looked equally worthless until the ring was
complete. The current arc potential uses a bounding-box × completion-ratio formula that fires
continuously as the ring is built, growing from the first arc placed to the completed ring.
Straight lines score zero (no 2D bounding box area); a ring half-built already scores half its
eventual reward.

### Personality profiles and non-determinism

Pure MCTS is deterministic: given the same board, it always makes the same choice. Two identical
instances playing each other would mirror every move perfectly. Two mechanisms break this:

**Dirichlet noise** — random perturbation of the priors each turn, where a few random positions
get a large boost and most get almost nothing (spiky, not uniform). The spikiness adapts to the
number of legal moves so the effect is consistent regardless of board state.

**MCTS vs MCTS personality profiles** — when two MCTS players face each other, they receive
deliberately different weight profiles so they pursue genuinely different strategies:

| | Opportunist (Player 1) | Strategist (Player 2) |
|---|---|---|
| Style | Grabs immediate territory aggressively | Builds large rings patiently |
| Immediate close weight | 5.5 | 2.5 |
| Arc extension weight | 2.5 | 7.0 |
| Dirichlet noise fraction | 35% | 18% |
| UCB exploration constant | 1.20 (narrower) | 1.65 (wider) |

The Opportunist closes squares quickly and plays noisily; the Strategist explores more of the
tree and invests in large ring constructions that pay off heavily when complete.

### Deep territory mode

The **DEEP TERR** toggle in Settings switches rollout scoring:

- **OFF (default, fast):** Lightweight square-counting, ~500 rollouts/second.
- **ON (slow, accurate):** Full flood-fill territory engine per simulated move — catches
  triangles, encirclement, and forbidden positions correctly, but ~100× slower. Only worthwhile
  at 2000 ms+ think time. Also available as a toggle in the training config screen.

### What it's good at

MCTS doesn't need training — it plays strategic ring-building from the first move. Gets noticeably
stronger with more thinking time. Games are varied because of Dirichlet noise and (in MCTS vs
MCTS) distinct personality profiles.

### What it struggles with

Tree search spreads too wide to plan many moves ahead. And even with strategic signals, the
rollouts are still partly random after the first few moves — there's a ceiling on quality that
more thinking time alone can't overcome.

---

## 2. Neural Network — The Intuition Machine

**Full name:** Actor-Critic Neural Network (pure NumPy)

### The core idea

A neural network learns by playing many games and slowly adjusting thousands of tiny internal
numbers based on whether each game was won or lost. After enough games, the network develops an
intuition — a mathematical pattern that maps board positions to good moves, much like how a human
player learns to *feel* that a position looks promising.

### What the network sees

The board is encoded as **15 layers** (like 15 transparent sheets stacked on top of each other):

| Layer | What it shows |
|-------|---------------|
| 1 | Where YOUR dots are |
| 2 | Where the OPPONENT'S dots are |
| 3 | Which positions are forbidden (inside claimed territory) |
| 4 | Which dots are connected to your other dots |
| 5 | Which dots are connected to the opponent's dots |
| 6 | Immediate closing opportunities FOR YOU |
| 7 | Immediate closing opportunities FOR THE OPPONENT |
| 8 | How much territory your ring arcs could enclose if completed (flood-fill) |
| 9 | How much territory the opponent's ring arcs could enclose |
| 10 | Where bridging your own arcs would create a larger ring |
| 11 | Where the opponent could bridge their arcs |
| 12 | Which positions disrupt territory the opponent is building toward |
| 13 | Which positions create two or more simultaneous threats (fork map) |
| 14 | How far through the game we are (broadcast to all positions) |
| 15 | How central each position is geometrically |

Layer 14 (game phase) lets the network learn different strategies for early vs late game. Layer 15
(centrality) gives a spatial prior that central moves have more long-term flexibility.

### The two outputs

**Policy** — a probability for every empty position. Higher = better move candidate.

**Value** — a number in [-1, 1] estimating how well-placed the current player is.

### How it learns — and the ring-building fix

After each game the network updates using **GAE** (Generalised Advantage Estimation): for each
move, it asks "how much better or worse did things turn out than expected?" and adjusts weights
accordingly.

The training signal has three components:

1. **Score margin** at game end — winning 45–10 gives a much stronger positive signal than barely
   winning 26–25. This teaches the network to prefer large enclosures over scraping to the win
   threshold with scattered small captures.

2. **Territory-delta shaping** — a small reward each time you gain territory this move. This
   gives the network fast feedback instead of waiting until the end of the game.

3. **Arc-potential-delta shaping** — a reward each time your move increases your ring-building
   potential, even if the ring isn't closed yet. This is the fix for the "small-grab bias"
   problem: without this signal, the network only sees a reward when a ring actually closes, so
   it learns to prefer small immediate captures (reliably rewarded) over patient ring-building
   (rewarded only once, many moves later). With arc shaping, every step that extends a ring arc
   is rewarded proportionally — encouraging the same ring-building instinct that MCTS has built
   in from its arc potential priors.

A small entropy bonus keeps the network from becoming too certain too quickly, preserving
exploration throughout training.

### What it's good at

Given enough training, it can develop strategic intuitions that MCTS misses — patterns spanning
many moves that emerge from experience rather than explicit lookahead. All 15 channels including
arc potential and bridge signals are visible to it.

### What it struggles with

Needs substantial training to be any good. A freshly created network plays almost randomly. It
also can't look ahead — it can only react to what it sees on the board right now.

---

## 3. PyTorch Network — The Deep Learner

**Full name:** Residual Convolutional Neural Network with PPO (Proximal Policy Optimisation)

### The core idea

Think of this as an upgraded Neural Network: instead of looking at the board as a flat list of
numbers, it looks at it like an *image* — using the same deep learning that made AI good at
recognising photos.

### How it sees the board differently

The PyTorch network receives the same 15 layers but processes them with *convolutional filters* —
a 3×3 sliding window scanning across the board for local patterns. Adjacent positions are
genuinely treated as spatially related, which is essential for a grid game. Ring arc shapes and
bridge opportunities are detected directly from spatial layout rather than from explicitly
engineered signals.

### The architecture

**The Stem** — converts the 15 input layers into 64 feature channels using 3×3 filters,
detecting basic patterns: dot clusters, arcs, gaps.

**The Residual Tower** — four stacked blocks (more for larger grids), each with two convolutional
layers and a shortcut connection. After four blocks the network sees a 19×19 neighbourhood per
output cell — enough to span most of a 10×10 board.

**Policy Head** — move recommendations (probability per position).

**Value Head** — position evaluation.

### How it learns — PPO

PPO's key constraint: don't change the policy so drastically that a single unusual game overrides
everything learned before. Updates are clipped to 20% per step. Combined with the same three-part
reward (score margin + territory shaping + arc-potential shaping), this produces smooth, stable
improvement across thousands of training games.

The arc-potential shaping reward has the same effect here as for the Neural Network: ring-building
behaviour emerges earlier in training and the network doesn't over-specialise in harvesting small
immediate captures.

### What makes it stronger than the Neural Network

**Spatial awareness** — CNN filters detect ring shapes, arc curves, and territory boundaries from
spatial layout rather than from explicitly enumerated features.

**Receptive field** — after four residual blocks, almost every decision is informed by the entire
board state, not just the local neighbourhood.

**Training stability** — PPO prevents catastrophic regressions that simpler algorithms can suffer.

### What it struggles with

Needs the most training time to reach its potential. Also requires PyTorch (`pip install torch`).

---

## 4. Neural MCTS — The Self-Improving Planner

**Full name:** AlphaZero-style MCTS guided by a residual CNN

### The core idea

Each of the first three AIs has one fatal weakness: MCTS has hand-crafted signals instead of
learned intuition; the Neural Network and PyTorch Network have learned intuition but no ability
to look ahead. Neural MCTS fixes both at once by making the CNN and the tree search partners
that continuously improve each other.

The basic concept comes from AlphaZero — the same architecture that mastered chess, Go, and
shogi without any human-provided strategy beyond the rules.

### How it works

**Step 1 — Network call at the root.**
At the start of each turn, the CNN evaluates the current board position and produces two outputs:
a *policy* (which moves look promising) and a *value* (how well-placed are we right now?). The
policy replaces the hand-crafted eight-signal priors that pure MCTS uses.

**Step 2 — PUCT search.**
Instead of branching equally into all legal moves, the tree focuses on moves the policy rates
highly. The formula is: `Q + c_puct × P × √parent_visits / (1 + child_visits)`. Q is the
average value from past simulations; P is the network's prior; the square root term decays so
well-visited children stop getting automatic priority.

**Step 3 — Expand, don't roll out.**
When the search reaches a position it has never seen before, instead of simulating the rest of
the game with random moves (like pure MCTS does), it simply calls the network once to get the
value of that position. One forward pass replaces hundreds of random moves. This lets the same
time budget explore far more tree nodes, because each new node costs microseconds rather than
milliseconds.

**Step 4 — Backpropagate.**
The value propagates back up the tree exactly like regular MCTS. Over many simulations the tree
learns which branches are genuinely better, even if the initial network estimates were wrong.

**Step 5 — Choose the most-visited move.**
After the time budget runs out, the move that was visited most often is chosen. Visit counts are
far more reliable than a single network guess.

### How it trains — the self-improvement loop

After each game, two things are recorded:
- The **visit distribution**: how many simulations visited each move at the root. This is better
  than the raw network prior — the tree searched deeper and found things the network alone missed.
- The **game outcome** plus shaping rewards (territory delta + arc-potential delta).

The network is then trained to predict both: make the policy head output look like the visit
distribution (cross-entropy loss), and make the value head output predict the actual returns
(mean-squared-error loss). No PPO or clipping is needed: the visit distribution is a stable
supervised target, not a noisy self-bootstrapped estimate.

This creates the self-improvement loop: a stronger network → better priors and leaf values →
better tree search → better visit distributions → stronger network → ...

### Why it beats the other AIs given enough training

**vs. pure MCTS:** Neural MCTS learns which moves are worth exploring rather than distributing
simulations based on hand-crafted formulas. It also learns accurate position values, so short
search trees give reliable estimates. The hand-crafted signals in pure MCTS encode specific
strategic patterns you thought of; the learned policy encodes everything the tree discovered.

**vs. Neural Network / PyTorch Net:** Same CNN, but paired with tree search. The network alone
reacts to the current position; Neural MCTS looks 10–20 moves ahead, guided by the network at
every step.

### What it struggles with

Needs meaningful training before it surpasses pure MCTS. A freshly created Neural MCTS with a
random network is actually *weaker* than pure MCTS — bad priors mislead the search. Train against
MCTS first (100–200 rounds) to bootstrap reasonable priors, then switch to self-play to enter the
improvement loop. Also requires PyTorch (`pip install torch`).

---

## Side-by-Side Comparison

| | MCTS | Neural Network | PyTorch Network | Neural MCTS |
|---|---|---|---|---|
| **Needs training to be good?** | No | Yes | Yes | Yes (bootstrap first) |
| **Gets stronger with thinking time?** | Strongly | No | No | Yes (more simulations) |
| **Plays ring-building strategy?** | Yes (built-in arc potential) | Yes, after training | Yes, after training | Yes (learns + searches) |
| **Arc-potential shaping reward?** | N/A | Yes | Yes | Yes |
| **Varies between games?** | Yes (noise + profiles) | Yes (sampling) | Yes (sampling) | Yes (noise + sampling) |
| **Trained on score margin?** | N/A | Yes | Yes | Yes |
| **Looks ahead explicitly?** | Yes (tree) | No | No | Yes (tree + network) |
| **Requires extra software?** | No | No | Yes (PyTorch) | Yes (PyTorch) |
| **Ceiling after extensive training?** | Fixed | Medium | High | Highest |

---

## How All Four React to Opportunities

All four AIs share a move-selection boost applied on top of their core decision-making:

**Closing boost (4×):** Completing a territory square or triangle → up to 4× multiplier (scales
with territory count: closing two squares at once → 7×; single triangle → 2.5×).

**Blocking boost (3×):** Opponent about to close → up to 3×.

For MCTS, tactical awareness works at three levels: priors (closing moves get explored first),
rollout policy (six-tier priority from fork to random), and final selection (win rate blended with
all prior signals). Each MCTS instance's exact weights depend on its personality profile.

For the Neural and PyTorch Networks, the boost is applied after the network outputs its policy.
As training progresses the network learns to assign high probabilities to good moves on its own,
so the boost becomes a safety net rather than the primary driver.

For Neural MCTS, the boost is applied to the root priors at the start of each turn as a safety
net during the early untrained phase. As the network matures it learns to rate closing and
blocking moves highly on its own, so the boost becomes redundant.

---

## Tips for Getting the Best from Each AI

**MCTS:** Give it more thinking time. At 100 ms it's decent; at 1000 ms (the default) it's quite
strong. In MCTS vs MCTS mode watch the two styles emerge: the Opportunist tends to scatter quick
captures early while the Strategist builds slower, larger rings. The **DEEP TERR** toggle in
Settings adds full flood-fill accuracy to rollouts — only worth enabling at 2000 ms+ think time.

**Neural Network:** Train for several hundred games before playing against it. The arc-potential
shaping reward means ring-building strategy now emerges in fewer games than before — you should
see genuine ring attempts after a few hundred rounds of training rather than purely reactive play.

**PyTorch Network:** Needs the most training but becomes the strongest pure-network AI. Train in
batches of 100–500 games. After 1000+ games it should play strategic, coherent Dot Grid — building
rings, bridging arcs, and claiming large territories rather than just harvesting small immediate
scores. The PPO training stability means you can leave it training overnight without worrying about
catastrophic regressions.

**Neural MCTS:** Start with `Neural MCTS vs MCTS` at 200–500 ms for 100–200 rounds. The MCTS
teacher bootstraps reasonable priors quickly. Then switch to `Neural MCTS vs Neural MCTS` self-play
to enter the improvement loop. Given 500–1000 self-play games it will surpass every other AI at
the same time budget — it gets both the positional intuition of the CNN and the lookahead of the
tree, and the two components keep making each other better.
