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

### What it knows about the game

The decisive plan in Dot Grid is a **big ring**: a closed chain of dots around half the board
wins on the spot and captures everything inside. Twenty well-placed dots are enough on a 10×10
board. An AI that only notices small squares being completed never sees this coming — which is
exactly how earlier versions of this AI could be beaten by anyone who quietly built one large
ring.

So before it imagines any futures, MCTS now measures the **ring race**:

- *How many more dots do I need to close a winning ring?*
- *How many does my opponent need?*
- *Which points do those rings pass through?*

It answers these exactly, including detours around the other side's dots. That last part
matters: dropping a single dot into the path of a ring usually achieves nothing, because the
ring simply steps around it. The AI knows the difference between a block that really costs the
opponent a move and one that doesn't.

### How it works, step by step

**Step 1 — Score every move.**
For each legal move the AI works out, with the real rules:

| Signal | What it rewards |
|--------|-----------------|
| Exact gain | Territory this move wins right now — a triangle, a square, or a 40-cell ring with captures |
| Threats left | The biggest thing the opponent could still do next move |
| Ring race | Who is ahead in the race for a winning ring after this move |
| Ring potential | How good both sides' best smaller rings are, when nobody can win outright |

A move that wins immediately is simply played.

**Step 2 — Build a tree, most promising moves first.**
At the top is the current position; each branch is a move, each branch of that a reply. The
scores from step 1 decide where the search spends its effort.

**Step 3 — Play out full games.**
From each new position the AI plays the game to the very end with random moves, many hundreds
of times per turn, using the *exact* rules — real territory, real captures, real win
conditions. The scores from step 1 say where to look; the playouts say what actually works.

**Step 4 — Update the tree.**
Results ripple back up the tree. Moves that keep leading to wins get explored more.

**Step 5 — Choose the most-visited move.**
A move tried 500 times with good results beats one tried 50 times with slightly better ones —
the second hasn't been tested enough to trust.

**Step 6 — Remembering for next time.**
MCTS keeps its tree between moves, and saves the results of finished games as a small opening
book that nudges its first few moves in later games.

### Personality profiles and non-determinism

Games differ because the playouts are random, and because a little random noise is mixed into
the move preferences each turn.

**MCTS vs MCTS personality profiles** — when two MCTS players face each other, they get
deliberately different priorities:

| | Opportunist (Player 1) | Strategist (Player 2) |
|---|---|---|
| Style | Takes territory as soon as it can | Invests in the ring race |
| Weight on immediate gains | 3.5 | 1.5 |
| Weight on the ring race | 0.8 | 1.2 |
| Weight on ring potential | 1.0 | 2.2 |
| Noise fraction | 25% | 10% |

### What it's good at

No training needed. It builds its own ring, cuts yours where it actually hurts, never misses a
capture, and gets stronger with more thinking time.

### What it struggles with

Its ring analysis looks for rings that curve around a handful of fixed centre points, so a very
irregular enclosure can be noticed late. And once the board is carved up and no big ring is
possible, it relies mostly on the playouts, which are random and therefore only a rough guide.

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

It can't look ahead — it reacts to the board as it is right now. To keep it from playing
randomly while it learns, the same move scores MCTS uses (step 1 above) are added to its
preferences: an untrained network already plays the best-looking move most of the time, and
training teaches it where that first impression is wrong.

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

Like the Neural Network it has no lookahead, and it starts from the same move scores. Needs the
most training time to improve on them. Also requires PyTorch (`pip install torch`).

---

## 4. Neural MCTS — The Planner That Learns

**Full name:** MCTS guided by a residual CNN

### The core idea

MCTS plans but never learns; the networks learn but never plan. Neural MCTS is the MCTS player
with a network watching over its shoulder: the search plays, the network studies what the search
found, and — once it has studied enough — starts telling the search where to look.

### How it works

**Step 1 — The search runs as in MCTS.**
Same ring-race analysis, same tree, same full-game playouts with the exact rules.

**Step 2 — The network adds its opinion at the start of each turn.**
Its *policy* output (which moves look promising) is blended into the search's move preferences,
and its *value* output rates the positions after the handful of best-looking moves.

**Step 3 — Trust is earned.**
How much the network's opinion counts depends on how much it has been trained. A brand-new
network has zero influence, so Neural MCTS starts out playing exactly like MCTS. Influence rises
steadily over the first few hundred training steps (about 150 games) up to a fixed maximum.

### How it trains

After each game, two things are recorded for every move:
- The **visit distribution**: how often the search tried each move. That is a better answer than
  the network's own first guess, because the search looked ahead.
- The **game outcome** plus shaping rewards.

The network is trained to predict both. The loop: a better network → a better-focused search →
better examples to learn from → a better network.

### What it struggles with

Until it has trained it is simply MCTS with a little overhead. Requires PyTorch
(`pip install torch`).

---

## Side-by-Side Comparison

| | MCTS | Neural Network | PyTorch Network | Neural MCTS |
|---|---|---|---|---|
| **Needs training to be good?** | No | No (improves with it) | No (improves with it) | No (improves with it) |
| **Gets stronger with thinking time?** | Strongly | Slightly (fuller analysis) | Slightly (fuller analysis) | Strongly |
| **Understands the ring race?** | Yes | Yes (move scores) | Yes (move scores) | Yes |
| **Plays with the exact rules?** | Yes | Yes (one move deep) | Yes (one move deep) | Yes |
| **Varies between games?** | Yes | Yes (sampling) | Yes (sampling) | Yes |
| **Looks ahead explicitly?** | Yes (tree) | No | No | Yes (tree) |
| **Requires extra software?** | No | No | Yes (PyTorch) | Yes (PyTorch) |

---

## How All Four React to Opportunities

All four AIs start from the same exact move scores: territory a move wins right now, the
biggest threat it leaves standing, and the ring race afterwards.

- **MCTS** and **Neural MCTS** use the scores to decide where to search, then test the
  candidates with thousands of full-game playouts.
- The **Neural** and **PyTorch Networks** add the scores to their own preferences and pick from
  the result — no lookahead beyond that.

A move that wins the game on the spot is always played by MCTS and Neural MCTS, and is by far
the most likely choice of the two networks.

---

## Tips for Getting the Best from Each AI

**MCTS:** Give it thinking time. Below 150 ms it uses a simplified analysis; at 1000 ms (the
default) it analyses fully and plays out many hundreds of games per move. In MCTS vs MCTS mode
watch the two styles: the Opportunist takes territory early, the Strategist invests in its ring.

**Neural Network / PyTorch Network:** They play sensibly untrained, because they start from the
exact move scores. Train them against MCTS (a few hundred games) so they learn where those
scores are wrong; they have no lookahead, so expect them to stay below MCTS in strength.

**Neural MCTS:** Plays like MCTS from the start. Train it by self-play or against MCTS at
200–500 ms; after about 150 games the network has full influence on where the search looks.
