"""Replay trainer.

Loads recorded human games and replays them through each AI player's
observe_opponent_move + record_outcome pipeline.  This gives the AI the
same learning signal as playing those games live, repeated as many times
as requested — without a human needing to be present.

Same start/cancel/is_running/on_progress/on_done interface as Trainer so
the existing training-progress UI in dotgame.py / draw.py works unchanged.

How replay learning works
─────────────────────────
For each saved game we re-simulate it move by move using SimGame, then:

  • For the AI player's own moves:
      observe_opponent_move(board_before, ..., player=AI, move=AI_move)
      — encodes the position from the AI's perspective, uses its actual move
        as a one-hot policy target, and stores its current value estimate.
      — equivalent to supervised imitation: "in this position, you did X, and
        the outcome was Y"

  • For the opponent's moves:
      observe_opponent_move(board_before, ..., player=opp, move=opp_move)
      — same symmetric learning the live runner already does

  • At game end:
      record_outcome(winner, final_scores=...)
      — triggers the gradient update with the correct score-margin terminal value

Training both players (player1 and player2) is optional but supported:
pass None for a player slot to skip it (e.g. when only one AI is being trained).
"""

import threading

import settings as S
from ai.game_recorder import load_buffer
from ai.trainer import SimGame          # reuse the headless sim engine


class ReplayTrainer:
    """Replay saved human games to train AI players offline."""

    def __init__(self, player1, player2, repeats: int = 1,
                 on_progress=None, on_done=None):
        """
        player1, player2 — AI player instances to train (either may be None
                           to skip training that side)
        repeats          — how many times to replay the full buffer
        on_progress      — callback(completed, total, wins_dict)
        on_done          — callback()
        """
        self._p1          = player1
        self._p2          = player2
        self._repeats     = max(1, repeats)
        self._on_progress = on_progress
        self._on_done     = on_done
        self._cancel      = False
        self._thread: threading.Thread | None = None
        self.wins         = {1: 0, 2: 0, 'draw': 0}

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def cancel(self):
        self._cancel = True

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _run(self):
        games = load_buffer()
        if not games:
            if self._on_done:
                self._on_done()
            return

        # Filter to current grid size
        games = [g for g in games if g.get('grid', S.GRID) == S.GRID]
        if not games:
            if self._on_done:
                self._on_done()
            return

        total = len(games) * self._repeats
        done  = 0

        for _rep in range(self._repeats):
            for game_rec in games:
                if self._cancel:
                    break

                self._replay_one(game_rec)
                done += 1

                if self._on_progress:
                    self._on_progress(done, total, dict(self.wins))

            if self._cancel:
                break

        if self._p1:
            self._p1.save()
        if self._p2:
            self._p2.save()

        if self._on_done:
            self._on_done()

    def _replay_one(self, game_rec: dict):
        """Replay a single saved game through both players' learning pipelines."""
        moves        = game_rec['moves']          # [[player, gx, gy], ...]
        winner       = game_rec['winner']          # int 1/2/0
        final_scores = game_rec.get('final_scores', {})

        # Re-simulate the game from scratch to get accurate board states
        # at each step (SimGame tracks forbidden zones, encirclement, etc.)
        sim = SimGame()

        for entry in moves:
            if self._cancel:
                return

            player_who_moved, gx, gy = entry

            # Snapshot board state BEFORE the move so observe_opponent_move
            # gets the pre-move position (same as in the live game loop)
            board_before = dict(sim.board)
            conn_before  = set(sim.connections)
            forb_before  = set(sim.forbidden)

            # Apply the move to advance sim state
            sim.place(gx, gy)

            # Notify both players about this move.
            # For each player:
            #   - their own moves: teaches "in this position I played here"
            #   - opponent moves:  symmetric learning already wired in live play
            for pid, player in ((1, self._p1), (2, self._p2)):
                if player is None:
                    continue
                if not hasattr(player, 'observe_opponent_move'):
                    continue
                player.observe_opponent_move(
                    board_before, conn_before,
                    player_who_moved, (gx, gy),
                    forb_before)

        # Compute outcome for win tracking
        if winner == 1:
            self.wins[1] = self.wins.get(1, 0) + 1
        elif winner == 2:
            self.wins[2] = self.wins.get(2, 0) + 1
        else:
            self.wins['draw'] = self.wins.get('draw', 0) + 1

        # Fire record_outcome on both players — triggers gradient update
        w_int = winner or 0
        fs    = {int(k): float(v) for k, v in final_scores.items()} if final_scores else {}
        if self._p1:
            self._p1.record_outcome(w_int, final_scores=fs or None)
        if self._p2:
            self._p2.record_outcome(w_int, final_scores=fs or None)
