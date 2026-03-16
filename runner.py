"""Background AI runner.

Runs BasePlayer.choose_move() in a daemon thread so the pygame main loop
stays responsive.  Check .pending_move each frame; it becomes non-None
when the AI has decided.
"""

import threading
from ai.base_player import BasePlayer


class AIRunner:
    """Wraps any BasePlayer and runs its thinking in a background thread."""

    def __init__(self, player: BasePlayer):
        self._player    = player
        self._move: tuple | None = None
        self._thinking  = False
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_thinking(self) -> bool:
        return self._thinking

    @property
    def pending_move(self) -> tuple | None:
        """Non-None once the background thread has finished."""
        return self._move

    def clear_move(self) -> None:
        self._move = None

    def start_thinking(
        self,
        board: dict,
        connections: set,
        player: int,
        scores: dict,
    ) -> None:
        """Kick off background computation.  Ignores call if already thinking."""
        if self._thinking:
            return
        self._move     = None
        self._thinking = True

        # Snapshot mutable state so the thread works on stable copies.
        # forbidden_positions must be snapshotted here too — it is derived
        # from the same board state and must not be read from live state
        # inside the background thread.
        import state as _state
        board_snap     = dict(board)
        conn_snap      = set(connections)
        scores_snap    = dict(scores)
        forbidden_snap = set(_state.forbidden_positions)

        def _run():
            try:
                move = self._player.choose_move(
                    board_snap, conn_snap, player, scores_snap,
                    forbidden_snap,
                )
                self._move = move
            except Exception as exc:
                import sys, random
                print(f"[AIRunner] choose_move raised: {exc}", file=sys.stderr)
                # Emergency fallback: pick a random legal position so the
                # game is never permanently stuck due to a crashed AI thread.
                occupied  = set(board_snap.keys())
                import settings as _S
                legal = [
                    (gx, gy)
                    for gx in range(_S.GRID)
                    for gy in range(_S.GRID)
                    if (gx, gy) not in occupied
                    and (gx, gy) not in forbidden_snap
                ]
                self._move = random.choice(legal) if legal else (0, 0)
            finally:
                self._thinking = False

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def observe_opponent_move(self, board: dict, connections: set,
                              player_who_moved: int, move: tuple,
                              forbidden=None) -> None:
        """Let the AI observe the board state that resulted from the opponent's move.

        Called immediately after the other player places a dot, before this
        player's own think cycle begins.  Runs synchronously (cheap — no tree
        search) so it can safely share the main-thread board snapshot.
        """
        if hasattr(self._player, 'observe_opponent_move'):
            self._player.observe_opponent_move(
                board, connections, player_who_moved, move, forbidden)

    def on_game_end(self, winner: int, final_scores: dict | None = None) -> None:
        """Forward outcome to the underlying player for learning."""
        self._player.record_outcome(winner, final_scores=final_scores)

    def save(self) -> None:
        self._player.save()

    def load(self) -> None:
        self._player.load()
