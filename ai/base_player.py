"""Abstract base class for every AI player.

To swap MCTS for a neural-network player, subclass BasePlayer,
implement the four abstract members, and pass your class to GameMode.
"""

from abc import ABC, abstractmethod


class BasePlayer(ABC):
    """Minimal interface every AI backend must satisfy."""

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    @abstractmethod
    def choose_move(
        self,
        board: dict,
        connections: set,
        player: int,
        scores: dict,
        forbidden: set | None = None,
    ) -> tuple[int, int]:
        """Return (gx, gy) — the best move for *player* in this position.

        *forbidden* is a snapshot of state.forbidden_positions taken at the
        moment thinking started.  Always use this instead of reading live
        state so the thread works on a consistent view of the board.

        This is called in a background thread so it may block freely.
        """

    # ------------------------------------------------------------------
    # Lifecycle hooks (optional to override)
    # ------------------------------------------------------------------

    def observe_opponent_move(
        self,
        board: dict,
        connections: set,
        player_who_moved: int,
        move: tuple,
        forbidden: set | None = None,
    ) -> None:
        """Called after the *opponent* makes a move.

        board / connections / forbidden are snapshots taken *before* the move.
        Override in subclasses to learn symmetrically from opponent play.
        Default is a no-op (human players, base class).
        """

    def record_outcome(self, winner: int, intermediate_rewards=None) -> None:
        """Called once the game ends.  Use to back-propagate the result."""

    def save(self) -> None:
        """Persist learned experience to disk."""

    def load(self) -> None:
        """Load previously saved experience from disk."""
