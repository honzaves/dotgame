"""Unit tests for state.py — board state, placement, win detection, snapshots."""
import pytest
import state
import settings as S


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def fresh_state():
    state.reset()
    yield
    state.reset()


# ── reset() ───────────────────────────────────────────────────────────────────

class TestReset:
    def test_board_empty(self):
        state.place(0, 0)
        state.reset()
        assert state.board == {}

    def test_connections_empty(self):
        state.place(0, 0)
        state.reset()
        assert state.connections == set()

    def test_scores_zeroed(self):
        state.reset()
        assert state.scores == {1: 0.0, 2: 0.0}

    def test_current_player_is_one(self):
        state.place(0, 0)          # now it's P2's turn
        state.reset()
        assert state.current_player == 1

    def test_game_over_cleared(self):
        # Force game over then reset
        state.game_over = True
        state.winner_player = 1
        state.reset()
        assert state.game_over is False
        assert state.winner_player is None

    def test_forbidden_positions_cleared(self):
        """forbidden_positions must be reset — missing from global decl causes stale state."""
        state.forbidden_positions = {(3, 3), (4, 4)}
        state.reset()
        assert state.forbidden_positions == set()

    def test_snapshot_created(self):
        """reset() must seed snapshots with the empty-board capture at index 0."""
        state.reset()
        assert len(state.snapshots) == 1

    def test_snapshot_index_live(self):
        state.reset()
        assert state.snapshot_index == -1

    def test_total_moves_zeroed(self):
        state.place(0, 0)
        state.reset()
        assert state.total_moves == 0

    def test_last_move_cleared(self):
        state.place(0, 0)
        state.reset()
        assert state.last_move is None


# ── place() ───────────────────────────────────────────────────────────────────

class TestPlace:
    def test_returns_true_on_valid(self):
        assert state.place(0, 0) is True

    def test_places_dot_for_current_player(self):
        state.place(2, 3)
        assert state.board[(2, 3)] == 1

    def test_alternates_players(self):
        state.place(0, 0)
        state.place(1, 1)
        assert state.board[(0, 0)] == 1
        assert state.board[(1, 1)] == 2

    def test_returns_false_on_duplicate(self):
        state.place(0, 0)
        state.place(1, 0)          # P2 somewhere else
        result = state.place(0, 0) # P1 tries to replay same spot
        assert result is False

    def test_returns_false_out_of_bounds_negative(self):
        assert state.place(-1, 0) is False

    def test_returns_false_out_of_bounds_high(self):
        assert state.place(S.GRID, 0) is False

    def test_returns_false_when_game_over(self):
        state.game_over = True
        assert state.place(0, 0) is False

    def test_returns_false_on_forbidden_position(self):
        state.forbidden_positions = {(5, 5)}
        assert state.place(5, 5) is False

    def test_increments_total_moves(self):
        state.place(0, 0)
        assert state.total_moves == 1
        state.place(1, 0)
        assert state.total_moves == 2

    def test_records_last_move(self):
        state.place(3, 4)
        assert state.last_move == (3, 4)

    def test_appends_snapshot(self):
        state.place(0, 0)
        assert len(state.snapshots) == 2

    def test_builds_connection_to_adjacent_same_player(self):
        state.place(0, 0)   # P1
        state.place(5, 5)   # P2 (far away)
        state.place(1, 0)   # P1 (adjacent to first P1 dot)
        conn = frozenset({(0, 0), (1, 0)})
        assert conn in state.connections

    def test_no_connection_to_opponent(self):
        state.place(0, 0)   # P1
        state.place(1, 0)   # P2 (adjacent to P1 but different player)
        conn = frozenset({(0, 0), (1, 0)})
        assert conn not in state.connections

    def test_diagonal_connection_built(self):
        state.place(0, 0)   # P1
        state.place(5, 5)   # P2
        state.place(1, 1)   # P1 diagonal to (0,0)
        conn = frozenset({(0, 0), (1, 1)})
        assert conn in state.connections

    def test_place_respects_forbidden_after_territory(self):
        """Positions enclosed by territory should become forbidden after closure."""
        # Build a 2×2 ring for P1 in the top-left corner
        _build_p1_ring_at_origin()
        # The interior of the ring is not a grid intersection here (the ring
        # IS the top-left unit square), but forbidden_positions should not include
        # any corner that the ring is built on.
        for pos in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            assert pos not in state.forbidden_positions


# ── check_win() ───────────────────────────────────────────────────────────────

class TestCheckWin:
    def test_no_win_on_empty_board(self):
        state.check_win()
        assert state.game_over is False

    def test_player1_wins_on_threshold(self):
        state.scores[1] = S.WIN_PCT * state.TOTAL_FIELDS
        state.check_win()
        assert state.game_over is True
        assert state.winner_player == 1

    def test_player2_wins_on_threshold(self):
        state.scores[2] = S.WIN_PCT * state.TOTAL_FIELDS
        state.check_win()
        assert state.game_over is True
        assert state.winner_player == 2

    def test_draw_when_board_full_equal_scores(self):
        """winner_player must be None on a draw — all consumers must handle None."""
        state.scores = {1: 10.0, 2: 10.0}
        # Fill the board
        for gx in range(S.GRID):
            for gy in range(S.GRID):
                state.board[(gx, gy)] = 1
        state.check_win()
        assert state.game_over is True
        assert state.winner_player is None

    def test_player1_wins_full_board(self):
        state.scores = {1: 15.0, 2: 10.0}
        for gx in range(S.GRID):
            for gy in range(S.GRID):
                state.board[(gx, gy)] = 1
        state.check_win()
        assert state.winner_player == 1

    def test_no_placeable_triggers_end(self):
        """Game ends when placeable positions drop to zero (all forbidden or filled)."""
        total = S.GRID * S.GRID
        # Place some dots and mark the rest forbidden
        state.board = {(0, 0): 1}
        state.forbidden_positions = {
            (gx, gy)
            for gx in range(S.GRID) for gy in range(S.GRID)
            if (gx, gy) != (0, 0)
        }
        state.scores = {1: 5.0, 2: 3.0}
        state.check_win()
        assert state.game_over is True
        assert state.winner_player == 1

    def test_check_win_not_triggered_below_threshold(self):
        state.scores = {1: 1.0, 2: 0.0}
        state.check_win()
        assert state.game_over is False


# ── Snapshot / replay helpers ─────────────────────────────────────────────────

class TestSnapshots:
    def test_snapshot_grows_with_moves(self):
        state.place(0, 0)
        state.place(1, 0)
        # index 0 = empty board, +1 per move
        assert len(state.snapshots) == 3

    def test_snapshot_captures_board(self):
        state.place(2, 3)
        snap = state.snapshots[-1]
        assert (2, 3) in snap['board']

    def test_replay_step_back(self):
        state.place(0, 0)
        state.place(1, 0)
        state.replay_step(-1)          # go back one move
        assert (1, 0) not in state.board

    def test_replay_step_forward(self):
        state.place(0, 0)
        state.place(1, 0)
        state.replay_step(-1)
        state.replay_step(1)           # back to latest
        assert (1, 0) in state.board

    def test_snapshot_index_returns_to_live(self):
        state.place(0, 0)
        state.replay_step(-1)
        state.replay_step(1)
        assert state.snapshot_index == -1

    def test_mid_replay_new_place_truncates_future(self):
        state.place(0, 0)   # move 1
        state.place(1, 0)   # move 2
        state.replay_step(-1)          # back to after move 1
        state.place(2, 0)              # branch: overwrites move 2
        # Now there should be 3 snapshots (empty + move1 + new move2)
        assert len(state.snapshots) == 3
        assert (1, 0) not in state.board   # old move 2 gone


# ── _capture / _restore ───────────────────────────────────────────────────────

class TestCaptureRestore:
    def test_capture_includes_forbidden(self):
        state.forbidden_positions = {(3, 3)}
        snap = state._capture()
        assert (3, 3) in snap['forbidden_positions']

    def test_restore_sets_forbidden(self):
        snap = state._capture()
        snap['forbidden_positions'] = {(7, 7)}
        state._restore(snap)
        assert (7, 7) in state.forbidden_positions

    def test_capture_deep_copies_board(self):
        state.board[(0, 0)] = 1
        snap = state._capture()
        snap['board'][(0, 0)] = 99   # mutate the snap copy
        assert state.board[(0, 0)] == 1  # original unchanged


# ── Integration: territory scoring via place() ────────────────────────────────

class TestTerritoryViaPlace:
    def test_four_corners_scores_one_full_square(self):
        """Closing a 2×2 ring must award exactly 1.0 to P1."""
        _build_p1_ring_at_origin()
        assert state.scores[1] == 1.0

    def test_opponent_dot_inside_ring_removed(self):
        """A P2 dot placed inside a subsequently closed P1 ring must be removed."""
        # P1 places 3 corners first, P2 sneaks inside, then P1 closes
        state.place(0, 0)   # P1
        state.place(0, 1)   # P2 — will end up inside P1's ring attempt
        # But actually (0,1) is a corner of the ring — let's use a larger ring.
        # Use a 3×3 ring: P1 at all 8 perimeter positions.
        state.reset()
        _build_p1_3x3_ring_with_interior_p2()
        # The P2 dot at (1,1) (interior) should have been removed
        assert (1, 1) not in state.board

    def test_interior_position_becomes_forbidden(self):
        """After encirclement, interior grid positions must enter forbidden_positions."""
        state.reset()
        _build_p1_3x3_ring_with_interior_p2()
        # (1,1) was P2's dot, now removed → that grid position should be forbidden
        assert (1, 1) in state.forbidden_positions


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_p1_ring_at_origin():
    """Place P1 at (0,0),(1,0),(0,1),(1,1) with P2 moves elsewhere."""
    moves_p1 = [(0, 0), (1, 0), (0, 1), (1, 1)]
    moves_p2 = [(9, 9), (9, 8), (9, 7), (9, 6)]
    for m1, m2 in zip(moves_p1, moves_p2):
        state.place(*m1)
        state.place(*m2)


def _build_p1_3x3_ring_with_interior_p2():
    """Build a 3×3 P1 ring (8 perimeter dots) with P2 dot at center (1,1).

    Move sequence interleaves P1 ring-building with P2 moves.  P2 places at
    (1,1) early so it lands inside the eventually-closed ring.
    """
    # Place P1 corners and sides, P2 at far corner then (1,1)
    seq = [
        (0, 0, 1), (9, 9, 2),
        (1, 0, 1), (9, 8, 2),
        (2, 0, 1), (9, 7, 2),
        (2, 1, 1), (9, 6, 2),
        (2, 2, 1), (9, 5, 2),
        (1, 2, 1), (9, 4, 2),
        (0, 2, 1), (9, 3, 2),
        (0, 1, 1), (1, 1, 2),   # P2 lands inside
    ]
    for gx, gy, expected_player in seq:
        assert state.current_player == expected_player, (
            f"Expected P{expected_player} to move, got P{state.current_player}")
        state.place(gx, gy)
