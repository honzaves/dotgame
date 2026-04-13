"""Unit tests for FastBoard in ai/mcts.py — the lightweight MCTS simulator."""
import pytest
import settings as S
from ai.mcts import FastBoard


N = S.GRID


# ── Helpers ───────────────────────────────────────────────────────────────────

def empty_fb(player=1):
    return FastBoard({}, set(), player, {1: 0.0, 2: 0.0})


def fb_with(board, connections=None, player=1, scores=None):
    return FastBoard(
        board,
        connections or set(),
        player,
        scores or {1: 0.0, 2: 0.0},
    )


# ── Construction ──────────────────────────────────────────────────────────────

class TestConstruction:
    def test_board_copied(self):
        d = {(0, 0): 1}
        fb = fb_with(d)
        d[(1, 1)] = 2
        assert (1, 1) not in fb.board

    def test_connections_copied(self):
        conns = {frozenset({(0, 0), (1, 0)})}
        fb = fb_with({}, conns)
        conns.add(frozenset({(2, 2), (3, 3)}))
        assert frozenset({(2, 2), (3, 3)}) not in fb.connections

    def test_forbidden_defaults_empty(self):
        fb = empty_fb()
        assert fb.forbidden == set()

    def test_forbidden_copied(self):
        forbidden = {(5, 5)}
        fb = FastBoard({}, set(), 1, {1: 0.0, 2: 0.0}, forbidden)
        forbidden.add((6, 6))
        assert (6, 6) not in fb.forbidden


# ── copy() ────────────────────────────────────────────────────────────────────

class TestCopy:
    def test_deep_copy_board(self):
        fb = fb_with({(0, 0): 1})
        copy = fb.copy()
        copy.board[(1, 1)] = 2
        assert (1, 1) not in fb.board

    def test_deep_copy_connections(self):
        conns = {frozenset({(0, 0), (1, 0)})}
        fb = fb_with({}, conns)
        copy = fb.copy()
        copy.connections.add(frozenset({(2, 2), (3, 3)}))
        assert frozenset({(2, 2), (3, 3)}) not in fb.connections

    def test_deep_copy_scores(self):
        fb = fb_with({}, scores={1: 5.0, 2: 3.0})
        copy = fb.copy()
        copy.scores[1] = 999.0
        assert fb.scores[1] == 5.0


# ── legal_moves() ─────────────────────────────────────────────────────────────

class TestLegalMoves:
    def test_empty_board_all_positions(self):
        fb = empty_fb()
        moves = fb.legal_moves()
        assert len(moves) == N * N

    def test_excludes_occupied(self):
        fb = fb_with({(0, 0): 1})
        moves = fb.legal_moves()
        assert (0, 0) not in moves
        assert len(moves) == N * N - 1

    def test_excludes_forbidden(self):
        fb = FastBoard({}, set(), 1, {1: 0.0, 2: 0.0}, {(3, 3)})
        moves = fb.legal_moves()
        assert (3, 3) not in moves

    def test_excludes_both_occupied_and_forbidden(self):
        fb = FastBoard({(0, 0): 1}, set(), 1, {1: 0.0, 2: 0.0}, {(1, 1)})
        moves = fb.legal_moves()
        assert (0, 0) not in moves
        assert (1, 1) not in moves
        assert len(moves) == N * N - 2

    def test_full_board_no_moves(self):
        board = {(gx, gy): 1 for gx in range(N) for gy in range(N)}
        fb = fb_with(board)
        assert fb.legal_moves() == []


# ── closing_moves() ───────────────────────────────────────────────────────────

class TestClosingMoves:
    def test_no_closing_on_empty_board(self):
        fb = empty_fb()
        assert fb.closing_moves(1) == []

    def test_detects_closing_move(self):
        """Three P1 dots at unit-square corners → the 4th is a closing move."""
        board = {(0, 0): 1, (1, 0): 1, (0, 1): 1}
        fb = fb_with(board)
        moves = fb.closing_moves(1)
        assert (1, 1) in moves

    def test_sorted_by_score_descending(self):
        """Higher-scoring positions should come first."""
        # Create two closing opportunities: one square (1.0) and one triangle (0.5)
        board = {
            (0, 0): 1, (1, 0): 1, (0, 1): 1,   # closing (1,1) scores 1.0
        }
        fb = fb_with(board)
        moves = fb.closing_moves(1)
        assert moves[0] == (1, 1)

    def test_does_not_include_occupied(self):
        board = {(0, 0): 1, (1, 0): 1, (0, 1): 1, (1, 1): 1}
        fb = fb_with(board)
        moves = fb.closing_moves(1)
        assert (1, 1) not in moves


# ── blocking_moves() ──────────────────────────────────────────────────────────

class TestBlockingMoves:
    def test_no_blocking_on_empty_board(self):
        fb = empty_fb()
        assert fb.blocking_moves(1) == []

    def test_detects_opponent_threat(self):
        """Three P2 dots → the 4th position is a blocking move for P1."""
        board = {(0, 0): 2, (1, 0): 2, (0, 1): 2}
        fb = fb_with(board)
        moves = fb.blocking_moves(1)
        assert (1, 1) in moves


# ── adjacent_own() ────────────────────────────────────────────────────────────

class TestAdjacentOwn:
    def test_empty_board_no_adjacent(self):
        fb = empty_fb()
        assert fb.adjacent_own(1) == []

    def test_detects_adjacent_empty(self):
        board = {(5, 5): 1}
        fb = fb_with(board)
        adj = fb.adjacent_own(1)
        # All 8 Chebyshev neighbours of (5,5) should be in adj
        expected = [
            (4, 4), (5, 4), (6, 4),
            (4, 5),         (6, 5),
            (4, 6), (5, 6), (6, 6),
        ]
        for pos in expected:
            assert pos in adj

    def test_excludes_occupied_positions(self):
        board = {(5, 5): 1, (5, 6): 1}
        fb = fb_with(board)
        adj = fb.adjacent_own(1)
        assert (5, 5) not in adj
        assert (5, 6) not in adj

    def test_excludes_forbidden_positions(self):
        board = {(5, 5): 1}
        fb = FastBoard(board, set(), 1, {1: 0.0, 2: 0.0}, {(5, 6)})
        adj = fb.adjacent_own(1)
        assert (5, 6) not in adj

    def test_excludes_opponent_adjacent(self):
        """Adjacent positions are adjacent to OWN dots, not opponent."""
        board = {(5, 5): 2}
        fb = fb_with(board)
        # No P1 dots → adj_own(1) should be empty
        assert fb.adjacent_own(1) == []


# ── setup_moves() ─────────────────────────────────────────────────────────────

class TestSetupMoves:
    def test_empty_board_no_setups(self):
        fb = empty_fb()
        assert fb.setup_moves(1) == []

    def test_detects_setup_position(self):
        """Two own dots → a third position creating a 3-corner setup detected."""
        board = {(0, 0): 1, (1, 0): 1}
        fb = fb_with(board)
        setups = fb.setup_moves(1)
        # Placing at (0,1): cell(0,0) gets (0,0),(1,0),(0,1) = 3 own → setup
        assert (0, 1) in setups

    def test_no_setup_if_opp_in_cell(self):
        """Opponent in the cell prevents a setup there."""
        board = {(0, 0): 1, (1, 0): 1, (1, 1): 2}
        fb = fb_with(board)
        setups = fb.setup_moves(1)
        # (0,1) in cell(0,0): corners (0,0)(1,0)(0,1)(1,1), opp at (1,1) → not a setup
        assert (0, 1) not in setups


# ── fork_moves() ──────────────────────────────────────────────────────────────

class TestForkMoves:
    def test_empty_board_no_forks(self):
        fb = empty_fb()
        assert fb.fork_moves(1) == []

    def test_single_threat_not_a_fork(self):
        board = {(0, 0): 1, (1, 0): 1}
        fb = fb_with(board)
        # Only one cell can be threatened; not a fork
        forks = fb.fork_moves(1)
        assert (0, 1) not in forks   # only 1 threat at (0,1)

    def test_two_threats_is_a_fork(self):
        """Position creating 2+ simultaneous threats qualifies as a fork."""
        # (1,1) is a corner of cells (0,0),(1,0),(0,1),(1,1).
        # Give 2 own corners in two different cells, each needing (1,1).
        board = {
            (0, 0): 1, (1, 0): 1,   # cell(0,0): placing(1,1) creates 3-own+empty(0,1) threat
            (2, 1): 1, (2, 2): 1,   # not helpful for (1,1)
        }
        # Better: use a position that creates threats in two cells simultaneously
        board2 = {
            (0, 0): 1, (1, 0): 1,   # cell(0,0): (1,1) would give 3 own + empty (0,1)
            (1, 2): 1, (2, 2): 1,   # cell(1,1): (1,1) would give 3 own + empty (2,1)... no
        }
        # Simplest fork: (1,1) as pivot with 2 own in cell(0,0) AND 2 own in cell(1,1)
        board3 = {
            (0, 0): 1, (1, 0): 1,   # cell(0,0): corners tl,tr own; (1,1) creates 3-own+empty(0,1)
            (2, 1): 1, (2, 2): 1,   # cell(1,1): corners (2,1),(2,2) → (1,1) creates (1,1),(2,1),(2,2)=3 + empty(1,2)
        }
        fb = fb_with(board3)
        forks = fb.fork_moves(1)
        assert (1, 1) in forks


# ── play() ────────────────────────────────────────────────────────────────────

class TestPlay:
    def test_places_dot(self):
        fb = empty_fb()
        fb.play(3, 4)
        assert fb.board[(3, 4)] == 1

    def test_switches_player(self):
        fb = empty_fb(player=1)
        fb.play(0, 0)
        assert fb.player == 2

    def test_switches_back(self):
        fb = empty_fb(player=1)
        fb.play(0, 0)
        fb.play(1, 0)
        assert fb.player == 1

    def test_builds_connection_to_adjacent_same_player(self):
        fb = empty_fb()
        fb.play(0, 0)   # P1
        fb.play(9, 9)   # P2 far away
        fb.play(1, 0)   # P1 adjacent to (0,0)
        assert frozenset({(0, 0), (1, 0)}) in fb.connections

    def test_no_connection_to_opponent(self):
        fb = empty_fb()
        fb.play(0, 0)   # P1
        fb.play(1, 0)   # P2 adjacent
        assert frozenset({(0, 0), (1, 0)}) not in fb.connections

    def test_scores_updated_on_square_close(self):
        """Closing a unit square increments the closing player's score."""
        fb = empty_fb()
        fb.play(0, 0)   # P1
        fb.play(9, 9)   # P2
        fb.play(1, 0)   # P1
        fb.play(9, 8)   # P2
        fb.play(0, 1)   # P1
        fb.play(9, 7)   # P2
        fb.play(1, 1)   # P1 closes cell(0,0)
        assert fb.scores[1] >= 1.0

    def test_diagonal_connection_built(self):
        fb = empty_fb()
        fb.play(0, 0)   # P1
        fb.play(9, 9)   # P2
        fb.play(1, 1)   # P1 diagonal to (0,0)
        assert frozenset({(0, 0), (1, 1)}) in fb.connections


# ── _check_terminal() ─────────────────────────────────────────────────────────

class TestCheckTerminal:
    def test_not_done_on_empty(self):
        fb = empty_fb()
        fb._check_terminal()
        assert fb.done is False

    def test_player1_wins_on_threshold(self):
        threshold = S.WIN_PCT * (N - 1) ** 2
        fb = fb_with({}, scores={1: threshold, 2: 0.0})
        fb._check_terminal()
        assert fb.done is True
        assert fb.winner == 1

    def test_player2_wins_on_threshold(self):
        threshold = S.WIN_PCT * (N - 1) ** 2
        fb = fb_with({}, scores={1: 0.0, 2: threshold})
        fb._check_terminal()
        assert fb.done is True
        assert fb.winner == 2

    def test_board_full_triggers_terminal(self):
        """Fast mode: board full (len(board) >= N²) means no moves left."""
        board = {(gx, gy): 1 for gx in range(N) for gy in range(N)}
        fb = fb_with(board, scores={1: 5.0, 2: 3.0})
        fb._check_terminal()
        assert fb.done is True

    def test_board_full_winner_by_score(self):
        board = {(gx, gy): 1 for gx in range(N) for gy in range(N)}
        fb = fb_with(board, scores={1: 10.0, 2: 5.0})
        fb._check_terminal()
        assert fb.winner == 1

    def test_below_threshold_not_done(self):
        fb = fb_with({}, scores={1: 1.0, 2: 0.0})
        fb._check_terminal()
        assert fb.done is False


# ── _update_scores_fast() ─────────────────────────────────────────────────────

class TestUpdateScoresFast:
    def test_no_square_no_score(self):
        fb = fb_with({(0, 0): 1, (1, 0): 1})
        fb._update_scores_fast(1, 0, 1)   # place (1,0) for P1 — only 2 corners
        assert fb.scores[1] == 0.0

    def test_four_own_corners_scores_one(self):
        """All four corners of a unit square owned by P → score += 1.0 per square."""
        board = {(0, 0): 1, (1, 0): 1, (0, 1): 1}
        fb = fb_with(board)
        fb.board[(1, 1)] = 1   # place the 4th dot directly
        fb._update_scores_fast(1, 1, 1)
        assert fb.scores[1] == 1.0

    def test_multiple_squares_closed_simultaneously(self):
        """A dot can be a corner of up to 4 unit squares — all count."""
        board = {
            (0, 0): 1, (1, 0): 1, (0, 1): 1,   # cell(0,0) — needs (1,1)
            (2, 0): 1, (2, 1): 1,               # cell(1,0) — needs (1,0) and (1,1)
        }
        # cell(1,0): corners (1,0),(2,0),(1,1),(2,1). Need all 4 to be P1.
        board[(1, 0)] = 1
        fb = fb_with(board)
        fb.board[(1, 1)] = 1
        fb._update_scores_fast(1, 1, 1)
        # cell(0,0) closed (all four P1) + cell(1,0) closed (all four P1) = 2.0
        assert fb.scores[1] == 2.0
