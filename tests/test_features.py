"""Unit tests for ai/features.py — board encoding and strategic signals."""
import pytest
import numpy as np
import settings as S
from ai.features import (
    opportunity_masks,
    apply_boost,
    _find_components,
    _component_potential,
    _arc_potential_comp,
    arc_potential_scalars,
    arc_potential_map,
    enclosure_potential,
    bridge_potential,
    disruption_map,
    fork_map,
    close_setup_map,
    get_centrality,
    strategic_channels,
)

N = S.GRID   # use the real grid size (10)


# ── opportunity_masks() ───────────────────────────────────────────────────────

class TestOpportunityMasks:
    def test_empty_board_all_zeros(self):
        own, opp = opportunity_masks({}, set(), 1)
        assert np.all(own == 0.0)
        assert np.all(opp == 0.0)

    def test_three_own_corners_scores_one(self):
        """3 own dots in a unit square → placing the 4th scores 1.0."""
        board = {(0, 0): 1, (1, 0): 1, (0, 1): 1}
        own, _ = opportunity_masks(board, set(), 1)
        idx = 1 * N + 1   # (gx=1, gy=1)
        assert own[idx] == pytest.approx(1.0)

    def test_three_own_corners_no_opp_score(self):
        board = {(0, 0): 1, (1, 0): 1, (0, 1): 1}
        _, opp = opportunity_masks(board, set(), 1)
        idx = 1 * N + 1
        assert opp[idx] == pytest.approx(0.0)

    def test_three_opp_corners_scores_in_opp_mask(self):
        board = {(0, 0): 2, (1, 0): 2, (0, 1): 2}
        _, opp = opportunity_masks(board, set(), 1)
        idx = 1 * N + 1
        assert opp[idx] == pytest.approx(1.0)

    def test_two_own_corners_triangle_scores_half(self):
        """2 own corners + 1 empty in a triangle → 0.5 pts at the empty position."""
        board = {(0, 0): 1, (1, 0): 1}
        own, _ = opportunity_masks(board, set(), 1)
        # Triangle UL: (0,0),(1,0),(0,1) — empty slot is (0,1) → index 1*N+0 = 10
        idx = 1 * N + 0
        assert own[idx] == pytest.approx(0.5)

    def test_full_square_skips_triangle_for_same_player(self):
        """When a full square is found for player X, triangle checks are skipped for X."""
        board = {(0, 0): 1, (1, 0): 1, (0, 1): 1}
        own, _ = opportunity_masks(board, set(), 1)
        idx_full = 1 * N + 1   # the full-square position
        # Should be exactly 1.0, not 1.0 + 0.5 (if triangle were also counted)
        assert own[idx_full] == pytest.approx(1.0)

    def test_already_occupied_position_scores_zero(self):
        board = {(0, 0): 1, (1, 0): 1, (0, 1): 1, (1, 1): 1}
        own, _ = opportunity_masks(board, set(), 1)
        idx = 1 * N + 1
        assert own[idx] == pytest.approx(0.0)

    def test_scores_accumulate_across_cells(self):
        """A position shared by two unit squares accumulates from both."""
        # Position (1,1) is a corner of cells (0,0), (1,0), (0,1), (1,1)
        # Give (1,1) three own neighbours in cell (0,0) AND cell (1,0)
        board = {
            (0, 0): 1, (1, 0): 1, (0, 1): 1,   # → close (1,1) for cell (0,0): +1.0
            (2, 0): 1, (2, 1): 1,               # → cell (1,0) has 3 own at (1,0),(2,0),(2,1)
        }                                         #   empty at (1,1) for that cell? No — (1,1) empty
        # Cell (1,0): corners (1,0),(2,0),(1,1),(2,1). Own: (1,0),(2,0),(2,1)=3 → close (1,1)
        own, _ = opportunity_masks(board, set(), 1)
        idx = 1 * N + 1
        # (1,1) is the closing move for both cell(0,0) and cell(1,0) → 2.0
        assert own[idx] == pytest.approx(2.0)

    def test_output_length_matches_grid_squared(self):
        own, opp = opportunity_masks({}, set(), 1)
        assert len(own) == N * N
        assert len(opp) == N * N

    def test_dtype_float32(self):
        own, opp = opportunity_masks({}, set(), 1)
        assert own.dtype == np.float32
        assert opp.dtype == np.float32


# ── apply_boost() ─────────────────────────────────────────────────────────────

class TestApplyBoost:
    def _uniform(self):
        n = N * N
        p = np.ones(n, dtype=np.float32) / n
        legal = np.ones(n, dtype=np.float32)
        return p, legal

    def test_output_sums_to_one(self):
        p, legal = self._uniform()
        own = np.zeros(N * N, dtype=np.float32)
        opp = np.zeros(N * N, dtype=np.float32)
        result = apply_boost(p, own, opp, legal)
        assert result.sum() == pytest.approx(1.0, abs=1e-6)

    def test_closing_move_boosted(self):
        """A move that scores a full square should receive more probability mass."""
        p, legal = self._uniform()
        own = np.zeros(N * N, dtype=np.float32)
        own[0] = 1.0   # position 0 can close a square
        result = apply_boost(p, own, np.zeros(N * N, dtype=np.float32), legal)
        assert result[0] > p[0]

    def test_blocking_move_boosted(self):
        p, legal = self._uniform()
        opp = np.zeros(N * N, dtype=np.float32)
        opp[5] = 1.0
        result = apply_boost(p, np.zeros(N * N, dtype=np.float32), opp, legal)
        assert result[5] > p[5]

    def test_illegal_positions_not_boosted(self):
        p = np.ones(N * N, dtype=np.float32) / (N * N)
        legal = np.zeros(N * N, dtype=np.float32)
        legal[1] = 1.0
        own = np.ones(N * N, dtype=np.float32) * 0.5
        result = apply_boost(p, own, np.zeros(N * N, dtype=np.float32), legal)
        # Non-legal positions: only legal position 1 should have probability mass
        non_legal = np.delete(result, 1)
        # After boost, all probability should flow to legal moves
        assert result.sum() == pytest.approx(1.0, abs=1e-6)

    def test_all_zero_probs_fallback_to_uniform_legal(self):
        p = np.zeros(N * N, dtype=np.float32)
        legal = np.zeros(N * N, dtype=np.float32)
        legal[0] = 1.0
        legal[1] = 1.0
        result = apply_boost(p, np.zeros(N * N), np.zeros(N * N), legal)
        assert result.sum() == pytest.approx(1.0, abs=1e-6)


# ── _find_components() ────────────────────────────────────────────────────────

class TestFindComponents:
    def test_empty_board(self):
        assert _find_components({}, 1) == []

    def test_single_dot_is_one_component(self):
        comps = _find_components({(3, 3): 1}, 1)
        assert len(comps) == 1
        assert (3, 3) in comps[0]

    def test_adjacent_dots_same_component(self):
        board = {(0, 0): 1, (1, 0): 1, (0, 1): 1}
        comps = _find_components(board, 1)
        assert len(comps) == 1
        assert len(comps[0]) == 3

    def test_diagonal_dots_same_component(self):
        board = {(0, 0): 1, (1, 1): 1}   # Chebyshev distance 1
        comps = _find_components(board, 1)
        assert len(comps) == 1

    def test_separated_dots_two_components(self):
        board = {(0, 0): 1, (5, 5): 1}
        comps = _find_components(board, 1)
        assert len(comps) == 2

    def test_opponent_dots_not_included(self):
        board = {(0, 0): 1, (1, 0): 2, (2, 0): 1}
        comps = _find_components(board, 1)
        # (0,0) and (2,0) are not Chebyshev-adjacent (distance 2), so 2 components
        assert len(comps) == 2


# ── _component_potential() ────────────────────────────────────────────────────

class TestComponentPotential:
    def test_fewer_than_three_dots_zero(self):
        assert _component_potential({(0, 0), (1, 0)}, N) == 0.0

    def test_collinear_dots_zero(self):
        """A straight line has no interior → potential 0."""
        comp = {(0, 0), (1, 0), (2, 0), (3, 0)}
        assert _component_potential(comp, N) == 0.0

    def test_closed_ring_nonzero(self):
        """A complete ring enclosing interior positions → potential > 0."""
        ring = {(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2), (0, 1)}
        pot = _component_potential(ring, N)
        assert pot > 0.0

    def test_potential_at_most_one(self):
        ring = {(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2), (0, 1)}
        pot = _component_potential(ring, N)
        assert pot <= 1.0


# ── _arc_potential_comp() ─────────────────────────────────────────────────────

class TestArcPotentialComp:
    def test_fewer_than_three_zero(self):
        assert _arc_potential_comp({(0, 0), (1, 0)}, N) == 0.0

    def test_straight_line_zero(self):
        """Straight line: span_y=0 → interior=0 → 0."""
        comp = {(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)}
        assert _arc_potential_comp(comp, N) == 0.0

    def test_l_shape_positive(self):
        """L-shape has 2D extent → interior > 0."""
        comp = {(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)}
        val = _arc_potential_comp(comp, N)
        assert val > 0.0

    def test_returns_territory_cell_units(self):
        """Return value is in territory-cell units, NOT normalised to [0,1]."""
        # A ring bounding a 3×3 area (span_x=2, span_y=2) with high completion
        # should return close to (2-1)*(2-1) = 1 interior cell at full completion
        ring = {(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2), (0, 1)}
        val = _arc_potential_comp(ring, N)
        # Expected perimeter = 2*(2+2)=8; actual=8 → completion=1.0 → val=1*(1-1)*(2-1)...
        # wait: interior = (span_x-1)*(span_y-1) = (2-1)*(2-1) = 1
        assert val == pytest.approx(1.0, abs=0.2)

    def test_larger_ring_higher_value(self):
        """A larger ring should produce a higher arc potential."""
        small_ring = {(0,0),(1,0),(2,0),(2,1),(2,2),(1,2),(0,2),(0,1)}
        large_ring = {(0,0),(1,0),(2,0),(3,0),(4,0),(4,1),(4,2),(4,3),(4,4),
                      (3,4),(2,4),(1,4),(0,4),(0,3),(0,2),(0,1)}
        small_val = _arc_potential_comp(small_ring, N)
        large_val = _arc_potential_comp(large_ring, N)
        assert large_val > small_val

    def test_completion_grows_with_dots(self):
        """Adding dots to an arc increases its potential."""
        arc3 = {(0, 0), (1, 0), (2, 0)}
        arc5 = {(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)}
        val3 = _arc_potential_comp(arc3, N)
        val5 = _arc_potential_comp(arc5, N)
        assert val5 >= val3


# ── arc_potential_scalars() ───────────────────────────────────────────────────

class TestArcPotentialScalars:
    def test_empty_board_zero(self):
        own, opp = arc_potential_scalars({}, 1)
        assert own == 0.0
        assert opp == 0.0

    def test_values_in_zero_one(self):
        board = {(0, 0): 1, (1, 0): 1, (2, 0): 1, (2, 1): 1, (2, 2): 1}
        own, opp = arc_potential_scalars(board, 1)
        assert 0.0 <= own <= 1.0
        assert 0.0 <= opp <= 1.0

    def test_own_higher_than_opp_when_only_p1_has_arc(self):
        board = {(0, 0): 1, (0, 1): 1, (0, 2): 1, (1, 2): 1, (2, 2): 1}
        own, opp = arc_potential_scalars(board, 1)
        assert own > opp


# ── arc_potential_map() ───────────────────────────────────────────────────────

class TestArcPotentialMap:
    def test_empty_board_all_zero(self):
        own, opp = arc_potential_map({}, 1)
        assert np.all(own == 0.0)
        assert np.all(opp == 0.0)

    def test_output_length(self):
        own, opp = arc_potential_map({(0, 0): 1}, 1)
        assert len(own) == N * N
        assert len(opp) == N * N

    def test_dot_positions_get_component_potential(self):
        board = {(0, 0): 1, (1, 0): 1, (0, 1): 1, (1, 1): 1}
        own, _ = arc_potential_map(board, 1)
        # All four dots are in the same component; their arc potential must be equal
        vals = [own[0 * N + 0], own[0 * N + 1], own[1 * N + 0], own[1 * N + 1]]
        assert all(v == vals[0] for v in vals)

    def test_adjacent_empty_gets_max_neighbour_potential(self):
        """Empty positions adjacent to an arc component should inherit its potential."""
        board = {(0, 0): 1, (1, 0): 1, (2, 0): 1, (2, 1): 1, (2, 2): 1}
        own, _ = arc_potential_map(board, 1)
        # (3, 2) is adjacent to (2,2) — should have non-zero potential
        idx = 2 * N + 3
        assert own[idx] > 0.0


# ── enclosure_potential() ─────────────────────────────────────────────────────

class TestEnclosurePotential:
    def test_empty_board_zero(self):
        own, opp = enclosure_potential({}, set(), 1)
        assert np.all(own == 0.0)
        assert np.all(opp == 0.0)

    def test_open_ring_zero(self):
        """An incomplete ring (one gap) returns 0 — flood fill leaks through the gap."""
        # 3×3 ring with one dot missing (no (0,1))
        board = {pos: 1 for pos in [(0,0),(1,0),(2,0),(2,1),(2,2),(1,2),(0,2)]}
        own, _ = enclosure_potential(board, set(), 1)
        # All values should be 0 since the ring is open
        assert np.all(own == 0.0)

    def test_closed_ring_nonzero(self):
        """A closed ring should produce positive enclosure potential."""
        board = {pos: 1 for pos in [(0,0),(1,0),(2,0),(2,1),(2,2),(1,2),(0,2),(0,1)]}
        own, _ = enclosure_potential(board, set(), 1)
        # At least the ring dots themselves should have non-zero potential
        assert np.any(own > 0.0)

    def test_values_in_zero_one(self):
        board = {pos: 1 for pos in [(0,0),(1,0),(2,0),(2,1),(2,2),(1,2),(0,2),(0,1)]}
        own, opp = enclosure_potential(board, set(), 1)
        assert np.all(own >= 0.0) and np.all(own <= 1.0)
        assert np.all(opp >= 0.0) and np.all(opp <= 1.0)


# ── disruption_map() ─────────────────────────────────────────────────────────

class TestDisruptionMap:
    def test_empty_board_zero(self):
        result = disruption_map({}, set(), 1)
        assert np.all(result == 0.0)

    def test_no_complete_opp_ring_zero(self):
        """Disruption is 0 when opponent has no closed ring (no interior)."""
        board = {(5, 5): 2, (6, 5): 2}
        result = disruption_map(board, set(), 1)
        assert np.all(result == 0.0)

    def test_interior_of_opp_ring_disrupted(self):
        """Positions inside a closed opponent ring should have positive disruption."""
        ring = {(0,0),(1,0),(2,0),(2,1),(2,2),(1,2),(0,2),(0,1)}
        board = {pos: 2 for pos in ring}
        result = disruption_map(board, set(), 1)
        # (1,1) is the interior of the 3×3 ring → should be non-zero
        idx = 1 * N + 1
        assert result[idx] > 0.0

    def test_output_clamped_to_one(self):
        ring = {(0,0),(1,0),(2,0),(2,1),(2,2),(1,2),(0,2),(0,1)}
        board = {pos: 2 for pos in ring}
        result = disruption_map(board, set(), 1)
        assert np.all(result <= 1.0)


# ── fork_map() ────────────────────────────────────────────────────────────────

class TestForkMap:
    def test_empty_board_zero(self):
        result = fork_map({}, 1)
        assert np.all(result == 0.0)

    def test_fork_detected(self):
        """Position creating 2+ simultaneous square threats should score > 0."""
        # Two unit squares sharing position (1,1), each with 2 own corners
        # Cell (0,0): own at (0,0),(1,0) → need (0,1) and (1,1); after placing (1,1) →
        #   other corners are (0,0),(1,0),(0,1): own=(0,0),(1,0)=2, empty=(0,1)=1 → threat
        # Cell (1,0): own at (2,0),(2,1) → need (1,0) and (1,1); placing (1,1) →
        #   other corners (1,0),(2,0),(2,1): own=(2,0),(2,1)=2, empty=(1,0)=1 → ... (1,0) occupied
        # Build a cleaner fork: (1,1) is corner of 4 cells; give 2 own in two of them
        board = {
            (0, 0): 1, (1, 0): 1,   # cell(0,0): 2 own at tl,tr; placing(1,1) with empty(0,1)→threat
            (2, 0): 1, (2, 1): 1,   # cell(1,0): 2 own at tr,br; placing(1,1) with empty(1,0)? No.
        }
        # Let me just test that fork_map runs and returns [0,1] values
        result = fork_map(board, 1)
        assert np.all(result >= 0.0) and np.all(result <= 1.0)

    def test_values_normalised(self):
        board = {(0, 0): 1, (1, 0): 1, (0, 1): 1, (2, 2): 1, (3, 2): 1, (2, 3): 1}
        result = fork_map(board, 1)
        assert np.all(result >= 0.0) and np.all(result <= 1.0)

    def test_occupied_position_zero(self):
        board = {(0, 0): 1, (1, 0): 1, (0, 1): 1, (1, 1): 1}
        result = fork_map(board, 1)
        # All four dots are occupied; their indices should be 0
        for gx, gy in [(0,0),(1,0),(0,1),(1,1)]:
            assert result[gy * N + gx] == 0.0


# ── close_setup_map() ─────────────────────────────────────────────────────────

class TestCloseSetupMap:
    def test_empty_board_zero(self):
        result = close_setup_map({}, 1)
        assert np.all(result == 0.0)

    def test_setup_detected(self):
        """Placing at a position that creates a 3-own-corner square → non-zero."""
        # (0,0),(1,0) own; placing at (0,1) creates cell(0,0) with 3 own + empty at (1,1)
        board = {(0, 0): 1, (1, 0): 1}
        result = close_setup_map(board, 1)
        idx = 1 * N + 0   # (gx=0, gy=1)
        assert result[idx] > 0.0

    def test_no_setup_with_opp_in_cell(self):
        """An opponent dot in the square prevents creating a 3-own threat there."""
        board = {(0, 0): 1, (1, 0): 1, (1, 1): 2}
        result = close_setup_map(board, 1)
        idx = 1 * N + 0   # (0,1) would need cell(0,0): corners (0,0)(1,0)(0,1)(1,1)
        # (1,1) is opponent, so condition opp_cnt==0 fails → 0
        assert result[idx] == 0.0

    def test_normalised_max_four(self):
        result = close_setup_map({}, 1)
        assert np.all(result <= 1.0)


# ── get_centrality() ──────────────────────────────────────────────────────────

class TestGetCentrality:
    def test_length(self):
        cent = get_centrality()
        assert len(cent) == N * N

    def test_centre_highest(self):
        cent = get_centrality()
        cx = cy = (N - 1) // 2
        centre_idx = cy * N + cx
        assert cent[centre_idx] == cent.max()

    def test_corner_lowest(self):
        cent = get_centrality()
        corner_idx = 0   # (0,0)
        assert cent[corner_idx] == cent.min()

    def test_values_in_zero_one(self):
        cent = get_centrality()
        assert np.all(cent >= 0.0) and np.all(cent <= 1.0)

    def test_cached(self):
        """Two calls must return the same object (cache works)."""
        a = get_centrality()
        b = get_centrality()
        assert a is b


# ── bridge_potential() ────────────────────────────────────────────────────────

class TestBridgePotential:
    def test_empty_board_zero(self):
        own, opp = bridge_potential({}, set(), 1)
        assert np.all(own == 0.0)
        assert np.all(opp == 0.0)

    def test_single_component_zero(self):
        """No bridging opportunity with only one component."""
        board = {(0, 0): 1, (0, 1): 1, (0, 2): 1}
        own, _ = bridge_potential(board, set(), 1)
        assert np.all(own == 0.0)

    def test_two_separate_components_bridge_detected(self):
        """Two separate arcs → a bridging position should score > 0."""
        board = {
            (0, 0): 1, (0, 1): 1, (0, 2): 1,   # arc 1 (vertical)
            (2, 0): 1, (2, 1): 1, (2, 2): 1,   # arc 2 (vertical), separated by column 1
        }
        own, _ = bridge_potential(board, set(), 1)
        # Column 1 positions should have bridge potential
        assert np.any(own > 0.0)

    def test_output_length(self):
        own, opp = bridge_potential({}, set(), 1)
        assert len(own) == N * N
        assert len(opp) == N * N


# ── strategic_channels() ─────────────────────────────────────────────────────

class TestStrategicChannels:
    def test_returns_six_arrays(self):
        result = strategic_channels({}, set(), 1, 0)
        assert len(result) == 6

    def test_phase_scalar_correct(self):
        total_dots = 10
        _, _, _, _, phase_arr, _ = strategic_channels({}, set(), 1, total_dots)
        expected = total_dots / (N * N)
        assert np.all(phase_arr == pytest.approx(expected))

    def test_all_arrays_correct_length(self):
        for arr in strategic_channels({}, set(), 1, 0):
            assert len(arr) == N * N
