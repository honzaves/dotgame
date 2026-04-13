"""Unit tests for territory.py — raster engine, flood fill, encirclement."""
import pytest
import state
import settings as S
import territory


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def fresh_state():
    state.reset()
    yield
    state.reset()


SCALE = territory.SCALE


# ── _build_walls() ────────────────────────────────────────────────────────────

class TestBuildWalls:
    def test_empty_board_no_walls(self):
        walls = territory._build_walls(1)
        assert walls == set()

    def test_dot_creates_wall_pixels(self):
        state.board[(2, 3)] = 1
        walls = territory._build_walls(1)
        # The scaled centre pixel must be in walls
        assert (SCALE * 2, SCALE * 3) in walls

    def test_wall_covers_3x3_block(self):
        """Each dot expands to at most a 3×3 block (9 pixels) around its scaled coord."""
        state.board[(3, 3)] = 1
        walls = territory._build_walls(1)
        sx, sy = SCALE * 3, SCALE * 3
        for ddx in range(-1, 2):
            for ddy in range(-1, 2):
                assert (sx + ddx, sy + ddy) in walls

    def test_no_negative_pixels_for_corner_dot(self):
        """3×3 expansion of dot at (0,0) must not write to pixel (-1, …) — critical
        for flood fill correctness: pixel (-1,-1) is the flood seed."""
        state.board[(0, 0)] = 1
        walls = territory._build_walls(1)
        neg = [(x, y) for (x, y) in walls if x < 0 or y < 0]
        assert neg == [], f"Negative wall pixels found: {neg}"

    def test_only_player_dots_create_walls(self):
        """Walls for player 1 must ignore player 2 dots."""
        state.board[(1, 1)] = 1
        state.board[(2, 2)] = 2
        walls1 = territory._build_walls(1)
        walls2 = territory._build_walls(2)
        assert (SCALE * 1, SCALE * 1) in walls1
        assert (SCALE * 2, SCALE * 2) not in walls1
        assert (SCALE * 2, SCALE * 2) in walls2

    def test_connection_fills_pixels_between_dots(self):
        """A horizontal connection between (0,0) and (1,0) must rasterise the gap."""
        state.board[(0, 0)] = 1
        state.board[(1, 0)] = 1
        state.connections.add(frozenset({(0, 0), (1, 0)}))
        walls = territory._build_walls(1)
        # Midpoint pixel between scaled (0,0)=(0,0) and scaled (1,0)=(6,0) is (3,0)
        assert (3, 0) in walls

    def test_diagonal_connection_seals_l_corners(self):
        """Diagonal connections must receive L-corner fills so 4-directional flood
        fill cannot slip through the gap at the bend."""
        state.board[(0, 0)] = 1
        state.board[(1, 1)] = 1
        state.connections.add(frozenset({(0, 0), (1, 1)}))
        walls = territory._build_walls(1)
        # First diagonal step goes from (0,0) to (1,1) in scaled space.
        # L-corner fills must appear at (1,0) and (0,1)
        assert (1, 0) in walls
        assert (0, 1) in walls


# ── _flood_outside() ──────────────────────────────────────────────────────────

class TestFloodOutside:
    def test_empty_walls_reaches_entire_border(self):
        sz = SCALE * (S.GRID - 1)
        outside = territory._flood_outside(set(), sz)
        # Seed (-1,-1) must always be outside
        assert (-1, -1) in outside
        # Corners of the extended region must be reachable
        assert (sz + 1, sz + 1) in outside

    def test_walls_block_flood(self):
        """A complete wall enclosure prevents flood from reaching the interior."""
        sz = 10   # small synthetic raster
        # Build a 3×3 box of walls that completely encloses pixel (5,5)
        walls = set()
        for x in range(3, 8):
            walls.add((x, 3))   # top
            walls.add((x, 7))   # bottom
        for y in range(3, 8):
            walls.add((3, y))   # left
            walls.add((7, y))   # right
        outside = territory._flood_outside(walls, sz)
        # Interior pixel should NOT be reachable from (-1,-1)
        assert (5, 5) not in outside

    def test_seed_always_outside(self):
        sz = 20
        outside = territory._flood_outside(set(), sz)
        assert (-1, -1) in outside

    def test_wall_pixel_not_in_outside(self):
        sz = 10
        walls = {(5, 5)}
        outside = territory._flood_outside(walls, sz)
        assert (5, 5) not in outside


# ── recompute_territories() ───────────────────────────────────────────────────

class TestRecomputeTerritories:
    def test_empty_board_zero_scores(self):
        territory.recompute_territories()
        assert state.scores == {1: 0.0, 2: 0.0}
        assert state.territories == {}

    def test_no_territory_with_isolated_dots(self):
        """Isolated dots not forming any enclosure must score nothing."""
        state.board[(0, 0)] = 1
        state.board[(5, 5)] = 1
        territory.recompute_territories()
        assert state.scores[1] == 0.0

    def test_2x2_ring_scores_one_full_square(self):
        """Four P1 dots at unit-square corners with all connections → 1.0 for P1."""
        _place_2x2_ring_p1()
        territory.recompute_territories()
        assert state.scores[1] == 1.0
        assert state.scores[2] == 0.0

    def test_2x2_ring_territory_key_present(self):
        _place_2x2_ring_p1()
        territory.recompute_territories()
        assert (0, 0) in state.territories
        owner, shape, _ = state.territories[(0, 0)]
        assert owner == 1
        assert shape == 'full'

    def test_encircled_dot_removed(self):
        """A P2 dot trapped inside a complete P1 ring must be deleted from the board."""
        _place_p1_ring_with_p2_inside()
        territory.recompute_territories()
        assert (1, 1) not in state.board

    def test_forbidden_positions_updated(self):
        """Grid positions strictly inside enclosed territory must become forbidden."""
        _place_p1_ring_with_p2_inside()
        territory.recompute_territories()
        # The trapped interior position should now be forbidden
        assert (1, 1) in state.forbidden_positions

    def test_no_forbidden_for_open_board(self):
        state.board[(0, 0)] = 1
        territory.recompute_territories()
        assert len(state.forbidden_positions) == 0

    def test_interior_dots_detected(self):
        """Dots completely surrounded by full territory of one owner are interior."""
        # Build 3×3 P1 ring so (1,1) becomes fully surrounded
        _place_p1_3x3_ring()
        territory.recompute_territories()
        # The ring closes 4 unit cells; (1,1) is surrounded on all sides by P1 territory
        assert (1, 1) in state.interior_dots

    def test_scores_are_floats(self):
        _place_2x2_ring_p1()
        territory.recompute_territories()
        assert isinstance(state.scores[1], float)

    def test_rebuild_after_encirclement(self):
        """After removing encircled dots the walls/fills are rebuilt — step 4 of the
        encirclement sequence. Verify scores reflect the post-removal geometry."""
        _place_p1_ring_with_p2_inside()
        territory.recompute_territories()
        # P1 should have positive score after encirclement
        assert state.scores[1] > 0.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def _connect_all_adjacent(dots, player):
    """Add connections between all Chebyshev-adjacent same-player dots."""
    dot_list = list(dots)
    for i, a in enumerate(dot_list):
        for b in dot_list[i + 1:]:
            if max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1:
                state.connections.add(frozenset({a, b}))


def _place_2x2_ring_p1():
    """Directly plant a P1 2×2 ring at (0,0)-(1,1) with all connections."""
    dots = [(0, 0), (1, 0), (0, 1), (1, 1)]
    for d in dots:
        state.board[d] = 1
    _connect_all_adjacent(dots, 1)


def _place_p1_ring_with_p2_inside():
    """3×3 P1 ring with P2 dot at interior (1,1)."""
    ring = [(0,0),(1,0),(2,0),(2,1),(2,2),(1,2),(0,2),(0,1)]
    for d in ring:
        state.board[d] = 1
    _connect_all_adjacent(ring, 1)
    state.board[(1, 1)] = 2


def _place_p1_3x3_ring():
    """3×3 P1 ring without any opponent dot inside."""
    ring = [(0,0),(1,0),(2,0),(2,1),(2,2),(1,2),(0,2),(0,1)]
    for d in ring:
        state.board[d] = 1
    _connect_all_adjacent(ring, 1)
