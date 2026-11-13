"""Tests for ai/rings.py — ring-race analysis."""
import numpy as np

import settings as S
from ai import rings
from ai.engine import BitGame, geo


def _mask(points, W):
    m = 0
    for gx, gy in points:
        m |= 1 << (gy * W + gx)
    return m


class TestProfile:
    def test_empty_board_profile(self):
        info = rings.analyse(0, 0, rings.centres()[0])
        p = info.profile
        assert p[0] == p[2] == 0.0
        assert p[3] == 0.5            # triangle around the centre
        assert p[4] == 2.0            # 4-dot diamond encloses 2 cells
        assert (np.diff(p) >= 0).all()

    def test_winning_octagon_distance_on_10x10(self):
        if S.GRID != 10:
            return
        info = rings.analyse(0, 0, rings.centres()[0])
        assert info.moves_to(S.WIN_PCT * 81) == 19

    def test_own_dots_are_free(self):
        W = geo().W
        cg = rings.centres()[0]
        base = rings.analyse(0, 0, cg).moves_to(8.0)
        # a 3x3-point square ring around the centre cell block, half built
        own = _mask([(4, 3), (5, 3), (6, 4), (6, 5)], W)
        assert rings.analyse(own, 0, cg).moves_to(8.0) < base

    def test_blocked_points_never_cheaper(self):
        W = geo().W
        cg = rings.centres()[0]
        free = rings.analyse(0, 0, cg).profile
        blocked = rings.analyse(0, _mask([(5, 3), (3, 5), (6, 6)], W), cg).profile
        assert (blocked <= free + 1e-6).all()

    def test_unreachable_area_reports_never(self):
        info = rings.analyse(0, 0, rings.centres()[0])
        assert info.moves_to(10_000) == rings.NEVER


class TestAgainstEngine:
    def test_closed_ring_area_matches_engine_score(self):
        """A complete ring costs 0 and its DP area equals the real score."""
        bg = BitGame()
        W = bg.g.W
        ring = [(4, 3), (5, 3), (6, 4), (6, 5), (5, 6), (4, 6), (3, 5), (3, 4)]
        # Build P1's ring on a board where P2 never interferes
        d1 = _mask(ring, W)
        probe = BitGame.from_board({p: 1 for p in ring}, 2)
        info = rings.analyse(d1, 0, rings.centres()[0])
        assert info.profile[0] == probe.s1 > 0

    def test_owned_territory_not_counted_again(self):
        ring = [(4, 3), (5, 3), (6, 4), (6, 5), (5, 6), (4, 6), (3, 5), (3, 4)]
        probe = BitGame.from_board({p: 1 for p in ring}, 2)
        info = rings.analyse(probe.d1, probe.forb, rings.centres()[0],
                             owned=probe.enclosed(1))
        assert info.profile[0] == 0.0        # nothing NEW to gain for free


class TestThroughPoint:
    def test_gap_point_is_on_cheapest_ring(self):
        W = geo().W
        ring = [(4, 3), (5, 3), (6, 4), (6, 5), (5, 6), (4, 6), (3, 5), (3, 4)]
        gap = ring[2]
        own = _mask([p for p in ring if p != gap], W)
        cg = rings.centres()[0]
        info = rings.analyse(own, 0, cg, backward=True)
        area = BitGame.from_board({p: 1 for p in ring}, 2).s1
        assert info.moves_to(area) == 1
        costs = rings.through_costs(rings.through_table(info), area)
        by_point = {cg.order[r]: int(c) for r, c in enumerate(costs)}
        assert by_point[gap[1] * W + gap[0]] == 1

    def test_through_cost_never_below_global_cost(self):
        cg = rings.centres()[0]
        info = rings.analyse(0, 0, cg, backward=True)
        target = 12.0
        costs = rings.through_costs(rings.through_table(info), target)
        assert costs.min() == info.moves_to(target)
