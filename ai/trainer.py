"""Self-play trainer.

Improvements over the original:
* Computes per-move territory-delta rewards and passes them to both
  players' record_outcome() calls, enabling reward shaping.
* Adds arc-potential-delta shaping: rewards ring-building progress each step,
  not just at ring completion.  Prevents the NN/PT small-grab bias.
* Supports asymmetric matchups (e.g. NN vs MCTS) for curriculum training.
* SimGame uses the same flood-fill territory engine as the real game.
"""

import threading
import random
from typing import Callable

import settings as S
from ai.base_player import BasePlayer
from ai.features import arc_potential_scalars


# ─────────────────────────────────────────────────────────────────────────────
# Isolated game simulation (no global state, no pygame)
# ─────────────────────────────────────────────────────────────────────────────

_SCALE = 6


def _build_walls(board, connections, player, sz):
    walls: set = set()
    for gx, gy in board:
        if board[(gx, gy)] != player:
            continue
        sx, sy = _SCALE * gx, _SCALE * gy
        for ddx in range(-1, 2):
            for ddy in range(-1, 2):
                wx, wy = sx + ddx, sy + ddy
                if 0 <= wx <= sz and 0 <= wy <= sz:
                    walls.add((wx, wy))
    for conn in connections:
        a, b = tuple(conn)
        if board[a] != player:
            continue
        ax, ay = _SCALE * a[0], _SCALE * a[1]
        bx, by = _SCALE * b[0], _SCALE * b[1]
        dx, dy = bx - ax, by - ay
        steps  = max(abs(dx), abs(dy))
        prev   = (ax, ay)
        for i in range(1, steps + 1):
            cur = (ax + dx * i // steps, ay + dy * i // steps)
            walls.add(cur)
            px, py   = prev
            cx2, cy2 = cur
            if px != cx2 and py != cy2:
                walls.add((cx2, py))
                walls.add((px, cy2))
            prev = cur
    return walls


def _flood_outside(walls, sz):
    outside = {(-1, -1)}
    stack   = [(-1, -1)]
    while stack:
        sx, sy = stack.pop()
        for ddx, ddy in ((1,0),(-1,0),(0,1),(0,-1)):
            nx, ny = sx+ddx, sy+ddy
            pos = (nx, ny)
            if -1<=nx<=sz+1 and -1<=ny<=sz+1 and pos not in outside and pos not in walls:
                outside.add(pos)
                stack.append(pos)
    return outside


def _remove_encircled(board, connections, winner, w_outside, w_walls):
    to_remove = [
        (gx, gy) for gx, gy in board
        if (_SCALE*gx, _SCALE*gy) not in w_outside
        and not (board[(gx,gy)] == winner
                 and (_SCALE*gx, _SCALE*gy) in w_walls)
    ]
    for dot in to_remove:
        del board[dot]
    dead = frozenset(to_remove)
    for conn in list(connections):
        if conn & dead:
            connections.discard(conn)


class SimGame:
    """Self-contained Dot Grid game (no pygame, no global state)."""

    def __init__(self):
        self.board:       dict  = {}
        self.connections: set   = set()
        self.scores:      dict  = {1: 0.0, 2: 0.0}
        self.territories: dict  = {}
        self.forbidden:   set   = set()
        self.current:     int   = 1
        self.game_over:   bool  = False
        self.winner:      int | None = None
        self._sz        = _SCALE * (S.GRID - 1)
        self._threshold = S.WIN_PCT * (S.GRID - 1) ** 2

    def place(self, gx, gy):
        if self.game_over or (gx, gy) in self.board:
            return False
        if (gx, gy) in self.forbidden:
            return False
        p = self.current
        self.board[(gx, gy)] = p
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                nb = (gx+dx, gy+dy)
                if self.board.get(nb) == p:
                    self.connections.add(frozenset({(gx, gy), nb}))
        self.current = 2 if p == 1 else 1
        self._recompute()
        self._check_win()
        return True

    def legal_moves(self):
        occ = self.board.keys()
        return [(gx, gy) for gx in range(S.GRID) for gy in range(S.GRID)
                if (gx, gy) not in occ and (gx, gy) not in self.forbidden]

    def _recompute(self):
        sz = self._sz
        w1 = _build_walls(self.board, self.connections, 1, sz)
        o1 = _flood_outside(w1, sz)
        w2 = _build_walls(self.board, self.connections, 2, sz)
        o2 = _flood_outside(w2, sz)

        p2_trap = [d for d in self.board if self.board[d]==2
                   and (_SCALE*d[0], _SCALE*d[1]) not in o1]
        p1_trap = [d for d in self.board if self.board[d]==1
                   and (_SCALE*d[0], _SCALE*d[1]) not in o2]

        if p2_trap:
            _remove_encircled(self.board, self.connections, 1, o1, w1)
            w1=_build_walls(self.board,self.connections,1,sz); o1=_flood_outside(w1,sz)
            w2=_build_walls(self.board,self.connections,2,sz); o2=_flood_outside(w2,sz)
        elif p1_trap:
            _remove_encircled(self.board, self.connections, 2, o2, w2)
            w1=_build_walls(self.board,self.connections,1,sz); o1=_flood_outside(w1,sz)
            w2=_build_walls(self.board,self.connections,2,sz); o2=_flood_outside(w2,sz)

        p2_in_enc1 = any((_SCALE*gx,_SCALE*gy) not in o1
                         for gx,gy in self.board if self.board[(gx,gy)]==2)
        p1_in_enc2 = any((_SCALE*gx,_SCALE*gy) not in o2
                         for gx,gy in self.board if self.board[(gx,gy)]==1)

        def owner(sx, sy):
            in1 = (sx,sy) not in o1 and (sx,sy) not in w1
            in2 = (sx,sy) not in o2 and (sx,sy) not in w2
            if in1 and in2:
                return 1 if p2_in_enc1 else (2 if p1_in_enc2 else 1)
            return 1 if in1 else (2 if in2 else None)

        new_terr = {}
        for cx in range(S.GRID-1):
            for cy in range(S.GRID-1):
                bx_,by_ = _SCALE*cx, _SCALE*cy
                tl=(cx,cy); tr=(cx+1,cy); bl=(cx,cy+1); br=(cx+1,cy+1)
                oT=owner(bx_+3,by_+1); oR=owner(bx_+5,by_+3)
                oB=owner(bx_+3,by_+5); oL=owner(bx_+1,by_+3)
                if oT is not None and oT==oR==oB==oL:
                    new_terr[(cx,cy)]=(oT,'full',[tl,tr,br,bl]); continue
                if frozenset({tr,bl}) in self.connections:
                    if oL is not None: new_terr[(cx,cy,'ul')]=(oL,'tri',[tl,tr,bl])
                    if oR is not None: new_terr[(cx,cy,'lr')]=(oR,'tri',[tr,br,bl])
                if frozenset({tl,br}) in self.connections:
                    if oT is not None: new_terr[(cx,cy,'ur')]=(oT,'tri',[tl,tr,br])
                    if oB is not None: new_terr[(cx,cy,'ll')]=(oB,'tri',[tl,bl,br])

        new_scores = {1:0.0, 2:0.0}
        for _,( own,shape,_) in new_terr.items():
            new_scores[own] += 1.0 if shape=='full' else 0.5
        self.scores     = new_scores
        self.territories = new_terr

        new_fp = set()
        for gx in range(S.GRID):
            for gy in range(S.GRID):
                if (gx,gy) in self.board: continue
                sx,sy = _SCALE*gx, _SCALE*gy
                if sx<0 or sy<0 or sx>sz or sy>sz: continue
                if ((sx,sy) not in o1 and (sx,sy) not in w1) or \
                   ((sx,sy) not in o2 and (sx,sy) not in w2):
                    new_fp.add((gx,gy))
        self.forbidden = new_fp

    def _check_win(self):
        for p in (1,2):
            if self.scores[p] >= self._threshold:
                self.game_over=True; self.winner=p; return
        total    = S.GRID*S.GRID
        placeable= total - len(self.board) - len(self.forbidden)
        if len(self.board)==total or placeable<=0:
            self.game_over=True
            s1, s2 = self.scores[1], self.scores[2]
            self.winner = None if s1 == s2 else (1 if s1 > s2 else 2)


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    """Runs N self-play games between two AI players.

    Computes per-move territory-delta rewards and passes them to both
    players via record_outcome(winner, intermediate_rewards).

    on_progress(completed, total, wins) is called after each game.
    on_done(wins) is called when all rounds finish or are cancelled.
    """

    def __init__(self, player1, player2, rounds,
                 on_progress=None, on_done=None):
        self._p1          = player1
        self._p2          = player2
        self._rounds      = rounds
        self._on_progress = on_progress
        self._on_done     = on_done
        self._cancel      = False
        self._thread      = None
        self.wins         = {1: 0, 2: 0, 'draw': 0}

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def cancel(self):
        self._cancel = True

    @property
    def is_running(self):
        return self._thread is not None and self._thread.is_alive()

    def _run(self):
        players = {1: self._p1, 2: self._p2}

        for rnd in range(self._rounds):
            if self._cancel:
                break

            game = SimGame()
            # Per-player move records: [(move_scores_before, move_scores_after)]
            move_scores: dict = {1: [], 2: []}

            while not game.game_over:
                if self._cancel:
                    break
                cp    = game.current
                p     = players[cp]
                scores_before = dict(game.scores)

                # Snapshot state BEFORE the move for symmetric observation
                board_snap     = dict(game.board)
                conn_snap      = set(game.connections)
                forbidden_snap = set(game.forbidden)

                # Snapshot arc potential before the move for shaping reward
                arc_before, _ = arc_potential_scalars(dict(game.board), cp)

                try:
                    move = p.choose_move(
                        dict(game.board), set(game.connections), cp,
                        dict(game.scores), set(game.forbidden),
                    )
                except Exception:
                    legal = game.legal_moves()
                    move  = random.choice(legal) if legal else (0, 0)

                if not game.place(*move):
                    legal = game.legal_moves()
                    if not legal:
                        break
                    game.place(*random.choice(legal))

                # Territory delta shaping reward
                terr_delta = (game.scores.get(cp, 0.0)
                              - scores_before.get(cp, 0.0))
                # Arc potential delta shaping reward: rewards ring-building
                # progress even before a ring is complete.  Without this,
                # the NN only sees territory reward at the instant of ring
                # closure, so it learns to prefer small immediate captures
                # over patient ring-building that pays off much later.
                arc_after, _ = arc_potential_scalars(dict(game.board), cp)
                arc_delta    = max(0.0, arc_after - arc_before)

                shaping = max(-1.0, min(1.0,
                    terr_delta * S.REWARD_TERRITORY
                    + arc_delta * S.REWARD_ARC_SHAPING))
                move_scores[cp].append(shaping)

                # Symmetric learning: let the OTHER player observe this move
                # from their own perspective.  Doubles training data and
                # teaches each player to recognise strong/weak opponent play.
                players[3 - cp].observe_opponent_move(
                    board_snap, conn_snap, cp, move, forbidden_snap)

                # Let the OTHER player observe the resulting board position.
                # They learn: "this board (created by the opponent's move) had
                # this eventual outcome from my perspective."
                opp = players[3 - cp]
                if hasattr(opp, 'observe_opponent_move'):
                    opp.observe_opponent_move(
                        dict(game.board), set(game.connections),
                        cp, move, set(game.forbidden))

            winner = game.winner   # None = draw, 1 or 2 = winner
            if winner:
                self.wins[winner] = self.wins.get(winner, 0) + 1
            else:
                self.wins['draw'] = self.wins.get('draw', 0) + 1

            # record_outcome expects int: 0 = draw/unknown
            w_int = winner or 0
            self._p1.record_outcome(w_int, move_scores[1] or None,
                                    final_scores=dict(game.scores))
            self._p2.record_outcome(w_int, move_scores[2] or None,
                                    final_scores=dict(game.scores))

            if self._on_progress:
                self._on_progress(rnd + 1, self._rounds, dict(self.wins))

        self._p1.save()
        self._p2.save()

        if self._on_done:
            self._on_done(dict(self.wins))
