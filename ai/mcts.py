"""Monte-Carlo Tree Search player.

A thin wrapper around ai/search.py, which does the real work:

EXACT RULES
    The tree and the playouts run on the bitboard engine (ai/engine.py) —
    real flood-fill territory, encirclement captures, forbidden points and
    both win conditions.  The player sees a 40-cell ring closing exactly as
    it sees a unit square closing.

RING-RACE GUIDANCE
    Priors come from ai/strategy.move_scores(): exact one-move tactics plus
    the ring race (how many dots each side still needs for a winning ring)
    after every candidate move.  Full-game playouts then judge the
    candidates.

TREE REUSE
    The chosen child's subtree is kept; if the next position is that subtree
    plus one opponent reply, its statistics are inherited.

EXPERIENCE / OPENING BOOK
    Outcomes of finished games are stored per position (canonical under the
    8 board symmetries).  In the opening, root moves leading to remembered
    positions are seeded with a few virtual visits at the remembered result.
"""

import json
import os
import random

import settings as S
from ai.base_player import BasePlayer
from ai.engine import BitGame, bits
from ai.paths import experience_path
from ai.search import Searcher, SearchConfig
from ai.strategy import Weights
from ai.symmetry import canonical_board_key

# Experience file format.  Version 2 stores every outcome from the
# perspective of the player who made the move leading to the position;
# older files mixed perspectives and are ignored.
_EXPERIENCE_VERSION = 2


class MCTSPlayer(BasePlayer):

    def __init__(self, player_id: int, profile: int = 0):
        """Create an MCTS player.

        profile:  0 = balanced defaults (settings.STRAT_* / MCTS_NOISE_*)
                  1 = "Opportunist"  — weighs immediate territory higher
                  2 = "Strategist"   — weighs the ring race higher
        Profiles give MCTS-vs-MCTS games two distinct styles.
        """
        self.player_id = player_id
        self.experience: dict = {}
        self._game_moves: list = []   # keys of positions after our moves
        self._obs_moves:  list = []   # keys of positions after opponent moves
        self._learning_shared: bool = False

        p = S.MCTS_PROFILES.get(profile, {})
        self._searcher = Searcher(SearchConfig(
            weights=Weights(race=p.get('race_w'),
                            potential=p.get('potential_w'),
                            gain=p.get('gain_w'),
                            threat=p.get('threat_w')),
            c_puct=p.get('exploration', S.SEARCH_C_PUCT),
            noise_alpha=p.get('noise_alpha', S.MCTS_NOISE_ALPHA),
            noise_frac=p.get('noise_frac', S.MCTS_NOISE_FRAC),
        ))
        self.last_search = None       # SearchResult of the latest move

        self.load()

    # ── BasePlayer interface ──────────────────────────────────────────────────

    def choose_move(self, board, connections, player, scores,
                    forbidden=None) -> tuple:
        bg = BitGame.from_board(board, player, scores, forbidden)
        if bg.done or not bg.legal():
            return (0, 0)

        res = self._searcher.search(
            bg, S.AI_THINK_MS / 1000.0, seed_visits=self._book(bg))
        self.last_search = res

        after = bg.copy()
        after.play(res.move)
        self._game_moves.append(canonical_board_key(after.board()))
        return bg.g.xy[res.move]

    def observe_opponent_move(self, board, connections, player_who_moved,
                              move, forbidden=None) -> None:
        """Remember the position the opponent's move leads to, so the book
        also learns from the other side's play (stored at record_outcome
        with the result seen from the opponent's side)."""
        if self._learning_shared:
            return
        new_board = dict(board)
        new_board[tuple(move)] = player_who_moved
        self._obs_moves.append(canonical_board_key(new_board))

    def share_learning_with(self, other: 'MCTSPlayer') -> None:
        """Share the experience dict with *other* (self-play pairs).

        Disables opponent-observation recording on both sides: with one
        shared dict, the partner's own-move recording already covers those
        positions (recording both would double-count every visit)."""
        other.experience = self.experience
        self._learning_shared = other._learning_shared = True

    def record_outcome(self, winner: int, intermediate_rewards=None,
                       final_scores: dict | None = None) -> None:
        # [0, 1] outcome encoding the margin of victory (score share), so a
        # 30–5 win (0.86) counts for more than a 16–15 squeaker (0.52).
        if final_scores and winner != 0:
            own_s = final_scores.get(self.player_id, 0.0)
            opp_s = final_scores.get(3 - self.player_id, 0.0)
            total = own_s + opp_s
            outcome = (own_s / total) if total > 0 else 0.5
        else:
            outcome = (1.0 if winner == self.player_id else
                       0.0 if winner != 0 else 0.5)

        # Each position is credited from the perspective of the player who
        # moved into it: ours for our moves, inverted for the opponent's.
        for keys, value in ((self._game_moves, outcome),
                            (self._obs_moves, 1.0 - outcome)):
            for key in keys:
                e = self.experience.setdefault(key, {'wins': 0.0, 'visits': 0})
                e['visits'] += 1
                e['wins'] += value
        self._game_moves.clear()
        self._obs_moves.clear()
        self._searcher.reset()

        if len(self.experience) > S.AI_MAX_EXPERIENCE:
            keys = list(self.experience.keys())
            random.shuffle(keys)
            for k in keys[S.AI_MAX_EXPERIENCE:]:
                del self.experience[k]
        self.save()

    def save(self):
        path = experience_path(S.AI_EXPERIENCE_BASE, '.json')
        try:
            with open(path, 'w') as fh:
                json.dump({'version': _EXPERIENCE_VERSION,
                           'positions': self.experience}, fh)
        except OSError:
            pass

    def load(self):
        path = experience_path(S.AI_EXPERIENCE_BASE, '.json')
        if os.path.exists(path):
            try:
                with open(path) as fh:
                    data = json.load(fh)
            except (OSError, json.JSONDecodeError):
                return
            if (isinstance(data, dict)
                    and data.get('version') == _EXPERIENCE_VERSION):
                self.experience = data.get('positions', {})

    # ── Internals ─────────────────────────────────────────────────────────────

    def _book(self, bg: BitGame) -> dict:
        """Virtual visits for root moves whose resulting position we have
        seen before.  Opening only — later positions practically never
        repeat, and the lookup costs a canonical key per legal move."""
        if not self.experience or (bg.d1 | bg.d2).bit_count() > S.MCTS_BOOK_MAX_DOTS:
            return {}
        board = bg.board()
        xy = bg.g.xy
        seed = {}
        for i in bits(bg.legal()):
            board[xy[i]] = bg.player
            e = self.experience.get(canonical_board_key(board))
            del board[xy[i]]
            if e and e['visits'] >= 2:
                n = min(e['visits'], S.MCTS_BOOK_MAX_VISITS)
                seed[i] = (n, e['wins'] / e['visits'])
        return seed
