"""Human game recorder.

Saves completed human-vs-AI games to a JSON replay buffer on disk so they
can be replayed later for offline training.  Each record stores the full
move sequence plus the final outcome; replaying it through observe_opponent_move
+ record_outcome is equivalent to re-experiencing the game live.

Buffer is capped at S.REPLAY_BUFFER_MAX games (oldest evicted first).
Filename embeds the grid size so buffers from different grid configs don't mix.
"""

import json
import os

import settings as S
from ai.paths import experience_path


# ── Path helper ───────────────────────────────────────────────────────────────

def _buffer_path() -> str:
    return experience_path('../replay_buffer', '.json')


# ── Public API ────────────────────────────────────────────────────────────────

def load_buffer() -> list:
    """Return the saved game list, or [] if none exist."""
    path = _buffer_path()
    if not os.path.exists(path):
        return []
    try:
        with open(path) as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def save_buffer(games: list) -> None:
    path = _buffer_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, 'w') as fh:
            json.dump(games, fh)
    except OSError:
        pass


def record_game(moves: list, winner: int, final_scores: dict) -> int:
    """Append one game to the buffer.  Returns new buffer length.

    moves       — list of [player, gx, gy] recorded during the game
    winner      — 1, 2, or 0 (draw)
    final_scores — {1: float, 2: float}
    """
    if not moves:
        return count_games()

    games = load_buffer()
    games.append({
        'moves':        moves,
        'winner':       winner,
        'final_scores': final_scores,
        'grid':         S.GRID,
    })

    # Evict oldest if over cap
    max_g = getattr(S, 'REPLAY_BUFFER_MAX', 20)
    if len(games) > max_g:
        games = games[-max_g:]

    save_buffer(games)
    return len(games)


def count_games() -> int:
    """Return number of saved games without loading full records."""
    return len(load_buffer())


def clear_buffer() -> None:
    save_buffer([])
