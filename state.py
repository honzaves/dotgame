"""Game state: board, connections, scores, snapshots and move logic."""

import copy

import settings as S

# ── Mutable game state ────────────────────────────────────────────────────────
board: dict         = {}
connections: set    = set()
territories: dict   = {}
interior_dots: set       = set()
interior_conns: set      = set()
forbidden_positions: set = set()  # grid coords where no dot may be placed
scores: dict        = {1: 0.0, 2: 0.0}
current_player: int = 1
last_move: tuple | None = None
total_moves: int    = 0
game_over: bool     = False
winner_player: int | None = None

# Replay
snapshots: list     = []   # list of state dicts, one per move
snapshot_index: int = -1   # -1 = live (not replaying)

TOTAL_FIELDS: int   = (S.GRID - 1) ** 2


def _capture() -> dict:
    """Return a deep copy of all displayable state."""
    return {
        'board':               dict(board),
        'connections':         set(connections),
        'territories':         dict(territories),
        'interior_dots':       set(interior_dots),
        'interior_conns':      set(interior_conns),
        'forbidden_positions': set(forbidden_positions),
        'scores':              dict(scores),
        'current_player':      current_player,
        'last_move':           last_move,
        'total_moves':         total_moves,
    }


def _restore(snap: dict) -> None:
    """Overwrite all displayable state from a snapshot (does NOT touch
    game_over / winner_player — those stay as-is so the win screen
    remains visible during replay)."""
    global board, connections, territories, interior_dots, interior_conns
    global scores, current_player, last_move, total_moves

    board               = dict(snap['board'])
    connections         = set(snap['connections'])
    territories         = dict(snap['territories'])
    interior_dots       = set(snap['interior_dots'])
    interior_conns      = set(snap['interior_conns'])
    forbidden_positions = set(snap.get('forbidden_positions', set()))
    scores              = dict(snap['scores'])
    current_player      = snap['current_player']
    last_move           = snap['last_move']
    total_moves         = snap['total_moves']


def reset() -> None:
    """Reset all state to an empty board."""
    global board, connections, territories, interior_dots, interior_conns
    global scores, current_player, last_move, total_moves
    global game_over, winner_player, snapshots, snapshot_index
    global forbidden_positions

    board           = {}
    connections     = set()
    territories     = {}
    interior_dots        = set()
    interior_conns       = set()
    forbidden_positions  = set()
    scores          = {1: 0.0, 2: 0.0}
    current_player  = 1
    last_move       = None
    total_moves     = 0
    game_over       = False
    winner_player   = None
    snapshots       = [_capture()]   # index 0 = empty board
    snapshot_index  = -1


def check_win() -> None:
    """Update game_over / winner_player if a win condition is met."""
    global game_over, winner_player

    threshold = S.WIN_PCT * TOTAL_FIELDS
    for p in (1, 2):
        if scores[p] >= threshold:
            game_over     = True
            winner_player = p
            return

    # All grid intersections occupied or all remaining ones are forbidden
    total_positions = S.GRID * S.GRID
    placeable = total_positions - len(board) - len(forbidden_positions)
    if len(board) == total_positions or placeable <= 0:
        game_over     = True
        winner_player = None if scores[1] == scores[2] else (
            1 if scores[1] > scores[2] else 2)


def place(gx: int, gy: int) -> bool:
    """Place a dot for the current player. Returns False if illegal."""
    global current_player, last_move, total_moves, snapshot_index

    if game_over:
        return False
    if (gx, gy) in board:
        return False
    if not (0 <= gx < S.GRID and 0 <= gy < S.GRID):
        return False

    if (gx, gy) in forbidden_positions:
        return False

    # If we're mid-replay, truncate future history
    if snapshot_index != -1:
        del snapshots[snapshot_index + 1:]
        snapshot_index = -1

    p              = current_player
    board[(gx, gy)] = p
    last_move      = (gx, gy)
    total_moves   += 1

    for dx in range(-1, 2):
        for dy in range(-1, 2):
            if dx == 0 and dy == 0:
                continue
            nb = (gx + dx, gy + dy)
            if board.get(nb) == p:
                connections.add(frozenset({(gx, gy), nb}))

    current_player = 2 if p == 1 else 1

    from territory import recompute_territories
    recompute_territories()
    check_win()

    snapshots.append(_capture())
    return True


# ── Replay helpers ────────────────────────────────────────────────────────────

def replay_at() -> int:
    """Current snapshot index, or last index if live."""
    if snapshot_index == -1:
        return len(snapshots) - 1
    return snapshot_index


def replay_step(delta: int) -> None:
    """Move replay cursor by *delta* (-1 = back, +1 = forward)."""
    global snapshot_index

    idx = replay_at() + delta
    idx = max(0, min(len(snapshots) - 1, idx))

    if idx == len(snapshots) - 1:
        _restore(snapshots[-1])
        snapshot_index = -1      # back to live view
    else:
        snapshot_index = idx
        _restore(snapshots[idx])
