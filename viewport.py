"""Viewport: cell size, pan offset, zoom and coordinate conversion."""

import settings as S

cell_size: float = float(S.CELL_INIT)
offset_x: float = 0.0
offset_y: float = 0.0


def board_w(win_w: int) -> int:
    """Width of the board area (window minus panel)."""
    return win_w - S.PANEL_W


def fit_cell(win_w: int, win_h: int) -> float:
    """Compute cell size so the whole grid fits in the board area."""
    bw = board_w(win_w)
    return min(
        (bw - 40) / (S.GRID - 1),
        (win_h - 40) / (S.GRID - 1),
    )


def center(win_w: int, win_h: int) -> None:
    """Centre the grid in the board area."""
    global offset_x, offset_y
    total = (S.GRID - 1) * cell_size
    offset_x = (board_w(win_w) - total) / 2
    offset_y = (win_h - total) / 2


def clamp(win_w: int, win_h: int) -> None:
    """Clamp the pan offset so the grid never leaves the viewport."""
    global offset_x, offset_y
    bw = board_w(win_w)
    total = (S.GRID - 1) * cell_size
    offset_x = max(bw - total - 40, min(80.0, offset_x))
    offset_y = max(win_h - total - 40, min(80.0, offset_y))


def g2s(gx: int, gy: int) -> tuple[float, float]:
    """Convert grid coordinates to screen (pixel) coordinates."""
    return offset_x + gx * cell_size, offset_y + gy * cell_size


def s2g(sx: float, sy: float) -> tuple[int, int]:
    """Convert screen coordinates to the nearest grid intersection."""
    return (
        round((sx - offset_x) / cell_size),
        round((sy - offset_y) / cell_size),
    )


def zoom_at(mx: float, my: float, factor: float, win_w: int, win_h: int) -> None:
    """Zoom in or out, keeping the point (mx, my) stationary."""
    global cell_size, offset_x, offset_y
    new_cs = max(S.CELL_MIN, min(S.CELL_MAX, cell_size * factor))
    ratio = new_cs / cell_size
    offset_x = mx - (mx - offset_x) * ratio
    offset_y = my - (my - offset_y) * ratio
    cell_size = new_cs
    clamp(win_w, win_h)
