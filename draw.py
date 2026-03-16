"""Rendering: board, panel, overlays (mode-picker, settings, arch, pause, replay, win)."""

import pygame

import settings as S
import state
import viewport as vp
import game_mode as gm
from game_mode import Mode


# ── Alpha helpers ─────────────────────────────────────────────────────────────

def draw_circle_alpha(surf, colour, cx, cy, radius):
    if radius < 1:
        return
    ri = int(radius) + 1
    tmp = pygame.Surface((ri * 2, ri * 2), pygame.SRCALPHA)
    pygame.draw.circle(tmp, colour, (ri, ri), int(radius))
    surf.blit(tmp, (int(cx) - ri, int(cy) - ri))


def draw_ring_alpha(surf, colour, cx, cy, radius, width=1):
    ri = int(radius) + int(width) + 2
    if ri < 1:
        return
    tmp = pygame.Surface((ri * 2, ri * 2), pygame.SRCALPHA)
    pygame.draw.circle(tmp, colour, (ri, ri), int(radius), int(width))
    surf.blit(tmp, (int(cx) - ri, int(cy) - ri))


def draw_poly_alpha(surf, pts, colour):
    if len(pts) < 3:
        return
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, y0 = min(xs) - 1, min(ys) - 1
    x1, y1 = max(xs) + 2, max(ys) + 2
    tmp = pygame.Surface((max(1, x1 - x0), max(1, y1 - y0)), pygame.SRCALPHA)
    pygame.draw.polygon(tmp, colour, [(p[0] - x0, p[1] - y0) for p in pts])
    surf.blit(tmp, (x0, y0))


def _fit_label(font, text, max_w):
    """Return text truncated with '…' so font.render fits within max_w px."""
    if font.render(text, True, (0, 0, 0)).get_width() <= max_w:
        return text
    while len(text) > 1:
        text = text[:-1]
        if font.render(text + "…", True, (0, 0, 0)).get_width() <= max_w:
            return text + "…"
    return "…"


def _button(surf, font, label, rect, active=False, col=None):
    """Draw a standard button. Text is ellipsis-clipped if too wide."""
    bg  = (40, 44, 60) if active else S.C_PANEL
    bdr = col if col else (S.C_GRID5 if not active else S.C_GOLD)
    pygame.draw.rect(surf, bg,  rect, border_radius=5)
    pygame.draw.rect(surf, bdr, rect, width=1, border_radius=5)
    safe = _fit_label(font, label, rect.w - 10)
    lbl  = font.render(safe, True, col if col else S.C_TEXT)
    surf.blit(lbl, (rect.x + (rect.w - lbl.get_width()) // 2,
                    rect.y + (rect.h - lbl.get_height()) // 2))
    return rect


# ── Board ─────────────────────────────────────────────────────────────────────

def draw_board(surf, hover, win_w, win_h):
    bw = vp.board_w(win_w)
    cs = vp.cell_size
    surf.fill(S.C_BG, (0, 0, bw, win_h))

    r0x = max(0, int((0 - vp.offset_x) / cs))
    r1x = min(S.GRID - 1, int((bw - vp.offset_x) / cs) + 1)
    r0y = max(0, int((0 - vp.offset_y) / cs))
    r1y = min(S.GRID - 1, int((win_h - vp.offset_y) / cs) + 1)

    for gx in range(r0x, r1x + 1):
        x   = int(vp.offset_x + gx * cs)
        col = S.C_GRID5 if gx % 5 == 0 else S.C_GRID
        pygame.draw.line(surf, col,
                         (x, int(vp.offset_y + r0y * cs)),
                         (x, int(vp.offset_y + r1y * cs)), 1)
    for gy in range(r0y, r1y + 1):
        y   = int(vp.offset_y + gy * cs)
        col = S.C_GRID5 if gy % 5 == 0 else S.C_GRID
        pygame.draw.line(surf, col,
                         (int(vp.offset_x + r0x * cs), y),
                         (int(vp.offset_x + r1x * cs), y), 1)

    ts = pygame.Surface((bw, win_h), pygame.SRCALPHA)
    for _key, (owner, _shape, corners) in state.territories.items():
        col = S.C_P1 if owner == 1 else S.C_P2
        pts = [(int(x), int(y))
               for gx_, gy_ in corners
               for x, y in [vp.g2s(gx_, gy_)]]
        draw_poly_alpha(ts, pts, (*col, 165))
    surf.blit(ts, (0, 0))

    lw = max(1, int(cs * 0.14))
    for conn in state.connections:
        if conn in state.interior_conns:
            continue
        a, b    = tuple(conn)
        ax_, ay_ = vp.g2s(*a)
        bx_, by_ = vp.g2s(*b)
        col = S.C_P1 if state.board[a] == 1 else S.C_P2
        pygame.draw.line(surf, col,
                         (int(ax_), int(ay_)), (int(bx_), int(by_)), lw)

    dot_r = max(2.5, cs * 0.36)
    for (gx, gy), p in state.board.items():
        if (gx, gy) in state.interior_dots:
            continue
        if not (r0x <= gx <= r1x and r0y <= gy <= r1y):
            continue
        cx_, cy_ = vp.g2s(gx, gy)
        col = S.C_P1 if p == 1 else S.C_P2
        if state.last_move == (gx, gy):
            draw_circle_alpha(surf, (*col, 50), cx_, cy_, dot_r * 2.8)
            draw_ring_alpha(surf, (*col, 180), cx_, cy_,
                            dot_r * 1.8, max(1, int(cs * 0.08)))
        pygame.draw.circle(surf, col,
                           (int(cx_), int(cy_)), max(2, int(dot_r)))

    if hover and not state.game_over and state.snapshot_index == -1:
        gx, gy = hover
        if (0 <= gx < S.GRID and 0 <= gy < S.GRID
                and (gx, gy) not in state.board
                and (gx, gy) not in state.forbidden_positions):
            hx, hy = vp.g2s(gx, gy)
            col    = S.C_P1 if state.current_player == 1 else S.C_P2
            draw_circle_alpha(surf, (*col, 120), hx, hy, max(2, dot_r))
            tmp = pygame.Surface((bw, win_h), pygame.SRCALPHA)
            cr  = (*S.C_GOLD, 50)
            pygame.draw.line(tmp, cr,
                             (int(hx), int(vp.offset_y + r0y * cs)),
                             (int(hx), int(vp.offset_y + r1y * cs)), 1)
            pygame.draw.line(tmp, cr,
                             (int(vp.offset_x + r0x * cs), int(hy)),
                             (int(vp.offset_x + r1x * cs), int(hy)), 1)
            surf.blit(tmp, (0, 0))


# ── Panel ─────────────────────────────────────────────────────────────────────

def draw_panel(surf, fonts, win_w, win_h, ai_thinking=False, mode=None):
    """Draw side panel.

    Returns (new_game_rect, zoom_rects, stop_rect, None, None).
    The trailing Nones preserve the 5-tuple contract; AI arch and deep-terr
    have moved to the Settings / AI Info screens.
    """
    bw  = vp.board_w(win_w)
    pygame.draw.rect(surf, S.C_PANEL, (bw, 0, S.PANEL_W, win_h))
    pygame.draw.line(surf, S.C_BORDER, (bw, 0), (bw, win_h), 1)

    ft = fonts['title']
    fb = fonts['big']
    fs = fonts['small']
    fx = fonts['tiny']
    pad = 24
    y   = 24

    t = ft.render("DOT  GRID", True, S.C_GOLD)
    surf.blit(t, (bw + (S.PANEL_W - t.get_width()) // 2, y))
    y += t.get_height() + 4

    sub = fx.render(f"{S.GRID} × {S.GRID}", True, S.C_DIM)
    surf.blit(sub, (bw + (S.PANEL_W - sub.get_width()) // 2, y))
    y += sub.get_height() + 20

    pygame.draw.line(surf, S.C_BORDER,
                     (bw + pad, y), (bw + S.PANEL_W - pad, y), 1)
    y += 16

    for p, col in ((1, S.C_P1), (2, S.C_P2)):
        active   = (state.current_player == p) and not state.game_over
        card_h   = 82
        card_col = (30, 33, 45) if active else S.C_PANEL
        card_x   = bw + pad - 8
        card_w   = S.PANEL_W - 2 * pad + 16
        pygame.draw.rect(surf, card_col, (card_x, y - 8, card_w, card_h),
                         border_radius=6)
        if active:
            pygame.draw.rect(surf, col, (card_x, y - 8, card_w, card_h),
                             width=1, border_radius=6)

        dx_, dy_ = bw + pad + 8, y + 18
        pygame.draw.circle(surf, col, (dx_, dy_), 7)
        if active:
            draw_circle_alpha(surf, (*col, 70), dx_, dy_, 15)

        lbl = fs.render(_fit_label(fs, gm.player_name(p), card_w - 50),
                        True, col if active else S.C_DIM)
        surf.blit(lbl, (dx_ + 22, y + 4))

        sv        = state.scores[p]
        score_str = f"{int(sv)}" if sv == int(sv) else f"{sv:.1f}"
        sc = fb.render(score_str, True, col)
        surf.blit(sc, (dx_ + 22, y + 22))

        if active and not state.game_over:
            tag_txt = "THINKING…" if ai_thinking else "YOUR TURN"
            tag_col = S.C_DIM    if ai_thinking else S.C_GOLD
            tag = fx.render(tag_txt, True, tag_col)
            surf.blit(tag, (bw + S.PANEL_W - pad - tag.get_width() - 4, y + 52))

        y += card_h + 12

    y += 6
    pygame.draw.line(surf, S.C_BORDER,
                     (bw + pad, y), (bw + S.PANEL_W - pad, y), 1)
    y += 14

    pct_val = int(S.WIN_PCT * 100)
    replay_idx = state.replay_at()
    move_label = (f"Move  {state.snapshots[replay_idx]['total_moves']}"
                  if state.snapshots else f"Move  {state.total_moves}")
    for txt in (move_label,
                f"Win at {pct_val}%  ({int(S.WIN_PCT * state.TOTAL_FIELDS)} fields)",
                "■ square = 1 pt",
                "▲ triangle = 0.5 pt"):
        r = fx.render(txt, True, S.C_DIM)
        surf.blit(r, (bw + (S.PANEL_W - r.get_width()) // 2, y))
        y += r.get_height() + 5

    y += 8
    pct = int(vp.cell_size / S.CELL_INIT * 100)
    zm  = fx.render(f"zoom  {pct}%", True, S.C_DIM)
    surf.blit(zm, (bw + (S.PANEL_W - zm.get_width()) // 2, y))
    y += zm.get_height() + 12

    # Zoom buttons
    btn_sz   = 32
    gap      = 10
    total_bw = btn_sz * 2 + gap
    bx_m     = bw + (S.PANEL_W - total_bw) // 2
    bx_p     = bx_m + btn_sz + gap
    by_      = y
    zoom_rects = []
    for bx_, label in ((bx_m, "−"), (bx_p, "+")):
        _button(surf, fs, label, pygame.Rect(bx_, by_, btn_sz, btn_sz))
        zoom_rects.append(pygame.Rect(bx_, by_, btn_sz, btn_sz))

    # Stop button (computer self-play only)
    stop_rect = None
    if mode == Mode.COMPUTER_LEARNING and not state.game_over:
        sr = pygame.Rect(bw + pad, by_ + btn_sz + 10,
                         S.PANEL_W - 2 * pad, 30)
        _button(surf, fx, "■  STOP", sr, col=(220, 80, 60))
        stop_rect = sr

    # New Game
    bw2, bh2 = 140, 36
    bx2 = bw + (S.PANEL_W - bw2) // 2
    by2 = win_h - bh2 - 24
    _button(surf, fs, "NEW  GAME", pygame.Rect(bx2, by2, bw2, bh2))

    for i, tip in enumerate(["scroll/+- = zoom", "rmb drag = pan", "Esc = menu"]):
        ht = fx.render(tip, True, S.C_DIM)
        surf.blit(ht, (bw + (S.PANEL_W - ht.get_width()) // 2, by2 - 46 + i * 15))

    return pygame.Rect(bx2, by2, bw2, bh2), zoom_rects, stop_rect, None, None


# ── Win screen ────────────────────────────────────────────────────────────────

def draw_win_screen(surf, fonts, win_w, win_h):
    overlay = pygame.Surface((win_w, win_h), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 120))
    surf.blit(overlay, (0, 0))

    p   = state.winner_player   # None = draw
    cx  = vp.board_w(win_w) // 2
    cy  = win_h // 2 - 40

    if p is None:
        col = S.C_GOLD
        t1  = fonts['title'].render("DRAW", True, col)
        t2  = fonts['big'].render("Equal score!", True, col)
    else:
        col = S.C_P1 if p == 1 else S.C_P2
        t1  = fonts['title'].render(gm.player_name(p), True, col)
        t2  = fonts['big'].render("WINS!", True, col)

    t3 = fonts['tiny'].render("Review moves with  ←  →   |   R = new game", True, S.C_DIM)
    surf.blit(t1, (cx - t1.get_width() // 2, cy))
    surf.blit(t2, (cx - t2.get_width() // 2, cy + t1.get_height() + 8))
    surf.blit(t3, (cx - t3.get_width() // 2, cy + t1.get_height() + t2.get_height() + 24))


# ── Replay controls ───────────────────────────────────────────────────────────

def draw_replay_controls(surf, fonts, win_w, win_h):
    bw  = vp.board_w(win_w)
    fx  = fonts['small']
    bsz = 40
    gap = 12
    by_ = win_h - bsz - 12

    idx   = state.replay_at()
    total = len(state.snapshots) - 1
    label = f"Move  {state.snapshots[idx]['total_moves']}  /  {total}"
    lbl   = fonts['tiny'].render(label, True, S.C_GOLD)
    surf.blit(lbl, (bw // 2 - lbl.get_width() // 2, by_ + (bsz - lbl.get_height()) // 2))

    bx_prev = bw // 2 - bsz - gap - lbl.get_width() // 2 - 20
    prev_r  = pygame.Rect(bx_prev, by_, bsz, bsz)
    active_prev = idx > 0
    _button(surf, fx, "◀", prev_r, active=active_prev,
            col=S.C_GOLD if active_prev else S.C_DIM)

    bx_next = bw // 2 + lbl.get_width() // 2 + gap + 20
    next_r  = pygame.Rect(bx_next, by_, bsz, bsz)
    active_next = idx < len(state.snapshots) - 1
    _button(surf, fx, "▶", next_r, active=active_next,
            col=S.C_GOLD if active_next else S.C_DIM)

    return prev_r, next_r


# ── Pause menu ────────────────────────────────────────────────────────────────

def draw_pause_menu(surf, fonts, win_w, win_h):
    overlay = pygame.Surface((win_w, win_h), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 180))
    surf.blit(overlay, (0, 0))

    fs  = fonts['small']
    ft  = fonts['title']
    cx  = win_w // 2
    bw_ = 220
    bh_ = 44
    gap = 14

    title = ft.render("PAUSED", True, S.C_GOLD)
    start_y = win_h // 2 - 80
    surf.blit(title, (cx - title.get_width() // 2, start_y - 44))

    rects = []
    for i, (label, col) in enumerate([
            ("RESUME", None), ("NEW  GAME", None), ("QUIT", (220, 80, 60))]):
        r = pygame.Rect(cx - bw_ // 2, start_y + i * (bh_ + gap), bw_, bh_)
        _button(surf, fs, label, r, col=col)
        rects.append(r)

    return tuple(rects)


# ── Mode picker (main menu) ───────────────────────────────────────────────────

from game_mode import MODE_LABELS
_MODE_OPTIONS = sorted(MODE_LABELS.items(), key=lambda kv: kv[1])


def draw_mode_picker(surf, fonts, win_w, win_h, name_inputs=None, error_msg=""):
    """Main menu overlay.

    Returns (mode_rects, name_rects, train_r, settings_r, arch_r).

    Think time has moved to the Settings screen.
    AI architecture has moved to the AI Info screen.
    """
    overlay = pygame.Surface((win_w, win_h), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 200))
    surf.blit(overlay, (0, 0))

    fs = fonts['small']
    ft = fonts['title']
    fx = fonts['tiny']

    bw_     = 300
    bh_     = 38
    btn_gap = 8
    pad_top = 50
    col_gap = 40

    n_modes  = len(_MODE_OPTIONS)
    col_h    = n_modes * (bh_ + btn_gap) - btn_gap
    left_cx  = win_w // 2 - col_gap // 2 - bw_ // 2 - 10
    right_cx = win_w // 2 + col_gap // 2 + bw_ // 2 + 10
    col_top  = max(pad_top + 36, win_h // 2 - col_h // 2)

    title = ft.render("SELECT  MODE", True, S.C_GOLD)
    surf.blit(title, (win_w // 2 - title.get_width() // 2, pad_top - 4))

    # Left: mode buttons
    mode_rects = []
    for i, (mode, label) in enumerate(_MODE_OPTIONS):
        r = pygame.Rect(left_cx - bw_ // 2, col_top + i * (bh_ + btn_gap), bw_, bh_)
        _button(surf, fs, label, r)
        mode_rects.append((r, mode))

    # Right column
    rx = right_cx - bw_ // 2
    rw = bw_
    ry = col_top

    # Player names
    name_inputs = name_inputs or {1: S.PLAYER_NAMES[1], 2: S.PLAYER_NAMES[2]}
    nl = fx.render("Player names:", True, S.C_TEXT)
    surf.blit(nl, (rx + rw // 2 - nl.get_width() // 2, ry))
    ry += nl.get_height() + 6

    name_rects = {}
    field_h    = 28
    for p, col in ((1, S.C_P1), (2, S.C_P2)):
        nfr = pygame.Rect(rx, ry, rw, field_h)
        pygame.draw.rect(surf, S.C_PANEL, nfr, border_radius=4)
        pygame.draw.rect(surf, col,       nfr, width=1, border_radius=4)
        txt_s = _fit_label(fs, name_inputs.get(p, ''), rw - 20)
        txt   = fs.render(txt_s, True, S.C_TEXT)
        surf.blit(txt, (nfr.x + 6, nfr.y + (field_h - txt.get_height()) // 2))
        cur = fx.render("|", True, col)
        surf.blit(cur, (nfr.x + 6 + txt.get_width() + 1,
                        nfr.y + (field_h - cur.get_height()) // 2))
        name_rects[p] = nfr
        ry += field_h + 6

    ry += 4
    pygame.draw.line(surf, S.C_BORDER, (rx, ry), (rx + rw, ry), 1)
    ry += 10

    # Error / done message
    if error_msg:
        for line in error_msg.split("\n"):
            if line.strip():
                em = fx.render(_fit_label(fx, line.strip(), rw), True, (220, 80, 60))
                surf.blit(em, (rx + rw // 2 - em.get_width() // 2, ry))
                ry += em.get_height() + 2
        ry += 4

    # Action buttons
    settings_r = pygame.Rect(rx, ry, rw, 34)
    _button(surf, fs, "⚙  SETTINGS", settings_r, col=(160, 160, 200))
    ry += 34 + 8

    arch_r = pygame.Rect(rx, ry, rw, 34)
    _button(surf, fs, "◉  AI  INFO", arch_r, col=(140, 180, 220))
    ry += 34 + 8

    train_r = pygame.Rect(rx, ry, rw, 34)
    _button(surf, fs, "TRAIN  AI  →", train_r, col=S.C_GOLD)
    ry += 34 + 10

    hint = fx.render("Esc  to cancel", True, S.C_DIM)
    surf.blit(hint, (rx + rw // 2 - hint.get_width() // 2, ry))

    return mode_rects, name_rects, train_r, settings_r, arch_r


# ── Settings screen ───────────────────────────────────────────────────────────

def draw_settings_screen(surf, fonts, win_w, win_h, think_ms, mcts_profile=0):
    """Settings overlay.

    Returns (back_r, prev_think_r, next_think_r, deep_terr_r, prev_prof_r, next_prof_r).
    """
    overlay = pygame.Surface((win_w, win_h), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 210))
    surf.blit(overlay, (0, 0))

    ft = fonts['title']
    fs = fonts['small']
    fx = fonts['tiny']
    cx = win_w // 2

    y = max(60, win_h // 2 - 170)

    title = ft.render("SETTINGS", True, S.C_GOLD)
    surf.blit(title, (cx - title.get_width() // 2, y))
    y += title.get_height() + 30

    # Think time
    lbl = fx.render("MCTS think time per move:", True, S.C_TEXT)
    surf.blit(lbl, (cx - lbl.get_width() // 2, y))
    y += lbl.get_height() + 8

    arr   = 32
    val_w = 180
    row_h = 34
    row_x = cx - (arr + 8 + val_w + 8 + arr) // 2

    prev_think_r = pygame.Rect(row_x, y, arr, row_h)
    next_think_r = pygame.Rect(row_x + arr + 8 + val_w + 8, y, arr, row_h)
    _button(surf, fs, "◀", prev_think_r)
    _button(surf, fs, "▶", next_think_r)

    vbox = pygame.Rect(row_x + arr + 8, y, val_w, row_h)
    pygame.draw.rect(surf, (30, 33, 45), vbox, border_radius=4)
    pygame.draw.rect(surf, S.C_BORDER,   vbox, width=1, border_radius=4)
    vt = fs.render(f"{think_ms} ms", True, S.C_TEXT)
    surf.blit(vt, (vbox.x + (val_w - vt.get_width()) // 2,
                   vbox.y + (row_h - vt.get_height()) // 2))
    y += row_h + 6

    hint = fx.render("Applies to MCTS play and MCTS self-play training", True, S.C_DIM)
    surf.blit(hint, (cx - hint.get_width() // 2, y))
    y += hint.get_height() + 20

    pygame.draw.line(surf, S.C_BORDER, (cx - 160, y), (cx + 160, y), 1)
    y += 16

    # MCTS profile selector
    lbl2 = fx.render("MCTS personality profile (single-player games):", True, S.C_TEXT)
    surf.blit(lbl2, (cx - lbl2.get_width() // 2, y))
    y += lbl2.get_height() + 8

    prev_prof_r = pygame.Rect(row_x, y, arr, row_h)
    next_prof_r = pygame.Rect(row_x + arr + 8 + val_w + 8, y, arr, row_h)
    _button(surf, fs, "◀", prev_prof_r)
    _button(surf, fs, "▶", next_prof_r)

    pbox = pygame.Rect(row_x + arr + 8, y, val_w, row_h)
    pygame.draw.rect(surf, (30, 33, 45), pbox, border_radius=4)
    pygame.draw.rect(surf, S.C_BORDER,   pbox, width=1, border_radius=4)
    plbl = S.MCTS_PROFILE_LABELS.get(mcts_profile, "Default")
    pt   = fs.render(plbl, True, S.C_TEXT)
    surf.blit(pt, (pbox.x + (val_w - pt.get_width()) // 2,
                   pbox.y + (row_h - pt.get_height()) // 2))
    y += row_h + 6

    prof_hint = fx.render("MCTS vs MCTS always uses Opportunist (P1) vs Strategist (P2)", True, S.C_DIM)
    surf.blit(prof_hint, (cx - prof_hint.get_width() // 2, y))
    y += prof_hint.get_height() + 20

    pygame.draw.line(surf, S.C_BORDER, (cx - 160, y), (cx + 160, y), 1)
    y += 16

    # Deep territory toggle
    _on   = getattr(S, 'MCTS_REAL_ROLLOUT', False)
    _tcol = (60, 180, 100) if _on else (130, 130, 130)
    _tlbl = "DEEP TERR:  ON" if _on else "DEEP TERR:  OFF"
    deep_w      = 260
    deep_terr_r = pygame.Rect(cx - deep_w // 2, y, deep_w, 36)
    _button(surf, fs, _tlbl, deep_terr_r, col=_tcol)
    y += 36 + 6

    for d in ("ON: full flood-fill scoring in rollouts (accurate territory + forbidden zones)",
              "OFF: fast square-counting only — default, ~3× more rollouts per second",
              "No extra delay either way — same think time, different rollout quality"):
        dt = fx.render(_fit_label(fx, d, win_w - 80), True, S.C_DIM)
        surf.blit(dt, (cx - dt.get_width() // 2, y))
        y += dt.get_height() + 2
    y += 18

    back_r = pygame.Rect(cx - 80, y, 160, 38)
    _button(surf, fs, "BACK", back_r)

    return back_r, prev_think_r, next_think_r, deep_terr_r, prev_prof_r, next_prof_r


# ── AI info: architecture select screen ──────────────────────────────────────

_ARCH_DESCRIPTIONS = {
    "MCTS": (
        "MCTS — Monte Carlo Tree Search",
        [
            "Explores thousands of imagined futures each turn by building a tree of moves.",
            "Strategic signals guide which branches matter: closing territory is first priority,",
            "building large rings is valued throughout construction (not just at completion),",
            "and Dirichlet noise ensures different games play out differently.",
            "Gets noticeably stronger with more thinking time. No training required.",
        ],
    ),
    "NN": (
        "Neural Network — learns over time (pure NumPy)",
        [
            "Learns by playing thousands of games. Sees the board as 15 overlapping",
            "feature maps: dot positions, territory threats, ring arcs in progress,",
            "bridge opportunities, game phase, and spatial centrality.",
            "Develops move intuition without looking ahead. Starts weak, improves with",
            "training. No extra software required beyond numpy.",
        ],
    ),
    "PT": (
        "PyTorch CNN — deepest, needs torch installed",
        [
            "A convolutional network that sees the board as an image. Sliding 3×3 filters",
            "detect local patterns; residual blocks stack them into a wide field of view.",
            "Trained with PPO — a stable algorithm that prevents large policy swings.",
            "Strongest AI given enough training. Requires: pip install torch",
        ],
    ),
    "NM": (
        "Neural MCTS — AlphaZero-style (needs torch, best AI)",
        [
            "Combines tree search with a learned CNN: the policy head guides which branches",
            "to explore (replacing hand-crafted priors), and the value head evaluates leaf",
            "nodes (replacing random rollouts). Trains on MCTS visit distributions, not raw",
            "policy outputs — the tree's opinion improves the network, which improves the tree.",
            "Requires: pip install torch",
        ],
    ),
}


def draw_arch_select(surf, fonts, win_w, win_h):
    """AI Info screen: pick which architecture to view.

    Returns (back_r, mcts_r, nn_r, pt_r, nm_r).
    """
    overlay = pygame.Surface((win_w, win_h), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 210))
    surf.blit(overlay, (0, 0))

    ft = fonts['title']
    fs = fonts['small']
    fx = fonts['tiny']
    cx = win_w // 2

    y = max(40, win_h // 4 - 60)

    title = ft.render("AI  INFO", True, (140, 180, 220))
    surf.blit(title, (cx - title.get_width() // 2, y))
    y += title.get_height() + 6

    sub = fx.render("Select an AI to see its architecture and how it works.", True, S.C_DIM)
    surf.blit(sub, (cx - sub.get_width() // 2, y))
    y += sub.get_height() + 16

    pygame.draw.line(surf, S.C_BORDER, (cx - 200, y), (cx + 200, y), 1)
    y += 14

    btn_w = min(320, win_w - 80)
    btn_h = 38
    gap   = 8
    rects = {}
    cols  = {
        "MCTS": (140, 180, 220),
        "NN":   (140, 200, 140),
        "PT":   (220, 180, 100),
        "NM":   (200, 140, 220),   # purple — fusion of MCTS + CNN
    }

    for ai_key in ("MCTS", "NN", "PT", "NM"):
        subtitle, desc_lines = _ARCH_DESCRIPTIONS[ai_key]
        r = pygame.Rect(cx - btn_w // 2, y, btn_w, btn_h)
        _button(surf, fs, subtitle, r, col=cols[ai_key])
        rects[ai_key] = r
        y += btn_h + 4
        for line in desc_lines[:2]:
            lt = fx.render(_fit_label(fx, line, btn_w), True, S.C_DIM)
            surf.blit(lt, (cx - lt.get_width() // 2, y))
            y += lt.get_height() + 1
        y += gap

    y += 8
    back_r = pygame.Rect(cx - 80, y, 160, 36)
    _button(surf, fs, "BACK", back_r)

    return back_r, rects["MCTS"], rects["NN"], rects["PT"], rects["NM"]


# ── Architecture overlay (scrollable) ────────────────────────────────────────

def draw_arch_overlay(surf, fonts, win_w, win_h, ai_type, scroll_y=0):
    """Scrollable overlay showing the architecture for one AI type.

    ai_type : "MCTS" | "NN" | "PT"
    scroll_y: vertical pixel offset (0 = top)

    Returns (close_r, up_r, down_r, content_height).
    up_r / down_r are None when scroll is not needed.
    """
    overlay = pygame.Surface((win_w, win_h), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 220))
    surf.blit(overlay, (0, 0))

    ft = fonts['title']
    fs = fonts['small']
    fx = fonts['tiny']
    cx = win_w // 2

    _AI_COL   = (140, 180, 220)
    _BOX_COL  = (30,  36,  52)
    _ACT_COL  = (80, 180, 120)
    _HEAD_COL = (220, 160, 80)

    arch     = S.pt_arch()
    hidden   = S.nn_hidden_sizes()
    n_blocks = arch['blocks']
    channels = arch['channels']
    G        = S.GRID

    # ── Fixed header ──────────────────────────────────────────────────────────
    header_h = 16
    titles_map = {
        "MCTS": "MCTS  —  Monte Carlo Tree Search",
        "NN":   f"Neural Network  —  {len(hidden)} hidden layers, 15 input channels",
        "PT":   f"PyTorch CNN  —  {n_blocks} res-blocks, {channels} channels, 15 inputs",
        "NM":   f"Neural MCTS  —  AlphaZero-style  ({n_blocks} res-blocks, {channels} ch)",
    }
    ttl = ft.render(titles_map.get(ai_type, ai_type), True, _AI_COL)
    surf.blit(ttl, (cx - ttl.get_width() // 2, header_h))
    header_h += ttl.get_height() + 6

    _, desc_lines = _ARCH_DESCRIPTIONS.get(ai_type, ("", []))
    for line in desc_lines:
        lt = fx.render(_fit_label(fx, line, win_w - 80), True, S.C_DIM)
        surf.blit(lt, (cx - lt.get_width() // 2, header_h))
        header_h += lt.get_height() + 2
    header_h += 8

    pygame.draw.line(surf, S.C_BORDER, (40, header_h), (win_w - 40, header_h), 1)
    header_h += 10

    # ── Footer ────────────────────────────────────────────────────────────────
    close_bw, close_bh = 120, 32
    close_r  = pygame.Rect(cx - close_bw // 2, win_h - close_bh - 8, close_bw, close_bh)
    _button(surf, fs, "CLOSE", close_r, col=(160, 160, 160))
    footer_h = close_bh + 16

    # ── Scrollable diagram ────────────────────────────────────────────────────
    scroll_top    = header_h
    scroll_bottom = win_h - footer_h
    visible_h     = max(1, scroll_bottom - scroll_top)

    # Render diagram to an off-screen surface
    est_h   = max(visible_h + 200, 1400)
    diagram = pygame.Surface((win_w, est_h), pygame.SRCALPHA)
    diagram.fill((0, 0, 0, 0))

    dy = 12

    col_w = win_w - 100
    bx    = 50 + (col_w - min(col_w - 20, 360)) // 2
    box_w = min(col_w - 20, 360)
    bh_b  = 44
    arr_l = 20
    gap_b = 10

    def _box(label, detail, bx_, by_, bw_, bh_,
             col=_BOX_COL, border=_AI_COL, lbl_col=None):
        pygame.draw.rect(diagram, col,    pygame.Rect(bx_, by_, bw_, bh_), border_radius=5)
        pygame.draw.rect(diagram, border, pygame.Rect(bx_, by_, bw_, bh_), width=1, border_radius=5)
        safe_lbl = _fit_label(fs, label, bw_ - 8)
        ls = fs.render(safe_lbl, True, lbl_col or S.C_TEXT)
        diagram.blit(ls, (bx_ + bw_//2 - ls.get_width()//2,
                          by_ + bh_//2 - ls.get_height()//2 - (8 if detail else 0)))
        if detail:
            safe_det = _fit_label(fx, detail, bw_ - 8)
            ds = fx.render(safe_det, True, S.C_DIM)
            diagram.blit(ds, (bx_ + bw_//2 - ds.get_width()//2,
                               by_ + bh_//2 + 4))

    def _arrow(sx, sy, ex, ey):
        if abs(ey - sy) < 2 and abs(ex - sx) > 2:
            pygame.draw.line(diagram, S.C_BORDER, (sx, sy), (ex, ey), 1)
            pygame.draw.polygon(diagram, S.C_BORDER,
                                [(ex, ey), (ex-6, ey-4), (ex-6, ey+4)])
        else:
            pygame.draw.line(diagram, S.C_BORDER, (sx, sy), (ex, ey), 1)
            pygame.draw.polygon(diagram, S.C_BORDER,
                                [(ex, ey), (ex-4, ey-6), (ex+4, ey-6)])

    if ai_type == "MCTS":
        for label, detail in [
            ("Board position",   f"{G}×{G} grid, up to {G*G} legal moves"),
            ("Strategic priors", "close(4×) block(3×) arc-ext(5×) bridge(3×) setup(2.5×) + noise"),
            ("UCB selection",    "win_rate + C · prior/√visits  (PUCT)"),
            ("6-tier rollout",   "fork → close → block → setup → adjacent → random"),
            ("Leaf evaluation",  "arc-potential 50% · territory 25% · dots 15% · conns 10%"),
            ("Backpropagation",  "result flows up the tree to all parent nodes"),
            ("Final move",       "best win_rate + territory blend, min 3 visits required"),
        ]:
            _box(label, detail, bx, dy, box_w, bh_b)
            _arrow(bx + box_w//2, dy + bh_b, bx + box_w//2, dy + bh_b + arr_l)
            dy += bh_b + arr_l
        dy -= arr_l   # remove trailing arrow

    elif ai_type == "NN":
        inp = 15 * G * G
        trunk = [("Input", f"15 channels × {G}×{G} = {inp} values")]
        for i, h in enumerate(hidden):
            trunk.append((f"Hidden layer {i+1}", f"{h} neurons  +  ReLU"))
        for label, detail in trunk:
            _box(label, detail, bx, dy, box_w, bh_b)
            _arrow(bx + box_w//2, dy + bh_b, bx + box_w//2, dy + bh_b + arr_l)
            dy += bh_b + arr_l
        split_y = dy
        hw = (box_w - 10) // 2
        _box("Policy head", f"{G*G} logits → softmax",
             bx,         split_y, hw, bh_b, border=_HEAD_COL, lbl_col=_HEAD_COL)
        _box("Value head",  "1 → tanh → win est.",
             bx+hw+10,   split_y, hw, bh_b, border=_HEAD_COL, lbl_col=_HEAD_COL)
        mid_x = bx + box_w // 2
        pygame.draw.line(diagram, S.C_BORDER,
                         (mid_x, split_y - arr_l), (bx+hw//2, split_y), 1)
        pygame.draw.line(diagram, S.C_BORDER,
                         (mid_x, split_y - arr_l), (bx+hw+10+hw//2, split_y), 1)
        dy = split_y + bh_b + 8
        for info in (f"Training: GAE λ={S.NN_GAE_LAMBDA}, Adam lr={S.NN_LEARNING_RATE}, entropy bonus",
                     "Reward: score margin (own−opp)/total at game end"):
            t = fx.render(_fit_label(fx, info, box_w), True, S.C_DIM)
            diagram.blit(t, (bx + box_w//2 - t.get_width()//2, dy))
            dy += t.get_height() + 3

    elif ai_type == "PT":
        _box("Input tensor", f"15 ch × {G}×{G} = {15*G*G} values", bx, dy, box_w, bh_b)
        _arrow(bx+box_w//2, dy+bh_b, bx+box_w//2, dy+bh_b+arr_l)
        dy += bh_b + arr_l
        _box("Stem", f"Conv2d(15→{channels}, 3×3) → BN → ReLU",
             bx, dy, box_w, bh_b, border=_ACT_COL)
        _arrow(bx+box_w//2, dy+bh_b, bx+box_w//2, dy+bh_b+arr_l)
        dy += bh_b + arr_l
        rb_h = 52
        for i in range(n_blocks):
            rr = pygame.Rect(bx, dy, box_w, rb_h)
            pygame.draw.rect(diagram, (28, 38, 55), rr, border_radius=5)
            pygame.draw.rect(diagram, _ACT_COL,     rr, width=1, border_radius=5)
            pygame.draw.arc(diagram,
                            (_ACT_COL[0]//2, _ACT_COL[1]//2, _ACT_COL[2]//2),
                            pygame.Rect(bx-18, dy, 18, rb_h), 1.57, 4.71, 2)
            ll = fs.render(f"ResBlock {i+1}", True, S.C_TEXT)
            diagram.blit(ll, (bx+box_w//2 - ll.get_width()//2, dy+6))
            dd = fx.render(_fit_label(fx,
                           f"Conv({channels}→{channels})→BN→ReLU→Conv→BN + skip→ReLU",
                           box_w-8), True, S.C_DIM)
            diagram.blit(dd, (bx+box_w//2 - dd.get_width()//2, dy+28))
            if i < n_blocks - 1:
                _arrow(bx+box_w//2, dy+rb_h, bx+box_w//2, dy+rb_h+gap_b)
            dy += rb_h + gap_b
        _arrow(bx+box_w//2, dy, bx+box_w//2, dy+arr_l)
        dy += arr_l
        split_y = dy
        hw = (box_w - 10) // 2
        _box("Policy head", f"Conv→Lin({G*G})",
             bx,       split_y, hw, bh_b, border=_HEAD_COL, lbl_col=_HEAD_COL)
        _box("Value head",  "Conv→Lin(64)→Lin(1)",
             bx+hw+10, split_y, hw, bh_b, border=_HEAD_COL, lbl_col=_HEAD_COL)
        mid_x = bx + box_w // 2
        pygame.draw.line(diagram, S.C_BORDER,
                         (mid_x, split_y-arr_l), (bx+hw//2, split_y), 1)
        pygame.draw.line(diagram, S.C_BORDER,
                         (mid_x, split_y-arr_l), (bx+hw+10+hw//2, split_y), 1)
        dy = split_y + bh_b + 6
        rf = 3 + 4 * n_blocks
        for note in (f"Receptive field: {rf}×{rf}  ({'covers' if rf>=G else 'partial'} {G}×{G})",
                     f"PPO clip={S.PT_PPO_CLIP}, GAE λ={S.PT_GAE_LAMBDA}, {S.PT_PPO_EPOCHS} epochs/game"):
            t = fx.render(_fit_label(fx, note, box_w), True, S.C_DIM)
            diagram.blit(t, (bx+box_w//2 - t.get_width()//2, dy))
            dy += t.get_height() + 3

    elif ai_type == "NM":
        # Neural MCTS: show the search loop diagram, then the same CNN as PT
        _NM_COL = (200, 140, 220)
        for label, detail in [
            ("Board state",      f"15 ch × {G}×{G} = {15*G*G} values"),
            ("Policy head",      f"→ prior P(s,a) for all {G*G} moves"),
            ("Value head",       "→ position value  ∈ [-1, 1]"),
            ("Root priors",      "Dirichlet noise + tactical boost applied"),
            ("PUCT selection",   "Q + c_puct · P · √N_parent / (1 + N_child)"),
            ("Expand leaf",      "network call → populate children with priors"),
            ("Backprop value",   "value head result flows up tree (no rollout)"),
            ("Visit distribution", "π_mcts = visit_counts / total visits"),
        ]:
            _box(label, detail, bx, dy, box_w, bh_b,
                 border=_NM_COL, lbl_col=_NM_COL if label in ("Policy head", "Value head") else None)
            _arrow(bx + box_w//2, dy + bh_b, bx + box_w//2, dy + bh_b + arr_l)
            dy += bh_b + arr_l
        dy -= arr_l

        dy += 16
        pygame.draw.line(diagram, S.C_BORDER, (40, dy), (win_w-40, dy), 1)
        t = fx.render("Training: CE(π_mcts, π_net) + MSE(v_pred, GAE_return) + entropy",
                      True, S.C_DIM)
        dy += 6
        diagram.blit(t, (cx - t.get_width()//2, dy))
        dy += t.get_height() + 4
        t2 = fx.render("No PPO needed — visit distribution is a stable supervised target",
                       True, S.C_DIM)
        diagram.blit(t2, (cx - t2.get_width()//2, dy))
        dy += t2.get_height() + 4
        t3 = fx.render(f"c_puct={S.NM_C_PUCT}  noise_frac={S.NM_NOISE_FRAC}"
                       f"  lr={S.NM_LEARNING_RATE}  γ={S.NM_DISCOUNT}  λ={S.NM_GAE_LAMBDA}",
                       True, S.C_DIM)
        diagram.blit(t3, (cx - t3.get_width()//2, dy))
        dy += t3.get_height() + 4

    # Channel legend for NN / PT / NM
    if ai_type in ("NN", "PT", "NM"):
        dy += 14
        pygame.draw.line(diagram, S.C_BORDER, (40, dy), (win_w-40, dy), 1)
        dy += 8
        lt = fx.render("15 input channels (shared by NN, PyTorch, and Neural MCTS):", True, _AI_COL)
        diagram.blit(lt, (cx - lt.get_width()//2, dy))
        dy += lt.get_height() + 4
        ch_desc = [
            "ch  0: own dots",              "ch  1: opponent dots",
            "ch  2: forbidden positions",   "ch  3: own connection mask",
            "ch  4: opp connection mask",   "ch  5: own closing score ÷ 4",
            "ch  6: opp threat score ÷ 4",  "ch  7: own enclosure potential",
            "ch  8: opp enclosure potential","ch  9: own bridge potential",
            "ch 10: opp bridge potential",  "ch 11: disruption map",
            "ch 12: fork map",              "ch 13: game phase",
            "ch 14: centrality",
        ]
        ncols  = 3
        item_w = (win_w - 80) // ncols
        rh2    = fx.size("x")[1] + 3
        for i, desc in enumerate(ch_desc):
            ix = 40 + (i % ncols) * item_w
            iy = dy + (i // ncols) * rh2
            diagram.blit(fx.render(desc, True, S.C_DIM), (ix, iy))
        dy += ((len(ch_desc) + ncols - 1) // ncols) * rh2 + 10

    content_h  = dy + 20
    max_scroll = max(0, content_h - visible_h)
    scroll_y   = max(0, min(scroll_y, max_scroll))

    surf.blit(diagram, (0, scroll_top),
              pygame.Rect(0, scroll_y, win_w, visible_h))

    up_r = down_r = None
    if max_scroll > 0:
        arr_sz = 28
        up_r   = pygame.Rect(win_w - arr_sz - 6, scroll_top + 4,        arr_sz, arr_sz)
        down_r = pygame.Rect(win_w - arr_sz - 6, scroll_bottom-arr_sz-4, arr_sz, arr_sz)
        _button(surf, fs, "▲", up_r,   col=S.C_GOLD if scroll_y > 0          else S.C_DIM)
        _button(surf, fs, "▼", down_r, col=S.C_GOLD if scroll_y < max_scroll  else S.C_DIM)
        # Scrollbar
        tx  = win_w - 5
        tt  = scroll_top  + arr_sz + 6
        tb  = scroll_bottom - arr_sz - 6
        th  = max(1, tb - tt)
        if th > 0 and content_h > 0:
            tmb_h = max(20, int(th * visible_h / content_h))
            tmb_y = tt + int((th - tmb_h) * scroll_y / max(max_scroll, 1))
            pygame.draw.rect(surf, S.C_BORDER, (tx, tt, 4, th),    border_radius=2)
            pygame.draw.rect(surf, S.C_GRID5,  (tx, tmb_y, 4, tmb_h), border_radius=2)

    return close_r, up_r, down_r, content_h


# ── Training config overlay ───────────────────────────────────────────────────

_TRAIN_AI_OPTIONS    = ["MCTS", "Neural Net", "PyTorch Net", "Neural MCTS"]
_TRAIN_OPP_OPTIONS   = ["Self (same type)", "MCTS", "Neural Net", "PyTorch Net", "Neural MCTS"]
_TRAIN_ROUND_PRESETS = [10, 50, 100, 500, 1000, 5000]


def draw_train_config(surf, fonts, win_w, win_h,
                      ai_idx, rounds_idx, opp_idx=0, think_idx=0,
                      ai_profile_idx=0, opp_profile_idx=0,
                      replay_count=0):
    """Training configuration overlay.

    Returns (back_rect, start_rect, replay_rect,
             prev_ai, next_ai, prev_opp, next_opp,
             prev_rnd, next_rnd, prev_think, next_think,
             deep_terr_r,
             prev_ai_prof, next_ai_prof,    # None when AI is not MCTS
             prev_opp_prof, next_opp_prof). # None when opp is not MCTS
    """
    overlay = pygame.Surface((win_w, win_h), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 200))
    surf.blit(overlay, (0, 0))

    ft  = fonts['title']
    fs  = fonts['small']
    fx  = fonts['tiny']
    cx  = win_w // 2
    arr = 30
    val_w = 200
    btn_w = arr + 6 + val_w + 6 + arr
    row_h = 30
    lbl_h = 14
    row_gap = 14

    # Determine which profile rows to show
    ai_label     = _TRAIN_AI_OPTIONS[ai_idx]
    opp_label    = _TRAIN_OPP_OPTIONS[opp_idx]
    resolved_opp = ai_label if opp_label == "Self (same type)" else opp_label
    show_ai_prof  = (ai_label == "MCTS")
    show_opp_prof = (resolved_opp == "MCTS")
    both_mcts     = show_ai_prof and show_opp_prof
    n_prof_rows   = (1 if show_ai_prof else 0) + (1 if show_opp_prof else 0)

    total_h = (lbl_h + row_h + row_gap) * (4 + n_prof_rows) + 36 + 20 + 60 + 38 + 30
    y       = max(30, win_h // 2 - total_h // 2)

    title = ft.render("TRAIN  AI", True, S.C_GOLD)
    surf.blit(title, (cx - title.get_width() // 2, y))
    y += title.get_height() + 20

    def _row(label, value_str, y_):
        lbl = fx.render(label, True, S.C_DIM)
        surf.blit(lbl, (cx - lbl.get_width() // 2, y_))
        y_ctrl = y_ + lbl.get_height() + 4
        bx = cx - btn_w // 2
        pv = pygame.Rect(bx,                       y_ctrl, arr, row_h)
        nv = pygame.Rect(bx + arr + 6 + val_w + 6, y_ctrl, arr, row_h)
        _button(surf, fs, "◀", pv)
        _button(surf, fs, "▶", nv)
        vbox = pygame.Rect(bx + arr + 6, y_ctrl, val_w, row_h)
        pygame.draw.rect(surf, (30, 33, 45), vbox, border_radius=4)
        pygame.draw.rect(surf, S.C_BORDER,   vbox, width=1, border_radius=4)
        vt = fs.render(_fit_label(fs, value_str, val_w - 8), True, S.C_TEXT)
        surf.blit(vt, (vbox.x + (val_w - vt.get_width()) // 2,
                       vbox.y + (row_h - vt.get_height()) // 2))
        return pv, nv, lbl.get_height() + 4 + row_h

    h = 0
    prev_ai,    next_ai,    h = _row("AI type",         _TRAIN_AI_OPTIONS[ai_idx],             y); y += h + row_gap
    prev_opp,   next_opp,   h = _row("Opponent",        _TRAIN_OPP_OPTIONS[opp_idx],           y); y += h + row_gap
    prev_think, next_think, h = _row("Think time (ms)", str(S.AI_THINK_PRESETS[think_idx]),    y); y += h + row_gap
    prev_rnd,   next_rnd,   h = _row("Rounds",          str(_TRAIN_ROUND_PRESETS[rounds_idx]), y); y += h + row_gap

    # Profile rows — only for MCTS players
    prev_ai_prof = next_ai_prof = None
    prev_opp_prof = next_opp_prof = None

    if show_ai_prof:
        ai_prof_label  = "P1 profile" if both_mcts else "MCTS profile"
        ai_prof_val    = S.MCTS_PROFILE_LABELS.get(ai_profile_idx, "Default")
        prev_ai_prof, next_ai_prof, h = _row(ai_prof_label, ai_prof_val, y); y += h + row_gap

    if show_opp_prof:
        opp_prof_label = "P2 profile" if both_mcts else "MCTS profile"
        opp_prof_val   = S.MCTS_PROFILE_LABELS.get(opp_profile_idx, "Default")
        prev_opp_prof, next_opp_prof, h = _row(opp_prof_label, opp_prof_val, y); y += h + row_gap

    y -= row_gap  # remove last extra gap before deep-terr toggle
    y += 6

    # Deep territory toggle
    _on   = getattr(S, 'MCTS_REAL_ROLLOUT', False)
    _tcol = (60, 180, 100) if _on else (110, 110, 110)
    _tlbl = "DEEP TERR: ON (accurate)" if _on else "DEEP TERR: OFF (fast)"
    deep_w      = btn_w
    deep_terr_r = pygame.Rect(cx - deep_w // 2, y, deep_w, 32)
    _button(surf, fx, _tlbl, deep_terr_r, col=_tcol)
    y += 32 + 6

    info = fx.render("Deep Terr: full flood-fill scoring per rollout step (~3× fewer rollouts, same think time)", True, S.C_DIM)
    surf.blit(info, (cx - info.get_width() // 2, y))
    y += info.get_height() + 4
    info2 = fx.render("Reward shaping + arc potential shaping enabled", True, S.C_DIM)
    surf.blit(info2, (cx - info2.get_width() // 2, y))
    y += info2.get_height() + 14

    bw_, bh_ = 120, 38
    gap = 16
    back_r  = pygame.Rect(cx - bw_ - gap // 2, y, bw_, bh_)
    start_r = pygame.Rect(cx + gap // 2,        y, bw_, bh_)
    _button(surf, fs, "BACK",  back_r)
    _button(surf, fs, "START", start_r, col=S.C_GOLD)
    y += bh_ + 10

    # Replay human games button
    if replay_count > 0:
        rp_lbl = f"REPLAY  {replay_count} HUMAN GAME{'S' if replay_count != 1 else ''}"
        rp_col = (80, 160, 220)
    else:
        rp_lbl = "REPLAY HUMAN GAMES  (none recorded)"
        rp_col = (80, 80, 80)
    rp_w = btn_w + arr
    replay_r = pygame.Rect(cx - rp_w // 2, y, rp_w, 32)
    _button(surf, fx, rp_lbl, replay_r, col=rp_col)
    hint = fx.render("Replays saved games × Rounds — same AI type as START", True, S.C_DIM)
    surf.blit(hint, (cx - hint.get_width() // 2, y + 34))

    return (back_r, start_r, replay_r,
            prev_ai, next_ai, prev_opp, next_opp,
            prev_rnd, next_rnd, prev_think, next_think,
            deep_terr_r,
            prev_ai_prof, next_ai_prof,
            prev_opp_prof, next_opp_prof)


# ── Training progress overlay ─────────────────────────────────────────────────

def draw_train_progress(surf, fonts, win_w, win_h,
                        completed, total, wins, ai_label, cancel_rect_out,
                        eta_secs=None):
    overlay = pygame.Surface((win_w, win_h), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 210))
    surf.blit(overlay, (0, 0))

    ft = fonts['title']
    fs = fonts['small']
    fx = fonts['tiny']
    cx = win_w // 2
    cy = win_h // 2 - 60

    title = ft.render(f"TRAINING  {ai_label}", True, S.C_GOLD)
    surf.blit(title, (cx - title.get_width() // 2, cy))
    cy += title.get_height() + 16

    bar_w, bar_h = 320, 18
    bar_x = cx - bar_w // 2
    pygame.draw.rect(surf, S.C_BORDER, (bar_x, cy, bar_w, bar_h), border_radius=4)
    pct = completed / max(total, 1)
    if pct > 0:
        pygame.draw.rect(surf, S.C_GOLD,
                         (bar_x, cy, int(bar_w * pct), bar_h), border_radius=4)
    pygame.draw.rect(surf, S.C_GRID5, (bar_x, cy, bar_w, bar_h), width=1, border_radius=4)
    cy += bar_h + 8

    pct_lbl = fx.render(f"{completed} / {total}  ({int(pct*100)}%)", True, S.C_TEXT)
    surf.blit(pct_lbl, (cx - pct_lbl.get_width() // 2, cy))
    cy += pct_lbl.get_height() + 6

    if eta_secs is not None and completed > 0:
        if eta_secs < 60:
            eta_str = f"About {int(eta_secs)+1}s remaining"
        elif eta_secs < 3600:
            m, s = divmod(int(eta_secs), 60)
            eta_str = f"About {m}m {s:02d}s remaining"
        else:
            h_, rem = divmod(int(eta_secs), 3600)
            eta_str = f"About {h_}h {rem//60}m remaining"
    else:
        eta_str = "Estimating time…"
    eta_lbl = fx.render(eta_str, True, S.C_GOLD)
    surf.blit(eta_lbl, (cx - eta_lbl.get_width() // 2, cy))
    cy += eta_lbl.get_height() + 10

    w1 = wins.get(1, 0); w2 = wins.get(2, 0); wd = wins.get('draw', 0)
    stat = fx.render(f"P1 wins: {w1}    P2 wins: {w2}    Draws: {wd}", True, S.C_DIM)
    surf.blit(stat, (cx - stat.get_width() // 2, cy))
    cy += stat.get_height() + 14

    bw_, bh_ = 120, 36
    cancel_r = pygame.Rect(cx - bw_ // 2, cy, bw_, bh_)
    _button(surf, fs, "CANCEL", cancel_r, col=(220, 80, 60))
    if cancel_rect_out is not None:
        if cancel_rect_out:
            cancel_rect_out[0] = cancel_r
        else:
            cancel_rect_out.append(cancel_r)
