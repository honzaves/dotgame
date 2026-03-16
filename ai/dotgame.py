"""
Dot Grid — Two Player
=====================
Install:  pip install pygame
Run:      python dotgame.py

Controls:
  Click intersection     — place dot
  Scroll / + / -         — zoom in / out
  Right-click drag       — pan
  R                      — new game (mode picker)
  Esc                    — pause menu
  ← / →                  — step through moves after game ends
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pygame

import settings as S
import state
import viewport as vp
import game_mode as gm
from game_mode import Mode, apply_names
from draw import (draw_board, draw_panel, draw_win_screen,
                  draw_replay_controls, draw_mode_picker, draw_pause_menu,
                  draw_settings_screen, draw_arch_select, draw_arch_overlay,
                  draw_train_config, draw_train_progress)
from ai.mcts import MCTSPlayer
from ai.nn_player import NNPlayer
from ai.runner import AIRunner
from ai.trainer import Trainer
try:
    from ai.pytorch_player import PyTorchPlayer
    _PT_AVAILABLE = True
except ImportError:
    _PT_AVAILABLE = False
    PyTorchPlayer = None

try:
    from ai.neural_mcts_player import NeuralMCTSPlayer
    _NM_AVAILABLE = True
except ImportError:
    _NM_AVAILABLE = False
    NeuralMCTSPlayer = None


# ── Font helper ───────────────────────────────────────────────────────────────

def _make_font(size, bold=False):
    for name in ("Menlo", "Consolas", "Courier New"):
        try:
            f = pygame.font.SysFont(name, size, bold=bold)
            if f:
                return f
        except Exception:
            pass
    return pygame.font.SysFont(None, size, bold=bold)


# ── AI setup ──────────────────────────────────────────────────────────────────

_MODE_PLAYERS = {
    Mode.HUMAN_VS_HUMAN: ('human', 'human'),
    Mode.HUMAN_VS_MCTS:  ('human', 'mcts'),
    Mode.MCTS_VS_MCTS:   ('mcts',  'mcts'),
    Mode.HUMAN_VS_NN:    ('human', 'nn'),
    Mode.NN_VS_NN:       ('nn',    'nn'),
    Mode.MCTS_VS_NN:     ('mcts',  'nn'),
    Mode.HUMAN_VS_PT:    ('human', 'pt'),
    Mode.PT_VS_PT:       ('pt',    'pt'),
    Mode.MCTS_VS_PT:     ('mcts',  'pt'),
    Mode.NN_VS_PT:       ('nn',    'pt'),
    Mode.HUMAN_VS_NM:    ('human', 'nm'),
    Mode.NM_VS_NM:       ('nm',    'nm'),
    Mode.MCTS_VS_NM:     ('mcts',  'nm'),
    Mode.NN_VS_NM:       ('nn',    'nm'),
    Mode.PT_VS_NM:       ('pt',    'nm'),
}



def _make_runner(player_id: int, ptype: str, profile: int = 0) -> AIRunner:
    if ptype == 'nn':
        return AIRunner(NNPlayer(player_id=player_id))
    if ptype == 'pt':
        if not _PT_AVAILABLE:
            raise RuntimeError("PyTorch is not installed.\n\nRun:  pip install torch")
        return AIRunner(PyTorchPlayer(player_id=player_id))
    if ptype == 'nm':
        if not _NM_AVAILABLE:
            raise RuntimeError("PyTorch is not installed.\n\nRun:  pip install torch")
        return AIRunner(NeuralMCTSPlayer(player_id=player_id))
    return AIRunner(MCTSPlayer(player_id=player_id, profile=profile))


def _make_runners(mode: Mode, profile: int = 0) -> dict:
    p1_type, p2_type = _MODE_PLAYERS.get(mode, ('mcts', 'mcts'))
    # MCTS vs MCTS always uses the two distinct personalities (Opportunist vs
    # Strategist) so the players pursue different strategies rather than mirroring.
    # All other MCTS instances use the player-selected profile from settings.
    if mode == Mode.MCTS_VS_MCTS:
        p1_profile = 1
        p2_profile = 2
    else:
        p1_profile = profile if p1_type == 'mcts' else 0
        p2_profile = profile if p2_type == 'mcts' else 0
    return {
        1: _make_runner(1, p1_type, p1_profile),
        2: _make_runner(2, p2_type, p2_profile),
    }


# ── New game ──────────────────────────────────────────────────────────────────

def _start_new_game(mode, think_ms, win_w, win_h, custom_names=None):
    gm.current_mode = mode
    S.AI_THINK_MS   = think_ms
    apply_names(mode, custom_names)
    state.reset()
    vp.cell_size = vp.fit_cell(win_w, win_h)
    vp.center(win_w, win_h)


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    pygame.init()
    screen = pygame.display.set_mode((S.WIN_W, S.WIN_H), pygame.RESIZABLE)
    pygame.display.set_caption(f"Dot Grid  {S.GRID}×{S.GRID}")
    clock = pygame.time.Clock()

    fonts = {
        'title': _make_font(20, bold=True),
        'big':   _make_font(34, bold=True),
        'small': _make_font(15),
        'tiny':  _make_font(13),
    }

    win_w, win_h = screen.get_size()
    vp.cell_size  = vp.fit_cell(win_w, win_h)
    vp.center(win_w, win_h)

    runners = _make_runners(gm.current_mode)

    # ── UI state ──────────────────────────────────────────────────────────────
    picking_mode  = True
    paused        = False
    settings_screen = False   # settings overlay
    arch_select     = False   # AI info: pick which arch to show
    arch_overlay    = False   # actual arch diagram
    arch_type       = "MCTS"  # which AI is shown in arch overlay
    arch_scroll     = 0       # scroll offset within arch overlay
    arch_content_h  = 0       # total content height (for clamping)

    think_presets = S.AI_THINK_PRESETS
    think_ms      = S.AI_THINK_DEFAULT
    think_idx     = (think_presets.index(think_ms)
                     if think_ms in think_presets else
                     think_presets.index(min(think_presets, key=lambda x: abs(x - think_ms))))
    game_mcts_profile = 0   # profile used for single-MCTS game modes (0=Default,1=Opp,2=Strat)

    name_inputs  = {1: S.PLAYER_NAMES[1], 2: S.PLAYER_NAMES[2]}
    active_input = None
    _error_msg   = ""

    # Training state
    training_mode   = False
    training        = False
    trainer         = None
    train_ai_idx    = 0
    train_opp_idx   = 0
    train_think_idx = 0
    train_rnd_idx   = 2
    train_ai_profile_idx  = 0   # profile index for AI player in training
    train_opp_profile_idx = 0   # profile index for opponent in training
    train_progress  = [0, 1, {}]
    train_done_msg  = ""
    train_cancel_r  = []
    train_cfg_rects = None
    train_start_time = None

    dragging   = False
    drag_start = (0, 0)
    drag_off   = (0.0, 0.0)
    hover      = None

    # Panel / overlay rects (populated each frame by draw calls)
    reset_rect   = None
    zoom_rects   = None
    stop_rect    = None
    mode_rects   = []
    name_rects   = {}
    train_btn_r  = None
    settings_r   = None
    arch_r       = None
    pause_rects  = ()
    prev_r = next_r = None
    settings_rects   = None   # (back, prev_think, next_think, deep_terr)
    arch_sel_rects   = None   # (back, mcts_r, nn_r, pt_r)
    arch_close_r     = None
    arch_up_r        = None
    arch_down_r      = None

    _SCROLL_STEP = 40

    pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_CROSSHAIR)

    while True:
        win_w, win_h = screen.get_size()
        mx, my       = pygame.mouse.get_pos()
        bw           = vp.board_w(win_w)
        on_board     = mx < bw

        # ── Hover ─────────────────────────────────────────────────────────────
        any_overlay = (picking_mode or training_mode or training or paused
                       or settings_screen or arch_select or arch_overlay)
        if (on_board and not dragging and not any_overlay
                and not state.game_over
                and state.snapshot_index == -1):
            gx, gy = vp.s2g(mx, my)
            if 0 <= gx < S.GRID and 0 <= gy < S.GRID:
                px, py = vp.g2s(gx, gy)
                dist   = ((mx - px) ** 2 + (my - py) ** 2) ** 0.5
                hover  = (gx, gy) if dist < vp.cell_size * 0.48 else None
            else:
                hover = None
        else:
            hover = None

        # ── AI trigger ────────────────────────────────────────────────────────
        cp     = state.current_player
        runner = runners[cp]
        ai_thinking = False

        if (not any_overlay
                and not state.game_over
                and not gm.human_controls(cp)
                and not runner.is_thinking
                and runner.pending_move is None):
            runner.start_thinking(
                state.board, state.connections, cp, state.scores)

        if not gm.human_controls(cp):
            ai_thinking = runner.is_thinking

        # Apply AI move
        if (not any_overlay
                and not state.game_over
                and not gm.human_controls(cp)
                and runner.pending_move is not None):
            move = runner.pending_move
            runner.clear_move()
            # Snapshot BEFORE placing — observe needs state prior to the move
            board_snap     = dict(state.board)
            conn_snap      = set(state.connections)
            forbidden_snap = set(state.forbidden_positions)
            state.place(*move)
            # Symmetric learning: other player observes the position they'll
            # now face, from their own encoding perspective.
            opp_runner = runners.get(3 - cp)
            if opp_runner and not gm.human_controls(3 - cp):
                opp_runner.observe_opponent_move(
                    board_snap, conn_snap, cp, move, forbidden_snap)
            if state.game_over:
                for r in runners.values():
                    r.on_game_end(state.winner_player,
                                  final_scores=dict(state.scores))

        # ── Events ────────────────────────────────────────────────────────────
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                for r in runners.values():
                    r.save()
                pygame.quit()
                sys.exit()

            elif event.type == pygame.VIDEORESIZE:
                win_w, win_h = event.w, event.h
                vp.cell_size = vp.fit_cell(win_w, win_h)
                vp.center(win_w, win_h)

            elif event.type == pygame.KEYDOWN:

                if event.key == pygame.K_ESCAPE:
                    if arch_overlay:
                        arch_overlay = False
                        arch_select  = True
                        arch_scroll  = 0
                    elif arch_select:
                        arch_select  = False
                        picking_mode = True
                    elif settings_screen:
                        settings_screen = False
                        picking_mode    = True
                    elif training:
                        if trainer:
                            trainer.cancel()
                        training      = False
                        training_mode = True
                    elif training_mode:
                        training_mode = False
                        picking_mode  = True
                    elif picking_mode:
                        picking_mode = False
                    elif paused:
                        paused = False
                    elif not state.game_over:
                        paused = True

                elif event.key == pygame.K_r and not paused and not any_overlay:
                    picking_mode = True
                    active_input = None

                elif picking_mode and active_input is not None:
                    if event.key == pygame.K_BACKSPACE:
                        name_inputs[active_input] = name_inputs[active_input][:-1]
                    elif event.key == pygame.K_RETURN:
                        active_input = None
                    elif event.unicode and len(name_inputs[active_input]) < 20:
                        name_inputs[active_input] += event.unicode

                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    vp.zoom_at(bw / 2, win_h / 2, S.ZOOM_STEP, win_w, win_h)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    vp.zoom_at(bw / 2, win_h / 2, 1 / S.ZOOM_STEP, win_w, win_h)

                elif event.key == pygame.K_LEFT and state.game_over:
                    state.replay_step(-1)
                elif event.key == pygame.K_RIGHT and state.game_over:
                    state.replay_step(1)

                # Scroll arch overlay with arrow keys
                elif arch_overlay:
                    if event.key == pygame.K_UP:
                        arch_scroll = max(0, arch_scroll - _SCROLL_STEP)
                    elif event.key == pygame.K_DOWN:
                        arch_scroll = min(
                            max(0, arch_content_h - (win_h - 200)),
                            arch_scroll + _SCROLL_STEP)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:   # scroll up
                    if arch_overlay:
                        arch_scroll = max(0, arch_scroll - _SCROLL_STEP)
                    elif on_board and not any_overlay:
                        vp.zoom_at(mx, my, S.ZOOM_STEP, win_w, win_h)
                elif event.button == 5:  # scroll down
                    if arch_overlay:
                        arch_scroll = min(
                            max(0, arch_content_h - (win_h - 200)),
                            arch_scroll + _SCROLL_STEP)
                    elif on_board and not any_overlay:
                        vp.zoom_at(mx, my, 1 / S.ZOOM_STEP, win_w, win_h)
                elif event.button in (2, 3) and on_board and not any_overlay:
                    dragging   = True
                    drag_start = event.pos
                    drag_off   = (vp.offset_x, vp.offset_y)
                    pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_SIZEALL)

            elif event.type == pygame.MOUSEMOTION:
                if dragging:
                    dx = event.pos[0] - drag_start[0]
                    dy = event.pos[1] - drag_start[1]
                    vp.offset_x = drag_off[0] + dx
                    vp.offset_y = drag_off[1] + dy
                    vp.clamp(win_w, win_h)

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button in (2, 3) and dragging:
                    dragging = False
                    pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_CROSSHAIR)

                elif event.button == 1:
                    pos = event.pos

                    # ── Arch overlay ──────────────────────────────────────────
                    if arch_overlay:
                        if arch_close_r and arch_close_r.collidepoint(pos):
                            arch_overlay = False
                            arch_select  = True
                            arch_scroll  = 0
                        elif arch_up_r and arch_up_r.collidepoint(pos):
                            arch_scroll = max(0, arch_scroll - _SCROLL_STEP)
                        elif arch_down_r and arch_down_r.collidepoint(pos):
                            arch_scroll = min(
                                max(0, arch_content_h - (win_h - 200)),
                                arch_scroll + _SCROLL_STEP)

                    # ── Arch select ───────────────────────────────────────────
                    elif arch_select and arch_sel_rects:
                        back_r, mcts_r, nn_r, pt_r, nm_r = arch_sel_rects
                        if back_r.collidepoint(pos):
                            arch_select  = False
                            picking_mode = True
                        elif mcts_r.collidepoint(pos):
                            arch_type    = "MCTS"
                            arch_overlay = True
                            arch_select  = False
                            arch_scroll  = 0
                        elif nn_r.collidepoint(pos):
                            arch_type    = "NN"
                            arch_overlay = True
                            arch_select  = False
                            arch_scroll  = 0
                        elif pt_r.collidepoint(pos):
                            arch_type    = "PT"
                            arch_overlay = True
                            arch_select  = False
                            arch_scroll  = 0
                        elif nm_r.collidepoint(pos):
                            arch_type    = "NM"
                            arch_overlay = True
                            arch_select  = False
                            arch_scroll  = 0

                    # ── Settings screen ───────────────────────────────────────
                    elif settings_screen and settings_rects:
                        back_r, prev_t, next_t, deep_r, prev_prof, next_prof = settings_rects
                        if back_r.collidepoint(pos):
                            settings_screen = False
                            picking_mode    = True
                        elif prev_t.collidepoint(pos):
                            think_idx = max(0, think_idx - 1)
                            think_ms  = think_presets[think_idx]
                            S.AI_THINK_MS = think_ms
                        elif next_t.collidepoint(pos):
                            think_idx = min(len(think_presets) - 1, think_idx + 1)
                            think_ms  = think_presets[think_idx]
                            S.AI_THINK_MS = think_ms
                        elif deep_r.collidepoint(pos):
                            S.MCTS_REAL_ROLLOUT = not S.MCTS_REAL_ROLLOUT
                        elif prev_prof.collidepoint(pos):
                            game_mcts_profile = max(0, game_mcts_profile - 1)
                        elif next_prof.collidepoint(pos):
                            game_mcts_profile = min(2, game_mcts_profile + 1)

                    # ── Training config ───────────────────────────────────────
                    elif training_mode and train_cfg_rects and not training:
                        (back_r, start_r,
                         pra, nxa, pro, nxo,
                         prr, nxr, prt, nxt,
                         train_deep_r,
                         pr_aip, nx_aip,
                         pr_opp_p, nx_opp_p) = train_cfg_rects
                        from draw import (_TRAIN_AI_OPTIONS, _TRAIN_ROUND_PRESETS,
                                          _TRAIN_OPP_OPTIONS)
                        if back_r.collidepoint(pos):
                            training_mode = False
                            picking_mode  = True
                        elif train_deep_r.collidepoint(pos):
                            S.MCTS_REAL_ROLLOUT = not S.MCTS_REAL_ROLLOUT
                        elif start_r.collidepoint(pos):
                            ai_label  = _TRAIN_AI_OPTIONS[train_ai_idx]
                            opp_label = _TRAIN_OPP_OPTIONS[train_opp_idx]

                            def _make_train_player(pid, label, profile=0):
                                if label == "MCTS":
                                    return MCTSPlayer(player_id=pid, profile=profile)
                                if label == "Neural Net":
                                    return NNPlayer(player_id=pid)
                                if label == "Neural MCTS":
                                    if not _NM_AVAILABLE:
                                        raise RuntimeError("PyTorch not installed. Run: pip install torch")
                                    return NeuralMCTSPlayer(player_id=pid)
                                if not _PT_AVAILABLE:
                                    raise RuntimeError("PyTorch not installed. Run: pip install torch")
                                return PyTorchPlayer(player_id=pid)

                            resolved_opp = ai_label if opp_label == "Self (same type)" else opp_label
                            try:
                                tp1 = _make_train_player(1, ai_label,   train_ai_profile_idx)
                                tp2 = _make_train_player(2, resolved_opp, train_opp_profile_idx)
                            except RuntimeError as exc:
                                _error_msg    = str(exc)
                                training_mode = False
                                picking_mode  = True
                            else:
                                S.AI_THINK_MS = S.AI_THINK_PRESETS[train_think_idx]
                                n_rounds = _TRAIN_ROUND_PRESETS[train_rnd_idx]
                                train_progress[:] = [0, n_rounds, {}]
                                train_cancel_r.clear()

                                def _on_progress(done, total, wins):
                                    train_progress[:] = [done, total, wins]

                                def _on_done(wins):
                                    train_progress[0] = train_progress[1]

                                trainer = Trainer(tp1, tp2, n_rounds,
                                                  on_progress=_on_progress,
                                                  on_done=_on_done)
                                trainer.start()
                                import time as _time_
                                train_start_time = _time_.time()
                                training = True
                        elif pra.collidepoint(pos):
                            train_ai_idx = max(0, train_ai_idx - 1)
                        elif nxa.collidepoint(pos):
                            train_ai_idx = min(len(_TRAIN_AI_OPTIONS)-1, train_ai_idx+1)
                        elif pro.collidepoint(pos):
                            train_opp_idx = max(0, train_opp_idx - 1)
                        elif nxo.collidepoint(pos):
                            train_opp_idx = min(len(_TRAIN_OPP_OPTIONS)-1, train_opp_idx+1)
                        elif prr.collidepoint(pos):
                            train_rnd_idx = max(0, train_rnd_idx - 1)
                        elif nxr.collidepoint(pos):
                            train_rnd_idx = min(len(_TRAIN_ROUND_PRESETS)-1, train_rnd_idx+1)
                        elif prt.collidepoint(pos):
                            train_think_idx = max(0, train_think_idx - 1)
                        elif nxt.collidepoint(pos):
                            train_think_idx = min(len(S.AI_THINK_PRESETS)-1,
                                                  train_think_idx+1)
                        elif pr_aip and pr_aip.collidepoint(pos):
                            train_ai_profile_idx = max(0, train_ai_profile_idx - 1)
                        elif nx_aip and nx_aip.collidepoint(pos):
                            train_ai_profile_idx = min(2, train_ai_profile_idx + 1)
                        elif pr_opp_p and pr_opp_p.collidepoint(pos):
                            train_opp_profile_idx = max(0, train_opp_profile_idx - 1)
                        elif nx_opp_p and nx_opp_p.collidepoint(pos):
                            train_opp_profile_idx = min(2, train_opp_profile_idx + 1)

                    # ── Training in progress ──────────────────────────────────
                    elif training and train_cancel_r:
                        if train_cancel_r[0].collidepoint(pos):
                            if trainer:
                                trainer.cancel()
                            training      = False
                            train_start_time = None
                            training_mode = True

                    # ── Mode picker ───────────────────────────────────────────
                    elif picking_mode:
                        for rect, mode in mode_rects:
                            if rect.collidepoint(pos):
                                try:
                                    new_runners = _make_runners(mode, game_mcts_profile)
                                except RuntimeError as exc:
                                    _error_msg = str(exc)
                                    break
                                picking_mode = False
                                active_input = None
                                _start_new_game(mode, think_ms, win_w, win_h,
                                                custom_names=dict(name_inputs))
                                runners    = new_runners
                                _error_msg = ""
                                break
                        if settings_r and settings_r.collidepoint(pos):
                            picking_mode    = False
                            settings_screen = True
                        elif arch_r and arch_r.collidepoint(pos):
                            picking_mode = False
                            arch_select  = True
                        elif train_btn_r and train_btn_r.collidepoint(pos):
                            picking_mode  = False
                            training_mode = True
                            _error_msg    = ""
                        if name_rects:
                            for p, nr in name_rects.items():
                                if nr.collidepoint(pos):
                                    active_input = p
                                    break

                    # ── Pause menu ────────────────────────────────────────────
                    elif paused:
                        if len(pause_rects) == 3:
                            resume_r, new_r, quit_r = pause_rects
                            if resume_r.collidepoint(pos):
                                paused = False
                            elif new_r.collidepoint(pos):
                                paused       = False
                                picking_mode = True
                            elif quit_r.collidepoint(pos):
                                for r in runners.values():
                                    r.save()
                                pygame.quit()
                                sys.exit()

                    else:
                        # ── Panel buttons ─────────────────────────────────────
                        if reset_rect and reset_rect.collidepoint(pos):
                            picking_mode = True

                        if zoom_rects:
                            cxp = bw + S.PANEL_W // 2
                            cyp = win_h // 2
                            if zoom_rects[0].collidepoint(pos):
                                vp.zoom_at(cxp, cyp, 1/S.ZOOM_STEP, win_w, win_h)
                            elif zoom_rects[1].collidepoint(pos):
                                vp.zoom_at(cxp, cyp, S.ZOOM_STEP, win_w, win_h)

                        if stop_rect and stop_rect.collidepoint(pos):
                            state.game_over     = True
                            s1, s2 = state.scores[1], state.scores[2]
                            state.winner_player = (None if s1 == s2
                                                   else (1 if s1 > s2 else 2))
                            for r in runners.values():
                                r.on_game_end(state.winner_player,
                                              final_scores=dict(state.scores))

                        if state.game_over:
                            if prev_r and prev_r.collidepoint(pos):
                                state.replay_step(-1)
                            if next_r and next_r.collidepoint(pos):
                                state.replay_step(1)

                        if (on_board and hover
                                and not state.game_over
                                and gm.human_controls(state.current_player)):
                            human_cp       = state.current_player
                            board_snap     = dict(state.board)
                            conn_snap      = set(state.connections)
                            forbidden_snap = set(state.forbidden_positions)
                            state.place(*hover)
                            # AI opponent observes the human's move
                            opp_runner = runners.get(3 - human_cp)
                            if opp_runner and not gm.human_controls(3 - human_cp):
                                opp_runner.observe_opponent_move(
                                    board_snap, conn_snap,
                                    human_cp, hover, forbidden_snap)
                            if state.game_over:
                                for r in runners.values():
                                    r.on_game_end(state.winner_player,
                                                  final_scores=dict(state.scores))

        # ── Render ────────────────────────────────────────────────────────────
        draw_board(screen, hover, win_w, win_h)
        (reset_rect, zoom_rects, stop_rect, _, _) = draw_panel(
            screen, fonts, win_w, win_h,
            ai_thinking=ai_thinking,
            mode=gm.current_mode,
        )

        if state.game_over:
            draw_win_screen(screen, fonts, win_w, win_h)
            prev_r, next_r = draw_replay_controls(screen, fonts, win_w, win_h)

        if paused:
            pause_rects = draw_pause_menu(screen, fonts, win_w, win_h)

        if picking_mode:
            (mode_rects, name_rects,
             train_btn_r, settings_r, arch_r) = draw_mode_picker(
                screen, fonts, win_w, win_h,
                name_inputs, error_msg=_error_msg)

        if settings_screen:
            settings_rects = draw_settings_screen(
                screen, fonts, win_w, win_h, think_ms, game_mcts_profile)

        if arch_select:
            arch_sel_rects = draw_arch_select(screen, fonts, win_w, win_h)

        if arch_overlay:
            (arch_close_r, arch_up_r,
             arch_down_r, arch_content_h) = draw_arch_overlay(
                screen, fonts, win_w, win_h,
                arch_type, scroll_y=arch_scroll)

        if training_mode and not training:
            train_cfg_rects = draw_train_config(
                screen, fonts, win_w, win_h,
                train_ai_idx, train_rnd_idx, train_opp_idx, train_think_idx,
                train_ai_profile_idx, train_opp_profile_idx)

        if training:
            from draw import _TRAIN_AI_OPTIONS, _TRAIN_ROUND_PRESETS
            import time as _time_
            ai_label = _TRAIN_AI_OPTIONS[train_ai_idx]
            _eta = None
            if train_start_time is not None and train_progress[0] > 0:
                _elapsed = _time_.time() - train_start_time
                _done    = train_progress[0]
                _total   = max(train_progress[1], 1)
                _eta     = (_elapsed / _done) * (_total - _done) if _done < _total else 0
            draw_train_progress(
                screen, fonts, win_w, win_h,
                train_progress[0], train_progress[1],
                train_progress[2], ai_label, train_cancel_r,
                eta_secs=_eta)

            if trainer and not trainer.is_running:
                training = False
                train_done_msg = (
                    f"Done! {train_progress[0]} games — "
                    f"P1: {train_progress[2].get(1,0)} wins  "
                    f"P2: {train_progress[2].get(2,0)} wins  "
                    f"Draws: {train_progress[2].get('draw',0)}"
                )
                _error_msg    = train_done_msg
                training_mode = False
                picking_mode  = True

        pygame.display.flip()
        clock.tick(S.FPS)


if __name__ == "__main__":
    main()
