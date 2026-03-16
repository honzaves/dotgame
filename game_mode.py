"""Game-mode selector, active-mode bookkeeping and player name resolution."""

from enum import Enum, auto


class Mode(Enum):
    HUMAN_VS_HUMAN      = auto()
    HUMAN_VS_MCTS       = auto()   # P1=human,  P2=MCTS
    MCTS_VS_MCTS        = auto()   # P1=MCTS,   P2=MCTS
    HUMAN_VS_NN         = auto()   # P1=human,  P2=NN
    NN_VS_NN            = auto()   # P1=NN,     P2=NN
    MCTS_VS_NN          = auto()   # P1=MCTS,   P2=NN
    HUMAN_VS_PT         = auto()   # P1=human,  P2=PyTorch
    PT_VS_PT            = auto()   # P1=PT,     P2=PT
    MCTS_VS_PT          = auto()   # P1=MCTS,   P2=PT
    NN_VS_PT            = auto()   # P1=NN,     P2=PT
    HUMAN_VS_NM         = auto()   # P1=human,  P2=Neural MCTS
    NM_VS_NM            = auto()   # P1=NM,     P2=NM
    MCTS_VS_NM          = auto()   # P1=MCTS,   P2=Neural MCTS
    NN_VS_NM            = auto()   # P1=NN,     P2=Neural MCTS
    PT_VS_NM            = auto()   # P1=PT,     P2=Neural MCTS

    # Legacy alias kept so old saves/references don't break
    HUMAN_VS_COMPUTER   = HUMAN_VS_MCTS
    COMPUTER_LEARNING   = MCTS_VS_MCTS


MODE_LABELS = {
    Mode.HUMAN_VS_HUMAN: "Human  vs  Human",
    Mode.HUMAN_VS_MCTS:  "Human  vs  MCTS",
    Mode.MCTS_VS_MCTS:   "MCTS  vs  MCTS  (self-play)",
    Mode.HUMAN_VS_NN:    "Human  vs  Neural Net",
    Mode.NN_VS_NN:       "Neural Net  vs  Neural Net",
    Mode.MCTS_VS_NN:     "MCTS  vs  Neural Net",
    Mode.HUMAN_VS_PT:    "Human  vs  PyTorch Net",
    Mode.PT_VS_PT:       "PyTorch Net  vs  PyTorch Net",
    Mode.MCTS_VS_PT:     "MCTS  vs  PyTorch Net",
    Mode.NN_VS_PT:       "Neural Net  vs  PyTorch Net",
    Mode.HUMAN_VS_NM:    "Human  vs  Neural MCTS",
    Mode.NM_VS_NM:       "Neural MCTS  vs  Neural MCTS",
    Mode.MCTS_VS_NM:     "MCTS  vs  Neural MCTS",
    Mode.NN_VS_NM:       "Neural Net  vs  Neural MCTS",
    Mode.PT_VS_NM:       "PyTorch Net  vs  Neural MCTS",
}

_MODE_TYPE_LABELS = {
    Mode.HUMAN_VS_HUMAN: ("Human",        "Human"),
    Mode.HUMAN_VS_MCTS:  ("Human",        "MCTS"),
    Mode.MCTS_VS_MCTS:   ("MCTS",         "MCTS"),
    Mode.HUMAN_VS_NN:    ("Human",        "Neural Net"),
    Mode.NN_VS_NN:       ("Neural Net",   "Neural Net"),
    Mode.MCTS_VS_NN:     ("MCTS",         "Neural Net"),
    Mode.HUMAN_VS_PT:    ("Human",        "PyTorch Net"),
    Mode.PT_VS_PT:       ("PyTorch Net",  "PyTorch Net"),
    Mode.MCTS_VS_PT:     ("MCTS",         "PyTorch Net"),
    Mode.NN_VS_PT:       ("Neural Net",   "PyTorch Net"),
    Mode.HUMAN_VS_NM:    ("Human",        "Neural MCTS"),
    Mode.NM_VS_NM:       ("Neural MCTS",  "Neural MCTS"),
    Mode.MCTS_VS_NM:     ("MCTS",         "Neural MCTS"),
    Mode.NN_VS_NM:       ("Neural Net",   "Neural MCTS"),
    Mode.PT_VS_NM:       ("PyTorch Net",  "Neural MCTS"),
}

current_mode: Mode = Mode.HUMAN_VS_HUMAN
active_names: dict = {1: "Player I", 2: "Player II"}


def apply_names(mode: Mode, custom: dict | None = None) -> None:
    import settings as S
    type_p1, type_p2 = _MODE_TYPE_LABELS[mode]
    base_names = dict(S.PLAYER_NAMES)

    def resolve(player: int, ptype: str) -> str:
        base = (custom or {}).get(player, '').strip() or base_names.get(player, f"Player {player}")
        if ptype != 'Human' and ptype not in base:
            return f"{base}  ({ptype})"
        return base

    active_names[1] = resolve(1, type_p1)
    active_names[2] = resolve(2, type_p2)


def player_name(player: int) -> str:
    return active_names.get(player, f"Player {player}")


def human_controls(player: int) -> bool:
    if current_mode == Mode.HUMAN_VS_HUMAN:
        return True
    if current_mode in (Mode.HUMAN_VS_MCTS, Mode.HUMAN_VS_NN,
                        Mode.HUMAN_VS_PT,   Mode.HUMAN_VS_NM):
        return player == 1
    return False
