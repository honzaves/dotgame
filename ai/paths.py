import os
import settings as S


def experience_path(base: str, ext: str) -> str:
    """Return an absolute path with the grid size embedded in the filename.

    Example: base="../ai_experience", ext=".json", GRID=25
             -> /abs/path/to/ai_experience_25x25.json
    """
    name = f"{os.path.basename(base)}_{S.GRID}x{S.GRID}{ext}"
    directory = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        os.path.dirname(base),
    )
    return os.path.normpath(os.path.join(directory, name))
