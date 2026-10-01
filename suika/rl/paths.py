"""Headless Suika engine wiring: locate part2 modules, force dummy SDL."""
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

_HERE = os.path.dirname(os.path.abspath(__file__))           # suika_dqn/
_PROJ = os.path.dirname(_HERE)                               # project root

if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def find_suika_root():
    """Directory that contains part2/ and blits/. Override with SUIKA_ROOT."""
    cand = os.environ.get("SUIKA_ROOT")
    if cand:
        return cand
    for c in (os.path.join(_PROJ, "suika"), _PROJ):
        if os.path.isdir(os.path.join(c, "part2")):
            return c
    raise RuntimeError("cannot locate suika engine root (part2/ missing)")


def setup_engine_path():
    root = find_suika_root()
    part2 = os.path.join(root, "part2")
    if part2 not in sys.path:
        sys.path.insert(0, part2)
    # part2/config.py opens "part2/config.yaml" and blits relative to root.
    os.chdir(root)
    return root
