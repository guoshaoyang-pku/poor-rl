"""Visual observation renderer for the vision arm (w6).

Renders the headless SuikaEnv board to a compact RGB image for the Qwen3.5
VL ViT (patch 16, merge 2 -> keep both sides multiples of 32 so the token
grid is exact: 288x416 -> 9x13 = 117 vision tokens).

Beyond suika_env.render() this adds the overlays a policy actually needs:
  - kill line at config.pad.killy (moves with the geometry);
  - current fruit: cloud + guide line + sprite at its x (PreParticle.draw);
  - next-fruit preview stamped inside the crop (top-right corner);
  - score text is skipped (V/pi heads do not need it).

Physics is untouched: drawing happens on the env's off-screen surface after
settling, identical to render(mode="rgb_array") plus overlays.
"""
import io

import numpy as np

from paths import setup_engine_path
setup_engine_path()

import pygame  # noqa: E402
from config import config  # noqa: E402

_CROP_PAD = 8          # pixels of margin around the play area
_KILL_COLOR = (220, 40, 40)
_PREVIEW_POS = (-44, 52)   # next-fruit stamp: offset from (pad.right, crop top)


def crop_box():
    """Current-geometry crop (x0, y0, x1, y1) around the play area."""
    return (int(config.pad.left) - _CROP_PAD, 0,
            int(config.pad.right) + _CROP_PAD, int(config.pad.bot) + 1)


def render_board(env, out_size=(288, 416)):
    """env: SuikaEnv (or DQNEnv.env). Returns uint8 [H, W, 3]."""
    if env._surface is None:
        env._surface = pygame.Surface(
            (config.screen.width, config.screen.height))
    surf = env._surface
    surf.blit(config.background_blit, (0, 0))
    for p in env._live_particles():
        p.draw(surf)
    # kill line
    pygame.draw.line(surf, _KILL_COLOR,
                     (config.pad.left, config.pad.killy),
                     (config.pad.right, config.pad.killy), 2)
    # current fruit: cloud + guide line + sprite (same call as the game UI)
    env.curr.draw(surf, wait=False)
    # next-fruit preview stamped inside the crop (game UI puts it at 1084,185,
    # which is outside the play-area crop)
    nxt = env.next
    px = config.pad.right + _PREVIEW_POS[0]
    py = _PREVIEW_POS[1]
    surf.blit(nxt.sprite, nxt._sprite_pos((px, py)))

    arr = pygame.surfarray.array3d(surf)          # (W, H, 3)
    arr = np.transpose(arr, (1, 0, 2))            # (H, W, 3)
    x0, y0, x1, y1 = crop_box()
    crop = arr[y0:y1, x0:x1]
    if out_size is not None:
        from PIL import Image
        img = Image.fromarray(crop).resize(out_size, Image.BILINEAR)
        return np.asarray(img, dtype=np.uint8)
    return np.ascontiguousarray(crop)


def encode_jpeg(img, quality=85):
    """uint8 [H,W,3] -> JPEG bytes."""
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def render_jpeg(env, out_size=(288, 416), quality=85):
    return encode_jpeg(render_board(env, out_size=out_size), quality=quality)
