"""Minimal 2D full-screen renderer for PySuika (part2).

This is an ALTERNATIVE entry point. It does NOT touch the original
``part2/main.py`` gameplay. It reuses the exact same physics, merge,
scoring and input logic (``config``, ``particle``, ``collision``,
``wall``, ``preparticle``), and only replaces the *rendering*:

  * full-screen, pure dark background, no clouds / no image sprites
  * each fruit is drawn as a solid 2D circle (one distinct colour per
    level) with a thin outline and the level number in the centre
  * the play-area (container) is centred and scaled up by screen height,
    leaving dark bars on the left/right
  * minimal HUD: score (corner) + next-fruit preview (small circle)
  * mouse to aim/drop, ESC to quit, R to restart after game over

Run interactively (full screen)::

    cd .../suika && /Users/.../envs/suika/bin/python part2/main_minimal.py

Headless smoke test (no window, writes preview PNGs)::

    SDL_VIDEODRIVER=dummy python part2/main_minimal.py --smoke
"""
import os
import sys

# --- make imports + relative asset paths work from any cwd -----------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))   # .../suika/part2
_ROOT = os.path.dirname(_THIS_DIR)                       # .../suika
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
os.chdir(_ROOT)

# In smoke mode force a headless SDL driver *before* importing pygame so no
# OS window is ever created.
_SMOKE = "--smoke" in sys.argv
if _SMOKE:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pygame
import pymunk

pygame.init()

from config import config, CollisionTypes
from particle import Particle
from preparticle import PreParticle
from wall import Wall
from collision import collide

# --------------------------------------------------------------------------- #
# Visual constants
# --------------------------------------------------------------------------- #
BG_COLOR = (30, 30, 35)
WALL_COLOR = (95, 95, 120)
DEATH_LINE_COLOR = (220, 60, 60)
AIM_LINE_COLOR = (110, 110, 130)
OUTLINE_COLOR = (18, 18, 22)
HUD_COLOR = (235, 235, 240)

# one clearly distinguishable colour per fruit level (0..10)
FRUIT_COLORS = [
    (220,  40,  40),   # 0  cherry      red
    (240,  90, 150),   # 1  strawberry  pink
    (150,  70, 200),   # 2  grapes      purple
    (250, 150,  40),   # 3  orange      orange
    (210, 110,  30),   # 4  persimmon   dark orange
    (235,  50,  60),   # 5  apple       crimson
    (180, 220,  70),   # 6  pear        yellow-green
    (250, 190, 200),   # 7  peach       light pink
    (245, 220,  60),   # 8  pineapple   gold
    (90,  200, 110),   # 9  melon       green
    (40,  150,  70),   # 10 watermelon  dark green
]

_FONT_CACHE = {}


def get_font(size, bold=True):
    size = max(8, int(size))
    key = (size, bold)
    if key not in _FONT_CACHE:
        _FONT_CACHE[key] = pygame.font.SysFont("arial", size, bold=bold)
    return _FONT_CACHE[key]


# --------------------------------------------------------------------------- #
# World -> screen transform (physics stays in original coordinates)
# --------------------------------------------------------------------------- #
class View:
    """Maps original world (physics) coordinates onto the centred,
    height-scaled screen region. Physics is never changed."""

    # world vertical window we want to show (a little above the container
    # top so the falling / preview fruit is visible, down to the floor)
    VIEW_TOP = 45
    VIEW_BOT = config.pad.bot + 10           # ~685
    HEIGHT_FILL = 0.94                       # fraction of screen height used

    def __init__(self, screen_w, screen_h):
        self.sw, self.sh = screen_w, screen_h
        view_h = self.VIEW_BOT - self.VIEW_TOP
        self.scale = (screen_h * self.HEIGHT_FILL) / view_h
        cx = (config.pad.left + config.pad.right) / 2.0
        self.ox = screen_w / 2.0 - cx * self.scale
        content_h = view_h * self.scale
        self.oy = (screen_h - content_h) / 2.0 - self.VIEW_TOP * self.scale

    def x(self, wx):
        return self.ox + wx * self.scale

    def y(self, wy):
        return self.oy + wy * self.scale

    def p(self, wx, wy):
        return (self.ox + wx * self.scale, self.oy + wy * self.scale)

    def r(self, radius):
        return radius * self.scale


# --------------------------------------------------------------------------- #
# Drawing primitives (shared by interactive + smoke test)
# --------------------------------------------------------------------------- #
def draw_fruit(screen, view, wx, wy, n):
    color = FRUIT_COLORS[n % 11]
    cx, cy = view.p(wx, wy)
    rad = view.r(config[n % 11, "radius"])
    pygame.draw.circle(screen, color, (cx, cy), rad)
    pygame.draw.circle(screen, OUTLINE_COLOR, (cx, cy), rad, max(1, int(rad * 0.08)))
    if rad >= 9:
        font = get_font(rad * 0.95)
        label = font.render(str(n % 11 + 1), True, OUTLINE_COLOR)
        screen.blit(label, (cx - label.get_width() / 2, cy - label.get_height() / 2))


def draw_scene(screen, view, space, curr, score_val, wait, game_over):
    screen.fill(BG_COLOR)

    l = view.x(config.pad.left)
    r = view.x(config.pad.right)
    t = view.y(config.pad.top)
    b = view.y(config.pad.bot)

    # container walls (open top): left, bottom, right
    wall_w = max(2, int(view.r(4)))
    pygame.draw.line(screen, WALL_COLOR, (l, t), (l, b), wall_w)
    pygame.draw.line(screen, WALL_COLOR, (l, b), (r, b), wall_w)
    pygame.draw.line(screen, WALL_COLOR, (r, b), (r, t), wall_w)

    # death line (red)
    ky = view.y(config.pad.killy)
    pygame.draw.line(screen, DEATH_LINE_COLOR, (l, ky), (r, ky), max(1, int(view.r(2))))

    # aim line + current preview fruit at the top
    if not wait and not game_over:
        ax = view.x(curr.x)
        pygame.draw.line(screen, AIM_LINE_COLOR, (ax, view.y(config.pad.line_top)), (ax, b), 1)
        draw_fruit(screen, view, curr.x, config.pad.top, curr.n)

    # all live fruits
    for p in space.shapes:
        if isinstance(p, Particle) and getattr(p, "alive", False):
            x, y = p.body.position
            draw_fruit(screen, view, x, y, p.n)

    draw_hud(screen, view, score_val)
    if game_over:
        draw_game_over(screen, view)


def draw_hud(screen, view, score_val, next_n=None):
    font = get_font(view.sh * 0.045)
    label = font.render(f"Score: {score_val}", True, HUD_COLOR)
    screen.blit(label, (int(view.sh * 0.03), int(view.sh * 0.03)))
    if next_n is not None:
        small = get_font(view.sh * 0.03)
        nlabel = small.render("Next", True, HUD_COLOR)
        nx = view.sw - int(view.sh * 0.12)
        ny = int(view.sh * 0.03)
        screen.blit(nlabel, (nx, ny))
        rad = view.sh * 0.025
        cx = nx + nlabel.get_width() + rad + 8
        cy = ny + nlabel.get_height() / 2
        pygame.draw.circle(screen, FRUIT_COLORS[next_n % 11], (cx, cy), rad)
        pygame.draw.circle(screen, OUTLINE_COLOR, (cx, cy), rad, 2)


def draw_game_over(screen, view):
    overlay = pygame.Surface((view.sw, view.sh), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 150))
    screen.blit(overlay, (0, 0))
    font = get_font(view.sh * 0.09)
    label = font.render("Game Over", True, (255, 90, 90))
    screen.blit(label, (view.sw / 2 - label.get_width() / 2,
                        view.sh / 2 - label.get_height() / 2))
    sub = get_font(view.sh * 0.035)
    s = sub.render("Press R to restart  /  ESC to quit", True, HUD_COLOR)
    screen.blit(s, (view.sw / 2 - s.get_width() / 2,
                    view.sh / 2 + label.get_height()))


# --------------------------------------------------------------------------- #
# Game state (reuses original physics / merge / scoring unchanged)
# --------------------------------------------------------------------------- #
class Game:
    def __init__(self):
        self.reset()

    def reset(self):
        self.space = pymunk.Space()
        self.space.gravity = (0, config.physics.gravity)
        self.space.damping = config.physics.damping
        self.space.collision_bias = config.physics.bias
        self.walls = [
            Wall(config.top_left, config.bot_left, self.space),
            Wall(config.bot_left, config.bot_right, self.space),
            Wall(config.bot_right, config.top_right, self.space),
        ]
        self.handler = self.space.add_collision_handler(
            CollisionTypes.PARTICLE, CollisionTypes.PARTICLE)
        self.handler.begin = collide
        self.handler.data["score"] = 0
        self.curr = PreParticle()
        self.next = PreParticle()
        self.wait_for_next = 0
        self.game_over = False

    @property
    def score(self):
        return self.handler.data["score"]

    def release(self):
        self.curr.release(self.space)
        self.wait_for_next = config.screen.delay

    def advance_preview(self):
        self.curr = self.next
        self.next = PreParticle()

    def check_game_over(self):
        for p in self.space.shapes:
            if isinstance(p, Particle) and getattr(p, "alive", False):
                if p.pos[1] < config.pad.killy and p.has_collided:
                    self.game_over = True
        return self.game_over


# --------------------------------------------------------------------------- #
# Interactive full-screen loop
# --------------------------------------------------------------------------- #
def open_display():
    info = pygame.display.Info()
    W, H = info.current_w, info.current_h
    mode_desc = "FULLSCREEN"
    try:
        screen = pygame.display.set_mode((W, H), pygame.FULLSCREEN)
    except Exception as exc:  # pragma: no cover - depends on SDL backend
        print(f"[minimal] fullscreen failed ({exc}); falling back to borderless window")
        screen = pygame.display.set_mode((W, H), pygame.NOFRAME)
        mode_desc = "NOFRAME"
    pygame.display.set_caption("PySuika - Minimal 2D")
    print(f"[minimal] display {W}x{H} ({mode_desc})")
    return screen, W, H


def run():
    screen, W, H = open_display()
    view = View(W, H)
    clock = pygame.time.Clock()
    game = Game()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r and game.game_over:
                    game.reset()
            elif (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1
                  and game.wait_for_next == 0 and not game.game_over):
                game.release()

        if not game.game_over:
            if game.wait_for_next > 1:
                game.wait_for_next -= 1
            elif game.wait_for_next == 1:
                game.advance_preview()
                game.wait_for_next = 0

            # map mouse x (screen) -> world x for aiming
            mx = pygame.mouse.get_pos()[0]
            world_x = (mx - view.ox) / view.scale
            game.curr.set_x(world_x)

            game.space.step(1 / config.screen.fps)
            game.check_game_over()

        draw_scene(screen, view, game.space, game.curr,
                   game.score, game.wait_for_next, game.game_over)
        draw_hud(screen, view, game.score, next_n=game.next.n)
        pygame.display.update()
        clock.tick(config.screen.fps)

    pygame.quit()


# --------------------------------------------------------------------------- #
# Headless smoke test (no OS window): drop fruits, render frames, save PNGs
# --------------------------------------------------------------------------- #
def smoke_test():
    import numpy as _np
    _preparticle = sys.modules["preparticle"]
    _preparticle.rng = _np.random.default_rng(7)  # deterministic

    W, H = 1280, 720
    pygame.display.set_mode((W, H))  # dummy driver -> off-screen
    surface = pygame.Surface((W, H))
    view = View(W, H)

    traces_dir = os.path.join(_ROOT, "..", "traces")
    traces_dir = os.path.abspath(traces_dir)
    os.makedirs(traces_dir, exist_ok=True)

    def settle(space, max_steps=180):
        for _ in range(max_steps):
            space.step(1 / config.screen.fps)
            vmax = max((p.body.velocity.length for p in space.shapes
                        if isinstance(p, Particle) and getattr(p, "alive", False)),
                       default=0.0)
            if vmax < 2.0:
                break

    # --- normal play: drop a handful of fruits across the board ---------
    game = Game()
    drop_xs = [480, 560, 640, 720, 800, 520, 760, 600]
    for i, wx in enumerate(drop_xs):
        game.curr.set_x(float(wx))
        game.release()
        game.advance_preview()
        settle(game.space)
        game.check_game_over()
    n_fruits = sum(1 for p in game.space.shapes
                   if isinstance(p, Particle) and getattr(p, "alive", False))
    draw_scene(surface, view, game.space, game.curr, game.score, 0, game.game_over)
    draw_hud(surface, view, game.score, next_n=game.next.n)
    preview_path = os.path.join(traces_dir, "minimal_preview.png")
    pygame.image.save(surface, preview_path)
    print(f"[smoke] normal: dropped={len(drop_xs)} live_fruits={n_fruits} "
          f"score={game.score} game_over={game.game_over}")
    print(f"[smoke] saved {preview_path}")

    # --- second preview frame: merge demo (same fruit stacked) ----------
    _preparticle.rng = _np.random.default_rng(0)
    game2 = Game()
    for wx in [600, 640, 680, 620, 660]:
        game2.curr.set_x(float(wx))
        game2.release()
        game2.advance_preview()
        settle(game2.space)
    draw_scene(surface, view, game2.space, game2.curr, game2.score, 0, game2.game_over)
    draw_hud(surface, view, game2.score, next_n=game2.next.n)
    preview2 = os.path.join(traces_dir, "minimal_preview_2.png")
    pygame.image.save(surface, preview2)
    print(f"[smoke] merge demo: score={game2.score} saved {preview2}")

    # --- game over logic: pile many fruits into one column --------------
    game3 = Game()
    for _ in range(40):
        if game3.game_over:
            break
        game3.curr.set_x(640.0)
        game3.release()
        game3.advance_preview()
        settle(game3.space, max_steps=120)
        game3.check_game_over()
    draw_scene(surface, view, game3.space, game3.curr, game3.score, 0, game3.game_over)
    print(f"[smoke] game_over reached={game3.game_over} score={game3.score}")

    assert n_fruits > 0, "no fruits rendered in normal play"
    assert os.path.exists(preview_path), "preview PNG not written"
    print("[smoke] ALL CHECKS PASSED")


if __name__ == "__main__":
    if _SMOKE:
        smoke_test()
    else:
        run()
