"""Watch the search-based AI play Suika live in 2D (part2).

This is a NEW, additive entry point. It does **not** modify the original
gameplay, the physics, or any existing module. It simply combines two
existing assets:

  * the minimal full-screen 2D renderer in ``part2/main_minimal.py``
    (``View`` world->screen transform + ``draw_*`` primitives + ``Game``
    state wrapper, all reused verbatim), and
  * the search agent ``HeuristicLookaheadAgent`` from ``part2/ai_agent.py``
    (heuristic + 1-step real-physics lookahead).

Instead of the mouse deciding where to drop the fruit, the agent's
``decide(state)`` chooses the drop column; the physics then runs in real
time at 60fps so you can watch every fruit fall, settle and merge. A small
HUD overlays the AI's current decision (chosen column / x, candidate
scores, lookahead status).

Interactive (a small, normal, resizable title-bar window by default, so it
is easy to dock on one side of a split screen; pass ``--fullscreen`` for an
immersive full-screen view, or ``--height`` / ``--scale`` to size it)::

    cd .../suika && /Users/.../envs/suika/bin/python part2/ai_play_live.py --seed 0 --delay 0.6

Keys:  ESC quit | SPACE pause/resume | +/- faster/slower | R restart after game over

Headless smoke test (no OS window, writes a preview PNG)::

    SDL_VIDEODRIVER=dummy python part2/ai_play_live.py --smoke
"""
import os
import sys
import argparse

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))   # .../suika/part2
_ROOT = os.path.dirname(_THIS_DIR)                       # .../suika
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# In smoke mode force a headless SDL driver *before* pygame is imported, so no
# OS window is ever created. (main_minimal does the same check on its own.)
_SMOKE = "--smoke" in sys.argv
if _SMOKE:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pygame

# Reuse the minimal renderer wholesale (this also runs pygame.init(), imports
# config/particle/etc. and chdir's to the project root).
import main_minimal as mm
from main_minimal import (
    View, Game, draw_scene, draw_hud, get_font,
    FRUIT_COLORS, OUTLINE_COLOR, HUD_COLOR, BG_COLOR,
)
from config import config
from particle import Particle
import preparticle as _preparticle_mod

# Reuse the search agent (and its internals so we can also surface the per
# candidate scores for the HUD, without ever modifying ai_agent.py).
from ai_agent import HeuristicLookaheadAgent, simulate_drop

# --------------------------------------------------------------------------- #
# AlphaZero agent: deploy a trained / in-progress checkpoint live. Reuses the
# rl/ PolicyValueNet + MCTS + environment model verbatim (no search rewrite).
# --------------------------------------------------------------------------- #
_RL_DIR = os.path.join(_ROOT, "rl")          # .../suika/rl
if _RL_DIR not in sys.path:
    sys.path.insert(0, _RL_DIR)

DEFAULT_CKPT = os.path.join(_RL_DIR, "checkpoints", "latest.pt")

# Play-speed ladder (multiplier); 1.0x is real-time. Keys [ and ] step it.
_SPEED_STEPS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]


def _bump_speed(speed, direction):
    """Step the play-speed multiplier along a fixed ladder (direction +/-1)."""
    idx = min(range(len(_SPEED_STEPS)),
              key=lambda i: abs(_SPEED_STEPS[i] - speed))
    idx = max(0, min(len(_SPEED_STEPS) - 1, idx + direction))
    return _SPEED_STEPS[idx]


def _robust_torch_load(path, map_location, retries=6, wait=0.4):
    """Load a checkpoint that the training process may be concurrently
    rewriting. ``save_checkpoint`` writes a ``.tmp`` then ``os.replace``s it
    (atomic rename), so a torn read is unlikely -- but we still retry on any
    transient read/unpickle error instead of giving up on the first failure."""
    import time as _time
    import torch
    last = None
    for _ in range(max(1, retries)):
        try:
            return torch.load(path, map_location=map_location)
        except Exception as exc:               # truncated / locked mid-write
            last = exc
            _time.sleep(wait)
    raise RuntimeError("could not read checkpoint %s after %d tries: %r"
                       % (path, retries, last))


class AlphaZeroAgent:
    """Runs the rl/ MCTS (PUCT + network priors + value + env-model rollouts)
    for ``visits`` simulations per move on a loaded checkpoint, then drops in
    the most-visited column. Exposes ``lookahead_used`` / ``lookahead_failures``
    so the existing HUD code keeps working unchanged."""

    def __init__(self, ckpt_path=None, visits=100, device="cpu", seed=0):
        import torch
        import yaml
        from net import build_net
        from mcts import MCTS
        from common import col_to_x

        ckpt_path = ckpt_path or DEFAULT_CKPT
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError("checkpoint not found: %s" % ckpt_path)

        dev = torch.device(device)
        ck = _robust_torch_load(ckpt_path, map_location=dev)

        # Network structure: prefer the cfg stored *inside* the checkpoint (it
        # is exactly what the weights were trained with); fall back to
        # rl/config.yaml. This keeps us K / max_fruits / hidden compatible.
        cfg = ck.get("cfg") if isinstance(ck, dict) else None
        if not cfg:
            with open(os.path.join(_RL_DIR, "config.yaml")) as f:
                cfg = yaml.safe_load(f)

        K = int(cfg["K"])
        max_fruits = int(cfg["max_fruits"])

        net = build_net(cfg)
        state_dict = ck["net"] if (isinstance(ck, dict) and "net" in ck) else ck
        net.load_state_dict(state_dict)
        net.to(dev)
        net.eval()

        self.net = net
        self.device = dev
        self.K = K
        self.max_fruits = max_fruits
        self.visits = int(visits)
        self.col_to_x = col_to_x
        self.step = ck.get("step") if isinstance(ck, dict) else None
        self.mcts = MCTS(
            net, dev, K=K, num_simulations=self.visits,
            c_puct=float(cfg.get("c_puct", 1.5)),
            max_fruits=max_fruits, fast_model=True, seed=seed,
            eval_batch=int(cfg.get("eval_batch", 16)),
        )
        # HUD-compatibility counters (mirror HeuristicLookaheadAgent fields).
        self.lookahead_used = 0
        self.lookahead_failures = 0

    def set_visits(self, visits):
        self.visits = int(visits)
        self.mcts.num_simulations = self.visits

    def decide_with_scores(self, state):
        """Run MCTS for ``visits`` sims and pick the most-visited column.
        Returns ``(pick_x, scored, used)`` where ``scored`` are per-column
        ``(visit_count, x)`` pairs, so the existing candidate-bar HUD now
        visualises the MCTS visit distribution (taller = more visits, gold =
        chosen). No Dirichlet noise at play time -> deterministic strongest
        move."""
        _counts, root = self.mcts.run(state, add_noise=False)
        N = root.N
        if N is None:                          # terminal root (shouldn't occur)
            best_col = self.K // 2
            return float(self.col_to_x(best_col, self.K)), [], True
        best_col = int(np.argmax(N))
        pick_x = float(self.col_to_x(best_col, self.K))
        scored = [(float(N[c]), float(self.col_to_x(c, self.K)))
                  for c in range(self.K)]
        self.lookahead_used += 1
        return pick_x, scored, True


def agent_decide(agent, state):
    """Dispatch one decision to whichever agent type is active."""
    if isinstance(agent, AlphaZeroAgent):
        return agent.decide_with_scores(state)
    return decide_with_scores(agent, state)


def make_agent(agent_kind, ckpt, visits, device, num_columns, lookahead_steps,
               seed):
    """Build the requested agent. AlphaZero falls back to the heuristic agent
    on any load/build error (with a clear printed notice). Returns
    ``(agent, agent_label)``."""
    if agent_kind == "alphazero":
        try:
            az = AlphaZeroAgent(ckpt_path=ckpt, visits=visits, device=device,
                                seed=seed)
            print("[ai-live] AlphaZero ready: ckpt=%s step=%s K=%d visits=%d "
                  "device=%s" % (ckpt or DEFAULT_CKPT, az.step, az.K,
                                 az.visits, device))
            return az, "alphazero"
        except Exception as exc:
            print("[ai-live] WARNING: failed to load AlphaZero checkpoint "
                  "(%r); falling back to heuristic agent." % (exc,))
    agent = HeuristicLookaheadAgent(num_columns=num_columns,
                                    lookahead_steps=lookahead_steps, seed=seed)
    return agent, "heuristic"


# HUD accent colours (kept minimal / non-flashy)
PICK_COLOR = (90, 220, 250)     # chosen drop column highlight (cyan)
CAND_COLOR = (120, 120, 140)    # candidate tick marks
CAND_BEST_COLOR = (250, 220, 60)
PAUSE_COLOR = (250, 200, 90)


# Default windowed size: a small, narrow-tall window (the container is a tall
# rectangle) that comfortably fits the game area + HUD and docks nicely on one
# half of a split screen. Width is derived from the height by this aspect.
DEFAULT_WINDOW_HEIGHT = 720
WINDOW_ASPECT = 0.70            # width / height  -> narrow & tall


# --------------------------------------------------------------------------- #
# Window-aware view: like main_minimal.View but scales the container to FIT the
# (small) window on BOTH axes and centres it, so the whole game area stays
# visible at any window size / aspect (physics & world coords are untouched).
# --------------------------------------------------------------------------- #
class FitView(View):
    HEIGHT_FILL = 0.94          # fraction of window height the view may use
    WIDTH_FILL = 0.94           # fraction of window width the container may use

    def __init__(self, screen_w, screen_h):
        self.sw, self.sh = screen_w, screen_h
        view_h = self.VIEW_BOT - self.VIEW_TOP
        cont_w = config.pad.right - config.pad.left
        scale_h = (screen_h * self.HEIGHT_FILL) / view_h
        scale_w = (screen_w * self.WIDTH_FILL) / cont_w
        self.scale = min(scale_h, scale_w)
        cx = (config.pad.left + config.pad.right) / 2.0
        self.ox = screen_w / 2.0 - cx * self.scale
        content_h = view_h * self.scale
        self.oy = (screen_h - content_h) / 2.0 - self.VIEW_TOP * self.scale


def open_window(height=DEFAULT_WINDOW_HEIGHT, width=None, fullscreen=False):
    """Open the display. Default = a small, normal, resizable title-bar window
    (NOT full-screen, NOT borderless). ``--fullscreen`` reuses main_minimal's
    full-screen path. Returns ``(screen, W, H, is_fullscreen)``."""
    if fullscreen:
        screen, W, H = mm.open_display()
        pygame.display.set_caption("Suika AI")
        return screen, W, H, True
    height = int(height)
    width = int(width) if width else max(360, int(round(height * WINDOW_ASPECT)))
    flags = 0
    try:
        flags = pygame.RESIZABLE
    except Exception:
        flags = 0
    screen = pygame.display.set_mode((width, height), flags)
    pygame.display.set_caption("Suika AI")
    print("[ai-live] window %dx%d (resizable=%s)"
          % (width, height, bool(flags)))
    return screen, width, height, False


# --------------------------------------------------------------------------- #
# State snapshot: build the dict shape that the agent expects (identical to
# SuikaEnv.get_state) directly from a live ``main_minimal.Game``.
# --------------------------------------------------------------------------- #
def build_state(game):
    fruits = []
    for p in game.space.shapes:
        if isinstance(p, Particle) and getattr(p, "alive", False):
            x, y = p.body.position
            fruits.append({
                "x": float(x), "y": float(y),
                "radius": float(p.radius), "type": int(p.n),
                "name": config.fruit_names[p.n],
            })
    return {
        "fruits": fruits,
        "current": {
            "type": int(game.curr.n),
            "name": config.fruit_names[game.curr.n],
            "radius": float(game.curr.radius),
        },
        "next": {
            "type": int(game.next.n),
            "name": config.fruit_names[game.next.n],
        },
        "score": int(game.score),
        "game_over": bool(game.game_over),
        "play_area": {
            "left": config.pad.left, "right": config.pad.right,
            "top": config.pad.top, "bottom": config.pad.bot,
        },
    }


# --------------------------------------------------------------------------- #
# Decision: mirror HeuristicLookaheadAgent.decide() exactly, but also return
# the per-candidate (score, x) list so we can visualise the search on screen.
# Falls back to the pure heuristic if the physics rebuild fails (same as the
# agent itself).
# --------------------------------------------------------------------------- #
def decide_with_scores(agent, state):
    cur_type = int(state["current"]["type"])
    radius = config[cur_type, "radius"]
    xs = agent._candidate_xs(radius)
    scored = []
    try:
        for x in xs:
            outcome = simulate_drop(state, x, max_steps=agent.lookahead_steps)
            scored.append((agent._score_outcome(outcome), x))
        agent.lookahead_used += 1
    except Exception:
        agent.lookahead_failures += 1
        return float(agent._heuristic_decide(state)), [], False

    best_val = max(v for v, _ in scored)
    center = (config.pad.left + config.pad.right) / 2
    eps = 1e-6
    best = [x for v, x in scored if v >= best_val - eps]
    best.sort(key=lambda x: abs(x - center))
    return float(best[0]), scored, True


# --------------------------------------------------------------------------- #
# AI decision HUD overlay (drawn on top of the reused draw_scene/draw_hud).
# --------------------------------------------------------------------------- #
def draw_ai_overlay(screen, view, game, decision, step_count, delay,
                    paused, lookahead_used, lookahead_failed,
                    agent_label="heuristic", visits=None, speed=1.0):
    pick_x, scored, used = decision if decision else (None, [], False)

    # 1) chosen drop column: a bright vertical aiming line + a marker fruit
    if pick_x is not None and not game.game_over:
        col_idx = _column_index(pick_x, len(scored)) if scored else None
        sx = view.x(pick_x)
        top_y = view.y(config.pad.line_top)
        bot_y = view.y(config.pad.bot)
        pygame.draw.line(screen, PICK_COLOR, (sx, top_y), (sx, bot_y), 2)
        # small triangle pointer at the top of the chosen column
        pygame.draw.polygon(screen, PICK_COLOR, [
            (sx - 8, top_y - 12), (sx + 8, top_y - 12), (sx, top_y)])

    # 2) candidate ticks along the bottom, height ~ normalized score, best gold
    if scored:
        vals = [v for v, _ in scored]
        vmin, vmax = min(vals), max(vals)
        span = (vmax - vmin) or 1.0
        base_y = view.y(config.pad.bot) + max(8, int(view.sh * 0.012))
        max_h = view.sh * 0.10
        for v, x in scored:
            h = max_h * (v - vmin) / span
            tx = view.x(x)
            is_best = (pick_x is not None and abs(x - pick_x) < 1e-6
                       and abs(v - vmax) < 1e-6)
            col = CAND_BEST_COLOR if v >= vmax - 1e-6 else CAND_COLOR
            pygame.draw.line(screen, col, (tx, base_y), (tx, base_y + h),
                             3 if v >= vmax - 1e-6 else 2)

    # 3) text HUD (top-left, under the reused Score label)
    font = get_font(view.sh * 0.030)
    small = get_font(view.sh * 0.024)
    x0 = int(view.sh * 0.03)
    y = int(view.sh * 0.03 + view.sh * 0.06)
    lines = []
    lines.append(("Step: %d" % step_count, font, HUD_COLOR))
    if pick_x is not None:
        col_txt = ""
        if scored:
            col_txt = "  (col %d/%d)" % (_column_index(pick_x, len(scored)) + 1,
                                         len(scored))
        lines.append(("AI drop x: %.0f%s" % (pick_x, col_txt), font, PICK_COLOR))
    if scored:
        best_v = max(v for v, _ in scored)
        lines.append(("best score: %.1f over %d cands" % (best_v, len(scored)),
                      small, HUD_COLOR))
    if agent_label == "alphazero":
        lines.append(("agent: AlphaZero   visits: %d" % (visits or 0),
                      small, PICK_COLOR))
        lines.append(("(MCTS visit-count search; higher=stronger&slower)",
                      small, HUD_COLOR))
    else:
        mode = "lookahead" if used else "heuristic-fallback"
        lines.append(("agent: heuristic   search: %s (ok %d/fail %d)"
                      % (mode, lookahead_used, lookahead_failed),
                      small, HUD_COLOR))
    lines.append(("speed: %.2fx  [ / ]    delay: %.2fs  +/-"
                  % (speed, delay), small, HUD_COLOR))
    for text, fnt, col in lines:
        label = fnt.render(text, True, col)
        screen.blit(label, (x0, y))
        y += int(label.get_height() * 1.1)

    # 4) paused banner
    if paused and not game.game_over:
        pb = get_font(view.sh * 0.05).render("PAUSED  (SPACE)", True, PAUSE_COLOR)
        screen.blit(pb, (view.sw / 2 - pb.get_width() / 2, int(view.sh * 0.04)))


def _column_index(x, num_columns):
    """Best-effort: map a chosen x back to its candidate column index."""
    if num_columns <= 0:
        return 0
    lo = config.pad.left
    hi = config.pad.right
    frac = (x - lo) / max(1e-9, (hi - lo))
    return int(np.clip(int(frac * num_columns), 0, num_columns - 1))


def render_frame(screen, view, game, decision, step_count, delay, paused,
                 agent, agent_label="heuristic", visits=None, speed=1.0):
    draw_scene(screen, view, game.space, game.curr,
               game.score, game.wait_for_next, game.game_over)
    draw_hud(screen, view, game.score, next_n=game.next.n)
    draw_ai_overlay(screen, view, game, decision, step_count, delay, paused,
                    agent.lookahead_used, agent.lookahead_failures,
                    agent_label=agent_label, visits=visits, speed=speed)


# --------------------------------------------------------------------------- #
# Interactive live loop: AI replaces the mouse.
# --------------------------------------------------------------------------- #
def run(seed=0, delay=0.6, num_columns=14, lookahead_steps=160,
        window_height=DEFAULT_WINDOW_HEIGHT, window_width=None,
        fullscreen=False, agent_kind="alphazero", ckpt=None, visits=100,
        speed=1.0, device="cpu"):
    _preparticle_mod.rng = np.random.default_rng(seed)
    screen, W, H, fullscreen = open_window(window_height, window_width,
                                           fullscreen)
    view = FitView(W, H)
    clock = pygame.time.Clock()
    game = Game()
    agent, agent_label = make_agent(agent_kind, ckpt, visits, device,
                                    num_columns, lookahead_steps, seed)
    visits_now = visits if agent_label == "alphazero" else None

    decision = None            # (pick_x, scored, used)
    decision_elapsed = 0.0     # seconds accumulated since the decision was made
    step_count = 0
    paused = False
    fps = config.screen.fps
    speed = max(0.25, float(speed))
    phys_accum = 0.0           # fractional physics-substep accumulator (speed)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE and not fullscreen:
                W, H = max(1, event.w), max(1, event.h)
                screen = pygame.display.set_mode((W, H), pygame.RESIZABLE)
                view = FitView(W, H)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS,
                                   pygame.K_KP_PLUS):
                    delay = max(0.0, delay - 0.1)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    delay = min(3.0, delay + 0.1)
                elif event.key == pygame.K_RIGHTBRACKET:
                    speed = _bump_speed(speed, +1)      # ] faster
                elif event.key == pygame.K_LEFTBRACKET:
                    speed = _bump_speed(speed, -1)      # [ slower
                elif event.key == pygame.K_r and game.game_over:
                    _preparticle_mod.rng = np.random.default_rng(seed)
                    game.reset()
                    agent, agent_label = make_agent(
                        agent_kind, ckpt, visits, device,
                        num_columns, lookahead_steps, seed)
                    visits_now = visits if agent_label == "alphazero" else None
                    decision = None
                    decision_elapsed = 0.0
                    step_count = 0
                    phys_accum = 0.0

        dt = clock.tick(fps) / 1000.0
        # physics substeps this frame scale with the speed multiplier; the
        # fractional accumulator lets non-integer speeds (e.g. 0.5x) work too.
        phys_accum += speed
        n_sub = int(phys_accum)
        phys_accum -= n_sub

        if not paused and not game.game_over:
            # advance the "next" preview countdown (scaled by speed)
            if game.wait_for_next > 0:
                game.wait_for_next = max(0,
                                         game.wait_for_next - max(1, n_sub))
                if game.wait_for_next == 0:
                    game.advance_preview()

            # when idle (ready to drop) and no pending decision: AI thinks.
            # MCTS thinking time is set by --visits, NOT by speed; speed only
            # accelerates the physics / drop cadence.
            if decision is None and game.wait_for_next == 0:
                pick_x, scored, used = agent_decide(agent, build_state(game))
                decision = (pick_x, scored, used)
                decision_elapsed = 0.0
                game.curr.set_x(pick_x)   # move the preview fruit to the column
            elif decision is not None and game.wait_for_next == 0:
                # show the aim for delay/speed seconds, then drop
                decision_elapsed += dt
                if decision_elapsed >= (delay / max(0.25, speed)):
                    game.release()
                    step_count += 1
                    decision = None

            # physics advances n_sub substeps per frame so the user still sees
            # fruits fall / settle / merge, just faster at higher speed.
            for _ in range(max(1, n_sub)):
                game.space.step(1 / fps)
                if game.check_game_over():
                    break

        render_frame(screen, view, game, decision, step_count, delay, paused,
                     agent, agent_label=agent_label, visits=visits_now,
                     speed=speed)
        pygame.display.update()

    pygame.quit()


# --------------------------------------------------------------------------- #
# Headless smoke test: a SHORT run (kept light so it doesn't fight the
# background training job), verifying the chosen agent drops, the HUD renders,
# the game-over branch is reachable, and a preview PNG is written.
# --------------------------------------------------------------------------- #
def smoke_test(seed=0, n_drops=6, agent_kind="alphazero", ckpt=None,
               visits=16, speed=1.0):
    _preparticle_mod.rng = np.random.default_rng(seed)
    # Render at the default small windowed size (narrow & tall) so the smoke
    # test exercises exactly what the user sees on screen.
    H = DEFAULT_WINDOW_HEIGHT
    W = max(360, int(round(H * WINDOW_ASPECT)))
    pygame.display.set_mode((W, H))     # dummy driver -> off-screen
    surface = pygame.Surface((W, H))
    view = FitView(W, H)

    # Light params keep CPU use tiny (training is running in the bg): small
    # visits for alphazero, small columns/lookahead for the heuristic fallback.
    agent, agent_label = make_agent(agent_kind, ckpt, visits, "cpu",
                                    num_columns=6, lookahead_steps=60,
                                    seed=seed)
    visits_now = visits if agent_label == "alphazero" else None
    game = Game()

    def settle(space, max_steps=120):
        for _ in range(max_steps):
            space.step(1 / config.screen.fps)
            vmax = max((p.body.velocity.length for p in space.shapes
                        if isinstance(p, Particle) and getattr(p, "alive", False)),
                       default=0.0)
            if vmax < 2.0:
                break

    # demonstrate that --speed changes the drop / physics rhythm
    speed = max(0.25, float(speed))
    eff_delay = 0.6 / speed
    n_sub = max(1, int(round(speed)))
    print("[smoke] agent=%s visits=%s speed=%.2fx -> eff_delay=%.3fs, "
          "physics_substeps/frame=%d"
          % (agent_label, visits_now, speed, eff_delay, n_sub))

    last_decision = None
    for i in range(n_drops):
        if game.game_over:
            break
        decision = agent_decide(agent, build_state(game))
        last_decision = decision
        pick_x, scored, used = decision
        game.curr.set_x(pick_x)
        # render one in-flight frame (HUD + aim) before releasing
        render_frame(surface, view, game, decision, i, 0.6, False, agent,
                     agent_label=agent_label, visits=visits_now, speed=speed)
        game.release()
        game.advance_preview()
        game.wait_for_next = 0
        settle(game.space)
        game.check_game_over()
        print("[smoke] drop %d -> x=%.0f cands=%d used=%s score=%d"
              % (i + 1, pick_x, len(scored), used, game.score))

    n_fruits = sum(1 for p in game.space.shapes
                   if isinstance(p, Particle) and getattr(p, "alive", False))

    # final live frame with HUD
    render_frame(surface, view, game, last_decision, n_drops, 0.6, False, agent,
                 agent_label=agent_label, visits=visits_now, speed=speed)

    traces_dir = os.path.abspath(os.path.join(_ROOT, "..", "traces"))
    os.makedirs(traces_dir, exist_ok=True)
    fname = ("ai_live_alphazero_preview.png" if agent_label == "alphazero"
             else "ai_live_window_preview.png")
    preview_path = os.path.join(traces_dir, fname)
    pygame.image.save(surface, preview_path)

    # exercise the game-over rendering branch without an expensive simulation
    game.game_over = True
    render_frame(surface, view, game, last_decision, n_drops, 0.6, False, agent,
                 agent_label=agent_label, visits=visits_now, speed=speed)
    print("[smoke] game_over branch rendered ok")

    print("[smoke] agent=%s drops=%d live_fruits=%d score=%d"
          % (agent_label, n_drops, n_fruits, game.score))
    print("[smoke] saved %s" % preview_path)
    assert os.path.exists(preview_path), "preview PNG not written"
    assert n_fruits >= 0
    print("[smoke] ALL CHECKS PASSED")


def main():
    parser = argparse.ArgumentParser(description="Watch the search/AlphaZero AI play Suika live (2D).")
    parser.add_argument("--agent", choices=["heuristic", "alphazero"],
                        default="alphazero",
                        help="which AI plays: 'alphazero' deploys a trained "
                             "checkpoint via MCTS (default); 'heuristic' uses "
                             "the 1-step real-physics lookahead agent")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="AlphaZero checkpoint path (default "
                             "rl/checkpoints/latest.pt)")
    parser.add_argument("--visits", type=int, default=None,
                        help="MCTS simulations per move for alphazero "
                             "(default 100; try 50/200/500 - higher = stronger "
                             "but slower). Smoke default = 16.")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="play-speed multiplier (default 1.0; e.g. 2/4/8 "
                             "accelerate physics + drop cadence). MCTS think "
                             "time is set by --visits, not --speed.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="torch device for the policy/value net "
                             "(default cpu - fastest for single-state realtime "
                             "inference; mps/cuda also accepted)")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed (default 0, matches earlier traces)")
    parser.add_argument("--delay", type=float, default=0.6,
                        help="pause in seconds after each AI pick so you can "
                             "watch (effective pause = delay / speed)")
    parser.add_argument("--sims", "--num-columns", dest="num_columns",
                        type=int, default=14,
                        help="heuristic-mode: number of candidate drop columns")
    parser.add_argument("--lookahead-steps", type=int, default=160,
                        help="heuristic-mode: physics steps per lookahead sim")
    parser.add_argument("--height", type=int, default=DEFAULT_WINDOW_HEIGHT,
                        help="windowed mode height in pixels (default %d; "
                             "width is derived narrow & tall)"
                             % DEFAULT_WINDOW_HEIGHT)
    parser.add_argument("--width", type=int, default=None,
                        help="override window width in pixels (default = "
                             "height * %.2f)" % WINDOW_ASPECT)
    parser.add_argument("--scale", type=float, default=None,
                        help="size multiplier applied to the default height "
                             "(e.g. 1.3 for a bigger window)")
    parser.add_argument("--fullscreen", action="store_true",
                        help="immersive full screen instead of the default "
                             "small window")
    parser.add_argument("--smoke", action="store_true",
                        help="headless short smoke test (writes a preview PNG)")
    parser.add_argument("--drops", type=int, default=6,
                        help="number of AI drops in the smoke test")
    args = parser.parse_args()

    visits = args.visits
    if visits is None:
        visits = 16 if args.smoke else 100

    if args.smoke:
        smoke_test(seed=args.seed, n_drops=args.drops, agent_kind=args.agent,
                   ckpt=args.ckpt, visits=visits, speed=args.speed)
    else:
        height = args.height
        if args.scale:
            height = int(round(DEFAULT_WINDOW_HEIGHT * args.scale))
        run(seed=args.seed, delay=args.delay,
            num_columns=args.num_columns, lookahead_steps=args.lookahead_steps,
            window_height=height, window_width=args.width,
            fullscreen=args.fullscreen, agent_kind=args.agent, ckpt=args.ckpt,
            visits=visits, speed=args.speed, device=args.device)


if __name__ == "__main__":
    main()
