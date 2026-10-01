"""MJ-mode Suika env: fixed 120-frame tempo + lenient 3s full-cross termination.

Physics and rules stay ours (part2, two-watermelon merge rule). Only the
action tempo and game-over condition follow mattjacobs30's env.
"""
from paths import setup_engine_path
setup_engine_path()

from config import config  # noqa: E402
from preparticle import PreParticle  # noqa: E402
from suika_env import SuikaEnv  # noqa: E402


class TempoSuikaEnv(SuikaEnv):
    def __init__(self, seed=None, frames_per_action=120, term_seconds=3.0):
        super().__init__(seed=seed)
        self.frames_per_action = int(frames_per_action)
        self.term_seconds = float(term_seconds)
        self.over_line_timer = 0.0

    def reset(self, seed=None):
        self.over_line_timer = 0.0
        return super().reset(seed=seed)

    def _any_fully_over_line(self):
        killy = config.pad.killy
        for p in self._live_particles():
            if p.has_collided and (p.pos[1] + p.radius) < killy:
                return True
        return False

    def step(self, action):
        """Drop at x=action, advance exactly `frames_per_action` physics
        frames (no settle wait), then reveal the next fruit."""
        if self.game_over:
            return self.get_state(), 0.0, True, {"score": self.score}
        prev_score = self.score
        self.curr.set_x(float(action))
        self.curr.release(self.space)
        dt = 1.0 / self.fps
        for _ in range(self.frames_per_action):
            self.space.step(dt)
            if self._any_fully_over_line():
                self.over_line_timer += dt
            else:
                self.over_line_timer = 0.0
            if self.over_line_timer > self.term_seconds:
                self.game_over = True
                break
        # cloud advances at the end, like theirs
        self.curr = self.next
        self.next = PreParticle()
        self.steps_taken += 1
        reward = float(self.score - prev_score)
        return self.get_state(), reward, self.game_over, {"score": self.score}
