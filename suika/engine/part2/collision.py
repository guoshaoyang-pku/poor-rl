import numpy as np

from config import config
from particle import Particle


MAX_FRUIT_N = 10


def resolve_collision(particle1, particle2, space):
    if particle1.n == particle2.n == MAX_FRUIT_N:
        return
    distance = np.linalg.norm(particle1.pos - particle2.pos)
    if distance < 2 * particle1.radius:
        particle1.kill(space)
        particle2.kill(space)
        new_particle = Particle(
            np.mean([particle1.pos, particle2.pos], axis=0),
            particle1.n + 1,
            space,
        )
        for p in space.shapes:
            if p is new_particle:
                continue
            if isinstance(p, Particle) and p.alive:
                vector = p.pos - new_particle.pos
                distance = np.linalg.norm(vector)
                if 0 < distance < new_particle.radius + p.radius:
                    impulse = config.physics.impulse * vector / (distance ** 2)
                    p.body.apply_impulse_at_local_point(tuple(impulse))


def collide(arbiter, space, data):
    particle1, particle2 = arbiter.shapes
    alive = particle1.alive and particle2.alive
    same = particle1.n == particle2.n
    max_pair = same and particle1.n == MAX_FRUIT_N

    if max_pair and getattr(config.rules, "two_watermelon",
                            "no_merge_no_score_keep_both") == "merge_disappear_score":
        # Official rule: two watermelons poof out of existence and score.
        particle1.has_collided = False
        particle2.has_collided = False
        if alive:
            particle1.kill(space)
            particle2.kill(space)
            data["score"] += config[MAX_FRUIT_N, "points"]
        return False

    if not same or max_pair:
        particle1.has_collided = True
        particle2.has_collided = True
        return alive

    particle1.has_collided = False
    particle2.has_collided = False
    if alive:
        resolve_collision(particle1, particle2, space)
        data["score"] += config[particle1.n, "points"]
    return False
