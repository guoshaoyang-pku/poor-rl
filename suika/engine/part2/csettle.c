/* csettle.c — stateless C scan for the Suika engine's per-frame settle loop.
 *
 * All chipmunk entry points are injected as function addresses via cs_init()
 * (no linking). The scan is PURE READ-ONLY over the space:
 *
 *   cs_scan(space) -> (vmax_sq, min_y) over every in-space shape whose
 *                     collision_type == particle_ctype
 *
 * The Python caller derives the two break predicates:
 *   settled  <=> vmax_sq < settle_velocity^2          (identical float math)
 *   possibly-over <=> min_y < killy, in which case Python runs its own
 *   _check_game_over() (alive + has_collided + y < killy) — min_y < killy is
 *   a necessary condition for game over, so the composite decision is
 *   bit-identical to the original two-scan loop, with zero per-frame Python
 *   work in the common case (nothing above the kill line).
 *
 * In-space particle == alive Particle after Space.step() returns: pymunk
 * defers callback-time remove()/add() to the end of the step, so membership
 * equals liveness at scan time (verified by suika_dqn/phy_ablation.py).
 *
 * Build:  cc -O2 -fPIC -shared -std=c99 csettle.c -o _csettle_c.so
 */
#include <stdint.h>
#include <stddef.h>

typedef struct cpSpace cpSpace;
typedef struct cpShape cpShape;
typedef struct cpBody  cpBody;
typedef struct { double x, y; } cpVect;

typedef void       (*fn_step)(cpSpace *, double);
typedef void       (*fn_each)(cpSpace *, void (*)(cpShape *, void *), void *);
typedef cpBody    *(*fn_body_of)(cpShape *);
typedef cpVect    (*fn_pos)(cpBody *);
typedef cpVect    (*fn_vel)(cpBody *);
typedef uintptr_t (*fn_ctype)(cpShape *);

static fn_step   c_step;
static fn_each   c_each;
static fn_body_of c_body;
static fn_pos    c_pos;
static fn_vel    c_vel;
static fn_ctype  c_ctype;
static uintptr_t c_particle_ctype;

void cs_init(uintptr_t step, uintptr_t each, uintptr_t body_of,
             uintptr_t pos, uintptr_t vel, uintptr_t collision_type,
             uintptr_t particle_ctype)
{
    c_step = (fn_step)step;
    c_each = (fn_each)each;
    c_body = (fn_body_of)body_of;
    c_pos  = (fn_pos)pos;
    c_vel  = (fn_vel)vel;
    c_ctype = (fn_ctype)collision_type;
    c_particle_ctype = particle_ctype;
}

typedef struct {
    double vmax_sq;
    double min_y;
    int    n;        /* particles seen (debug) */
} cs_ctx;

static void cs_scan_cb(cpShape *shape, void *data)
{
    cs_ctx *c = (cs_ctx *)data;
    if (c_ctype(shape) != c_particle_ctype) return;   /* walls etc. */
    c->n++;
    cpBody *b = c_body(shape);
    cpVect p = c_pos(b);
    if (p.y < c->min_y) c->min_y = p.y;
    cpVect v = c_vel(b);
    double sq = v.x * v.x + v.y * v.y;
    if (sq > c->vmax_sq) c->vmax_sq = sq;
}

static cs_ctx g_last;

void cs_scan(cpSpace *space, double *vmax_sq_out, double *min_y_out)
{
    cs_ctx c;
    c.vmax_sq = 0.0;
    c.min_y = 1e30;
    c.n = 0;
    c_each(space, cs_scan_cb, &c);
    g_last = c;
    *vmax_sq_out = c.vmax_sq;
    *min_y_out = c.min_y;
}

void cs_debug_stats(int *n_particles, double *vmax_sq, double *min_y)
{
    *n_particles = g_last.n;
    *vmax_sq = g_last.vmax_sq;
    *min_y = g_last.min_y;
}
