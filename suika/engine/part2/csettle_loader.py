"""Load the self-contained csettle C extension (stateless settle scan in C).

All chipmunk entry points are injected as plain addresses (no linking); the
scan is read-only. If anything is off (missing .so, private ABI change),
``ok`` stays False and the engine transparently keeps the Python fast scan.

Build first:  python build_csettle.py   (or cc -O2 -fPIC -shared ...)
"""
import os

from cffi import FFI
import pymunk
from pymunk._chipmunk import lib as _cp, ffi as _cpffi

from config import CollisionTypes

_PARTICLE_CTYPE = CollisionTypes.PARTICLE

_HERE = os.path.dirname(os.path.abspath(__file__))
_SO = os.path.join(_HERE, "_csettle_c.so")

_ffi = FFI()
_ffi.cdef("""
    void cs_init(uintptr_t step, uintptr_t each, uintptr_t body_of,
                 uintptr_t pos, uintptr_t vel, uintptr_t collision_type,
                 uintptr_t particle_ctype);
    void cs_scan(void *space, double *vmax_sq_out, double *min_y_out);
    void cs_debug_stats(int *n_particles, double *vmax_sq, double *min_y);
""")

_lib = None
_vbuf = None
_ybuf = None
_last_space = None        # 1-entry cache: pymunk Space object -> void* cdata
_last_space_ptr = None


def _fnaddr(f):
    return int(_cpffi.cast("uintptr_t", f))


def _ptr(cdata):
    return _ffi.cast("void *", int(_cpffi.cast("uintptr_t", cdata)))


try:
    _lib = _ffi.dlopen(_SO)
    _lib.cs_init(_fnaddr(_cp.cpSpaceStep), _fnaddr(_cp.cpSpaceEachShape),
                 _fnaddr(_cp.cpShapeGetBody), _fnaddr(_cp.cpBodyGetPosition),
                 _fnaddr(_cp.cpBodyGetVelocity),
                 _fnaddr(_cp.cpShapeGetCollisionType),
                 _PARTICLE_CTYPE)
    # ABI probe: collision_type must round-trip through the injected fn.
    _b = pymunk.Body(1, 1)
    _c = pymunk.Circle(_b, 3.0)
    _c.collision_type = 7
    assert int(_cpffi.cast("uintptr_t",
                           _cp.cpShapeGetCollisionType(_c._shape))) == 7
    del _c, _b
    _vbuf = _ffi.new("double *")
    _ybuf = _ffi.new("double *")
except Exception:
    _lib = None

ok = _lib is not None


def scan(space):
    """(vmax_sq, min_y) over in-space particles; read-only."""
    global _last_space, _last_space_ptr
    p = _last_space_ptr if space is _last_space else _ptr(space._space)
    _last_space, _last_space_ptr = space, p
    _lib.cs_scan(p, _vbuf, _ybuf)
    return _vbuf[0], _ybuf[0]
