"""E8 lattice codes: a 2-bit weight format that rounds 8 values at once.

Plain 2-bit CQ snaps every value of a rotated 128-group to one of 4 levels. Here each block of
8 values snaps to the nearest point of the E8 lattice instead (the densest packing in 8-D, as in
SQuAT / QuIP#). Only points with norm^2 <= 10 get a code: 56,881 of them, so one 16-bit code
covers 8 weights, about 2 bits each. Per 128-group we store one FP16 gain, fitted by least
squares to the snapped points.

A point's coordinates are multiples of 1/2 in [-3, 3]: 13 levels. The engine decodes each code to
8 nibbles (level index 0-12) once at load, so E8 tensors run on the 4-bit kernels with LEVELS as
their codebook (a table lookup per code at inference would be about 2x slower). On the release weights this cuts the 2-bit rounding error by 10% (0.344 -> 0.310 per group).

Every function takes the array module (`np` or `jax.numpy`), so training and export round the same way.
"""
import functools

import numpy as np

E8 = 8                 # the record `bits` value of an E8 tensor (16-bit code per 8 weights)
SCALE = 11.0           # unit groups are scaled by this before snapping (best on the release weights)
RADIUS2 = 10           # only points with norm^2 <= 10 have a code
SHRINK = (1, .95, .9, .85, .8, .75, .7, .65, .6, .5)
LEVELS = np.concatenate([np.arange(-6, 7) / 2, np.zeros(3)]).astype(np.float32)  # nibble -> coordinate


def _d8(x, xp):
    """Nearest point of D8 (integer vectors with an even sum): round, then fix the worst value."""
    r = xp.round(x)
    odd = (xp.sum(r, -1, keepdims=True) % 2) != 0
    err = x - r
    worst = xp.arange(8) == xp.argmax(xp.abs(err), -1)[..., None]
    return xp.where(odd & worst, r + xp.where(err >= 0, 1.0, -1.0), r)


def _nearest(x, xp):
    """Nearest point of E8 = D8 and D8 + 1/2 (Conway & Sloane)."""
    a, b = _d8(x, xp), _d8(x - 0.5, xp) + 0.5
    closer = xp.sum((x - a) ** 2, -1, keepdims=True) <= xp.sum((x - b) ** 2, -1, keepdims=True)
    return xp.where(closer, a, b)


def snap(blocks, xp=np):
    """[..., 8] -> a nearby coded point (norm^2 <= 10). About 1 block in 5 has its nearest lattice
    point outside the ball; for those, the block is shrunk by each factor in SHRINK and the in-ball
    point closest to the original block wins (the zero point if none fits)."""
    best, dist = xp.zeros_like(blocks), xp.sum(blocks ** 2, -1, keepdims=True)
    for factor in SHRINK:
        point = _nearest(blocks * factor, xp)
        d = xp.sum((blocks - point) ** 2, -1, keepdims=True)
        take = (xp.sum(point ** 2, -1, keepdims=True) <= RADIUS2) & (d < dist)
        best, dist = xp.where(take, point, best), xp.where(take, d, dist)
    return best


def quantize_groups(rotated, xp=np):
    """Rotated groups [..., 128] -> (lattice points [..., 128], FP16-rounded gain [..., 1])."""
    norm = xp.sqrt(xp.sum(rotated ** 2, -1, keepdims=True))
    unit = rotated / xp.maximum(norm, 1e-12)
    points = snap((unit * SCALE).reshape(*unit.shape[:-1], -1, 8), xp).reshape(unit.shape)
    gain = xp.sum(rotated * points, -1, keepdims=True) / xp.maximum(xp.sum(points ** 2, -1, keepdims=True), 1e-12)
    return points, gain.astype(xp.float16).astype(xp.float32)


def _key(doubled):
    """Doubled coordinates [..., 8] (-6..6) -> base-13 key; sorted keys give the code order."""
    return ((doubled.astype(np.int64) + 6) * 13 ** np.arange(7, -1, -1)).sum(-1)


@functools.lru_cache(maxsize=1)
def table():
    """(doubled coordinates [56881, 8] int8, sorted keys): code i is point table[i] / 2."""
    grids = []
    for values in (np.arange(-6, 7, 2), np.arange(-5, 6, 2)):  # integer points, half-integer points
        grid = np.stack(np.meshgrid(*[values] * 8, indexing='ij'), -1).reshape(-1, 8).astype(np.int8)
        g = grid.astype(np.int32)
        grids.append(grid[((g ** 2).sum(1) <= 4 * RADIUS2) & (g.sum(1) % 4 == 0)])
    points = np.concatenate(grids)
    keys = _key(points)
    order = np.argsort(keys)
    return points[order], keys[order]


def encode(points):
    """Lattice points [..., 8] -> uint16 codes."""
    doubled = np.rint(np.asarray(points) * 2)
    _, keys = table()
    key = _key(doubled)
    codes = np.searchsorted(keys, key)
    if np.any(keys[np.minimum(codes, len(keys) - 1)] != key):
        raise ValueError('point outside the E8 code table')
    return codes.astype(np.uint16)


def decode(codes):
    """uint16 codes [...] -> lattice points [..., 8]."""
    points, _ = table()
    return points[np.asarray(codes, np.int64)].astype(np.float32) / 2
