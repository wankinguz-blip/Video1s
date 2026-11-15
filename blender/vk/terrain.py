"""Analytic terrain: the three stage hills on a rolling meadow.  height(x, y) is the single source
of truth for the ground (mesh, grass placement, character marks, rocket launch points).

Contract (do not break): height(mark) == config.HILL_TOP_Z[...] exactly at the three marks, and
the hill tops are almost flat within ~0.5 m of the marks (feet stay planted without ground IK).

Everything except the stage hills is laid out in *screen* terms: a "band" is a rolling ridge at a
given depth whose crest is described by the stage-plane height (zs) it reaches in the picture as a
function of the stage-plane x it projects to (xs).  That makes the composition (how much of the
frame every depth layer occupies) directly tunable; see _BANDS.
"""
import numpy as np

from . import config as C

S0 = -1.5            # meadow level around the stage hills (below the bottom frame edge)
_K = 1.25            # dome roundness


def _dome(u):
    """Rounded hill profile: 1 at the centre, 0 with zero slope at u >= 1."""
    u = np.clip(u, 0.0, 1.0)
    return np.cos(0.5 * np.pi * u) ** _K


def _hill_u(x, y, hx, hy, radius):
    # hills are a little deeper (in y) than wide so the crown reads round from the front
    return np.sqrt(((x - hx) / radius) ** 2 + ((y - hy) / (radius * 1.25)) ** 2)


def _smooth(a, b, v):
    s = np.clip((v - a) / (b - a), 0.0, 1.0)
    return s * s * (3 - 2 * s)


# ---------------------------------------------------------------- depth layers ("bands")
# Each band: depth y0, depth sigma sy, crest height zs0 on the stage plane (picture coordinates),
# crest undulation [(amplitude, frequency rad/m of stage x, phase)], local bumps [(xs, width, dz)].
_BANDS = [
    # foreground knolls: only peek into the bottom of the frame in the gaps between the hills
    dict(y0=-11.0, sy=3.4, zs0=-1.75, waves=[(0.03, 0.9, 0.3)],
         bumps=[(-10.9, 3.3, 0.86), (10.2, 3.1, 0.84), (-29.2, 2.6, 0.9), (27.9, 2.4, 0.9)]),
    # mid-ground swells right behind the stage hills (the "meadow" the hills sit on)
    dict(y0=12.0, sy=4.2, zs0=-0.66, waves=[(0.07, 0.31, 0.9), (0.035, 0.83, 2.2), (0.018, 1.9, 4.0)],
         bumps=[(-10.6, 4.0, 0.10), (10.4, 3.6, 0.08), (-27.5, 3.0, 0.06), (26.0, 3.0, 0.07)]),
    dict(y0=30.0, sy=7.5, zs0=-0.40, waves=[(0.07, 0.23, 2.9), (0.04, 0.61, 0.4), (0.02, 1.37, 1.1)],
         bumps=[(-15.5, 4.5, 0.10), (6.5, 4.0, 0.09), (22.5, 4.0, 0.07)]),
    dict(y0=62.0, sy=13.0, zs0=-0.17, waves=[(0.08, 0.19, 0.2), (0.045, 0.47, 3.3), (0.02, 1.13, 5.2)],
         bumps=[(-24.0, 5.0, 0.12), (-4.5, 4.5, 0.06), (14.0, 5.0, 0.12)]),
    dict(y0=118.0, sy=22.0, zs0=0.06, waves=[(0.085, 0.15, 4.4), (0.05, 0.39, 1.0), (0.022, 0.97, 2.7)],
         bumps=[(-11.5, 5.5, 0.12), (24.5, 5.0, 0.14), (-29.0, 4.0, 0.08)]),
    # far ridge: closes the horizon everywhere (trees stand on it)
    dict(y0=215.0, sy=38.0, zs0=0.30, waves=[(0.09, 0.12, 1.6), (0.05, 0.33, 0.1), (0.025, 0.81, 3.9)],
         bumps=[(-19.0, 6.0, 0.13), (3.5, 6.0, 0.08), (18.5, 5.0, 0.10)]),
]
_SMAX_P = 5.0        # bands are combined with a soft maximum so they do not pile up


def band_crest(band, xs):
    """Stage-plane crest height (zs) of a band for stage-plane x (xs)."""
    zs = np.full_like(np.asarray(xs, dtype=np.float64), band["zs0"])
    for a, f, p in band["waves"]:
        zs = zs + a * np.sin(f * xs + p)
    for bx, bw, dz in band["bumps"]:
        zs = zs + dz * np.exp(-0.5 * ((xs - bx) / bw) ** 2)
    return zs


def stage_x(x, y):
    """Stage-plane x that world point (x, y) projects to (static camera)."""
    return C.CAM_X + (x - C.CAM_X) * C.CAM_DIST / (y + C.CAM_DIST)


def height(x, y):
    """Ground height z at world (x, y).  Accepts scalars or numpy arrays."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x, y = np.broadcast_arrays(x, y)
    # gentle meadow undulation (deterministic)
    und = (0.14 * np.sin(0.21 * x + 0.9) * np.cos(0.17 * y - 0.4)
           + 0.07 * np.sin(0.53 * x - 1.3 + 0.31 * y)
           + 0.035 * np.sin(1.07 * x + 0.47 * y + 2.1)
           + 0.018 * np.sin(2.3 * x - 1.7 * y + 0.6))
    depth = np.maximum(y + C.CAM_DIST, 1.0)
    k = depth / C.CAM_DIST
    xs = C.CAM_X + (x - C.CAM_X) / k
    acc = np.zeros_like(x)
    for b in _BANDS:
        kc = (b["y0"] + C.CAM_DIST) / C.CAM_DIST
        crest = C.CAM_Z + (band_crest(b, xs) - C.CAM_Z) * kc          # world z of the crest
        rise = np.clip(crest - S0, 0.0, None) * np.exp(-0.5 * ((y - b["y0"]) / b["sy"]) ** 2)
        acc = acc + rise ** _SMAX_P
    rise = acc ** (1.0 / _SMAX_P)
    und = und * np.clip(k, 0.6, 2.5)                                   # broader relief far away
    mask = np.ones_like(x)
    hills = np.zeros_like(x)
    lump = np.zeros_like(x)
    for _role, hx, hy, top, radius, _ in C.HILLS:
        u = _hill_u(x, y, hx, hy, radius)
        mask = mask * _smooth(0.55, 1.30, u)
        hills = np.maximum(hills, (top - S0) * _dome(u))
        # subtle surface variation on the slopes (never on the flat top)
        lx, ly = x - hx, y - hy
        w = _smooth(0.22, 0.55, u) * (1.0 - _smooth(0.9, 1.25, u))
        lump = lump + w * (0.035 * np.sin(1.9 * lx + 0.7 * ly + hx) * np.cos(1.3 * ly - 0.5 * lx + 1.0)
                           + 0.02 * np.sin(3.7 * lx - 2.1 * ly + 2.0 * hx))
    return S0 + mask * (und + rise) + hills + lump


def height1(x, y):
    """Scalar convenience wrapper."""
    return float(height(x, y))


def normal(x, y, eps=0.05):
    """Unit surface normals (…, 3) by central differences."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    dzdx = (height(x + eps, y) - height(x - eps, y)) / (2 * eps)
    dzdy = (height(x, y + eps) - height(x, y - eps)) / (2 * eps)
    n = np.stack([-dzdx, -dzdy, np.ones_like(dzdx)], axis=-1)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def mark(name):
    """World position (x, y, z) of a character mark ('emo' | 'singer' | 'sax')."""
    x, y = C.MARKS[name]
    return (x, y, height1(x, y))
