"""Fireworks and the particle word VANKING.

Everything here is CLOSED FORM: position, size, colour and brightness of every spark are pure
functions of loop time t, so any frame renders independently and frame 450 == frame 0.

Geometry
  One mesh (`VK_FW_Sparks`) with pre-allocated, camera-parallel geometry that is rewritten each frame
  with numpy (`foreach_set`):
    * sprites  – quads in the XZ plane (the camera has no tilt, so they are exactly screen-parallel)
    * ribbons  – strips that follow the real past trajectory of a spark (curved, tapering trails)
  The material is pure additive emission (Emission + Transparent, blended) that reads the per-vertex
  colour attribute `col` (RGB = radiance, A = profile selector) and builds a soft profile from the UVs:
  hot core + wide halo for sparks (A = 0), a wide bell for glows (A = 1).  All glow is geometry, so
  there is no compositor / bloom / screen-space effect and the three screen slices match at the seams.

Choreography of one cycle (times from config.FIREWORK_CYCLES)
  launch → rockets rise from behind the hills (3 main ones behind the central hill, 4 small satellites
  behind the side hills) with a hot tail and a falling sparkle trail;
  burst  → flash + spherical 3-D shells (drag + gravity), coloured trails, pistil, crackle;
  gather → every letter spark is a real spark of one of the three main shells: it leaves the shell,
  turns white and flies into its target point of the word;
  hold   → the word shimmers for ~1 s;
  dissolve → sparks detach left → right, fall, shrink, cool down and vanish above the characters.

Lights
  Every burst drives two point lights: a "rim" light at the true burst position (behind the hills) and
  a "wash" light pulled towards the camera so that the camera-facing sides of the grass, characters
  and instruments are visibly tinted for a moment.  A faint white light is on while the word is on.
  All energies are exactly zero outside the active windows.
"""
import json
import math
import os

import bpy
import numpy as np
from mathutils import Vector

from . import camera as cam_mod
from . import config as C
from . import util

# ====================================================================================== tunables
DEPTH = C.FIREWORK_DEPTH          # plane of the word (world y)

# ---- colours (scene-linear).  Saturated primaries: AgX desaturates the hot cores by itself.
WHITE = np.array([1.0, 1.0, 1.0])
WARM = np.array([1.0, 0.68, 0.32])           # cooling ember tint of the crumbling letters
COLOR_GAIN = {"blue": 1.55, "red": 1.2, "green": 0.95}     # perceptual balance of the three hues


def _lin(c):
    return np.array(util.srgb(*c))


def _burst_color(name):
    c = _lin(C.COLORS[name])
    return c / c.max() * COLOR_GAIN[name]


# ---- burst physics
KD = 2.6                # air drag (1/s): shells reach ~90 % of their radius in 0.9 s
STAR_G = 0.75           # gravity on shell stars, in burst radii per s² (slow, "large scale" feel)

# ---- layout of the bursts: stage-plane position (x, z), depth y, radius (stage metres)
#      main bursts sit over the centre screen around the word, satellites flank the side characters
_EMO_X = C.MARKS["emo"][0]
_SAX_X = C.MARKS["sax"][0]
MAIN_BURSTS = [
    dict(x=-6.1, z=1.78, y=12.5, r=2.35, launch_x=-2.55, stars=720, light_x=-3.7),
    dict(x=1.0, z=2.62, y=15.5, r=1.85, launch_x=1.75, stars=460, light_x=0.0),     # x / launch_x mirrored on odd cycles
    dict(x=6.3, z=1.86, y=13.5, r=2.35, launch_x=2.6, stars=720, light_x=3.7),
]
SAT_BURSTS = [            # col = index into the cycle's (left, centre, right) colours
    dict(x=_EMO_X - 3.95, z=1.92, y=19.0, r=1.42, launch_x=_EMO_X - 2.0, stars=300, col=1, delay=0.20, light_x=_EMO_X - 2.3),
    dict(x=_EMO_X + 3.55, z=2.02, y=17.0, r=1.30, launch_x=_EMO_X + 1.9, stars=280, col=2, delay=0.11, light_x=_EMO_X + 2.1),
    dict(x=_SAX_X - 3.60, z=2.00, y=17.5, r=1.30, launch_x=_SAX_X - 1.9, stars=280, col=0, delay=0.14, light_x=_SAX_X - 2.1),
    dict(x=_SAX_X + 3.75, z=1.88, y=19.5, r=1.42, launch_x=_SAX_X + 2.0, stars=300, col=1, delay=0.24, light_x=_SAX_X + 2.3),
]
LAUNCH_Z = -1.55          # rockets start below the frame, hidden behind the hills (stage z)
CRACKLE_PER_STAR = 0.55   # crackle sparks per shell star
ROCKET_SPARKS_MAIN = 110
ROCKET_SPARKS_SAT = 60

# ---- sprite sizes (stage metres = apparent size; the quad half-size, the hot core is ~1/6 of it)
STAR_SIZE = 0.085
TEXT_SIZE = 0.050
TEXT_EDGE_GAIN = 1.25     # outline points of the letters are a little brighter → crisp silhouettes
TEXT_GLOW_EVERY = 9       # every n-th letter spark carries a wide soft glow sprite
TEXT_GLOW_SIZE = 0.20
DEPTH_CUE = 5.0           # exaggerates perspective inside a shell (near sparks bigger / brighter)

# ---- brightness (emission radiance at the sprite centre)
STAR_GAIN = 9.0
TEXT_GAIN = 9.0
TEXT_GLOW_GAIN = 0.012
ROCKET_GAIN = 30.0
FLASH_GAIN = 15.0
CLOUD_GAIN = 0.55

# ---- trails
STAR_TRAIL = 0.16         # seconds of trajectory shown by a star ribbon
TEXT_TRAIL = 0.05
ROCKET_TRAIL = 0.34
S_STAR, S_TEXT, S_ROCKET, S_RAY = 7, 6, 12, 4      # samples per ribbon
FLASH_RAYS = 16           # needle rays of the burst flash

# ---- word gather
GATHER_SPREAD = 0.36      # share of the gather window over which sparks leave their shell (rest = flight)

# ---- word dissolve
FALL_G = 6.0              # gravity on crumbling letter sparks (stage m/s²)
FALL_FLOOR_NEAR = 1.97    # letter sparks must be gone above this height near the singer (hat ≈ 1.9)
FALL_FLOOR_FAR = 1.45

# ---- lights (Watts, at the flash peak).  Per burst: a small point light at the true burst position
#      (lights the meadow behind the stage) + two soft spots aimed at the stage under the burst:
#      "back" from above / behind (rim + tops of grass, hats, shoulders) and "front" (camera-facing sides).
LIGHT_LOCAL = (1800.0, 1100.0)        # (main, satellite)
LIGHT_BACK = (26000.0, 18000.0)
LIGHT_FRONT = (11000.0, 8000.0)
LIGHT_TEXT = 2200.0                   # faint white glow while the word is on
BACK_POS = (9.0, 15.0)                # spot position: (world y, world z)
FRONT_POS = (-13.0, 10.5)
SPOT_SIZE = math.radians(36.0)
ROCKET_LIGHT = 0.05                   # share of the local light that follows a rising rocket

# ====================================================================================== helpers
_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
_CORNERS_X = np.array([-1.0, 1.0, 1.0, -1.0])
_CORNERS_Z = np.array([-1.0, -1.0, 1.0, 1.0])


def _sstep(a, b, v):
    u = np.clip((v - a) / (b - a), 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def _drag(tau):
    return (1.0 - np.exp(-KD * tau)) / KD


def _sphere_dirs(n, rs):
    """n well-spread unit vectors (Fibonacci sphere + jitter, random orientation, shuffled)."""
    i = np.arange(n) + 0.5
    zz = 1.0 - 2.0 * i / n
    r = np.sqrt(np.clip(1.0 - zz * zz, 0.0, 1.0))
    th = i * 2.399963229728653 + rs.rand(n) * 0.35
    d = np.stack([r * np.cos(th), r * np.sin(th), zz], axis=1)
    a, b, c = rs.rand(3) * math.tau                       # random rotation → every burst looks different
    for ang, (p, q) in ((a, (0, 1)), (b, (1, 2)), (c, (0, 2))):
        cs, sn = math.cos(ang), math.sin(ang)
        dp, dq = d[:, p].copy(), d[:, q].copy()
        d[:, p], d[:, q] = cs * dp - sn * dq, sn * dp + cs * dq
    rs.shuffle(d)
    return d


def _load_text():
    with open(os.path.join(_DATA, "vanking_text.json"), "r", encoding="utf-8") as f:
        d = json.load(f)
    pts = np.array(d["points"], dtype=np.float64)
    return pts, float(d["widthInCapHeights"])


class _Cam:
    """Stage-plane ↔ world mapping for the camera position of this frame."""

    def __init__(self, t=0.0):
        self.x, self.y, self.z = cam_mod.camera_position(t)

    def k(self, y):
        return (y - self.y) / (0.0 - self.y)

    def to_stage(self, p):
        """World points (..., 3) → stage-plane x, z and depth factor k."""
        k = (p[..., 1] - self.y) / (0.0 - self.y)
        return self.x + (p[..., 0] - self.x) / k, self.z + (p[..., 2] - self.z) / k, k

    def world(self, xs, zs, y):
        k = self.k(y)
        return np.stack(np.broadcast_arrays(self.x + (xs - self.x) * k, y, self.z + (zs - self.z) * k), axis=-1)


# ====================================================================================== cycle data
class _Cycle:
    pass


def _make_cycle(ci, text_pts, text_w):
    """Pre-compute every per-particle constant of firework cycle `ci` (deterministic seeds)."""
    cyc = C.FIREWORK_CYCLES[ci]
    rs = np.random.RandomState(9100 + 131 * ci)
    cam = _Cam(0.0)
    c = _Cycle()
    c.index = ci
    c.launch, c.burst = cyc["launch"], cyc["burst"]
    c.g0, c.g1 = cyc["gather"]
    c.h0, c.h1 = cyc["hold"]
    c.d0, c.d1 = cyc["dissolve"]
    c.speed = cyc.get("speed", 1.0)
    c.end = c.d1
    names = C.CYCLE_COLORS[ci]
    life_k = min(1.0, 1.0 / c.speed + 0.15)

    # ---------------------------------------------------------------- bursts
    bursts = []
    mirror = -1.0 if ci % 2 else 1.0
    for k, m in enumerate(MAIN_BURSTS):
        b = dict(m)
        if k == 1:
            b["x"], b["launch_x"] = b["x"] * mirror, b["launch_x"] * mirror
        b.update(main=True, color=_burst_color(names[k]), delay=0.0, text_group=k)
        bursts.append(b)
    for s in SAT_BURSTS:
        b = dict(s)
        b.update(main=False, color=_burst_color(names[s["col"]]), delay=s["delay"] / c.speed, text_group=-1)
        bursts.append(b)
    for b in bursts:
        b["x"] += (rs.rand() - 0.5) * (0.5 if b["main"] else 0.7)
        b["z"] += (rs.rand() - 0.5) * (0.16 if b["main"] else 0.22)
        b["r"] *= 0.95 + 0.1 * rs.rand()
        b["launch_x"] += (rs.rand() - 0.5) * 0.3
        b["t_burst"] = c.burst + b["delay"]
        b["t_launch"] = c.launch + b["delay"]
        b["k"] = cam.k(b["y"])
        b["O"] = cam.world(b["x"], b["z"], b["y"])                     # world burst centre
        b["S"] = cam.world(b["launch_x"], LAUNCH_Z, b["y"])            # world launch point
        b["R"] = b["r"] * b["k"]                                       # world radius
        if b["text_group"] == 1:             # the centre rocket bows away from the singer
            b["curve"] = (0.38 + 0.1 * rs.rand()) * (1.0 if b["launch_x"] > 0 else -1.0)
        else:
            b["curve"] = (rs.rand() - 0.5) * 0.9 + (0.45 if b["x"] > b["launch_x"] else -0.45)
        b["wob"] = (rs.rand() * math.tau, 5.0 + 3.0 * rs.rand())
        b["ray_ang"] = (np.arange(FLASH_RAYS) + rs.rand(FLASH_RAYS) * 0.8) / FLASH_RAYS * math.tau
        b["ray_len"] = 0.35 + 1.0 * rs.rand(FLASH_RAYS) ** 1.5
    c.bursts = bursts

    # ---------------------------------------------------------------- letter targets → three groups
    cap = C.TEXT_CAP_H
    tx = C.TEXT_CENTER[0] + text_pts[:, 0] * cap
    tz = C.TEXT_CENTER[1] - text_pts[:, 1] * cap
    xn = (text_pts[:, 0] + 0.5 * text_w) / text_w                      # 0 … 1 across the word
    grp = np.clip(np.floor((xn + (rs.rand(len(xn)) - 0.5) * 0.07) * 3.0), 0, 2).astype(int)

    T = dict(O=[], V=[], g=[], kref=[], col=[], tgt=[], xn=[], edge=[], t0=[], size0=[])
    St = dict(O=[], V=[], g=[], kref=[], col=[], t0=[], life=[], size=[], gain=[], bi=[])
    Cr = dict(O=[], V=[], g=[], kref=[], col=[], t0=[], pop=[], size=[])
    for bi, b in enumerate(bursts):
        idx = np.nonzero(grp == b["text_group"])[0] if b["main"] else np.zeros(0, dtype=int)
        nt, ns = len(idx), b["stars"]
        dirs = _sphere_dirs(nt + ns, rs)
        # shell structure: most stars on a thin outer shell (reads as a sphere), the rest fill the inside
        u = rs.rand(nt + ns)
        sp = np.where(u < 0.68, 0.84 + 0.16 * rs.rand(nt + ns), 0.30 + 0.60 * np.sqrt(rs.rand(nt + ns)))
        V = dirs * (sp * b["R"] * KD)[:, None]
        if nt:
            # match directions to targets by screen angle so that paths do not cross (elegant flow)
            ctr = np.array([tx[idx].mean() - b["x"], tz[idx].mean() - b["z"]])
            a0 = math.atan2(ctr[1], ctr[0])
            d_ang = np.angle(np.exp(1j * (np.arctan2(dirs[:nt, 2], dirs[:nt, 0]) - a0)))
            t_ang = np.angle(np.exp(1j * (np.arctan2(tz[idx] - b["z"], tx[idx] - b["x"]) - a0)))
            od, ot = np.argsort(d_ang), np.argsort(t_ang)
            order = np.empty(nt, dtype=int)
            order[ot] = od                                              # target j gets direction order[j]
            T["O"].append(np.repeat(b["O"][None, :], nt, 0))
            T["V"].append(V[:nt][order])
            T["g"].append(np.full(nt, STAR_G * b["R"]))
            T["kref"].append(np.full(nt, b["k"]))
            T["col"].append(np.repeat(b["color"][None, :], nt, 0))
            T["tgt"].append(np.stack([tx[idx], tz[idx]], 1))
            T["xn"].append(xn[idx])
            T["edge"].append(text_pts[idx, 3])
            T["t0"].append(np.full(nt, b["t_burst"]))
            T["size0"].append(STAR_SIZE * (0.8 + 0.4 * rs.rand(nt)))
        # decorative shell stars
        inner = sp[nt:] < 0.6
        St["O"].append(np.repeat(b["O"][None, :], ns, 0))
        St["V"].append(V[nt:] * 1.04)
        St["g"].append(np.full(ns, STAR_G * b["R"]))
        St["kref"].append(np.full(ns, b["k"]))
        pist = np.where(inner, 0.5, 0.0)[:, None]                      # pistil: paler, hotter core stars
        St["col"].append(b["color"][None, :] * (1 - pist) + pist * np.array([1.0, 0.92, 0.8])[None, :] * b["color"].max())
        St["t0"].append(np.full(ns, b["t_burst"]))
        base_life = (0.95 + 0.75 * rs.rand(ns)) if bi != 1 else (0.75 + 0.5 * rs.rand(ns))
        St["life"].append(base_life * life_k * np.where(inner, 0.8, 1.0))
        St["size"].append(STAR_SIZE * (0.8 + 0.55 * rs.rand(ns)) * (1.0 if b["main"] else 0.82))
        St["gain"].append((0.75 + 0.5 * rs.rand(ns)) * (1.0 if bi != 1 else 0.85))
        St["bi"].append(np.full(ns, bi))
        # crackle: invisible stars that pop into short white-gold sparkles late in the shell's life
        nc = int(ns * CRACKLE_PER_STAR)
        cd = _sphere_dirs(nc, rs)
        csp = 0.45 + 0.6 * np.sqrt(rs.rand(nc))
        Cr["O"].append(np.repeat(b["O"][None, :], nc, 0))
        Cr["V"].append(cd * (csp * b["R"] * KD)[:, None])
        Cr["g"].append(np.full(nc, STAR_G * b["R"] * 1.3))
        Cr["kref"].append(np.full(nc, b["k"]))
        Cr["col"].append(np.repeat((0.35 * b["color"] + 0.65 * np.array([1.0, 0.86, 0.6]))[None, :], nc, 0))
        Cr["t0"].append(np.full(nc, b["t_burst"]))
        Cr["pop"].append((0.42 + 0.85 * rs.rand(nc) ** 0.8) * life_k * (1.0 if bi != 1 else 0.8))
        Cr["size"].append(STAR_SIZE * (0.55 + 0.5 * rs.rand(nc)) * (1.0 if b["main"] else 0.85))

    def cat(d):
        return {k: np.concatenate(v) for k, v in d.items()}

    T, St, Cr = cat(T), cat(St), cat(Cr)

    # ---------------------------------------------------------------- letter sparks: gather / hold / dissolve
    n = len(T["xn"])
    glen, dlen = c.g1 - c.g0, c.d1 - c.d0
    T["target"] = cam.world(T["tgt"][:, 0], T["tgt"][:, 1], DEPTH + (rs.rand(n) - 0.5) * 0.12)
    # where the spark is when gathering starts → far sparks leave first
    f0 = _drag(c.g0 - T["t0"])
    p0 = T["O"] + T["V"] * f0[:, None]
    dist = np.linalg.norm(p0 - T["target"], axis=1)
    order = np.clip(0.70 * rs.rand(n) + 0.30 * (1.0 - np.clip(dist / 6.5, 0.0, 1.0)), 0.0, 1.0)
    T["ga"] = c.g0 + order * glen * GATHER_SPREAD                      # sparks peel off the shells one by one
    T["gb"] = np.minimum(T["ga"] + glen * (1.0 - GATHER_SPREAD) * (0.9 + 0.1 * rs.rand(n)), c.g1)
    T["swirl"] = (rs.rand(n) - 0.5) * 2.0 * np.clip(dist, 0.5, 5.0) * 0.22
    T["swirl_y"] = (rs.rand(n) - 0.5) * 1.6                            # arc in depth while travelling
    T["ph"] = rs.rand(n) * math.tau
    T["fq"] = 7.0 + 9.0 * rs.rand(n)
    T["size"] = TEXT_SIZE * (0.9 + 0.25 * rs.rand(n)) * np.where(T["edge"] > 0, 1.05, 1.0)
    T["gain"] = (0.82 + 0.3 * rs.rand(n)) * np.where(T["edge"] > 0, TEXT_EDGE_GAIN, 1.0)
    T["glint_t"] = c.h0 + 0.05 + rs.rand(n) * (c.h1 - c.h0)            # one bright twinkle per spark
    T["glint_a"] = np.where(rs.rand(n) < 0.35, 1.2 + 1.6 * rs.rand(n), 0.0)
    # dissolve: detach left → right (plus noise), fall, shrink, fade.  Everything is gone at d1.
    T["da"] = c.d0 + T["xn"] * dlen * 0.25 + rs.rand(n) * dlen * 0.13
    T["dlife"] = np.minimum(dlen * (0.42 + 0.20 * rs.rand(n)), c.d1 - T["da"] - 1e-3)
    T["dvx"] = 0.10 + (rs.rand(n) - 0.5) * 0.42
    T["dvz"] = -0.14 + rs.rand(n) * 0.34                                # initial downward speed (may hop up a little)
    T["dvy"] = (rs.rand(n) - 0.5) * 1.2
    # fall is limited so that sparks die above the hat of the singer / above the characters
    near = 1.0 - _sstep(1.0, 2.2, np.abs(T["tgt"][:, 0] - C.MARKS["singer"][0]))
    floor = FALL_FLOOR_FAR + (FALL_FLOOR_NEAR - FALL_FLOOR_FAR) * near
    want = T["dvz"] * T["dlife"] + 0.5 * FALL_G * T["dlife"] ** 2
    room = np.maximum(T["tgt"][:, 1] - floor, 0.02)
    T["fall_k"] = np.minimum(1.0, room / np.maximum(want, 1e-3))
    T["glow_idx"] = np.arange(0, n, TEXT_GLOW_EVERY)
    c.T = T

    # ---------------------------------------------------------------- stars / crackle extras
    ns = len(St["life"])
    St["ph"] = rs.rand(ns) * math.tau
    St["fq"] = 10.0 + 14.0 * rs.rand(ns)
    St["trail"] = STAR_TRAIL * (0.7 + 0.6 * rs.rand(ns))
    c.St = St
    nc = len(Cr["pop"])
    Cr["ph"] = rs.rand(nc) * math.tau
    Cr["gap"] = 0.07 + 0.07 * rs.rand(nc)
    c.Cr = Cr

    # ---------------------------------------------------------------- rocket sparkle trails
    Rk = dict(bi=[], te=[], jit=[], vel=[], life=[], size=[], white=[], ph=[])
    for bi, b in enumerate(bursts):
        m = ROCKET_SPARKS_MAIN if b["main"] else ROCKET_SPARKS_SAT
        dur = b["t_burst"] - b["t_launch"]
        Rk["bi"].append(np.full(m, bi))
        Rk["te"].append(b["t_launch"] + (np.arange(m) + rs.rand(m)) / m * dur * 0.97)
        Rk["jit"].append((rs.rand(m, 3) - 0.5) * 0.06)
        v = (rs.rand(m, 3) - 0.5) * np.array([0.9, 0.9, 0.5])
        v[:, 2] -= 0.25
        Rk["vel"].append(v)
        Rk["life"].append((0.28 + 0.42 * rs.rand(m)) * life_k)
        Rk["size"].append(STAR_SIZE * (0.45 + 0.5 * rs.rand(m)) * (1.0 if b["main"] else 0.8))
        Rk["white"].append((rs.rand(m) < 0.4).astype(float))
        Rk["ph"].append(rs.rand(m) * math.tau)
    c.Rk = cat(Rk)
    return c


def _rocket_pos(b, tt):
    """World position of the rocket of burst b at times tt (array); clamps to launch / burst point."""
    u = np.clip((tt - b["t_launch"]) / (b["t_burst"] - b["t_launch"]), 0.0, 1.0)
    e = 1.0 - (1.0 - u) ** 2.1
    p = b["S"][None, :] + (b["O"] - b["S"])[None, :] * e[:, None]
    bow = np.sin(np.pi * e) * b["curve"] + 0.05 * np.sin(b["wob"][1] * e + b["wob"][0]) * (1.0 - e)
    p[:, 0] += bow * b["k"]
    return p, u


def _ballistic(O, V, g, tau):
    """Drag + gravity trajectory.  tau: (S, N) seconds since burst (clamped ≥ 0); returns (S, N, 3)."""
    tau = np.maximum(tau, 0.0)
    f = _drag(tau)
    p = O[None, :, :] + V[None, :, :] * f[:, :, None]
    p[:, :, 2] -= (g[None, :] / KD) * (tau - f)
    return p


def _text_pos(c, tt):
    """World positions (S, N, 3) of the letter sparks at times tt (S, N)."""
    T = c.T
    p = _ballistic(T["O"], T["V"], T["g"], tt - T["t0"][None, :])
    gu = np.clip((tt - T["ga"][None, :]) / (T["gb"] - T["ga"])[None, :], 0.0, 1.0)
    e = 1.0 - (1.0 - gu) ** 4 * (1.0 + 4.0 * gu)          # zero speed at both ends, soft landing
    d = T["target"][None, :, :] - p
    dl = np.sqrt(d[:, :, 0] ** 2 + d[:, :, 2] ** 2) + 1e-6
    sw = np.sin(np.pi * e) * T["swirl"][None, :]
    q = p + d * e[:, :, None]
    q[:, :, 0] -= d[:, :, 2] / dl * sw
    q[:, :, 2] += d[:, :, 0] / dl * sw
    q[:, :, 1] += np.sin(np.pi * e) * T["swirl_y"][None, :]
    # shimmer (grows a little right before the spark lets go)
    loosen = _sstep(c.d0 - 0.15, 1.0, 0.0) * 0.0 + np.clip((tt - (c.d0 - 0.15)) / (T["da"][None, :] - (c.d0 - 0.15)), 0.0, 1.0)
    amp = (0.0035 + 0.012 * loosen * loosen) * e ** 6 * T["kref"][None, :]
    q[:, :, 0] += amp * np.sin(T["fq"][None, :] * tt + T["ph"][None, :])
    q[:, :, 2] += amp * np.cos(0.83 * T["fq"][None, :] * tt + 1.3 * T["ph"][None, :])
    # crumble
    age = np.maximum(tt - T["da"][None, :], 0.0)
    kk = T["kref"][None, :]
    q[:, :, 0] += T["dvx"][None, :] * age * kk
    q[:, :, 1] += T["dvy"][None, :] * age
    q[:, :, 2] -= (T["dvz"][None, :] * age + 0.5 * FALL_G * age * age) * T["fall_k"][None, :] * kk
    return q


# ====================================================================================== geometry buffers
class _Buf:
    """Vertex / colour buffers of the spark mesh + writers for sprites and ribbons."""

    def __init__(self):
        self.n_sprites = 0
        self.n_verts = 0
        self.faces = []
        self.uv = []

    # ---- allocation (build time)
    def alloc_sprites(self, n):
        first = self.n_verts
        self.n_verts += 4 * n
        return (first, n)

    def alloc_ribbons(self, n, s):
        first = self.n_verts
        self.n_verts += 2 * s * n
        return (first, n, s)

    def finish(self, blocks):
        faces, uvs = [], []
        for blk in blocks:
            if len(blk) == 2:
                first, n = blk
                base = first + 4 * np.arange(n)[:, None]
                faces.append((base + np.arange(4)[None, :]).reshape(-1, 4))
                uvs.append(np.tile(np.stack([_CORNERS_X, _CORNERS_Z], 1), (n, 1)))
            else:
                first, n, s = blk
                base = first + 2 * s * np.arange(n)[:, None, None] + 2 * np.arange(s - 1)[None, :, None]
                quad = np.array([0, 1, 3, 2])[None, None, :]
                faces.append((base + quad).reshape(-1, 4))
                uvs.append(np.tile(np.array([[-1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]]), (n * (s - 1), 1)))
        self.faces = np.concatenate(faces).astype(np.int32)
        self.uv = np.concatenate(uvs).astype(np.float32)
        self.co = np.zeros((self.n_verts, 3), dtype=np.float32)
        self.col = np.zeros((self.n_verts, 4), dtype=np.float32)

    def clear(self):
        self.co[:] = 0.0
        self.col[:] = 0.0

    # ---- writers (per frame)
    def sprites(self, blk, cam, pos, half, rgb, soft=0.0, sub=None):
        """pos: world (n, 3); half: apparent half-size (stage m, 0 → hidden); rgb: (n, 3) radiance."""
        first, n = blk
        if sub is not None:
            first, n = first + 4 * sub[0], sub[1]
        xs, zs, k = cam.to_stage(pos)
        co = self.co[first:first + 4 * n].reshape(n, 4, 3)
        hk = (half * k)[:, None]
        co[:, :, 0] = pos[:, 0:1] + hk * _CORNERS_X[None, :]
        co[:, :, 1] = pos[:, 1:2]
        co[:, :, 2] = pos[:, 2:3] + hk * _CORNERS_Z[None, :]
        col = self.col[first:first + 4 * n].reshape(n, 4, 4)
        col[:, :, :3] = rgb[:, None, :]
        col[:, :, 3] = soft

    def ribbons(self, blk, cam, pos, half, rgb, taper=0.55, power=1.6, sub=None):
        """pos: world (S, N, 3), sample 0 = head; half: (N,) apparent half-width at the head;
        rgb: (N, 3) radiance at the head.  Width and brightness taper towards the tail."""
        first, n, s = blk
        if sub is not None:
            first, n = first + 2 * s * sub[0], sub[1]
        xs, zs, k = cam.to_stage(pos)                               # (S, N)
        tx = np.empty_like(xs)
        tz = np.empty_like(zs)
        tx[0], tz[0] = xs[0] - xs[1], zs[0] - zs[1]
        tx[1:-1], tz[1:-1] = xs[:-2] - xs[2:], zs[:-2] - zs[2:]
        tx[-1], tz[-1] = xs[-2] - xs[-1], zs[-2] - zs[-1]
        ln = np.sqrt(tx * tx + tz * tz)
        ok = ln > 1e-5
        nx = np.where(ok, -tz / np.maximum(ln, 1e-9), 0.0)
        nz = np.where(ok, tx / np.maximum(ln, 1e-9), 0.0)
        for j in range(1, s):                                       # degenerate samples inherit the direction
            bad = ~ok[j]
            nx[j, bad], nz[j, bad] = nx[j - 1, bad], nz[j - 1, bad]
        for j in range(s - 2, -1, -1):
            bad = ~ok[j] & (nx[j] == 0.0) & (nz[j] == 0.0)
            nx[j, bad], nz[j, bad] = nx[j + 1, bad], nz[j + 1, bad]
        u = np.arange(s)[:, None] / (s - 1.0)
        w = half[None, :] * (1.0 - taper * u) * k                   # world half-width per sample
        co = self.co[first:first + 2 * s * n].reshape(n, s, 2, 3)
        px, py, pz = pos[:, :, 0].T, pos[:, :, 1].T, pos[:, :, 2].T    # (N, S)
        ox, oz = (nx * w).T, (nz * w).T
        co[:, :, 0, 0], co[:, :, 0, 1], co[:, :, 0, 2] = px - ox, py, pz - oz
        co[:, :, 1, 0], co[:, :, 1, 1], co[:, :, 1, 2] = px + ox, py, pz + oz
        fade = ((1.0 - u) ** power).T                               # (1 | N, S)
        col = self.col[first:first + 2 * s * n].reshape(n, s, 2, 4)
        col[:, :, :, :3] = (rgb[:, None, :] * fade[:, :, None])[:, :, None, :]
        col[:, :, :, 3] = 0.0


# ====================================================================================== material / mesh / lights
def _material():
    mat = bpy.data.materials.new("VK_FW_Spark")
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    N, L = nt.nodes, nt.links

    def math_(op, a=None, b=None, clamp=False):
        n = N.new("ShaderNodeMath")
        n.operation = op
        n.use_clamp = clamp
        for i, v in enumerate((a, b)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                n.inputs[i].default_value = v
            else:
                L.new(v, n.inputs[i])
        return n.outputs[0]

    uv = N.new("ShaderNodeUVMap")
    uv.uv_map = "uv"
    dot = N.new("ShaderNodeVectorMath")
    dot.operation = "DOT_PRODUCT"
    L.new(uv.outputs["UV"], dot.inputs[0])
    L.new(uv.outputs["UV"], dot.inputs[1])
    r2 = dot.outputs["Value"]
    win = math_("SUBTRACT", 1.0, r2, clamp=True)                    # 1 at the centre, 0 at the quad edge
    core = math_("EXPONENT", math_("MULTIPLY", r2, -1.0 / (2 * 0.085 ** 2)))
    halo = math_("MULTIPLY", math_("EXPONENT", math_("MULTIPLY", r2, -1.0 / (2 * 0.30 ** 2))), 0.085)
    wide = math_("MULTIPLY", math_("POWER", win, 3.0), 0.022)       # faint skirt → soft, bloom-like falloff
    spark = math_("MULTIPLY", math_("ADD", math_("ADD", core, halo), wide), math_("MULTIPLY", win, win))
    soft = math_("DIVIDE", math_("MULTIPLY", win, win), math_("ADD", math_("MULTIPLY", r2, 1.0 / 0.02), 1.0))   # peaked glare
    attr = N.new("ShaderNodeAttribute")
    attr.attribute_type = "GEOMETRY"
    attr.attribute_name = "col"
    mix = N.new("ShaderNodeMix")
    mix.data_type = "FLOAT"
    L.new(attr.outputs["Alpha"], mix.inputs[0])
    L.new(spark, mix.inputs[2])
    L.new(soft, mix.inputs[3])
    scale = N.new("ShaderNodeVectorMath")
    scale.operation = "SCALE"
    L.new(attr.outputs["Color"], scale.inputs[0])
    L.new(mix.outputs[0], scale.inputs["Scale"])
    em = N.new("ShaderNodeEmission")
    L.new(scale.outputs["Vector"], em.inputs["Color"])
    em.inputs["Strength"].default_value = 1.0
    tr = N.new("ShaderNodeBsdfTransparent")
    add = N.new("ShaderNodeAddShader")
    L.new(em.outputs[0], add.inputs[0])
    L.new(tr.outputs[0], add.inputs[1])
    out = N.new("ShaderNodeOutputMaterial")
    L.new(add.outputs[0], out.inputs["Surface"])
    mat.surface_render_method = "BLENDED"       # additive: transparent (white) + emission, order independent
    mat.use_transparency_overlap = True
    mat.use_backface_culling = False
    mat.use_transparent_shadow = True
    return mat


def _mesh(buf, col):
    me = bpy.data.meshes.new("VK_FW_Sparks")
    nf = len(buf.faces)
    me.vertices.add(buf.n_verts)
    me.loops.add(nf * 4)
    me.polygons.add(nf)
    me.loops.foreach_set("vertex_index", buf.faces.ravel())
    me.polygons.foreach_set("loop_start", np.arange(0, nf * 4, 4, dtype=np.int32))
    me.polygons.foreach_set("loop_total", np.full(nf, 4, dtype=np.int32))
    uv = me.uv_layers.new(name="uv")
    uv.data.foreach_set("uv", buf.uv.ravel())
    me.attributes.new("col", "FLOAT_COLOR", "POINT")
    me.update()
    obj = bpy.data.objects.new("VK_FW_Sparks", me)
    col.objects.link(obj)
    obj.visible_shadow = False
    for a in ("visible_diffuse", "visible_glossy", "visible_transmission", "visible_volume_scatter"):
        if hasattr(obj, a):
            setattr(obj, a, False)
    return obj


def _light(name, col, kind="POINT", radius=1.0):
    li = bpy.data.lights.new(name, kind)
    li.energy = 0.0
    li.shadow_soft_size = radius
    li.use_shadow = False            # flashes are short; shadow-less lights are cheap and seam-safe
    li.specular_factor = 0.6
    if kind == "SPOT":
        li.spot_size = SPOT_SIZE
        li.spot_blend = 1.0
    ob = bpy.data.objects.new(name, li)
    col.objects.link(ob)
    return ob


def _aim(ob, pos, target):
    ob.location = pos
    d = Vector(target) - Vector(pos)
    ob.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


# ====================================================================================== build / update
class _State:
    pass


def build(ctx):
    st = _State()
    col = ctx.collection("Fireworks")
    pts, width = _load_text()
    st.cycles = [_make_cycle(ci, pts, width) for ci in range(len(C.FIREWORK_CYCLES))]
    c0 = st.cycles[0]
    nt, ns, nc, nr = len(c0.T["xn"]), len(c0.St["life"]), len(c0.Cr["pop"]), len(c0.Rk["te"])
    nb = len(c0.bursts)
    for c in st.cycles:                       # the same geometry is reused by every cycle
        assert (len(c.T["xn"]), len(c.St["life"]), len(c.Cr["pop"]), len(c.Rk["te"])) == (nt, ns, nc, nr)

    buf = _Buf()
    st.b_cloud = buf.alloc_sprites(nb)                    # big soft glows first (drawn "behind")
    st.b_flash = buf.alloc_sprites(nb * 2)
    st.b_text_glow = buf.alloc_sprites(len(c0.T["glow_idx"]))
    st.b_star_rib = buf.alloc_ribbons(ns, S_STAR)
    st.b_text_rib = buf.alloc_ribbons(nt, S_TEXT)
    st.b_rocket_rib = buf.alloc_ribbons(nb, S_ROCKET)
    st.b_ray = buf.alloc_ribbons(nb * FLASH_RAYS, S_RAY)
    st.b_star = buf.alloc_sprites(ns)
    st.b_crackle = buf.alloc_sprites(nc)
    st.b_rspark = buf.alloc_sprites(nr)
    st.b_rocket = buf.alloc_sprites(nb * 2)
    st.b_text = buf.alloc_sprites(nt)
    buf.finish([st.b_cloud, st.b_flash, st.b_text_glow, st.b_star_rib, st.b_text_rib, st.b_rocket_rib, st.b_ray,
                st.b_star, st.b_crackle, st.b_rspark, st.b_rocket, st.b_text])
    st.buf = buf
    st.obj = _mesh(buf, col)
    st.obj.data.materials.append(_material())
    st.dirty = True

    st.local = [_light(f"VK_FW_Local{i}", col, "POINT", 1.0) for i in range(nb)]
    st.back = [_light(f"VK_FW_Back{i}", col, "SPOT", 1.5) for i in range(nb)]
    st.front = [_light(f"VK_FW_Front{i}", col, "SPOT", 1.5) for i in range(nb)]
    st.text_light = _light("VK_FW_TextGlow", col, "SPOT", 2.0)
    st.lights = st.local + st.back + st.front + [st.text_light]
    if ctx.engine == "cycles":                # thousands of overlapping additive sprites
        ctx.scene.cycles.transparent_max_bounces = max(ctx.scene.cycles.transparent_max_bounces, 256)
    return st


def _active_cycle(st, t):
    for c in st.cycles:
        if c.launch <= t <= c.end:
            return c
    return None


def _write(st):
    me = st.obj.data
    me.vertices.foreach_set("co", st.buf.co.ravel())
    me.attributes["col"].data.foreach_set("color", st.buf.col.ravel())
    me.update()


def _face_fade(xs, zs, near=1.1, far=2.0):
    """1 everywhere except close to / below the characters' heads (non-letter sparks only)."""
    f = np.ones_like(xs)
    fz = _sstep(C.FACE_SAFE_Z + 0.02, C.FACE_SAFE_Z + 0.42, zs)
    for mx, _my in C.MARKS.values():
        w = 1.0 - _sstep(near, far, np.abs(xs - mx))
        f *= 1.0 - w * (1.0 - fz)
    return f * _sstep(-0.95, -0.45, zs)          # … and evaporate before reaching the ground


def update(st, ctx, t):
    t = float(t) % C.LOOP
    c = _active_cycle(st, t)
    buf = st.buf
    if c is None:
        if st.dirty:
            buf.clear()
            _write(st)
            st.dirty = False
        for ob in st.lights:
            ob.data.energy = 0.0
        return
    st.dirty = True
    buf.clear()
    cam = _Cam(t)
    nb = len(c.bursts)

    # ------------------------------------------------------------------ rockets, flashes, lights
    for bi, b in enumerate(c.bursts):
        tau = t - b["t_burst"]
        dur = b["t_burst"] - b["t_launch"]
        scale_b = 1.0 if b["main"] else 0.72
        colr = b["color"]
        # hot tail: ribbon along the real path; after the burst it shrinks into the burst point
        tt = t - np.arange(S_ROCKET) * (ROCKET_TRAIL * min(1.0, dur) / (S_ROCKET - 1))
        tt = np.maximum(tt, b["t_launch"])
        pos, u = _rocket_pos(b, tt)
        rise = _sstep(0.0, 0.10, u[0]) * (1.0 - _sstep(0.0, 0.10, tau))
        flick = 0.82 + 0.18 * math.sin(61.0 * t + 1.7 * bi) * math.sin(23.0 * t + bi)
        hot = 0.45 * colr + 0.55 * WHITE * colr.max()
        if rise > 0.0:
            ff = float(_face_fade(*cam.to_stage(pos[0:1])[:2], near=0.6, far=1.1)[0])
            buf.ribbons(st.b_rocket_rib, cam, pos[:, None, :], np.array([STAR_SIZE * 1.15 * scale_b]),
                        (hot * ROCKET_GAIN * 0.55 * rise * flick * ff)[None, :], taper=0.75, power=1.4, sub=(bi, 1))
            head = pos[0:1] if tau < 0 else b["O"][None, :]
            buf.sprites(st.b_rocket, cam, np.repeat(head, 2, 0),
                        np.array([STAR_SIZE * 1.9, STAR_SIZE * 9.0]) * scale_b,
                        np.stack([hot * ROCKET_GAIN * rise * flick * ff, colr * 0.9 * rise * flick * ff]), sub=(2 * bi, 2))
            buf.col[st.b_rocket[0] + 4 * (2 * bi + 1): st.b_rocket[0] + 4 * (2 * bi + 2), 3] = 1.0   # soft glow
        # flash: white-hot glare + needle rays + coloured bloom, then a cloud glow that sinks with the shell
        if 0.0 <= tau < 2.2:
            fa = (1.0 - min(tau / 0.26, 1.0)) ** 2.0
            fr = (1.0 - min(tau / 0.17, 1.0)) ** 1.5
            fb = math.exp(-tau / 0.16) * (1.0 - _sstep(0.5, 0.9, tau))
            buf.sprites(st.b_flash, cam, np.repeat(b["O"][None, :], 2, 0),
                        np.array([b["r"] * (0.16 + 2.4 * tau) * (1.0 if fa > 0 else 0.0), b["r"] * (0.8 + 2.4 * min(tau, 0.5))]),
                        np.stack([(0.2 * colr + 0.8 * WHITE * colr.max()) * FLASH_GAIN * fa,
                                  colr * 2.6 * fb]), soft=1.0, sub=(2 * bi, 2))
            if fr > 0.0:
                grow = 1.0 - math.exp(-tau / 0.05)
                ln = b["r"] * b["ray_len"] * grow                       # stage metres
                u = np.linspace(0.0, 1.0, S_RAY)[:, None]
                rx = b["x"] + np.cos(b["ray_ang"])[None, :] * ln[None, :] * u
                rz = b["z"] + np.sin(b["ray_ang"])[None, :] * ln[None, :] * u
                rp = cam.world(rx, rz, b["y"])
                buf.ribbons(st.b_ray, cam, rp, np.full(FLASH_RAYS, 0.085 * scale_b),
                            np.repeat(((0.35 * colr + 0.65 * WHITE * colr.max()) * FLASH_GAIN * 0.6 * fr)[None, :], FLASH_RAYS, 0),
                            taper=0.9, power=1.3, sub=(bi * FLASH_RAYS, FLASH_RAYS))
            f = _drag(tau)
            cpos = b["O"].copy()
            cpos[2] -= (STAR_G * b["R"] / KD) * (tau - f) * 0.8
            ca = _sstep(0.0, 0.12, tau) * math.exp(-tau / 0.55) * (1.0 - _sstep(1.2, 2.0, tau))
            buf.sprites(st.b_cloud, cam, cpos[None, :], np.array([b["r"] * (0.5 + 1.15 * KD * f)]),
                        (colr * CLOUD_GAIN * ca)[None, :], soft=1.0, sub=(bi, 1))
        # lights
        if tau < 0:
            pl, _ = _rocket_pos(b, np.array([t]))
            env = ROCKET_LIGHT * float(_sstep(0.05, 0.5, u[0]))
            lpos = pl[0]
        else:
            env = float(_sstep(0.0, 0.035, tau)) * (0.62 * math.exp(-tau / 0.13) + 0.38 * math.exp(-tau / 0.55)) \
                * (1.0 - float(_sstep(1.1, 1.9, tau)))
            lpos = b["O"]
        m = 0 if b["main"] else 1
        lc = colr / colr.max()
        st.local[bi].location = lpos
        st.local[bi].data.color = lc
        st.local[bi].data.energy = LIGHT_LOCAL[m] * env
        flash = env if tau >= 0 else 0.0
        tgt = (b["light_x"], 0.0, 0.5)
        _aim(st.back[bi], (b["O"][0], BACK_POS[0], BACK_POS[1]), tgt)
        _aim(st.front[bi], (b["light_x"], FRONT_POS[0], FRONT_POS[1]), tgt)
        for ob, e in ((st.back[bi], LIGHT_BACK[m]), (st.front[bi], LIGHT_FRONT[m])):
            ob.data.color = lc
            ob.data.energy = e * flash

    # faint white light while the word is on
    txt = float(_sstep(c.g0 + 0.5 * (c.g1 - c.g0), c.h0, t) * (1.0 - _sstep(c.d0, c.d0 + 0.6 * (c.d1 - c.d0), t)))
    _aim(st.text_light, (0.0, 6.0, 14.0), (0.0, 0.0, 0.6))
    st.text_light.data.color = (0.93, 0.96, 1.0)
    st.text_light.data.energy = LIGHT_TEXT * txt

    # ------------------------------------------------------------------ rocket sparkle trails
    Rk = c.Rk
    age = t - Rk["te"]
    live = (age >= 0.0) & (age < Rk["life"])
    if live.any():
        pos = np.zeros((len(age), 3))
        colr = np.zeros((len(age), 3))
        for bi, b in enumerate(c.bursts):
            m = Rk["bi"] == bi
            p0, _ = _rocket_pos(b, Rk["te"][m])
            pos[m] = p0
            white = Rk["white"][m][:, None]
            colr[m] = (b["color"] * 0.55 + 0.45 * b["color"].max()) * (1 - white) + white * np.array([1.0, 0.9, 0.72]) * b["color"].max()
        a = np.maximum(age, 0.0)
        pos += Rk["jit"] + Rk["vel"] * a[:, None]
        pos[:, 2] -= 0.5 * 1.6 * a * a
        u = np.clip(a / Rk["life"], 0.0, 1.0)
        br = (1.0 - u) ** 1.6 * (0.55 + 0.45 * np.sin(38.0 * t + Rk["ph"])) * live
        xs, zs, _k = cam.to_stage(pos)
        br *= _face_fade(xs, zs, near=0.6, far=1.1)
        buf.sprites(st.b_rspark, cam, pos, Rk["size"] * (1.0 - 0.5 * u) * (br > 0), colr * (STAR_GAIN * 0.8 * br)[:, None])

    # ------------------------------------------------------------------ shell stars
    St = c.St
    tau = t - St["t0"]
    live = (tau >= 0.0) & (tau < St["life"])
    if live.any():
        tt = tau[None, :] - np.arange(S_STAR)[:, None] * (St["trail"] * (1.0 + 1.2 * _sstep(0.2, 1.0, tau)) / (S_STAR - 1))[None, :]
        pos = _ballistic(St["O"], St["V"], St["g"], tt)
        xs, zs, k = cam.to_stage(pos[0])
        cue = np.clip((St["kref"] / k) ** DEPTH_CUE, 0.6, 1.6)
        tq = np.maximum(tau, 0.0)
        a = (0.30 + 0.70 * np.exp(-1.5 * tq)) * (1.0 - _sstep(St["life"] - 0.4, St["life"], tq))
        tw = np.where(tq > 0.4, 0.68 + 0.32 * np.sin(St["fq"] * tq + St["ph"]), 1.0)
        early = 0.13 + 0.87 * _sstep(0.04, 0.62, tq)                    # the flash, not a white blob of stars
        a = a * tw * St["gain"] * cue * _face_fade(xs, zs) * live * early
        hotk = 0.5 * np.exp(-tq / 0.08)[:, None]                        # hotter right after the burst
        rgb = St["col"] * (1 - hotk) + hotk * St["col"].max(axis=1, keepdims=True)
        size = St["size"] * cue * (0.75 + 0.25 * np.exp(-1.2 * tq)) * (0.45 + 0.55 * _sstep(0.0, 0.22, tq)) * (a > 0)
        buf.sprites(st.b_star, cam, pos[0], size, rgb * (STAR_GAIN * a)[:, None])
        ta = a * (0.25 + 0.75 * np.exp(-1.6 * tq))                      # trails are strongest while the shell expands
        buf.ribbons(st.b_star_rib, cam, pos, size * 0.62, rgb * (STAR_GAIN * 0.5 * ta)[:, None])

    # ------------------------------------------------------------------ crackle
    Cr = c.Cr
    tau = t - Cr["t0"]
    d1 = (tau - Cr["pop"]) / 0.045
    d2 = (tau - Cr["pop"] - Cr["gap"]) / 0.04
    d3 = (tau - Cr["pop"] - 2.1 * Cr["gap"]) / 0.035
    a = np.exp(-d1 * d1) + 0.75 * np.exp(-d2 * d2) + 0.5 * np.exp(-d3 * d3)
    live = (a > 0.02) & (tau > 0)
    if live.any():
        pos = _ballistic(Cr["O"], Cr["V"], Cr["g"], tau[None, :])[0]
        xs, zs, k = cam.to_stage(pos)
        a = a * _face_fade(xs, zs) * live
        buf.sprites(st.b_crackle, cam, pos, Cr["size"] * (0.7 + 0.5 * a) * (a > 0), Cr["col"] * (STAR_GAIN * 1.5 * a)[:, None])

    # ------------------------------------------------------------------ letter sparks
    T = c.T
    if t >= c.burst:
        n = len(T["xn"])
        tt = t - np.arange(S_TEXT)[:, None] * np.full((1, n), TEXT_TRAIL / (S_TEXT - 1))
        pos = _text_pos(c, tt)
        xs, zs, k = cam.to_stage(pos[0])
        tq = t - T["t0"]
        gu = np.clip((t - T["ga"]) / (T["gb"] - T["ga"]), 0.0, 1.0)
        e = 1.0 - (1.0 - gu) ** 4 * (1.0 + 4.0 * gu)
        white = _sstep(0.30, 0.97, e)
        cue = np.clip((T["kref"] / k) ** DEPTH_CUE, 0.6, 1.6)
        cue = 1.0 + (cue - 1.0) * (1.0 - e)
        hotk = 0.5 * np.exp(-tq / 0.08)[:, None]
        base = T["col"] * (1 - hotk) + hotk * T["col"].max(axis=1, keepdims=True)
        rgb = base * (1.0 - white)[:, None] + white[:, None] * WHITE[None, :]
        a_star = (0.30 + 0.70 * np.exp(-1.5 * tq)) * STAR_GAIN * cue * (0.13 + 0.87 * _sstep(0.04, 0.62, tq))
        tw = 0.86 + 0.14 * np.sin(T["fq"] * 1.7 * t + T["ph"])
        dg = (t - T["glint_t"]) / 0.07
        dl = (t - T["gb"]) / 0.06                                       # every spark flares as it lands
        glint = 1.0 + T["glint_a"] * np.exp(-dg * dg) + 1.3 * np.exp(-dl * dl)
        a_txt = TEXT_GAIN * T["gain"] * tw * glint
        a = a_star * (1.0 - e) + a_txt * e
        size = T["size0"] * cue * (1.0 - e) + T["size"] * e * (1.0 + 0.12 * (glint - 1.0))
        # crumble: shrink, dim, cool down
        u = np.clip((t - T["da"]) / T["dlife"], 0.0, 1.0)
        gone = u >= 1.0
        falling = _sstep(0.0, 0.12, u)
        size = size * (1.0 - u ** 1.5) ** 0.6
        flick = 1.0 - falling * 0.45 * (0.5 + 0.5 * np.sin(T["fq"] * 5.0 * t + 2.0 * T["ph"]))   # embers glitter
        a = a * (1.0 - u ** 3) ** 1.3 * flick * (1.0 + 0.8 * falling)
        warm = _sstep(0.05, 0.95, u)[:, None]
        rgb = rgb * (1.0 - warm) + warm * WARM[None, :]
        a = np.where(gone, 0.0, a)
        buf.sprites(st.b_text, cam, pos[0], size * (a > 0), rgb * a[:, None])
        # streak while flying / falling (zero length while the word holds)
        buf.ribbons(st.b_text_rib, cam, pos, size * 0.62, rgb * (a * (0.32 + 0.30 * falling))[:, None])
        # soft bloom around the letters
        gi = T["glow_idx"]
        ga = e[gi] ** 3 * (1.0 - u[gi]) ** 1.5 * TEXT_GLOW_GAIN * TEXT_GAIN * (~gone[gi])
        buf.sprites(st.b_text_glow, cam, pos[0][gi], np.full(len(gi), TEXT_GLOW_SIZE) * (ga > 0),
                    rgb[gi] * ga[:, None], soft=1.0)

    _write(st)
