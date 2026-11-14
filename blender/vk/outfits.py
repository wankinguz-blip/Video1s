"""Outfits of the three moles: lead singer, saxophonist, emo dancer.

Every garment is a sculpted solid in rest space skinned by Body.weights (see vk.mole).  Layering
is done with radial offsets from the bare body ("grow", metres):
    shirt / tee +0.003   trousers +0.009   belt +0.014   jacket +0.018   lapels +0.024
Colour areas inside one garment (shirt front inside a jacket, lapels, cuffs' dark insides …) are
painted with smooth per-vertex fields that the shaders threshold (see vk.charmat).
"""
import math

import numpy as np

from . import charmat as M
from . import meshkit as mk
from . import props
from . import rig as R
from . import skin
from .skin import smoothstep

SIDES = R.SIDES
sgn = R.side_sign

G_SHIRT, G_TROUSERS, G_BELT, G_JACKET = 0.003, 0.009, 0.013, 0.015


def extra_bones(variant):
    """(name, parent, rest position, options) — see vk.characters.follow_through."""
    H = np.array((0.0, 0.0, R.HEAD_Z))
    if variant == "emo":
        return [("fringe", "head", tuple(H + (-0.05, -0.27, 0.44)), dict(lag=0.10, dir=(-0.35, 0.0, -1.0), len=0.24, gain=1.3, limit=0.6))]
    return []


def build(b):
    {"singer": singer, "sax": sax, "emo": emo}[b.variant](b)


# ---------------------------------------------------------------- shared pieces
def cap_field(mesh, direction, point, tol=0.012, cos=0.6):
    """Field (boundary 0.5) selecting the flat end cap of a sculpted tube: faces whose normal
    points along `direction` near the plane through `point`."""
    v, f = mesh
    n = skin.vertex_normals(v, f)
    d = np.asarray(direction, dtype=np.float64)
    near = 1 - smoothstep(tol * 0.6, tol, np.abs((v - np.asarray(point)) @ d))
    return np.clip(0.5 + (n @ d - cos) * 4.0, 0, 1) * near


def trousers(b, mat, cuff_z=0.14, waist_z=0.555, name="trousers", leg=(0.097, 0.080, 0.068), paint_extra=None):
    """Pelvis + both legs as one surface.  paint: B = dark inside of the cuffs."""
    body = b.body
    parts = [body.torso_mesh(0.425, waist_z, grow=G_TROUSERS)]
    for s in SIDES:
        parts.append(body.leg_mesh(s, [(0.55, leg[0]), (0.50, leg[0]), (body.knee_z, leg[1]), (cuff_z + 0.06, leg[2] * 1.0), (cuff_z, leg[2])]))
    mesh = b.sculpt(parts, voxel=0.009, smooth=6)
    v, f = mesh
    cap = cap_field(mesh, (0, 0, -1), (0, 0, cuff_z)) * smoothstep(0.15, 0.05, np.abs(np.abs(v[:, 0]) - 0.11))
    paint = [np.zeros(len(v)), np.zeros(len(v)), cap]
    if paint_extra is not None:
        paint_extra(v, paint)
    return b.skinned(name, mesh, mat, paint=paint, arms=False)


def socks(b, mat, top_z=0.21, radius=0.053, name="socks"):
    parts = []
    for s in SIDES:
        x = b.body.hip[s][0]
        parts.append(skin.tube([(x, 0, top_z), (x, 0, 0.12), (x, 0, 0.04)], [radius, radius, radius * 1.04], sides=18, per_seg=3, cap_start="flat",
                               cap_end="flat"))
    return b.skinned(name, skin.merge(*parts), mat, arms=False)


def shoe_mesh(b, side, kind="loafer"):
    """Shoe in rest space + weights (foot / toe, bending at the ball).  Foot space: origin at the
    ankle, toes towards -Y, ground at z = -ANKLE_H."""
    k = sgn(side)
    g = -R.ANKLE_H
    if kind == "sneaker":
        parts = [skin.rounded_box((0, -0.092, g + 0.018), (0.068, 0.172, 0.018), 0.016, 3),             # chunky sole
                 skin.ellipsoid((0, -0.165, g + 0.056), (0.070, 0.096, 0.044), 24, 14),                  # toe box
                 skin.ellipsoid((0, -0.06, g + 0.070), (0.066, 0.105, 0.054), 24, 14),                   # instep
                 skin.ellipsoid((0, 0.012, g + 0.086), (0.064, 0.064, 0.072), 24, 14),                   # heel + collar
                 skin.tube([(0, -0.04, g + 0.105), (0, -0.012, g + 0.150)], [(0.040, 0.022), (0.044, 0.020)], sides=14, per_seg=2)]   # tongue
    else:
        parts = [skin.rounded_box((0, -0.098, g + 0.010), (0.056, 0.160, 0.010), 0.009, 3),              # thin leather sole
                 skin.rounded_box((0, 0.026, g + 0.016), (0.052, 0.042, 0.016), 0.010, 3),               # heel block
                 skin.ellipsoid((0, -0.172, g + 0.043), (0.064, 0.090, 0.036), 24, 14),                  # toe
                 skin.ellipsoid((0, -0.075, g + 0.054), (0.063, 0.110, 0.046), 24, 14),                  # vamp
                 skin.ellipsoid((0, 0.008, g + 0.062), (0.060, 0.064, 0.056), 24, 14)]                   # heel cup
    v, f = b.sculpt(parts, voxel=0.0048, smooth=3)
    v = v * [k, 1, 1]
    if k < 0:
        f = f[:, ::-1]
    local = v.copy()
    v = v + b.J("foot." + side)
    toe = smoothstep(-R.FOOT_BALL + 0.035, -R.FOOT_BALL - 0.035, local[:, 1])
    return (v, np.ascontiguousarray(f)), {"foot." + side: 1 - toe, "toe." + side: toe}, local


def hands(b, mats):
    """Both hands.  mats: {side: (material, grow)} — grow > 0 makes a glove."""
    from . import mole
    for s in SIDES:
        mat, grow = mats[s]
        mesh, w = mole.hand_mesh(b, s, grow)
        b.skinned("hand" + s, mesh, mat, weights=w)


def torso_garment(b, z0, z1, grow, sleeve, shoulder=0.078, hem_flare=0.0, voxel=0.009, smooth=5, collar=None):
    """Torso + sleeves fused into one surface.  sleeve: [(distance from the shoulder, radius), …]
    (the last entry is the open end)."""
    body = b.body
    flare = (lambda z: hem_flare * smoothstep(z0 + 0.10, z0, z)) if hem_flare else None
    parts = [body.torso_mesh(z0, z1, grow=grow, grow_fn=flare)]
    for s in SIDES:
        parts.append(skin.ellipsoid(body.arm_point(s, 0.0) + (0, 0, 0.004), (shoulder, shoulder * 0.95, shoulder), 20, 14))
        parts.append(body.arm_mesh(s, sleeve))
    if collar:
        parts.extend(collar)
    return b.sculpt(parts, voxel=voxel, smooth=smooth)


def sleeve_caps(b, mesh, end):
    """Field selecting the open ends of both sleeves (painted dark like a shadowed opening)."""
    out = 0
    for s in SIDES:
        out = out + cap_field(mesh, b.body.arm_dir[s], b.body.arm_point(s, end), tol=0.014)
    return out


def front_opening(v, body, half_width, z_bottom=0.0, z_top=2.0):
    """Signed distance-like field (metres, + inside) of a jacket's front opening: |x| narrower
    than half_width(z), front side only."""
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    d = np.minimum(half_width(z) - np.abs(x), -y - 0.03)
    return np.minimum(d, np.minimum(z - z_bottom, z_top - z))


def belt(b, z, height, mat_strap, mat_buckle, grow=G_BELT, studs=None, name="belt"):
    """Belt around the waist with a buckle; optional pyramid studs (dict(n, mat))."""
    body = b.body
    th = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    loop = body.torso_point(th, z, grow - 0.004)
    strap = skin.ring(loop, [(0.006, height / 2)] * len(loop), sides=8, ref=(0, 0, 1))
    parts = [(strap, mat_strap)]
    c = body.torso_point(0.0, z, grow + 0.003)
    bk = skin.ring(skin.superellipse_loop(c, height * 0.78, height * 0.62, 4.0, 24), 0.0045, sides=6)
    parts.append((bk, mat_buckle))
    parts.append((skin.tube([c + (-height * 0.3, -0.002, 0), c + (height * 0.75, -0.002, 0)], [0.003, 0.003], sides=6, per_seg=1), mat_buckle))
    if studs:
        for a in np.linspace(0, 2 * np.pi, studs["n"], endpoint=False):
            if abs(math.remainder(a, 2 * np.pi)) < 0.22:
                continue
            p = body.torso_point(a, z, grow + 0.0035)
            n = R.norm(p - np.array((0.0, body.torso_r(z)[2], z)))
            parts.append((_stud(p, n, height * 0.27), studs["mat"]))
    return b.skinned(name, parts, arms=False, legs=False)


def _stud(p, n, r):
    """Small pyramid stud at p pointing along n."""
    t1 = R.norm(np.cross(n, (0, 0, 1)))
    t2 = np.cross(n, t1)
    base = [p + t1 * r, p + t2 * r, p - t1 * r, p - t2 * r]
    v = np.array(base + [p + n * r * 0.9])
    return v, [(0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4), (3, 2, 1, 0)]


# ---------------------------------------------------------------- lead singer
def singer(b):
    body = b.body
    black = M.BLACK_CLOTH
    # trousers (slightly short), white socks, loafers
    m_tr = M.cloth("vk_singer_trousers", [dict(black, rough=0.42), {}, {}, M.SHADOW], weave=900.0)
    trousers(b, m_tr, cuff_z=0.175, leg=(0.095, 0.078, 0.066))
    socks(b, M.cloth("vk_sock_white", [dict(M.WHITE_CLOTH, emit=0.0)], weave=1400.0, sparkle=dict(scale=260.0, density=0.10, strength=5.0)))
    m_shoe = M.leather("vk_loafer", [dict(color=(0.010, 0.010, 0.012), rough=0.22), dict(color=(0.02, 0.02, 0.02), rough=0.5, coat=0.1)])
    for s in SIDES:
        mesh, w, local = shoe_mesh(b, s, "loafer")
        sole = 0.5 + ((-R.ANKLE_H + 0.016) - local[:, 2]) / 0.006
        b.skinned("shoe" + s, mesh, m_shoe, paint=(np.clip(sole, 0, 1),), weights=w)
    belt(b, 0.545, 0.030, M.leather("vk_belt_black", [dict(color=(0.012, 0.012, 0.014), rough=0.3)]), M.metal("vk_silver", M.SILVER, 0.22))

    # jacket over a white tee: one sculpted surface, the tee is the recessed, painted front panel
    sleeve = [(-0.02, 0.074), (0.0, 0.074), (0.12, 0.071), (0.25, 0.067), (0.40, 0.062), (0.458, 0.060)]
    mesh = torso_garment(b, 0.488, 0.972, G_JACKET, sleeve, hem_flare=0.010)
    v, f = mesh
    half = lambda z: 0.066 + 0.016 * smoothstep(0.78, 0.50, z) + 0.035 * smoothstep(0.87, 0.96, z)
    d = front_opening(v, body, half)
    lw = 0.052 * smoothstep(0.60, 0.76, v[:, 2]) * (1 - 0.6 * smoothstep(0.90, 0.97, v[:, 2])) + 0.012
    lap = np.minimum(-d, lw + d)
    relief = -(G_JACKET - G_SHIRT) * smoothstep(-0.004, 0.005, d) + 0.006 * smoothstep(0.0, 0.007, lap)
    mesh = mk.displace(mesh, relief)
    caps = sleeve_caps(b, mesh, 0.458) + cap_field(mesh, (0, 0, -1), (0, 0, 0.488), tol=0.02)
    paint = (np.clip(0.5 + d / 0.010, 0, 1), np.clip(0.5 + lap / 0.008, 0, 1) * (d < 0.002), np.clip(caps, 0, 1))
    m_jacket = M.cloth("vk_singer_jacket", [dict(black), dict(M.WHITE_CLOTH), dict(color=(0.01, 0.01, 0.012), rough=0.22, sheen=0.3, coat=0.8), M.SHADOW],
                       weave=700.0, sparkle=dict(layers=(0,), scale=170.0, density=0.14, strength=10.0, color=(0.85, 0.92, 1.0)))
    b.skinned("jacket", mesh, m_jacket, paint=paint, legs=False)

    # hands: bare right hand (holds the microphone), sparkling white glove on the left
    m_glove = M.cloth("vk_glove", [dict(color=(0.9, 0.9, 0.92), rough=0.35, sheen=0.6)], weave=1200.0, bump=0.15,
                      sparkle=dict(scale=210.0, density=0.34, strength=10.0, size=0.45))
    hands(b, {"L": (m_glove, 0.0022), "R": (M.skin_pad(), 0.0)})
    cuff = body.arm_mesh("L", [(0.425, 0.050), (0.47, 0.051), (0.497, 0.049)], sides=20, cap0="flat", cap1="flat", per_seg=2)
    b.skinned("glove_cuff", cuff, m_glove, at=body.arm_point("L", 0.44))

    # fedora
    hat(b, "fedora")
    props.mic(b)


def hat(b, kind):
    hb = b.hp(R.HEAD["hat_base"]) + (0.0, -0.03, 0.0)      # the crown swallows the top of the skull
    felt = M.cloth("vk_felt_black", [dict(M.BLACK_CLOTH, rough=0.55, sheen=1.0, tint=(0.6, 0.72, 1.0))], bump=0.15)
    band = M.cloth("vk_hat_band", [dict(color=(0.85, 0.85, 0.88), rough=0.28, sheen=0.5, coat=0.3)], weave=1500.0, bump=0.1)
    # crown: lathe, then the classic centre dent and front pinch
    prof = [(0.0, -0.01), (0.243, -0.01), (0.243, 0.0), (0.240, 0.05), (0.232, 0.11), (0.218, 0.16), (0.192, 0.195), (0.14, 0.212), (0.07, 0.216), (0.0, 0.216)]
    cv, cf = skin.lathe(prof, 48, scale=(1.0, 0.99))
    top = smoothstep(0.10, 0.20, cv[:, 2])
    cv[:, 2] -= 0.040 * top * np.exp(-(cv[:, 0] / 0.075) ** 2) * smoothstep(0.20, 0.0, np.abs(cv[:, 1]) - 0.02)
    pinch = top * smoothstep(0.02, -0.16, cv[:, 1]) * np.exp(-((np.abs(cv[:, 0]) - 0.11) / 0.07) ** 2)
    cv[:, 0] -= np.sign(cv[:, 0]) * 0.035 * pinch
    # brim: curled up at the sides, level at the front (the left hand pinches it at brim_front_L)
    r0, r1, th = 0.232, 0.368, 0.0085
    bp = [(r0, 0.0)] + [(r0 + (r1 - r0) * u, 0.0) for u in (0.25, 0.5, 0.75, 0.93)] + [(r1, th * 0.2), (r1 + 0.004, th * 0.5), (r1, th * 0.8)] + \
         [(r0 + (r1 - r0) * u, th) for u in (0.93, 0.75, 0.5, 0.25, 0.0)]
    bv, bf = skin.lathe(bp, 64, closed=True, scale=(1.0, 0.99))
    rr = np.hypot(bv[:, 0], bv[:, 1])
    u = np.clip((rr - r0) / (r1 - r0), 0, 1.05)
    sin2 = (bv[:, 0] / np.maximum(rr, 1e-6)) ** 2
    back = np.clip(bv[:, 1] / np.maximum(rr, 1e-6), 0, 1)
    bv[:, 2] += u ** 1.6 * (0.058 * sin2 + 0.030 * back ** 2 - 0.004)
    bandm = skin.lathe([(0.2445, 0.004), (0.2465, 0.006), (0.2445, 0.040), (0.2405, 0.042), (0.238, 0.040), (0.238, 0.004)], 48, closed=True, scale=(1.0, 0.99))
    parts = [((cv + hb, cf), felt), ((bv + hb, bf), felt), ((bandm[0] + hb, bandm[1]), band)]
    return b.rigid("hat", parts, bone="head")


def bare_hands(b):
    hands(b, {s: (M.skin_pad(), 0.0) for s in SIDES})


def cuffs(b, mat, name, s0=0.425, s1=0.495, radius=0.051, sides=SIDES):
    """Short bands around the wrists (shirt cuffs, wristbands); they ride on the forearm."""
    for s in sides:
        band = b.body.arm_mesh(s, [(s0, radius - 0.001), ((s0 + s1) / 2, radius), (s1, radius - 0.002)], sides=20, cap0="flat", cap1="flat", per_seg=2)
        b.skinned(f"{name}{s}", band, mat, at=b.body.arm_point(s, s0 + 0.01))


# ---------------------------------------------------------------- saxophonist
WINE = dict(color=M.S("#8a1a3c"), rough=0.50, sheen=0.9, tint=(1.0, 0.55, 0.68))
CREAM = dict(color=M.S("#f1e6cb"), rough=0.68, sheen=0.4, tint=(1.0, 1.0, 1.0))
OLD_GOLD = M.S("#d9a23a")


def sax(b):
    """Classic jazz musician: burgundy suit, cream shirt, golden bow tie, pork-pie hat, two-tone
    spectator shoes, golden alto saxophone."""
    body = b.body
    m_tr = M.cloth("vk_sax_trousers", [dict(WINE, rough=0.46), {}, {}, M.SHADOW], weave=820.0)
    trousers(b, m_tr, cuff_z=0.128, leg=(0.097, 0.082, 0.074))
    socks(b, M.cloth("vk_sock_wine", [dict(color=M.S("#3a0a1a"), rough=0.7, sheen=0.5)], weave=1400.0), top_z=0.20)
    # spectator shoes: cream leather with oxblood toe caps and heel counters
    m_shoe = M.leather("vk_spectator", [dict(color=M.S("#efe4cf"), rough=0.26), dict(color=M.S("#571022"), rough=0.22),
                                         dict(color=(0.02, 0.015, 0.012), rough=0.5, coat=0.1)])
    for s in SIDES:
        mesh, w, local = shoe_mesh(b, s, "loafer")
        cap = np.maximum(0.5 + (-0.176 - local[:, 1]) / 0.010, 0.5 + (local[:, 1] - 0.012) / 0.010)
        sole = 0.5 + ((-R.ANKLE_H + 0.016) - local[:, 2]) / 0.006
        b.skinned("shoe" + s, mesh, m_shoe, paint=(np.clip(cap, 0, 1), np.clip(sole, 0, 1)), weights=w)

    # suit jacket buttoned over a cream shirt: V opening from the top button to the collar
    sleeve = [(-0.02, 0.074), (0.0, 0.074), (0.12, 0.071), (0.25, 0.067), (0.40, 0.062), (0.452, 0.060)]
    mesh = torso_garment(b, 0.452, 0.972, G_JACKET, sleeve, hem_flare=0.012)
    v, f = mesh
    half = lambda z: 0.098 * smoothstep(0.655, 0.955, z)
    d = front_opening(v, body, half, z_bottom=0.655)
    lw = 0.050 * smoothstep(0.66, 0.80, v[:, 2]) * (1 - 0.5 * smoothstep(0.90, 0.97, v[:, 2])) + 0.010
    lap = np.minimum(-d, lw + d) * (v[:, 1] < -0.03) * (v[:, 2] > 0.64)
    relief = -(G_JACKET - G_SHIRT) * smoothstep(-0.004, 0.005, d) + 0.006 * smoothstep(0.0, 0.007, lap)
    mesh = mk.displace(mesh, relief)
    caps = sleeve_caps(b, mesh, 0.452) + cap_field(mesh, (0, 0, -1), (0, 0, 0.452), tol=0.02)
    paint = (np.clip(0.5 + d / 0.010, 0, 1), np.clip(0.5 + lap / 0.008, 0, 1) * (d < 0.002), np.clip(caps, 0, 1))
    m_jacket = M.cloth("vk_sax_jacket", [dict(WINE), dict(CREAM), dict(color=M.S("#4d0b1f"), rough=0.24, sheen=0.4, coat=0.6), M.SHADOW],
                       weave=760.0)
    b.skinned("jacket", mesh, m_jacket, paint=paint, legs=False)
    m_gold = M.metal("vk_gold", M.GOLD, 0.20, tint2=M.GOLD_DEEP)
    buttons = [(props._puck(body.torso_point(0.0, z, G_JACKET + 0.002), (0.0, -1.0, 0.05), 0.0125, 0.005, 14, dome=0.002), m_gold) for z in (0.625, 0.560)]
    b.skinned("buttons", buttons, arms=False, legs=False)

    # shirt collar band and cuffs, bow tie
    m_shirt = M.cloth("vk_sax_shirt", [dict(CREAM)], weave=1500.0)
    th = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    collar = skin.ring(body.torso_point(th, 0.968, 0.010), [(0.006, 0.021)] * len(th), sides=8, ref=(0, 0, 1))
    b.skinned("collar", collar, m_shirt, arms=False, legs=False)
    cuffs(b, m_shirt, "cuff", s0=0.440, s1=0.492, radius=0.0515)
    m_bow = M.cloth("vk_bowtie", [dict(color=OLD_GOLD, rough=0.3, sheen=0.5, coat=0.4)], weave=1700.0, bump=0.1)
    c = np.array((0.0, -0.126, 0.930))
    bow = [(skin.rounded_box(c + (0, -0.006, 0), (0.014, 0.012, 0.016), 0.008, 3), m_bow)]
    for k in (1, -1):
        wing = skin.tube([c + (k * 0.008, -0.004, 0), c + (k * 0.040, -0.008, 0.002), c + (k * 0.082, -0.002, 0.004)],
                         [(0.013, 0.008), (0.026, 0.008), (0.034, 0.007)], sides=14, per_seg=4, cap_start="flat", cap_end="round")
        bow.append((wing, m_bow))
    b.skinned("bowtie", bow, at=(0.0, -0.11, 0.93), arms=False, legs=False)

    bare_hands(b)
    porkpie(b)
    props.sax(b)


def porkpie(b):
    """Pork-pie jazz hat: low telescope crown, short brim curled up all round."""
    hb = b.hp(R.HEAD["hat_base"]) + (0.0, -0.03, 0.0)
    felt = M.cloth("vk_felt_wine", [dict(color=M.S("#571226"), rough=0.6, sheen=1.0, tint=(1.0, 0.6, 0.7))], bump=0.15)
    band = M.cloth("vk_hat_band_gold", [dict(color=OLD_GOLD, rough=0.3, sheen=0.5, coat=0.3)], weave=1500.0, bump=0.1)
    prof = [(0.0, -0.01), (0.240, -0.01), (0.240, 0.0), (0.238, 0.06), (0.233, 0.115), (0.226, 0.148), (0.214, 0.163), (0.198, 0.166),
            (0.184, 0.156), (0.172, 0.140), (0.12, 0.136), (0.0, 0.138)]
    cv, cf = skin.lathe(prof, 48, scale=(1.0, 0.97))
    r0, r1, th = 0.232, 0.305, 0.009
    bp = [(r0, 0.0)] + [(r0 + (r1 - r0) * u, 0.0) for u in (0.3, 0.6, 0.9)] + [(r1, th * 0.2), (r1 + 0.004, th * 0.5), (r1, th * 0.8)] + \
         [(r0 + (r1 - r0) * u, th) for u in (0.9, 0.6, 0.3, 0.0)]
    bv, bf = skin.lathe(bp, 64, closed=True, scale=(1.0, 0.97))
    rr = np.hypot(bv[:, 0], bv[:, 1])
    u = np.clip((rr - r0) / (r1 - r0), 0, 1.05)
    bv[:, 2] += 0.030 * u ** 1.8 - 0.004 * u
    bandm = skin.lathe([(0.2420, 0.004), (0.2440, 0.006), (0.2420, 0.036), (0.2385, 0.038), (0.236, 0.036), (0.236, 0.004)], 48, closed=True, scale=(1.0, 0.97))
    parts = [((cv + hb, cf), felt), ((bv + hb, bf), felt), ((bandm[0] + hb, bandm[1]), band)]
    return b.rigid("hat", parts, bone="head")


# ---------------------------------------------------------------- emo dancer
def emo(b):
    """2000s emo: black tee with a pink heart, skinny black jeans, studded belt and wallet chain,
    black wristbands, sneakers with pink laces, long black fringe with a pink streak."""
    body = b.body
    m_jeans = M.cloth("vk_emo_jeans", [dict(color=(0.022, 0.022, 0.030), rough=0.5, sheen=0.6, tint=(0.6, 0.7, 1.0)), {}, {}, M.SHADOW], weave=1100.0)
    trousers(b, m_jeans, cuff_z=0.130, waist_z=0.605, leg=(0.090, 0.069, 0.057))
    m_silver = M.metal("vk_silver", M.SILVER, 0.22)
    belt(b, 0.545, 0.036, M.leather("vk_belt_black", [dict(color=(0.012, 0.012, 0.014), rough=0.3)]), m_silver, studs=dict(n=26, mat=m_silver))
    # wallet chain on the right hip
    u = np.linspace(0.0, 1.0, 22)
    th = -0.62 - 1.75 * u
    chain = body.torso_point(th, 0.530, G_BELT + 0.016)
    chain[:, 2] -= 0.150 * np.sin(np.pi * u) ** 0.75
    b.skinned("chain", skin.tube(chain, [0.0048] * len(chain), sides=6, per_seg=1), m_silver, at=(-0.20, 0.0, 0.50), arms=False, legs=False)

    # sneakers: black canvas, white rubber sole and toe cap, pink laces
    m_shoe = M.leather("vk_sneaker", [dict(color=(0.016, 0.016, 0.020), rough=0.6), dict(color=(0.85, 0.85, 0.83), rough=0.42)], coat=0.15)
    m_lace = M.simple("vk_lace_pink", M.PINK, roughness=0.5, emission=0.25, emission_color=M.PINK)
    g = -R.ANKLE_H
    for s in SIDES:
        mesh, w, local = shoe_mesh(b, s, "sneaker")
        rubber = np.maximum(0.5 + ((g + 0.036) - local[:, 2]) / 0.006, 0.5 + (-0.212 - local[:, 1]) / 0.008)
        b.skinned("shoe" + s, mesh, m_shoe, paint=(np.clip(rubber, 0, 1),), weights=w)
        J = b.J("foot." + s)
        laces = []
        for y in (-0.150, -0.122, -0.094, -0.066):
            pts = []
            for x in (-0.036, -0.018, 0.0, 0.018, 0.036):
                near = np.hypot(local[:, 0] - x, local[:, 1] - y) < 0.011
                pts.append(J + (x, y, float(local[near, 2].max()) + 0.0015))
            laces.append(skin.tube(pts, [0.0048] * len(pts), sides=6, per_seg=2))
        lv, lf = skin.merge(*laces)
        b.skinned("laces" + s, (lv, lf), m_lace, weights={"foot." + s: np.ones(len(lv))})

    # bare furry arms under a short-sleeved tee
    for s in SIDES:
        arm = body.arm_mesh(s, [(0.05, 0.064), (0.25, 0.056), (0.487, 0.046)], sides=20, cap0="round", cap1="flat")
        n = len(arm[0])
        b.skinned("arm" + s, arm, M.fur(), paint=(np.zeros(n), np.zeros(n)), legs=False)
    z0 = 0.578
    sleeve = [(-0.02, 0.078), (0.0, 0.078), (0.09, 0.076), (0.172, 0.075)]
    mesh = torso_garment(b, z0, 0.972, 0.013, sleeve, shoulder=0.081, voxel=0.0075, smooth=5)
    v, f = mesh
    # the print: a pink heart on the chest, violet bands round the sleeve hems
    hs = 0.066
    hu, hw = v[:, 0] / hs, (v[:, 2] - 0.775) / hs + 0.12
    heart = (hu * hu + hw * hw - 1.0) ** 3 - hu * hu * hw ** 3
    heart = np.clip(0.5 - 2.0 * heart, 0, 1) * (v[:, 1] < -0.04)
    stripe = 0
    for s in SIDES:
        sa = (v - body.shoulder[s]) @ body.arm_dir[s]
        dist, _ = skin.seg_dist(v, body.shoulder[s], body.shoulder[s] + body.arm_dir[s] * 0.3)
        stripe = stripe + np.clip(0.5 + (0.016 - np.abs(sa - 0.140)) / 0.006, 0, 1) * (dist < 0.11) * (dist > 0.05)
    caps = sleeve_caps(b, mesh, 0.172) + cap_field(mesh, (0, 0, -1), (0, 0, z0), tol=0.02)
    m_tee = M.cloth("vk_emo_tee", [dict(color=(0.011, 0.011, 0.014), rough=0.62, sheen=0.7, tint=(0.75, 0.6, 1.0)),
                                   dict(color=M.PINK, rough=0.5, sheen=0.3, emit=0.35), dict(color=M.S("#9a4dff"), rough=0.5, sheen=0.3, emit=0.25),
                                   M.SHADOW], weave=1300.0)
    b.skinned("tee", mesh, m_tee, paint=(heart, np.clip(stripe, 0, 1) * (1 - np.clip(caps, 0, 1)), np.clip(caps, 0, 1)), legs=False)

    bare_hands(b)
    m_band = M.cloth("vk_wristband", [dict(color=(0.012, 0.012, 0.015), rough=0.7, sheen=0.8, tint=(0.8, 0.6, 1.0))], weave=900.0, bump=0.4)
    cuffs(b, m_band, "wristband", s0=0.405, s1=0.468, radius=0.054)
    emo_hair(b)


def emo_hair(b):
    """Asymmetric emo cut: a choppy black mop (rigid on the head) and a long fringe swept from a
    parting on the character's left across the right eye; the fringe bends with the bone
    'fringe' (automatic follow-through), one lock is pink."""
    hp = b.hp
    m_hair = M.simple("vk_hair_black", (0.011, 0.011, 0.017), roughness=0.36, sheen=0.8, sheen_tint=(0.55, 0.55, 1.0), coat=0.3)
    m_pink = M.simple("vk_hair_pink", M.PINK, roughness=0.4, sheen=0.5, emission=0.3, emission_color=M.PINK)
    # the mop: a dome set back and up on the skull, so it leaves the face free, with choppy ends
    c, r = np.array((0.0, 0.035, 0.335)), np.array((0.320, 0.274, 0.264))
    parts = [skin.ellipsoid(c, r, 40, 26)]
    for deg in range(70, 291, 20):                                    # sides and back: layered ends hanging to the collar
        a = math.radians(deg)
        d = np.array((math.sin(a), -math.cos(a), 0.0))
        k = 1.0 + 0.10 * math.sin(deg * 0.9)
        base = c + d * r * 0.86 + (0, 0, -0.07)
        tip = c + d * r * (1.06 + 0.05 * math.cos(deg * 0.37)) + (0, 0, -0.275 * k)
        parts.append(skin.spike(base, tip, 0.062, bend=tuple(d * 0.035), sides=10, flat=0.55, ref=tuple(d), tip_radius=0.004))
    for x, y, tx, ty, tz in ((0.10, 0.02, 0.19, 0.10, 0.66), (-0.08, 0.06, -0.16, 0.17, 0.655), (0.0, 0.12, 0.03, 0.27, 0.61),
                             (-0.16, -0.02, -0.27, 0.03, 0.625), (0.19, 0.10, 0.30, 0.19, 0.57)):
        parts.append(skin.spike((x, y, 0.555), (tx, ty, tz), 0.058, bend=(0.0, 0.0, 0.03), sides=10, flat=0.7, tip_radius=0.004))
    for z0, x1, z1 in ((0.50, 0.335, 0.150), (0.44, 0.345, 0.215)):   # two locks in front of the left ear
        parts.append(skin.spike((0.235, -0.06, z0), (x1, -0.105, z1), 0.050, bend=(0.03, -0.02, 0.0), sides=10, flat=0.55, ref=(1.0, 0.0, 0.0),
                                tip_radius=0.004))
    v, f = b.sculpt(parts, voxel=0.0085, smooth=3)
    b.rigid("hair", (v + b.H, f), m_hair, "head")

    # the fringe: locks from the parting (upper left) down across the forehead and the right eye
    locks = [((0.165, -0.115, 0.538), (-0.290, -0.300, 0.150), 0.052, False),
             ((0.135, -0.150, 0.548), (-0.240, -0.336, 0.185), 0.054, False),
             ((0.105, -0.172, 0.553), (-0.180, -0.350, 0.212), 0.050, True),
             ((0.072, -0.188, 0.556), (-0.118, -0.356, 0.250), 0.052, False),
             ((0.040, -0.198, 0.556), (-0.052, -0.356, 0.300), 0.050, False),
             ((0.008, -0.203, 0.553), (0.022, -0.342, 0.388), 0.046, False)]
    black, pink, wb, wp = [], [], [], []
    for base, tip, rad, is_pink in locks:
        base, tip = np.array(base), np.array(tip)
        mid_y = -0.362 - 0.5 * (base[1] + tip[1])                    # arc out in front of the brow and the glasses
        lv, lf = skin.spike(hp(base), hp(tip), rad, bend=(0.0, mid_y, 0.035), sides=12, flat=0.34, tip_radius=0.005, per_seg=6)
        _, along = skin.seg_dist(lv, hp(base), hp(tip))
        w = smoothstep(0.22, 0.95, along)
        (pink if is_pink else black).append((lv, lf))
        (wp if is_pink else wb).append(w)
    for name, meshes, ws, mat in (("fringe", black, wb, m_hair), ("streak", pink, wp, m_pink)):
        w = np.concatenate(ws)
        b.skinned(name, skin.merge(*meshes), mat, weights={"head": 1.0 - w, "fringe": w})
