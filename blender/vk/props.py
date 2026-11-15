"""Hand props, modelled in the local spaces fixed by vk.propspec: microphone and saxophone."""
import numpy as np

from . import charmat as M
from . import propspec
from . import rig as R
from . import skin


def _frame(origin, z_dir, x_hint=(1.0, 0.0, 0.0)):
    """4×4 frame with +Z along z_dir."""
    z = R.norm(np.asarray(z_dir, dtype=np.float64))
    x = np.asarray(x_hint, dtype=np.float64)
    x = R.norm(x - z * np.dot(x, z))
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, np.cross(z, x), z, origin
    return m


def _puck(origin, normal, radius, height, segments=16, bevel=0.3, dome=0.0):
    """Short bevelled cylinder standing on `origin` along `normal` (key cups, buttons, rings)."""
    bv = min(radius, height) * bevel
    prof = [(0.0, 0.0), (radius, 0.0), (radius, height - bv), (radius - bv, height), (radius * 0.5, height + dome * 0.75), (0.0, height + dome)]
    return skin.xform(skin.lathe(prof, segments), _frame(origin, normal))


# ---------------------------------------------------------------- microphone
def mic(b):
    """Stage microphone.  Local space: origin = centre of the grip, +Z towards the head."""
    S = propspec.MIC
    r = S["handle_radius"]
    z0, z1 = S["handle_z"]
    hc, hr = np.array(S["head_center"]), S["head_radius"]
    m_body = M.simple("vk_mic_body", (0.035, 0.036, 0.042), roughness=0.32, metallic=0.85, coat=0.4)
    m_ring = M.metal("vk_silver", M.SILVER, 0.22)
    m_grille = M.metal("vk_mic_grille", (0.80, 0.82, 0.88), 0.34, studio=0.8, bump_scale=420.0)
    handle = skin.lathe([(0.0, z0), (r * 0.80, z0), (r * 0.92, z0 + 0.006), (r * 0.96, z0 + 0.05), (r, 0.0), (r * 1.06, z1 - 0.02),
                         (r * 1.22, z1), (r * 1.22, z1 + 0.004), (0.0, z1 + 0.004)], 24)
    collar = skin.lathe([(r * 1.20, z1 - 0.002), (r * 1.42, z1 + 0.002), (r * 1.46, z1 + 0.010), (r * 1.30, z1 + 0.016), (0.0, z1 + 0.016),
                         (0.0, z1 - 0.002)], 24)
    tail = skin.lathe([(0.0, z0 - 0.004), (r * 0.70, z0 - 0.004), (r * 0.84, z0), (r * 0.84, z0 + 0.008), (0.0, z0 + 0.008)], 24)
    head = skin.ellipsoid(hc, (hr, hr, hr * 0.98), 32, 20)
    band = skin.lathe([(hr * 0.985, -0.005), (hr * 1.05, -0.004), (hr * 1.05, 0.004), (hr * 0.985, 0.005)], 32, center=hc, closed=True)
    o = b.obj("mic", [(handle, m_body), (collar, m_ring), (tail, m_ring), (head, m_grille), (band, m_ring)])
    b.rig.add_prop("mic", [o])
    return o


# ---------------------------------------------------------------- saxophone
def sax_axis():
    """Centre line of the saxophone tube and its radius: control points from the mouthpiece to
    the rim of the bell, in prop space."""
    P = propspec.sax_point
    r0, r1 = propspec.SAX["body_radius"]
    n = R.norm(P("bell_normal"))
    bell = P("bell_center")
    pts = [
        (P("mouthpiece") + (0, 0.004, -0.002), 0.0085),                 # tip of the mouthpiece (goes into the mouth)
        (np.array((0.0, -0.022, 0.006)), 0.0125),
        (np.array((0.0, -0.045, 0.012)), 0.0125),                       # end of the mouthpiece / cork
        (P("neck_top"), 0.0135),
        (np.array((0.0, -0.103, -0.012)), 0.0155),
        (P("body_top"), r0),
        (P("upper_grip"), r0 + (r1 - r0) * 0.33),
        (P("lower_grip"), r0 + (r1 - r0) * 0.74),
        (P("body_bottom"), r1),
        (np.array((0.0, -0.160, -0.553)), r1 * 1.07),
        (P("bow_bottom"), r1 * 1.12),
        (np.array((0.0, -0.243, -0.540)), r1 * 1.20),
        (bell - n * 0.115, r1 * 1.32),
        (bell - n * 0.045, r1 * 1.50),
        (bell - n * 0.012, propspec.SAX["bell_radius"] * 0.90),
        (bell, propspec.SAX["bell_radius"]),
    ]
    return [p for p, _ in pts], [r for _, r in pts]


def sax(b):
    """Alto saxophone.  Local space: origin = tip of the mouthpiece, +Z up, the player on +Y,
    the bell opens forward-up (see vk.propspec.SAX)."""
    P = propspec.sax_point
    gold = M.metal("vk_gold", M.GOLD, 0.20, tint2=M.GOLD_DEEP)
    dark = M.metal("vk_gold_inner", (0.55, 0.30, 0.07), 0.45, studio=0.25)
    ebon = M.simple("vk_ebonite", (0.012, 0.012, 0.014), roughness=0.25, coat=0.6)
    pearl = M.simple("vk_pearl", (0.92, 0.90, 0.84), roughness=0.2, coat=0.8, emission=0.05, emission_color=(1.0, 1.0, 1.0))
    silver = M.metal("vk_silver", M.SILVER, 0.22)
    parts = []
    pts, rad = sax_axis()
    n = R.norm(P("bell_normal"))
    bell, br = P("bell_center"), propspec.SAX["bell_radius"]
    side = np.array((1.0, 0.0, 0.0))

    # main tube: neck → body → bow → bell (open end), mouthpiece separately in black ebonite
    parts.append((skin.tube(pts[2:], rad[2:], sides=28, per_seg=6, cap_start="flat", cap_end=None, ref=(1.0, 0.0, 0.0)), gold))
    parts.append((skin.tube([pts[0], pts[1], pts[2] + (0, -0.004, 0.001)], [(0.0085, 0.0045), (0.0125, 0.0105), (0.0128, 0.0128)], sides=16, per_seg=4,
                            cap_start="round", cap_end="flat", ref=(0.0, 0.0, 1.0)), ebon))
    parts.append((skin.tube([pts[1] + (0, 0.004, -0.001), pts[1] + (0, -0.008, 0.002)], [0.0140, 0.0140], sides=16, per_seg=1, cap_start="flat",
                            cap_end="flat"), silver))                                                    # ligature
    # rolled rim + inside of the bell
    loop = skin.superellipse_loop(bell, br, br, 2.0, 40, axis_x=side, axis_z=np.cross(n, side))
    parts.append((skin.ring(loop, 0.0048, sides=8), gold))
    inner_pts = [bell + n * 0.001, bell - n * 0.012, bell - n * 0.045, bell - n * 0.115]
    iv, i_f = skin.tube(inner_pts, [br - 0.003, br * 0.90 - 0.003, rad[13] - 0.003, rad[12] - 0.004], sides=28, per_seg=3, cap_start=None, cap_end="flat",
                        ref=(1.0, 0.0, 0.0))
    parts.append(((iv, [f[::-1] for f in i_f]), dark))

    # reinforcement rings where the sections join
    for p, d, r in ((P("body_top"), pts[5] - pts[4], rad[5] + 0.0025), (P("body_bottom"), pts[8] - pts[7], rad[8] + 0.003),
                    (pts[12], n, rad[12] + 0.003), (pts[4], pts[5] - pts[3], rad[4] + 0.002)):
        fr = _frame(p, d)
        lp = skin.superellipse_loop(p, r, r, 2.0, 32, axis_x=fr[:3, 0], axis_z=fr[:3, 1])
        parts.append((skin.ring(lp, (0.0032, 0.0055), sides=8, ref=fr[:3, 2]), gold))

    # keys on the front of the body: cups on short posts, pearl touches where the fingers go
    top, bot = P("body_top"), P("body_bottom")
    axis = R.norm(bot - top)
    key = R.norm(np.array(propspec.SAX["key_side"]) - axis * np.dot(propspec.SAX["key_side"], axis))
    length = float(np.linalg.norm(bot - top))
    r0, r1 = propspec.SAX["body_radius"]

    def on_body(u, ang=0.0, lift=0.0):
        c = top + axis * (u * length)
        d = key * np.cos(ang) + side * np.sin(ang)
        return c + d * (r0 + (r1 - r0) * u + lift), d

    for u, cr in ((0.10, 0.0125), (0.19, 0.0135), (0.37, 0.0150), (0.46, 0.0160), (0.60, 0.0175), (0.84, 0.0200), (0.95, 0.0215)):
        p, d = on_body(u, 0.0, 0.004)
        parts.append((_puck(p, d, cr, 0.0055, 18, dome=0.002), gold))
    for u in (0.245, 0.30, 0.355, 0.66, 0.72, 0.78):                                               # pearl touches (3 per hand)
        p, d = on_body(u, 0.0, 0.003)
        parts.append((_puck(p, d, 0.0092, 0.006, 14), gold))
        parts.append((_puck(p + d * 0.0055, d, 0.0075, 0.0022, 14, dome=0.0015), pearl))
    for ang in (0.75, -0.75):                                                                       # long rods with posts
        a, _ = on_body(0.04, ang, 0.008)
        z, _ = on_body(0.97, ang, 0.008)
        parts.append((skin.tube([a, z], [0.0026, 0.0026], sides=6, per_seg=1), gold))
        for u in np.linspace(0.04, 0.97, 7):
            p, d = on_body(u, ang, -0.002)
            parts.append((_puck(p, d, 0.0036, 0.012, 8), gold))
    for u, ang, cr in ((0.30, 1.45, 0.011), (0.42, 1.5, 0.012), (0.56, -1.45, 0.013), (0.90, 1.35, 0.015)):       # side / palm keys
        p, d = on_body(u, ang, 0.003)
        parts.append((_puck(p, d, cr, 0.005, 14, dome=0.002), gold))
    p, d = on_body(0.44, np.pi, 0.0)                                                                # strap ring on the player's side
    parts.append((skin.ring(skin.superellipse_loop(p + d * 0.009, 0.009, 0.009, 2.0, 16, axis_x=d, axis_z=axis), 0.0024, sides=6), gold))
    p, d = on_body(0.63, np.pi, 0.0)                                                                # thumb hook
    parts.append((skin.tube([p, p + d * 0.016, p + d * 0.02 + axis * 0.016], [0.004, 0.004, 0.0035], sides=8, per_seg=3), gold))
    # octave key on the neck
    parts.append((_puck(pts[3] + (0, 0.0, rad[3] + 0.001), (0.0, -0.2, 1.0), 0.007, 0.004, 12), gold))
    parts.append((skin.tube([pts[3] + (0, 0.0, rad[3] + 0.005), pts[4] + (0, -0.004, rad[4] + 0.006), pts[5] + key * (rad[5] + 0.006)],
                            [0.0022] * 3, sides=6, per_seg=4), gold))

    # two big key cups on the bell with guards, brace between the bell and the body
    b_axis = R.norm(pts[13] - pts[11])
    for k, off in ((1, 0.04), (-1, 0.105)):
        c = pts[11] + b_axis * off
        d = R.norm(side * k * 0.9 - key * 0.25)
        rr = rad[11] + (rad[13] - rad[11]) * off / 0.19
        parts.append((_puck(c + d * (rr + 0.002), d, 0.021, 0.006, 18, dome=0.003), gold))
    mid_bell = pts[12] - n * 0.02
    body_pt = top + axis * (np.dot(mid_bell - top, axis))
    parts.append((skin.tube([body_pt, (body_pt + mid_bell) / 2, mid_bell], [(0.005, 0.009)] * 3, sides=8, per_seg=1, ref=(1.0, 0.0, 0.0)), gold))
    o = b.obj("sax", parts)
    b.rig.add_prop("sax", [o])
    return o
