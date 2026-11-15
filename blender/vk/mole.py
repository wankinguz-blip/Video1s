"""The VANKING mole mascot as a real 3D character: body, head, face controls and hands.
Outfits live in vk.outfits, hand props in vk.props.

API used by vk.characters:
    extra_bones(variant)                     -> [(name, parent_bone, rest_pos, options), ...]
    build(ctx, rig, variant)                 -> state
    update(state, rig, pose, frames, t)      -> None      (pure function of pose and t)
variant is 'singer' | 'sax' | 'emo'.

How the character is put together
---------------------------------
* Everything is modelled procedurally in CHARACTER REST SPACE (A-pose of vk.rig.Skeleton).
* Organic volumes (torso + sleeves, pelvis + legs, head + muzzle, palm + fingers) are fused by
  meshkit.sculpt() into single watertight surfaces, so there is no seam where a limb starts.
* ONE analytic weight function, Body.weights(points), skins every layer (fur, shirt, jacket, belt,
  buttons …).  Layers therefore deform identically and cannot slide apart or poke through, and
  limbs stay attached in every pose: shoulders and hips blend smoothly between torso and limb.
* The head is rigid (bone 'head'), the lower jaw follows bone 'jaw'; expressions are shape keys
  (smile, round mouth, puffed cheeks) plus small rigid parts driven in update(): eyelids, pupils
  and eyebrows.
"""
import math
import os
import time

import numpy as np

from . import charmat as M
from . import meshkit as mk
from . import rig as R
from . import skin
from .skin import smoothstep

SIDES = R.SIDES
sgn = R.side_sign

# ---------------------------------------------------------------- body proportions (rest space, metres)
# torso profile: z, half width, half depth, y of the centre line (front is -Y)
TORSO = np.array([
    (0.40, 0.120, 0.115, 0.000),
    (0.46, 0.185, 0.156, 0.000),
    (0.52, 0.205, 0.172, -0.004),
    (0.60, 0.212, 0.186, -0.010),
    (0.70, 0.192, 0.174, -0.008),
    (0.75, 0.172, 0.162, -0.005),
    (0.80, 0.152, 0.150, -0.003),
    (0.88, 0.136, 0.134, 0.000),
    (0.93, 0.120, 0.114, 0.000),
    (0.97, 0.100, 0.095, 0.000),
    (1.08, 0.092, 0.090, 0.000),
])
ARM_R = [(-0.03, 0.066), (0.0, 0.066), (0.25, 0.056), (0.485, 0.046)]       # distance from the shoulder joint → radius
ARM_S, ARM_RADIUS = [r[0] for r in ARM_R], [r[1] for r in ARM_R]
ARMPIT = 0.045
LEG_R = [(0.56, 0.092), (0.52, 0.092), (0.285, 0.074), (0.07, 0.056)]       # height → radius


def _interp(table, x):
    t = np.asarray(table, dtype=np.float64)
    order = np.argsort(t[:, 0])
    return np.interp(x, t[order, 0], t[order, 1])


class Body:
    """Rest-space anatomy shared by all layers: reference surfaces and THE skin weight function."""

    def __init__(self, sk):
        self.sk = sk
        J = sk.joint
        self.zj = {b: float(J(b)[2]) for b in ("hips", "spine", "chest", "neck", "head")}
        self.shoulder = {s: J("upper_arm." + s) for s in SIDES}
        self.wrist = {s: J("hand." + s) for s in SIDES}
        self.arm_dir = {s: R.norm(self.wrist[s] - self.shoulder[s]) for s in SIDES}
        self.hip = {s: J("thigh." + s) for s in SIDES}
        self.knee_z = float(J("shin.L")[2])
        self.ankle_z = float(J("foot.L")[2])

    # -- reference surfaces
    @staticmethod
    def torso_r(z):
        """Half width, half depth and centre-line y of the torso at height z."""
        return np.interp(z, TORSO[:, 0], TORSO[:, 1]), np.interp(z, TORSO[:, 0], TORSO[:, 2]), np.interp(z, TORSO[:, 0], TORSO[:, 3])

    def torso_point(self, theta, z, grow=0.0):
        """Point on the torso surface: theta = 0 is the front centre line, + towards the
        character's left; grow = distance outside the bare body."""
        rx, ry, yc = self.torso_r(z)
        return np.stack([(rx + grow) * np.sin(theta), yc - (ry + grow) * np.cos(theta), np.asarray(z, dtype=np.float64) + 0 * theta], axis=-1)

    def torso_mesh(self, z0, z1, grow=0.0, sides=40, step=0.02, cap0="flat", cap1="flat", grow_fn=None):
        zs = np.linspace(z0, z1, max(2, int(round(abs(z1 - z0) / step)) + 1))
        rx, ry, yc = self.torso_r(zs)
        g = grow + (grow_fn(zs) if grow_fn is not None else 0.0)
        pts = np.stack([np.zeros_like(zs), yc, zs], axis=1)
        return skin.tube(pts, np.stack([rx + g, ry + g], axis=1), sides=sides, per_seg=1, cap_start=cap0, cap_end=cap1)

    def arm_point(self, side, s):
        return self.shoulder[side] + self.arm_dir[side] * s

    def arm_mesh(self, side, radii, sides=20, cap0="round", cap1="flat", per_seg=3):
        """Tube along the arm.  radii: [(distance from the shoulder joint, radius), ...]."""
        pts = [self.arm_point(side, s) for s, _ in radii]
        return skin.tube(pts, [r for _, r in radii], sides=sides, per_seg=per_seg, cap_start=cap0, cap_end=cap1)

    def leg_mesh(self, side, radii, sides=22, cap0="round", cap1="flat", per_seg=3):
        """Tube down the leg.  radii: [(height z, radius or (rx, ry)), ...] from the top down."""
        x = self.hip[side][0]
        pts = [(x, 0.0, z) for z, _ in radii]
        return skin.tube(pts, [r for _, r in radii], sides=sides, per_seg=per_seg, cap_start=cap0, cap_end=cap1)

    # -- skin weights
    def weights(self, p, arms=True, legs=True):
        """Skin weights {bone: (N,)} for rest-space points p (N, 3).

        spine chain: smooth partition by height;  arms: a point belongs to the arm when it lies
        beyond the shoulder joint along the arm AND is nearer to the arm axis than to the torso
        axis (relative to their radii) — this gives a clean armpit and a soft shoulder cap;
        legs: split by an inclined plane through the hip joints (the inguinal line)."""
        p = np.atleast_2d(np.asarray(p, dtype=np.float64))
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        zj = self.zj
        up = [smoothstep(zc - b, zc + b, z) for zc, b in ((zj["spine"], 0.05), (zj["chest"], 0.06), (zj["neck"], 0.035), (zj["head"], 0.03))]
        w = {"hips": 1 - up[0], "spine": up[0] * (1 - up[1]), "chest": up[1] * (1 - up[2]), "neck": up[2] * (1 - up[3]), "head": up[3]}
        limb = {}
        taken = np.zeros(len(p))
        arm_mask = np.zeros(len(p))         # hands hang below the hips: arm points must never get leg weights
        if arms:
            rx, ry, yc = self.torso_r(z)
            dt = np.sqrt((x / rx) ** 2 + ((y - yc) / ry) ** 2)
            for s in SIDES:
                S, d = self.shoulder[s], self.arm_dir[s]
                sa = (p - S) @ d
                dist, _ = skin.seg_dist(p, S, S + d * 0.65)
                # distance outside the torso surface minus distance outside the arm surface: 0 in
                # the armpit crease, where the two surfaces meet; the blend zone is ±ARMPIT wide
                gap = (dt - 1.0) * 0.5 * (rx + ry) - (dist - np.interp(sa, ARM_S, ARM_RADIUS))
                m = smoothstep(-0.07, 0.06, sa) * smoothstep(-ARMPIT, ARMPIT, gap) ** 1.5 * (1 - smoothstep(0.13, 0.20, dist))
                fore = smoothstep(R.UPPER_ARM - 0.05, R.UPPER_ARM + 0.05, sa)
                hand = smoothstep(R.UPPER_ARM + R.FOREARM - 0.012, R.UPPER_ARM + R.FOREARM + 0.028, sa)
                # the shoulder is spread over helper bones (see HELPERS): torso → armA (half of the
                # swing) → armB (full swing, part of the twist) → upper arm
                half = 1 - np.abs(2 * m - 1)
                full = np.clip(2 * m - 1, 0.0, 1.0)
                top = 1 - smoothstep(0.03, 0.19, sa)
                limb["armA." + s] = half
                limb["armB." + s] = full * (1 - fore) * top
                limb["upper_arm." + s] = full * (1 - fore) * (1 - top)
                limb["forearm." + s] = full * fore * (1 - hand)
                limb["hand." + s] = full * hand
                # the shoulder girdle (shrug / push forward) carries the nearby torso
                near = (1 - smoothstep(0.07, 0.23, np.linalg.norm(p - S, axis=1))) * np.clip(1 - 2 * m, 0.0, 1.0)
                limb["shoulder." + s] = near * (x * sgn(s) > 0)
                taken += half + full + limb["shoulder." + s]
                arm_mask = np.maximum(arm_mask, m)
        if legs:
            for s in SIDES:
                H = self.hip[s]
                q = z - 0.5 * (np.abs(x) - abs(H[0])) - 0.25 * np.maximum(y, 0.0)       # buttocks stay with the pelvis longer
                m = (1 - smoothstep(H[2] - 0.075, H[2] + 0.04, q)) * smoothstep(-0.03, 0.03, x * sgn(s)) * (1 - arm_mask)
                shin = 1 - smoothstep(self.knee_z - 0.045, self.knee_z + 0.045, z)
                foot = 1 - smoothstep(self.ankle_z - 0.03, self.ankle_z + 0.03, z)
                limb["thigh." + s] = m * (1 - shin)
                limb["shin." + s] = m * shin * (1 - foot)
                limb["foot." + s] = m * foot
                taken += m
        rest = np.clip(1 - taken, 0.0, 1.0)
        out = {k: v * rest for k, v in w.items()}
        out.update(limb)
        total = sum(out.values())
        return {k: v / total for k, v in out.items() if float(v.max()) > 1e-4}

    def weights_at(self, point, n, **kw):
        """Constant weights (those of one point) for the n vertices of a small rigid accessory."""
        w = self.weights(np.asarray(point, dtype=np.float64)[None, :], **kw)
        return {k: np.full(n, float(v[0])) for k, v in w.items()}


# ---------------------------------------------------------------- builder
def join(parts):
    """[(mesh, material), ...] → (verts, faces list, materials, per-face material index)."""
    verts, faces, mats, idx = [], [], [], []
    off = 0
    for mesh, mat in parts:
        v, f = mesh
        if mat not in mats:
            mats.append(mat)
        mi = mats.index(mat)
        fl = f.tolist() if isinstance(f, np.ndarray) else f
        faces.extend(tuple(int(i) + off for i in face) for face in fl)
        idx.extend([mi] * len(fl))
        verts.append(np.asarray(v, dtype=np.float64))
        off += len(v)
    return np.concatenate(verts), faces, mats, np.array(idx, dtype=np.int32)


class Builder:
    """Per-character build context: creates objects, binds or attaches them, counts triangles."""

    def __init__(self, ctx, rig, variant):
        self.ctx, self.rig, self.sk, self.variant = ctx, rig, rig.sk, variant
        self.draft = bool(getattr(ctx, "draft", False))
        self.J = rig.sk.joint
        self.H = self.J("head")
        self.body = Body(rig.sk)
        self.tris = 0
        self.objects = []
        self.state = dict(variant=variant, name=rig.name)

    def vox(self, size):
        return size * (1.6 if self.draft else 1.0)

    def sculpt(self, parts, voxel=0.008, **kw):
        return mk.sculpt(parts, voxel=self.vox(voxel), **kw)

    def hp(self, p):
        """Head space → rest space."""
        return np.asarray(p, dtype=np.float64) + self.H

    def obj(self, name, mesh, mat=None, paint=None):
        """Create an object.  mesh: (verts, faces) with `mat`, or a list of (mesh, material)."""
        if mat is None:
            v, f, mats, idx = join(mesh)
        else:
            (v, f), mats, idx = mesh, [mat], None
        o = mk.new_object(f"{self.rig.name}_{name}", (v, f), self.rig.col)
        for m in mats:
            o.data.materials.append(m)
        if idx is not None and len(mats) > 1:
            o.data.polygons.foreach_set("material_index", idx)
        if paint is not None:
            mk.set_fields(o.data, "paint", *paint)
        self.tris += mk.tris(f)
        self.objects.append(o)
        return o

    def skinned(self, name, mesh, mat=None, paint=None, weights=None, at=None, **wkw):
        """Object deformed by the armature.  Default weights: Body.weights of every vertex;
        `at` = a rest-space point whose weights are used for the whole (rigid-looking) part."""
        o = self.obj(name, mesh, mat, paint)
        n = len(o.data.vertices)
        if weights is None:
            if at is not None:
                weights = self.body.weights_at(at, n, **wkw)
            else:
                co = np.empty(n * 3, dtype=np.float32)
                o.data.vertices.foreach_get("co", co)
                weights = self.body.weights(co.reshape(n, 3), **wkw)
        self.rig.bind(o, weights)
        return o

    def rigid(self, name, mesh, mat=None, bone="head", paint=None):
        o = self.obj(name, mesh, mat, paint)
        self.rig.attach(o, bone)
        return o


# ---------------------------------------------------------------- head
# all coordinates below are in HEAD space (origin = head pivot); R.HEAD holds the landmarks
HC = np.array(R.HEAD["center"])
HR = np.array(R.HEAD["radii"])
PAD_C, PAD_R = np.array((0.070, -0.220, 0.155)), np.array((0.105, 0.075, 0.068))     # upper-lip pad (mirrored)
JAW_C, JAW_R = np.array((0.0, -0.172, 0.068)), np.array((0.125, 0.108, 0.066))
EYE_R = 0.072
EYE_C = {s: np.array((sgn(s) * 0.115, -0.212, 0.33)) for s in SIDES}
LENS_C = {s: np.array((sgn(s) * 0.118, -0.303, 0.328)) for s in SIDES}
LENS_RX, LENS_RZ, LENS_YAW = 0.097, 0.087, math.radians(12)
LID_OPEN, LID_SHUT = math.radians(40), math.radians(-44)        # elevation of the upper lid's edge


def lip_line(x):
    """Height (head space) of the lower edge of the upper lip as seen from the front."""
    a = (np.abs(x) - PAD_C[0]) / PAD_R[0]
    return PAD_C[2] - PAD_R[2] * np.sqrt(np.clip(1 - a * a, 0.0, 1.0))


def _ell(p, c, r):
    """1 at the centre of an ellipsoid, 0 on its surface, negative outside."""
    return 1.0 - np.linalg.norm((p - c) / r, axis=1)


def _skull(seg=56, rings=36):
    v, f = skin.ellipsoid(HC, HR, seg, rings)
    zn = (v[:, 2] - HC[2]) / HR[2]
    v[:, 0] *= 1.0 - 0.09 * zn - 0.05 * np.clip(zn, 0, 1) ** 2          # wide cheeks, narrower dome
    v[:, 1] = HC[1] + (v[:, 1] - HC[1]) * (1.0 - 0.04 * zn)
    return v, f


def build_head(b):
    """Head, face and all face controls.  Returns the state used by update()."""
    hp = b.hp
    fur = M.fur()
    st = {}

    # --- skull + muzzle + cheek fur, one sculpted surface
    parts = [_skull()]
    for s in (1, -1):
        parts.append(skin.ellipsoid(PAD_C * [s, 1, 1], PAD_R, 28, 18))
        for (z0, z1, x1, r) in ((0.235, 0.215, 0.365, 0.060), (0.150, 0.105, 0.375, 0.064), (0.075, 0.015, 0.335, 0.058)):
            parts.append(skin.spike((s * 0.22, -0.06, z0), (s * x1, -0.075, z1), r, bend=(0.0, 0.0, -0.012), sides=12, tip_radius=0.006))
    parts.append(skin.ellipsoid((0.0, -0.245, 0.205), (0.075, 0.075, 0.06), 20, 14))        # bridge under the nose
    parts.append(skin.ellipsoid((0.0, -0.215, 0.42), (0.21, 0.058, 0.04), 24, 12))          # soft brow ridge
    v, f = b.sculpt(parts, voxel=0.0068, smooth=7)
    tan = np.maximum.reduce([_ell(v, (0.0, -0.265, 0.115), (0.225, 0.20, 0.135)),
                             _ell(v, (0.118, -0.275, 0.335), (0.110, 0.15, 0.105)),
                             _ell(v, (-0.118, -0.275, 0.335), (0.110, 0.15, 0.105))])
    inside = _ell(v, JAW_C, JAW_R) - 0.10
    paint = (np.clip(0.5 + tan * 2.2, 0, 1), np.clip(0.5 + inside * 8.0, 0, 1))
    head = b.obj("head", (v + b.H, f), fur, paint)
    b.rig.attach(head, "head")
    st["head"] = head
    # expression shape keys (deltas in head space)
    mouth = skin.gauss(v, (0.0, -0.27, 0.12), (0.20, 0.10, 0.09))
    smile = np.zeros_like(v)
    for s in (1, -1):
        g = skin.gauss(v, (s * 0.165, -0.235, 0.135), (0.075, 0.09, 0.07))[:, None]
        smile += g * np.array((s * 0.016, 0.010, 0.024))
    rnd = np.zeros_like(v)
    rnd[:, 0] = -v[:, 0] * 0.30 * mouth
    rnd[:, 1] = -0.014 * mouth
    puff = np.zeros_like(v)
    nrm = skin.vertex_normals(v, f)
    for s in (1, -1):
        g = skin.gauss(v, (s * 0.215, -0.17, 0.125), (0.115, 0.14, 0.105))[:, None]
        puff += g * (nrm * 0.050 + np.array((s * 0.012, -0.006, 0.0)))
    st["head_keys"] = _shape_keys(head, v + b.H, dict(smile=smile, round=rnd, cheeks=puff))

    # --- lower jaw (bone 'jaw'): chin + lower lip, dark lining where the closed mouth hides it
    v, f = b.sculpt([skin.ellipsoid(JAW_C, JAW_R, 32, 20)], voxel=0.0068, smooth=3)
    lining = 0.5 + (v[:, 2] - (lip_line(v[:, 0]) - 0.0035)) / 0.012
    jaw = b.obj("jaw", (v + b.H, f), fur, (np.ones(len(v)), np.clip(lining, 0, 1)))
    b.rig.attach(jaw, "jaw")
    smile = np.zeros_like(v)
    edge = smoothstep(0.03, 0.115, np.abs(v[:, 0])) * smoothstep(0.02, 0.10, v[:, 2])
    smile[:, 2] = 0.020 * edge
    smile[:, 0] = np.sign(v[:, 0]) * 0.010 * edge
    rnd = np.zeros_like(v)
    rnd[:, 0] = -v[:, 0] * 0.36
    rnd[:, 1] = -0.012
    st["jaw_keys"] = _shape_keys(jaw, v + b.H, dict(smile=smile, round=rnd))
    tongue = skin.ellipsoid(hp((0.0, -0.165, 0.118)), (0.066, 0.082, 0.026), 20, 12)
    b.rigid("tongue", tongue, M.simple("vk_tongue", M.TONGUE, roughness=0.35, coat=0.3), "jaw")

    # --- static face parts (one object, several materials)
    nose_v, nose_f = skin.ellipsoid((0.0, 0.0, 0.0), (0.064, 0.050, 0.046), 28, 18)
    nose_v[:, 0] *= 1.0 + 0.22 * nose_v[:, 2] / 0.046           # wider on top: rounded triangle
    nose = (nose_v + hp((0.0, -0.313, 0.207)), nose_f)
    m_nose = M.simple("vk_nose", M.NOSE, roughness=0.32, coat=0.55, sheen=0.3)
    m_dark = M.simple("vk_nostril", (0.03, 0.012, 0.01), roughness=0.5)
    m_tooth = M.simple("vk_tooth", (0.93, 0.92, 0.86), roughness=0.25, coat=0.5, emission=0.06, emission_color=(1.0, 1.0, 0.95))
    m_whisker = M.simple("vk_whisker", (0.05, 0.03, 0.02), roughness=0.4)
    parts = [(nose, m_nose)]
    for s in (1, -1):
        parts.append((skin.ellipsoid(hp((s * 0.024, -0.350, 0.190)), (0.012, 0.010, 0.009), 10, 6), m_dark))
        parts.append((skin.rounded_box(hp((s * 0.0225, -0.287, 0.078)), (0.0205, 0.009, 0.034), 0.0075, 3), m_tooth))
        for (dz, tip, droop) in ((0.012, (0.385, -0.285, 0.215), -0.012), (-0.008, (0.40, -0.270, 0.135), -0.02), (-0.03, (0.365, -0.265, 0.062), -0.022)):
            base = (s * 0.150, -0.262, 0.150 + dz)
            parts.append((skin.spike(hp(base), hp((s * tip[0], tip[1], tip[2])), 0.0038, bend=(0.0, -0.01, -droop), sides=6,
                                     tip_radius=0.0012, per_seg=3), m_whisker))
    b.rigid("face", parts, bone="head")

    # --- eyes: fixed white ball, pupil / lids rotate about the eye centre (driven in update)
    m_white, m_pupil = M.eye_white(), M.pupil()
    m_lid = M.simple("vk_lid", M.LID, roughness=0.55, sheen=0.5)
    m_lash = M.simple("vk_lash", (0.02, 0.012, 0.01), roughness=0.5)
    eyes = []
    for s in SIDES:
        c = hp(EYE_C[s])
        eyes.append((skin.ellipsoid(c, (EYE_R,) * 3, 28, 18), m_white))
    b.rigid("eyes", eyes, bone="head")
    st["eye"] = {}
    for s in SIDES:
        c = hp(EYE_C[s])
        pv, pf, hl = _pupil_cap(EYE_R + 0.0012, 0.50)
        pupil = b.obj("pupil" + s, (pv + c, pf), m_pupil, (hl,))
        lid = b.obj("lid" + s, [(_lid_shell(EYE_R + 0.0042, 0.0, 2.35, c), m_lid), (_lid_edge(EYE_R + 0.0046, c, 0.0042), m_lash)])
        low = b.obj("lowlid" + s, [(_lid_shell(EYE_R + 0.0036, -2.3, 0.0, c), m_lid)])
        st["eye"][s] = dict(center=c, pupil=pupil, lid=lid, low=low)

    # --- eyebrows (driven in update)
    m_brow = M.simple("vk_brow", M.BROW, roughness=0.6, sheen=0.6)
    st["brow"] = {}
    for s in SIDES:
        k = sgn(s)
        pts = [hp((k * x, y, z)) for x, y, z in ((0.040, -0.266, 0.418), (0.085, -0.268, 0.442), (0.140, -0.258, 0.448), (0.198, -0.232, 0.428))]
        mesh = skin.tube(pts, [(0.011, 0.012), (0.014, 0.016), (0.013, 0.015), (0.006, 0.008)], sides=10, per_seg=5, ref=(0.0, 0.0, 1.0))
        st["brow"][s] = dict(obj=b.obj("brow" + s, mesh, m_brow), pivot=hp((k * 0.12, -0.26, 0.44)))

    # --- glasses: the brand's big dark-green frames
    m_frame = M.simple("vk_glasses", M.GLASSES, roughness=0.22, coat=1.0, spec=0.6)
    frame, lenses = [], []
    for s in SIDES:
        k = sgn(s)
        ax = (math.cos(LENS_YAW), k * math.sin(LENS_YAW), 0.0)
        c = hp(LENS_C[s])
        loop = skin.superellipse_loop(c, LENS_RX, LENS_RZ, 3.2, 56, axis_x=ax)
        frame.append(skin.ring(loop, 0.0105, sides=10))
        rings = [c + (loop - c) * u for u in (1.0, 0.75, 0.5, 0.25)]
        lv = np.concatenate(rings + [c[None, :]])
        n = len(loop)
        lf = [(r * n + i, r * n + (i + 1) % n, (r + 1) * n + (i + 1) % n, (r + 1) * n + i) for r in range(3) for i in range(n)]
        lf += [(3 * n + i, 3 * n + (i + 1) % n, 4 * n) for i in range(n)]
        lenses.append((lv, lf))
        outer = c + np.array(ax) * k * (LENS_RX + 0.004)
        frame.append(skin.tube([outer, outer + (k * 0.035, 0.07, 0.004), hp((k * 0.292, -0.05, 0.335)), hp((k * 0.300, 0.03, 0.31))],
                               [(0.008, 0.011), (0.006, 0.010), (0.005, 0.008), (0.005, 0.007)], sides=8, per_seg=4, ref=(0.0, 0.0, 1.0)))
    inner = [hp(LENS_C[s]) - np.array((sgn(s) * math.cos(LENS_YAW), math.sin(LENS_YAW), 0.0)) * (LENS_RX - 0.002) + (0, 0, 0.012) for s in SIDES]
    frame.append(skin.tube([inner[0], (inner[0] + inner[1]) / 2 + (0, -0.004, 0.012), inner[1]], [0.009, 0.008, 0.009], sides=8, per_seg=4))
    b.rigid("glasses", skin.merge(*frame), m_frame, "head")
    lens = skin.merge(*lenses)
    lx = np.abs(lens[0][:, 0]) - LENS_C["L"][0]
    d = np.where(lens[0][:, 0] > 0, lx, -lx) + (lens[0][:, 2] - b.H[2] - LENS_C["L"][2])        # diagonal coordinate inside each lens
    glint = np.maximum(1 - np.abs(d - 0.075) / 0.011, 1 - np.abs(d - 0.105) / 0.005)
    b.rigid("lenses", lens, M.lens(), "head", paint=(np.clip(0.5 + glint, 0, 1),))
    return st


def _shape_keys(obj, base, deltas):
    obj.shape_key_add(name="Basis")
    keys = {}
    for name, d in deltas.items():
        kb = obj.shape_key_add(name=name)
        kb.slider_min, kb.slider_max = -1.0, 1.5
        kb.data.foreach_set("co", (base + d).astype(np.float32).ravel())
        keys[name] = kb
    return keys


def _pupil_cap(radius, ang, rings=6, seg=24):
    """Spherical cap facing -Y (the pupil), with the catch-light field for the pupil shader."""
    verts = [np.array([[0.0, -radius, 0.0]])]
    for i in range(1, rings + 1):
        a = ang * i / rings
        ph = np.linspace(0, 2 * np.pi, seg, endpoint=False)
        verts.append(np.stack([np.sin(a) * np.cos(ph), -np.full_like(ph, np.cos(a)), np.sin(a) * np.sin(ph)], axis=1) * radius)
    v = np.concatenate(verts)
    faces = [(0, 1 + (k + 1) % seg, 1 + k) for k in range(seg)]
    for i in range(rings - 1):
        a0, b0 = 1 + i * seg, 1 + (i + 1) * seg
        faces += [(a0 + k, a0 + (k + 1) % seg, b0 + (k + 1) % seg, b0 + k) for k in range(seg)]
    hl = 0.5 + (1.0 - np.linalg.norm(v - np.array((-0.011, -radius, 0.012)), axis=1) / 0.0085)
    return v, faces, np.clip(hl, 0, 1)


def _lid_shell(radius, a0, a1, center, nb=20, na=14):
    """Part of a sphere swept about the eye's X axis from elevation a0 to a1 (radians; 0 = the
    horizontal line through the front of the eye, + = up and over the top)."""
    beta = np.linspace(-1.45, 1.45, nb)
    alpha = np.linspace(a0, a1, na)
    A, Bt = np.meshgrid(alpha, beta, indexing="ij")
    v = np.stack([np.sin(Bt), -np.cos(Bt) * np.cos(A), np.cos(Bt) * np.sin(A)], axis=-1).reshape(-1, 3) * radius + center
    f = [(i * nb + j, i * nb + j + 1, (i + 1) * nb + j + 1, (i + 1) * nb + j) for i in range(na - 1) for j in range(nb - 1)]
    return v, f


def _lid_edge(radius, center, thick):
    beta = np.linspace(-1.45, 1.45, 20)
    pts = np.stack([np.sin(beta), -np.cos(beta), np.zeros_like(beta)], axis=1) * radius + center
    return skin.tube(pts, [thick * 0.5] + [thick] * 18 + [thick * 0.5], sides=6, per_seg=1, ref=(0.0, 0.0, 1.0))


# ---------------------------------------------------------------- hands
FINGER_R = 0.0122


def hand_mesh(b, side, grow=0.0):
    """Palm + thumb + three fingers as one sculpted surface with weights for the finger bones.
    Returns (mesh in rest space, weights)."""
    sk = b.sk
    k = sgn(side)
    F = sk.rest["hand." + side]
    r = FINGER_R + grow
    # palm (hand space: +Y to the finger tips, thumb on -Z, the palm faces -X on the left hand)
    palm = skin.rounded_box((k * 0.004, 0.041, 0.001), (0.0185 + grow, 0.043 + grow, 0.046 + grow), 0.0175 + grow, 4)
    mound = skin.ellipsoid((-k * 0.002, 0.030, -0.036), (0.019 + grow, 0.030, 0.020 + grow), 16, 10)
    wrist = skin.tube([(0.0, -0.045, 0.0), (0.0, 0.0, 0.0), (0.0, 0.03, 0.0)], [(0.030 + grow, 0.036 + grow)] * 2 + [(0.022, 0.036)], sides=18,
                      per_seg=2, ref=(0.0, 0.0, 1.0))
    parts = [skin.xform(m, F) for m in (palm, mound, wrist)]
    chains = []
    for i, (fname, _base, seg) in enumerate(R.FINGERS):
        fa, fb = sk.rest[f"{fname}a.{side}"], sk.rest[f"{fname}b.{side}"]
        pa, pb, ya, yb = fa[:3, 3], fb[:3, 3], fa[:3, 1], fb[:3, 1]
        tip = pb + yb * (seg - 0.002)
        rr = r * (1.12 if i == 0 else 1.0)
        parts.append(skin.tube([pa - ya * 0.012, pa, pb, tip], [rr * 1.04, rr * 1.04, rr, rr * 0.93], sides=14, per_seg=3))
        chains.append((f"{fname}a.{side}", f"{fname}b.{side}", pa, pb, tip, ya))
    v, f = b.sculpt(parts, voxel=0.0026, smooth=3, inflate=0.0006)
    # weights: soft nearest-finger selection, then hand → first → second phalanx along the finger
    dist = np.stack([np.minimum(skin.seg_dist(v, pa, pb)[0], skin.seg_dist(v, pb, tip)[0]) for (_a, _b, pa, pb, tip, _y) in chains], axis=1)
    sel = np.exp(-(dist - dist.min(axis=1, keepdims=True)) / 0.0025)
    sel /= sel.sum(axis=1, keepdims=True)
    w = {"hand." + side: np.zeros(len(v))}
    for j, (na, nb, pa, pb, tip, ya) in enumerate(chains):
        along = (v - pa) @ ya
        on = smoothstep(-0.009, 0.006, along) * (1 - smoothstep(r * 1.5, r * 2.3, dist[:, j]))
        second = smoothstep(-0.008, 0.008, (v - pb) @ R.norm(tip - pb))
        w[na] = sel[:, j] * on * (1 - second)
        w[nb] = sel[:, j] * on * second
    w["hand." + side] = np.clip(1 - sum(w[n] for n in w if n != "hand." + side), 0, 1)
    return (v, f), w


# ---------------------------------------------------------------- character assembly
# Helper bones of the shoulders, driven in update(): they carry part of the upper arm's rotation
# so that a raised or twisted arm bends the sleeve over a wide zone instead of creasing it.
_CHEST = np.array((0.0, 0.0, R.HIP_Z + R.SPINE_DZ + R.CHEST_DZ))
HELPERS = [(f"{n}.{s}", "chest", tuple(_CHEST + R.shoulder_rest(s)), {}) for s in SIDES for n in ("armA", "armB")]


def extra_bones(variant):
    from . import outfits
    return HELPERS + outfits.extra_bones(variant)


def _quat(m):
    """Rotation matrix → quaternion (w, x, y, z)."""
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0:
        r = math.sqrt(1.0 + t) * 2
        q = (0.25 * r, (m[2, 1] - m[1, 2]) / r, (m[0, 2] - m[2, 0]) / r, (m[1, 0] - m[0, 1]) / r)
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        r = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = ((m[2, 1] - m[1, 2]) / r, 0.25 * r, (m[0, 1] + m[1, 0]) / r, (m[0, 2] + m[2, 0]) / r)
    elif m[1, 1] > m[2, 2]:
        r = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = ((m[0, 2] - m[2, 0]) / r, (m[0, 1] + m[1, 0]) / r, 0.25 * r, (m[1, 2] + m[2, 1]) / r)
    else:
        r = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = ((m[1, 0] - m[0, 1]) / r, (m[0, 2] + m[2, 0]) / r, (m[1, 2] + m[2, 1]) / r, 0.25 * r)
    return np.array(q)


def _half_rotation(m):
    """4×4 rotation by half the angle of rotation matrix m (same axis)."""
    q = _quat(m)
    if q[0] < 0:
        q = -q
    q = q + (1.0, 0.0, 0.0, 0.0)
    w, x, y, z = q / np.linalg.norm(q)
    out = np.eye(4)
    out[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                   [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                   [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    return out


def shoulder_helpers(sk, frames, side):
    """Skin matrices of the helper bones armA / armB of one side.

    The upper arm's rotation relative to the shoulder girdle is split into a swing (measured
    from the T-pose direction, so its singular direction — the arm pointing through the chest —
    is never reached) and a twist about the arm.  armB takes the whole swing and 60 % of the
    twist, armA half of that."""
    k = sgn(side)
    S = sk.joint("upper_arm." + side)
    a_s = frames["shoulder." + side] @ sk.rest_inv["shoulder." + side]
    a_u = frames["upper_arm." + side] @ sk.rest_inv["upper_arm." + side]
    q = (R.inv(a_s) @ a_u)[:3, :3]
    d_rest = R.norm(sk.joint("forearm." + side) - S)
    d_t = np.array((k, 0.0, 0.0))
    r0 = R.rot_between(d_rest, d_t)[:3, :3]
    swing = R.rot_between(d_t, q @ d_rest)[:3, :3]
    tw = swing.T @ q @ r0.T                              # rotation about the X axis
    ang = math.atan2(tw[2, 1], tw[1, 1])
    mb = np.eye(4)
    mb[:3, :3] = swing @ R.Rx(0.6 * ang)[:3, :3] @ r0
    ma = _half_rotation(mb[:3, :3])
    about = lambda m: a_s @ R.T(S) @ m @ R.T(-S)
    return about(ma), about(mb)


def build(ctx, rig, variant):
    from . import outfits
    t0 = time.perf_counter()
    b = Builder(ctx, rig, variant)
    b.state["face"] = build_head(b)
    # neck: reaches into the skull and into the collar, weighted chest → neck → head
    neck = b.body.torso_mesh(0.86, 1.07, grow=0.0, sides=24)
    v, f = neck
    b.skinned("neck", neck, M.fur(), paint=(np.zeros(len(v)), np.zeros(len(v))), arms=False, legs=False)
    outfits.build(b)
    b.state["tris"] = b.tris
    b.state["sk"] = rig.sk
    b.state["build_seconds"] = time.perf_counter() - t0
    if os.environ.get("VK_PROFILE"):
        print(f"[mole] {variant}: built in {b.state['build_seconds']:.2f} s, {b.tris} triangles, {len(b.objects)} objects")
    return b.state


def _about(pivot, rot):
    return R.T(pivot) @ rot @ R.T(-np.asarray(pivot))


def update(state, rig, pose, frames, t):
    from mathutils import Matrix
    sk = rig.sk
    face = state["face"]
    A = frames["head"] @ sk.rest_inv["head"]
    smile = (float(pose.smile) - 0.5) * 2.0
    rnd = float(np.clip(pose.mouth_round, 0.0, 1.0))
    hk, jk = face["head_keys"], face["jaw_keys"]
    hk["smile"].value = smile * (1.0 - 0.6 * rnd)
    hk["round"].value = rnd
    hk["cheeks"].value = float(np.clip(pose.cheeks, 0.0, 1.2))
    jk["smile"].value = smile * (1.0 - 0.6 * rnd)
    jk["round"].value = rnd
    look = pose.look
    for s in SIDES:
        e = face["eye"][s]
        k = sgn(s)
        blink = float(np.clip(getattr(pose, "blink_" + s), 0.0, 1.0))
        brow = float(np.clip(getattr(pose, "brow_" + s), -1.0, 1.0))
        # pupils: slightly converged at rest, clamped so they never leave the visible eye
        yaw = float(np.clip(look[0], -1, 1)) * 0.34 - k * 0.07
        pitch = -float(np.clip(look[1], -1, 1)) * 0.26 + 0.04
        e["pupil"].matrix_world = Matrix((A @ _about(e["center"], R.Rz(yaw) @ R.Rx(pitch))).tolist())
        open_ang = LID_OPEN + max(brow, 0.0) * 0.16 - max(-brow, 0.0) * 0.10
        ang = open_ang + (LID_SHUT - open_ang) * blink
        e["lid"].matrix_world = Matrix((A @ _about(e["center"], R.Rx(-ang))).tolist())
        low = math.radians(-37) + 0.14 * max(smile, 0.0) + 0.10 * blink + 0.12 * float(pose.cheeks)
        e["low"].matrix_world = Matrix((A @ _about(e["center"], R.Rx(-low))).tolist())
        br = face["brow"][s]
        dz = 0.024 * brow if brow > 0 else 0.016 * brow
        roll = k * (0.10 * max(brow, 0.0) + 0.38 * min(brow, 0.0))
        br["obj"].matrix_world = Matrix((A @ R.T((0.0, 0.0, dz)) @ _about(br["pivot"], R.Ry(roll))).tolist())
    # shoulder helper bones
    for s in SIDES:
        for name, m in zip(("armA." + s, "armB." + s), shoulder_helpers(sk, frames, s)):
            rig.pbones[name].matrix_basis = Matrix((rig.rest_local_inv[name] @ m @ rig.rest_local[name]).tolist())
    hook = state.get("update")
    if hook is not None:
        hook(state, rig, pose, frames, t)
