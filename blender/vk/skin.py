"""Mesh + skin-weight helpers for building characters procedurally (numpy, no bpy)."""
import numpy as np


def catmull(points, per_seg=6):
    """Resample a polyline with a centripetal-ish Catmull-Rom spline.  Returns (M, 3)."""
    p = np.asarray(points, dtype=np.float64)
    if len(p) < 3 or per_seg <= 1:
        return p
    out = []
    n = len(p)
    for i in range(n - 1):
        p0, p1, p2, p3 = p[max(i - 1, 0)], p[i], p[i + 1], p[min(i + 2, n - 1)]
        for k in range(per_seg):
            u = k / per_seg
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3))
    out.append(p[-1])
    return np.array(out)


def _interp_radii(points, radii, path):
    """Interpolate per-control-point radii (scalar or (rx, rz) pairs) onto a resampled path."""
    p = np.asarray(points, dtype=np.float64)
    r = np.asarray(radii, dtype=np.float64)
    if r.ndim == 1:
        r = np.stack([r, r], axis=1)
    s_ctrl = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    s_path = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    s_path = s_path / s_path[-1] * s_ctrl[-1]
    return np.stack([np.interp(s_path, s_ctrl, r[:, 0]), np.interp(s_path, s_ctrl, r[:, 1])], axis=1)


def tube(points, radii, sides=16, per_seg=5, cap_start="round", cap_end="round", ref=(0.0, -1.0, 0.0)):
    """Sweep an elliptical ring along a smoothed path.

    points: control points (K, 3); radii: K scalars or K (r_side, r_ref) pairs, where r_ref is
    the radius along the `ref` direction (kept as constant as possible along the path) and
    r_side the radius across it.  cap_*: 'round' | 'flat' | None.
    Returns (verts (N,3) float64, faces list of tuples).
    """
    path = catmull(points, per_seg)
    rad = _interp_radii(points, radii, path)
    n = len(path)
    tang = np.gradient(path, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True)
    ref = np.asarray(ref, dtype=np.float64)
    rings = []
    b_prev = None
    for i in range(n):
        t = tang[i]
        b = ref - t * np.dot(ref, t)
        if np.linalg.norm(b) < 1e-4:
            b = b_prev if b_prev is not None else np.cross(t, [1.0, 0.0, 0.0])
        b = b / np.linalg.norm(b)
        b_prev = b
        a = np.cross(t, b)
        ang = np.linspace(0, 2 * np.pi, sides, endpoint=False)
        rings.append(path[i] + np.outer(np.cos(ang), a) * rad[i, 0] + np.outer(np.sin(ang), b) * rad[i, 1])
    verts = [np.concatenate(rings)]
    faces = []
    for i in range(n - 1):
        for k in range(sides):
            k2 = (k + 1) % sides
            faces.append((i * sides + k, i * sides + k2, (i + 1) * sides + k2, (i + 1) * sides + k))
    count = n * sides

    def cap(idx, direction, mode, flip):
        nonlocal count
        base = idx * sides
        if mode is None:
            return
        steps = 4 if mode == "round" else 0
        prev = base
        r0 = rad[idx]
        for s in range(1, steps + 1):
            ang = (s / (steps + 1)) * np.pi / 2
            ring = path[idx] + (rings[idx] - path[idx]) * np.cos(ang) + direction * (np.sin(ang) * float(min(r0)))
            verts.append(ring)
            cur = count
            count += sides
            for k in range(sides):
                k2 = (k + 1) % sides
                f = (prev + k, prev + k2, cur + k2, cur + k)
                faces.append(f[::-1] if flip else f)
            prev = cur
        tip = path[idx] + direction * (float(min(r0)) if mode == "round" else 0.0)
        verts.append(tip[None, :])
        ti = count
        count += 1
        for k in range(sides):
            k2 = (k + 1) % sides
            f = (prev + k, prev + k2, ti)
            faces.append(f[::-1] if flip else f)

    cap(0, -tang[0], cap_start, True)
    cap(n - 1, tang[-1], cap_end, False)
    return np.concatenate(verts), [f[::-1] for f in faces]      # reversed: normals point outwards


def ellipsoid(center, radii, segments=24, rings=16):
    """UV ellipsoid.  Returns (verts, faces)."""
    c = np.asarray(center, dtype=np.float64)
    r = np.asarray(radii, dtype=np.float64)
    verts = [c + r * [0, 0, 1]]
    for i in range(1, rings):
        th = np.pi * i / rings
        ph = np.linspace(0, 2 * np.pi, segments, endpoint=False)
        verts.extend(c + r * np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.full_like(ph, np.cos(th))], axis=1))
    verts.append(c + r * [0, 0, -1])
    verts = np.array(verts)
    faces = []
    for k in range(segments):
        faces.append((0, 1 + k, 1 + (k + 1) % segments))
    for i in range(rings - 2):
        a, b = 1 + i * segments, 1 + (i + 1) * segments
        for k in range(segments):
            k2 = (k + 1) % segments
            faces.append((a + k, b + k, b + k2, a + k2))
    last = len(verts) - 1
    a = 1 + (rings - 2) * segments
    for k in range(segments):
        faces.append((last, a + (k + 1) % segments, a + k))
    return verts, faces


def smoothstep(a, b, x):
    u = np.clip((x - a) / (b - a), 0.0, 1.0)
    return u * u * (3 - 2 * u)


def chain_weights(verts, joints, bones, blend=0.05):
    """Skin weights for vertices around a bone chain.

    joints: K+1 rest joint positions (start of bone 0 … end of bone K-1); bones: K bone names.
    Around each interior joint the weight passes from one bone to the next over ±blend metres
    (measured along the bisecting direction), which gives a smooth elbow / knee.
    Returns {bone: weights (N,)}.
    """
    v = np.asarray(verts, dtype=np.float64)
    j = np.asarray(joints, dtype=np.float64)
    k = len(bones)
    assert len(j) == k + 1
    w_next = []
    for i in range(1, k):
        d0 = j[i] - j[i - 1]
        d1 = j[i + 1] - j[i]
        axis = d0 / np.linalg.norm(d0) + d1 / np.linalg.norm(d1)
        axis /= np.linalg.norm(axis)
        w_next.append(smoothstep(-blend, blend, (v - j[i]) @ axis))
    out = {}
    for i, b in enumerate(bones):
        w = np.ones(len(v))
        if i > 0:
            w = w * w_next[i - 1]
        if i < k - 1:
            w = w * (1 - w_next[i])
        out[b] = out.get(b, 0) + w
    return out


def blend_weights(*parts):
    """Normalise a sum of weight dicts: blend_weights((mask, weights), ...) with per-vertex masks."""
    total = {}
    for mask, w in parts:
        for b, arr in w.items():
            total[b] = total.get(b, 0) + np.asarray(arr) * mask
    s = sum(total.values())
    s = np.where(s > 1e-9, s, 1.0)
    return {b: a / s for b, a in total.items()}


def merge(*meshes):
    """Concatenate (verts, faces) meshes."""
    verts, faces, off = [], [], 0
    for v, f in meshes:
        verts.append(np.asarray(v, dtype=np.float64))
        faces.extend(tuple(i + off for i in face) for face in f)
        off += len(v)
    return np.concatenate(verts), faces


# ---------------------------------------------------------------- more primitives (character modelling)
def xform(mesh, m=None, scale=None, offset=None):
    """Transform a (verts, faces) mesh by a 4×4 matrix and / or a per-axis scale and an offset."""
    v = np.asarray(mesh[0], dtype=np.float64)
    if scale is not None:
        v = v * np.asarray(scale, dtype=np.float64)
    if m is not None:
        m = np.asarray(m, dtype=np.float64)
        v = v @ m[:3, :3].T + m[:3, 3]
    if offset is not None:
        v = v + np.asarray(offset, dtype=np.float64)
    return v, mesh[1]


def mirror_x(mesh):
    """Mirror a mesh in the plane x = 0 (faces are flipped so normals stay outward)."""
    v = np.asarray(mesh[0], dtype=np.float64) * [-1.0, 1.0, 1.0]
    return v, [tuple(f[::-1]) for f in mesh[1]]


def superellipsoid(center, radii, e_ns=1.0, e_ew=1.0, segments=32, rings=20):
    """Ellipsoid with squareness control: exponents 1 = round, < 1 = boxy, > 1 = pinched.
    e_ns shapes the vertical profile, e_ew the horizontal section."""
    v, f = ellipsoid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), segments, rings)

    def sp(x, e):
        return np.sign(x) * np.abs(x) ** e

    z = v[:, 2]
    rxy = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    ang = np.arctan2(v[:, 1], v[:, 0])
    out = np.stack([sp(rxy, e_ns) * sp(np.cos(ang), e_ew), sp(rxy, e_ns) * sp(np.sin(ang), e_ew), sp(z, e_ns)], axis=1)
    return out * np.asarray(radii, dtype=np.float64) + np.asarray(center, dtype=np.float64), f


def lathe(profile, segments=32, center=(0.0, 0.0, 0.0), scale=(1.0, 1.0), closed=False):
    """Revolve a profile [(radius, z), ...] about the Z axis.  A radius of 0 at either end makes
    a pole (closed cap); `closed` joins the last profile point back to the first (ring shapes
    such as hat brims with thickness).  scale = (sx, sy) makes the section elliptical."""
    prof = [(float(r), float(z)) for r, z in profile]
    c = np.asarray(center, dtype=np.float64)
    ang = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    cs, sn = np.cos(ang) * scale[0], np.sin(ang) * scale[1]
    verts, index = [], []
    count = 0
    for r, z in prof:
        if r <= 1e-9:
            verts.append(np.array([[c[0], c[1], c[2] + z]]))
            index.append((count, 1))
            count += 1
        else:
            verts.append(np.stack([c[0] + r * cs, c[1] + r * sn, np.full_like(ang, c[2] + z)], axis=1))
            index.append((count, segments))
            count += segments
    faces = []
    pairs = list(zip(range(len(prof) - 1), range(1, len(prof))))
    if closed:
        pairs.append((len(prof) - 1, 0))
    for i, j in pairs:
        (a, na), (b, nb) = index[i], index[j]
        for k in range(segments):
            k2 = (k + 1) % segments
            if na == 1 and nb == 1:
                continue
            if na == 1:
                faces.append((a, b + k2, b + k))
            elif nb == 1:
                faces.append((a + k, a + k2, b))
            else:
                faces.append((a + k, a + k2, b + k2, b + k))
    # profiles are usually given bottom → top, which makes the faces above point inwards
    v = np.concatenate(verts)
    if _signed_volume(v, faces) < 0:
        faces = [tuple(f[::-1]) for f in faces]
    return v, faces


def _signed_volume(v, faces):
    vol = 0.0
    for f in faces:
        a = v[f[0]]
        for k in range(1, len(f) - 1):
            vol += float(np.dot(a, np.cross(v[f[k]], v[f[k + 1]])))
    return vol


def outward(mesh):
    """Flip the faces of a closed mesh if they point inwards."""
    v, f = mesh
    return (v, [tuple(x[::-1]) for x in f]) if _signed_volume(np.asarray(v), f) < 0 else (v, f)


def ring(points, radii, sides=10, ref=None):
    """Closed tube through a closed loop of points (no smoothing: pass a dense loop).  `ref` is the
    direction of the second radius (default: the normal of the loop's plane).
    radii: scalar, one (r_side, r_ref) pair for the whole loop, per-point scalars, or per-point
    (r_side, r_ref) pairs."""
    p = np.asarray(points, dtype=np.float64)
    n = len(p)
    r = np.asarray(radii, dtype=np.float64)
    if r.ndim == 0:
        r = np.full(n, float(r))
    if r.ndim == 1 and len(r) == 2 and n != 2:
        r = np.tile(r, (n, 1))
    if r.ndim == 1:
        r = np.stack([r, r], axis=1)
    tang = np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True)
    if ref is None:     # normal of the loop's plane (Newell)
        ref = np.cross(p, np.roll(p, -1, axis=0)).sum(axis=0)
        ref = ref / np.linalg.norm(ref)
    ref = np.asarray(ref, dtype=np.float64)
    ang = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    verts = []
    b_prev = None
    for i in range(n):
        b = ref - tang[i] * np.dot(ref, tang[i])
        if np.linalg.norm(b) < 1e-5:
            b = b_prev if b_prev is not None else np.cross(tang[i], [1.0, 0.0, 0.0])
        b /= np.linalg.norm(b)
        b_prev = b
        a = np.cross(tang[i], b)
        verts.append(p[i] + np.outer(np.cos(ang), a) * r[i, 0] + np.outer(np.sin(ang), b) * r[i, 1])
    faces = []
    for i in range(n):
        j = (i + 1) % n
        for k in range(sides):
            k2 = (k + 1) % sides
            faces.append((i * sides + k, i * sides + k2, j * sides + k2, j * sides + k))
    return outward((np.concatenate(verts), faces))


def superellipse_loop(center, rx, rz, exponent=2.0, n=48, axis_x=(1.0, 0.0, 0.0), axis_z=(0.0, 0.0, 1.0)):
    """Closed loop of points on a superellipse (exponent 2 = ellipse, 4 = rounded square)."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    e = 2.0 / exponent
    cx = np.sign(np.cos(t)) * np.abs(np.cos(t)) ** e * rx
    cz = np.sign(np.sin(t)) * np.abs(np.sin(t)) ** e * rz
    return np.asarray(center, dtype=np.float64) + np.outer(cx, axis_x) + np.outer(cz, axis_z)


def disc(loop, center=None):
    """Triangle fan over a closed loop of points (flat lens / cap).  Returns (verts, faces)."""
    p = np.asarray(loop, dtype=np.float64)
    c = p.mean(axis=0) if center is None else np.asarray(center, dtype=np.float64)
    n = len(p)
    return np.concatenate([p, c[None, :]]), [(i, (i + 1) % n, n) for i in range(n)]


def rounded_box(center, half, radius, steps=4):
    """Box with rounded edges and corners (a sphere with its octants pushed apart)."""
    seg = 4 * steps
    v, f = ellipsoid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), seg, 2 * steps)
    h = np.maximum(np.asarray(half, dtype=np.float64) - radius, 0.0)
    out = v * radius + np.sign(np.round(v, 9)) * h
    return out + np.asarray(center, dtype=np.float64), f


def spike(base, tip, radius, bend=(0.0, 0.0, 0.0), sides=10, flat=1.0, ref=(0.0, -1.0, 0.0), tip_radius=0.0015, per_seg=5):
    """Tapered, optionally curved and flattened tuft / lock of hair from base to tip."""
    b, t = np.asarray(base, dtype=np.float64), np.asarray(tip, dtype=np.float64)
    mid = (b + t) / 2 + np.asarray(bend, dtype=np.float64)
    q1, q3 = b + (mid - b) * 0.55 + (t - b) * 0.0, mid + (t - mid) * 0.5 + np.asarray(bend) * 0.25
    pts = [b, q1, mid, q3, t]
    rr = [radius, radius * 0.95, radius * 0.72, radius * 0.42, tip_radius]
    return tube(pts, [(r, r * flat) for r in rr], sides=sides, per_seg=per_seg, ref=ref)


def vertex_normals(verts, faces):
    """Area-weighted vertex normals for a uniform (M, k) face array."""
    v = np.asarray(verts, dtype=np.float64)
    f = np.asarray(faces)
    n = np.zeros_like(v)
    k = f.shape[1]
    fn = np.zeros((len(f), 3))
    for i in range(1, k - 1):
        fn += np.cross(v[f[:, i]] - v[f[:, 0]], v[f[:, i + 1]] - v[f[:, 0]])
    for i in range(k):
        np.add.at(n, f[:, i], fn)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.where(ln > 1e-12, ln, 1.0)


def seg_dist(p, a, b):
    """Distance of points p (N, 3) to segment a-b, and the parameter along it (0 at a … 1 at b,
    unclamped).  Returns (distance to the clamped segment, t)."""
    p = np.asarray(p, dtype=np.float64)
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    ab = b - a
    t = (p - a) @ ab / float(ab @ ab)
    q = a + np.clip(t, 0.0, 1.0)[:, None] * ab
    return np.linalg.norm(p - q, axis=1), t


def gauss(p, center, radius):
    """exp(-|p - c|² / r²) falloff, radius may be per-axis."""
    d = (np.asarray(p, dtype=np.float64) - np.asarray(center, dtype=np.float64)) / np.asarray(radius, dtype=np.float64)
    return np.exp(-np.sum(d * d, axis=1))
