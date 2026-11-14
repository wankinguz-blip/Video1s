"""Environment of the show: moonlit meadow with the three stage hills, layered depth, sky.

Layout is done in *picture* terms wherever possible: `xs`, `zs` are stage-plane coordinates (what
the camera sees: 1 m = config.PPM px, frame = zs in [-1.0, 3.2]) and `y` is depth.  A thing at
depth y that should cover the stage-plane point (xs, zs) sits at world
(xs * k, y, CAM_Z + (zs - CAM_Z) * k) with k = (y + CAM_DIST) / CAM_DIST.

Layers, near → far
  ground        one fan-shaped height-field mesh (uniform resolution on screen) from terrain.height
  grass         numpy-scattered points (visibility-culled, density falls with distance) instanced
                by Geometry Nodes; wind = instance rotation driven by the modifier input "Time"
  details       flowers, clover, mushrooms, molehills, stones (same scatter machinery)
  bushes/trees  rounded "puffy" canopies on the mid-ground swells and far ridges
  mist cards    soft alpha cards between the depth layers
  mountains     two silhouette strips with moonlit snow
  sky           emissive card (gradient), stars (twinkling quads), moon + glow cards
  fireflies     small additive quads on closed Lissajous paths (numpy, update())

No compositor / volumetrics / screen-space effects: every glow is geometry, haze is a per-material
mix towards the horizon colour by camera depth, so the three screen slices match at the seams.
Everything animated is a pure function of loop time (integer harmonics of the 15 s loop).
"""
import math

import bpy
import numpy as np

from . import config as C
from . import terrain, util

TAU = math.tau
D = C.CAM_DIST

# ================================================================ look-dev parameters
SEED = 7

PAL = dict(
    # sky gradient, sRGB hex (display intent; emitted through SKY_GAIN), bottom → top
    sky=[(-0.40, "#5a4aa0"), (0.15, "#5a4aa2"), (0.55, "#4a429c"), (1.05, "#383e96"),
         (1.70, "#28348a"), (2.45, "#1c2b7c"), (3.30, "#15236e")],
    haze="#4c4698",                 # far things fade to this (≈ sky just above the horizon)
    mist="#8088d8",
    grass_dark="#0c3a24", grass_mid="#1c6430", grass_light="#408c3a", grass_tip="#88c25e",
    grass_dry="#8a9a4a",
    soil="#3a2a1e",
)
SKY_GAIN = 0.55                     # scene-linear multiplier of the sky colours
HAZE_GAIN = 0.50
HAZE_START = 68.0                   # camera depth (m) where haze begins (stage = 60)
HAZE_LENGTH = 190.0                 # e-folding distance
HAZE_MAX = 0.93

# The moon hangs over the left hill, in the gap between the two satellite firework bursts of the left
# screen (they occupy about xs -26.3..-23.4 and -18.6..-16.0), clear of the screen seams and of the text.
MOON = dict(xs=-21.3, zs=2.50, radius=0.39, y=2300.0, strength=4.6)
STARS = dict(count=6000, y=2400.0, strength=3.2, sparkles=40)
SKY_Y = 2600.0


# ================================================================ small helpers
def kd(y):
    """Size multiplier at depth y (1 on the stage plane)."""
    return (np.asarray(y, dtype=np.float64) + D) / D


def to_world(xs, zs, y):
    k = kd(y)
    return C.CAM_X + (np.asarray(xs) - C.CAM_X) * k, C.CAM_Z + (np.asarray(zs) - C.CAM_Z) * k


def to_stage(x, y, z):
    k = kd(y)
    return C.CAM_X + (np.asarray(x) - C.CAM_X) / k, C.CAM_Z + (np.asarray(z) - C.CAM_Z) / k


def lin(hexcol, gain=1.0):
    r, g, b = util.srgb(hexcol)
    return (r * gain, g * gain, b * gain)


def rgba(c, a=1.0):
    return (c[0], c[1], c[2], a)


def _hash(ix, iy, seed):
    h = (ix.astype(np.int64) * 374761393 + iy.astype(np.int64) * 668265263 + int(seed) * 1442695041) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return (h & 0xFFFFFF) / float(0x1000000)


def vnoise(x, y, seed=0):
    """2-D value noise in 0..1 (numpy)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ix, iy = np.floor(x), np.floor(y)
    fx, fy = x - ix, y - iy
    fx = fx * fx * (3 - 2 * fx)
    fy = fy * fy * (3 - 2 * fy)
    a = _hash(ix, iy, seed)
    b = _hash(ix + 1, iy, seed)
    c = _hash(ix, iy + 1, seed)
    d = _hash(ix + 1, iy + 1, seed)
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def fbm(x, y, octaves=4, seed=0, gain=0.5, lac=2.03):
    v = 0.0
    amp = 1.0
    norm = 0.0
    for o in range(octaves):
        v = v + amp * vnoise(x, y, seed + 17 * o)
        norm += amp
        amp *= gain
        x = np.asarray(x) * lac
        y = np.asarray(y) * lac
    return v / norm


def smooth(a, b, v):
    s = np.clip((np.asarray(v, dtype=np.float64) - a) / (b - a), 0.0, 1.0)
    return s * s * (3 - 2 * s)


# ================================================================ node building helper
class NB:
    """Tiny helper to build shader / geometry node trees without the usual boilerplate."""

    def __init__(self, tree):
        self.tree = tree
        self.n = tree.nodes
        self.l = tree.links

    def node(self, kind, **props):
        nd = self.n.new(kind)
        for k, v in props.items():
            setattr(nd, k, v)
        return nd

    def set(self, sock, v):
        if v is None:
            return
        if isinstance(v, bpy.types.NodeSocket):
            self.l.new(v, sock)
        elif isinstance(v, bpy.types.Node):
            self.l.new(v.outputs[0], sock)
        else:
            if hasattr(sock.default_value, "__len__") and not hasattr(v, "__len__"):
                v = (v,) * len(sock.default_value)
            elif hasattr(sock.default_value, "__len__") and len(v) == 3 and len(sock.default_value) == 4:
                v = (*v, 1.0)
            sock.default_value = v

    def math(self, op, a, b=None, c=None, clamp=False):
        if op == "SMOOTHSTEP":                      # smoothstep(edge0=a, edge1=b, value=c)
            return self.maprange(c, a, b, 0.0, 1.0, interp="SMOOTHSTEP")
        nd = self.node("ShaderNodeMath", operation=op, use_clamp=clamp)
        self.set(nd.inputs[0], a)
        self.set(nd.inputs[1], b)
        self.set(nd.inputs[2], c)
        return nd.outputs[0]

    def vmath(self, op, a, b=None, scale=None):
        nd = self.node("ShaderNodeVectorMath", operation=op)
        self.set(nd.inputs[0], a)
        self.set(nd.inputs[1], b)
        if scale is not None:
            self.set(nd.inputs["Scale"], scale)
        return nd.outputs["Value"] if op in ("DOT_PRODUCT", "LENGTH", "DISTANCE") else nd.outputs["Vector"]

    def mix(self, fac, a, b, blend="MIX", clamp=True):
        nd = self.node("ShaderNodeMix", data_type="RGBA", blend_type=blend, clamp_factor=clamp)
        self.set(nd.inputs[0], fac)
        self.set(nd.inputs[6], a)
        self.set(nd.inputs[7], b)
        return nd.outputs[2]

    def ramp(self, fac, stops, interp="LINEAR"):
        """stops: [(position 0..1, (r, g, b) linear)]"""
        nd = self.node("ShaderNodeValToRGB")
        cr = nd.color_ramp
        cr.interpolation = interp
        while len(cr.elements) < len(stops):
            cr.elements.new(0.5)
        for e, (p, c) in zip(cr.elements, stops):
            e.position = p
            e.color = rgba(c) if len(c) == 3 else c
        self.set(nd.inputs["Fac"], fac)
        return nd.outputs["Color"]

    def maprange(self, v, a, b, c=0.0, d=1.0, clamp=True, interp="LINEAR"):
        nd = self.node("ShaderNodeMapRange", clamp=clamp, interpolation_type=interp)
        self.set(nd.inputs["Value"], v)
        for name, val in (("From Min", a), ("From Max", b), ("To Min", c), ("To Max", d)):
            self.set(nd.inputs[name], val)
        return nd.outputs[0]

    def attr(self, name, kind="GEOMETRY"):
        return self.node("ShaderNodeAttribute", attribute_type=kind, attribute_name=name)

    def sep(self, v):
        nd = self.node("ShaderNodeSeparateXYZ")
        self.set(nd.inputs[0], v)
        return nd.outputs

    def comb(self, x=0.0, y=0.0, z=0.0):
        nd = self.node("ShaderNodeCombineXYZ")
        for s, v in zip(nd.inputs, (x, y, z)):
            self.set(s, v)
        return nd.outputs[0]

    def noise(self, vec, scale, detail=2.0, rough=0.5, dims="3D"):
        nd = self.node("ShaderNodeTexNoise", noise_dimensions=dims)
        self.set(nd.inputs["Vector"], vec)
        self.set(nd.inputs["Scale"], scale)
        self.set(nd.inputs["Detail"], detail)
        self.set(nd.inputs["Roughness"], rough)
        return nd.outputs[0]

    def emission(self, color, strength=1.0):
        nd = self.node("ShaderNodeEmission")
        self.set(nd.inputs["Color"], color)
        self.set(nd.inputs["Strength"], strength)
        return nd.outputs[0]

    def phase(self):
        return util.time_node(self.n).outputs["Phase"]


def new_material(name, blended=False, shadow=True, cull=False):
    """Empty node material → (material, NB, output node)."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.node_tree.nodes.clear()
    nb = NB(mat.node_tree)
    out = nb.node("ShaderNodeOutputMaterial")
    mat.use_backface_culling = cull
    if blended:
        mat.surface_render_method = "BLENDED"
        mat.use_transparency_overlap = True
        mat.use_transparent_shadow = False
    return mat, nb, out


def haze_factor(nb, amount=1.0):
    """0..HAZE_MAX by camera depth (identical in every screen slice: depth along the view axis)."""
    depth = nb.node("ShaderNodeCameraData").outputs["View Z Depth"]
    d = nb.math("MAXIMUM", nb.math("SUBTRACT", depth, HAZE_START), 0.0)
    f = nb.math("SUBTRACT", 1.0, nb.math("EXPONENT", nb.math("MULTIPLY", d, -1.0 / HAZE_LENGTH)))
    return nb.math("MINIMUM", nb.math("MULTIPLY", f, amount), HAZE_MAX)


def with_haze(nb, shader, amount=1.0):
    """Mix a lit shader towards the (emissive) haze colour by depth."""
    mixs = nb.node("ShaderNodeMixShader")
    nb.l.new(haze_factor(nb, amount), mixs.inputs[0])
    nb.l.new(shader, mixs.inputs[1])
    nb.l.new(nb.emission(rgba(lin(PAL["haze"], HAZE_GAIN))), mixs.inputs[2])
    return mixs.outputs[0]


def additive(nb, out, color, strength):
    """Emission + transparent = pure additive card (use with new_material(blended=True))."""
    add = nb.node("ShaderNodeAddShader")
    nb.l.new(nb.emission(color, strength), add.inputs[0])
    nb.l.new(nb.node("ShaderNodeBsdfTransparent").outputs[0], add.inputs[1])
    nb.l.new(add.outputs[0], out.inputs["Surface"])


# ================================================================ mesh helpers
def mesh_from(name, verts, faces, col=None, uv=None, smooth_shade=True, attrs=None, loc=(0.0, 0.0, 0.0)):
    """Mesh object from numpy arrays.  faces: (F, 3|4) int array (uniform arity) or list of tuples.
    uv: (V, 2) per-vertex UVs.  attrs: {name: (V,) float | (V, 3) vector | (V, 4) colour}."""
    verts = np.asarray(verts, dtype=np.float32)
    me = bpy.data.meshes.new(name)
    if isinstance(faces, np.ndarray):
        nf, ar = faces.shape
        me.vertices.add(len(verts))
        me.vertices.foreach_set("co", verts.ravel())
        me.loops.add(nf * ar)
        me.loops.foreach_set("vertex_index", faces.astype(np.int32).ravel())
        me.polygons.add(nf)
        me.polygons.foreach_set("loop_start", np.arange(0, nf * ar, ar, dtype=np.int32))
        me.polygons.foreach_set("loop_total", np.full(nf, ar, dtype=np.int32))
        me.update()
        loop_v = faces.astype(np.int32).ravel()
    else:
        me.from_pydata([tuple(map(float, v)) for v in verts], [], [tuple(int(i) for i in f) for f in faces])
        me.update()
        loop_v = np.zeros(len(me.loops), dtype=np.int32)
        me.loops.foreach_get("vertex_index", loop_v)
    if smooth_shade and len(me.polygons):
        me.polygons.foreach_set("use_smooth", np.ones(len(me.polygons), dtype=bool))
    if uv is not None:
        layer = me.uv_layers.new(name="UVMap")
        layer.data.foreach_set("uv", np.asarray(uv, dtype=np.float32)[loop_v].ravel())
    for an, av in (attrs or {}).items():
        set_attr(me, an, av)
    obj = None
    if col is not None:
        obj = bpy.data.objects.new(name, me)
        obj.location = loc
        col.objects.link(obj)
    return obj if obj is not None else me


def set_attr(me, name, values, domain="POINT"):
    values = np.asarray(values)
    if values.dtype.kind in "iu":
        a = me.attributes.new(name, "INT", domain)
        a.data.foreach_set("value", values.astype(np.int32))
    elif values.ndim == 1:
        a = me.attributes.new(name, "FLOAT", domain)
        a.data.foreach_set("value", values.astype(np.float32))
    elif values.shape[1] == 3:
        a = me.attributes.new(name, "FLOAT_VECTOR", domain)
        a.data.foreach_set("vector", values.astype(np.float32).ravel())
    else:
        a = me.attributes.new(name, "FLOAT_COLOR", domain)
        a.data.foreach_set("color", values.astype(np.float32).ravel())
    return a


def grid_faces(nx, ny):
    i, j = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1))
    a = (j * nx + i).ravel()
    return np.stack([a, a + 1, a + nx + 1, a + nx], axis=1)


def quads_mesh(name, centers, half_w, half_h, col, attrs=None, loc=(0.0, 0.0, 0.0)):
    """Camera-facing quads (the camera never rotates → quads lie in XZ planes).
    centers: (N, 3) local coords; half_w / half_h: scalars or (N,).  Per-quad attrs are repeated on
    the 4 corners.  UV runs -1..1 across each quad (stored 0..1)."""
    centers = np.asarray(centers, dtype=np.float64)
    n = len(centers)
    hw = np.broadcast_to(np.asarray(half_w, dtype=np.float64), (n,))
    hh = np.broadcast_to(np.asarray(half_h, dtype=np.float64), (n,))
    cx = np.array([-1.0, 1.0, 1.0, -1.0])
    cz = np.array([-1.0, -1.0, 1.0, 1.0])
    v = np.repeat(centers, 4, axis=0)
    v[:, 0] += (hw[:, None] * cx[None, :]).ravel()
    v[:, 2] += (hh[:, None] * cz[None, :]).ravel()
    faces = np.arange(4 * n, dtype=np.int32).reshape(n, 4)
    uv = np.stack([np.tile(cx * 0.5 + 0.5, n), np.tile(cz * 0.5 + 0.5, n)], axis=1)
    rep = {k: np.repeat(np.asarray(a), 4, axis=0) for k, a in (attrs or {}).items()}
    obj = mesh_from(name, v, faces, col, uv=uv, smooth_shade=False, attrs=rep, loc=loc)
    obj.visible_shadow = False
    return obj


def card_uv(nb):
    """→ (u, v, r2) with u, v in -1..1 across a quads_mesh card."""
    uv = nb.node("ShaderNodeUVMap").outputs[0]
    s = nb.sep(uv)
    u = nb.math("MULTIPLY_ADD", s[0], 2.0, -1.0)
    v = nb.math("MULTIPLY_ADD", s[1], 2.0, -1.0)
    r2 = nb.math("ADD", nb.math("MULTIPLY", u, u), nb.math("MULTIPLY", v, v))
    return u, v, r2


# ================================================================ visibility (what the camera can see of the ground)
class Visibility:
    """Horizon map of the bare terrain in (stage x, depth) space → cheap occlusion / frustum tests
    for scattering (nothing is placed where the camera cannot see it)."""

    def __init__(self):
        self.xs0, self.dxs = C.PANO_X0 - 2.5, 0.1
        self.xs = np.arange(self.xs0, C.PANO_X1 + 2.5, self.dxs)
        self.ys = np.concatenate([np.arange(-30.0, 12.0, 0.1), np.arange(12.0, 70.0, 0.25), np.arange(70.0, 360.0, 1.5)])
        XS, Y = np.meshgrid(self.xs, self.ys)
        X, _ = to_world(XS, 0.0, Y)
        Z = terrain.height(X, Y)
        _, ZS = to_stage(X, Y, Z)
        hor = np.maximum.accumulate(ZS, axis=0)
        self.hor = np.vstack([np.full((1, len(self.xs)), -1e9), hor[:-1]])   # horizon of everything nearer

    def test(self, x, y, z, top=0.2, margin=0.03):
        """True where an object of height `top` standing at (x, y, z) is at least partly in frame
        and not hidden behind nearer terrain."""
        xs, zs0 = to_stage(x, y, z)
        _, zs1 = to_stage(x, y, np.asarray(z) + top)
        ci = np.clip(np.round((xs - self.xs0) / self.dxs).astype(int), 0, len(self.xs) - 1)
        ri = np.clip(np.searchsorted(self.ys, y), 0, len(self.ys) - 1)
        hor = np.maximum(self.hor[ri, ci], C.FRAME_Z0)
        inside = (xs > C.PANO_X0 - 0.6) & (xs < C.PANO_X1 + 0.6) & (zs0 < C.FRAME_Z1)
        return inside & (zs1 > hor - margin)


# ================================================================ ground
def tint_field(x, y):
    """Large-scale colour variation of the meadow shared by the ground and the grass:
    (patch 0..1 dark→light green, dry 0..1 straw-coloured patches)."""
    k = kd(y)
    patch = fbm(x * 0.32 / k + 3.1, y * 0.32 / k - 7.7, 4, seed=11)
    patch = np.clip((patch - 0.5) * 2.1 + 0.5, 0.0, 1.0)
    dry = smooth(0.62, 0.80, fbm(x * 0.21 / k - 9.0, y * 0.27 / k + 4.0, 3, seed=29))
    return patch, dry


def hill_u(x, y):
    """Smallest normalised hill radius (0 = on a mark, 1 = hill foot) and the hill index."""
    best = np.full(np.shape(x), 1e9)
    for _role, hx, hy, _top, radius, _ in C.HILLS:
        best = np.minimum(best, terrain._hill_u(x, y, hx, hy, radius))
    return best


def mark_dist(x, y):
    d = np.full(np.shape(x), 1e9)
    for mx, my in C.MARKS.values():
        d = np.minimum(d, np.hypot(np.asarray(x) - mx, np.asarray(y) - my))
    return d


def build_ground(ctx, col):
    step = 0.16 if ctx.draft else 0.08
    xs = np.arange(C.PANO_X0 - 3.0, C.PANO_X1 + 3.0 + step, step)
    ys = [np.arange(-27.0, -6.0, 0.3), np.arange(-6.0, 8.0, 0.2 if ctx.draft else 0.1)]
    y, dy = 8.0, 0.1
    far = []
    while y < 345.0:
        far.append(y)
        dy *= 1.07 if ctx.draft else 1.035
        y += dy
    ys = np.concatenate(ys + [np.array(far)])
    XS, Y = np.meshgrid(xs, ys)
    X, _ = to_world(XS, 0.0, Y)
    Z = terrain.height(X, Y)
    patch, dry = tint_field(X, Y)
    hu = hill_u(X, Y)
    tint = np.stack([patch.ravel(), dry.ravel(), np.clip(hu, 0, 2).ravel() * 0.5, np.ones(X.size)], axis=1)
    verts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    obj = mesh_from("VK_Ground", verts, grid_faces(len(xs), len(ys)), col, attrs={"tint": tint})

    mat, nb, out = new_material("VK_GroundMat")
    geo = nb.node("ShaderNodeNewGeometry")
    depth = nb.node("ShaderNodeCameraData").outputs["View Z Depth"]
    kinv = nb.math("DIVIDE", D, nb.math("MAXIMUM", depth, 20.0))
    pos_s = nb.vmath("SCALE", geo.outputs["Position"], scale=kinv)        # constant texture size on screen
    tint_n = nb.attr("tint")
    ts = nb.sep(tint_n.outputs["Vector"])
    n1 = nb.noise(pos_s, 5.5, 3.0, 0.55)
    n2 = nb.noise(pos_s, 23.0, 2.0, 0.6)
    t = nb.math("ADD", ts[0], nb.math("MULTIPLY_ADD", n1, 0.5, -0.25), clamp=True)
    base = nb.ramp(t, [(0.0, lin(PAL["grass_dark"])), (0.5, lin(PAL["grass_mid"])), (1.0, lin(PAL["grass_light"]))])
    base = nb.mix(nb.math("MULTIPLY", ts[1], 0.55), base, rgba(lin(PAL["grass_dry"])))
    # fine mottling: reads as blades from a distance, hides under the real grass close up
    base = nb.mix(nb.maprange(n2, 0.35, 0.7, 0.0, 0.5), base, rgba(lin(PAL["grass_dark"])))
    bsdf = nb.node("ShaderNodeBsdfPrincipled")
    nb.set(bsdf.inputs["Base Color"], base)
    bsdf.inputs["Roughness"].default_value = 0.95
    bsdf.inputs["Specular IOR Level"].default_value = 0.1
    nb.l.new(with_haze(nb, bsdf.outputs[0]), out.inputs["Surface"])
    obj.data.materials.append(mat)
    return obj


# ================================================================ sky, stars, moon
def stage_coords(nb, y):
    """(xs, zs) of the shaded point for geometry that lives on the plane world-y = y."""
    k = float(kd(y))
    p = nb.sep(nb.node("ShaderNodeNewGeometry").outputs["Position"])
    xs = nb.math("ADD", nb.math("MULTIPLY", nb.math("SUBTRACT", p[0], C.CAM_X), 1.0 / k), C.CAM_X)
    zs = nb.math("ADD", nb.math("MULTIPLY", nb.math("SUBTRACT", p[2], C.CAM_Z), 1.0 / k), C.CAM_Z)
    return xs, zs


def build_sky(ctx, col):
    x0, z0 = to_world(C.PANO_X0 - 6.0, C.FRAME_Z0 - 1.5, SKY_Y)
    x1, z1 = to_world(C.PANO_X1 + 6.0, C.FRAME_Z1 + 1.0, SKY_Y)
    verts = np.array([[x0, 0, z0], [x1, 0, z0], [x1, 0, z1], [x0, 0, z1]], dtype=np.float64)
    obj = mesh_from("VK_Sky", verts, np.array([[0, 1, 2, 3]]), col, smooth_shade=False, loc=(0, SKY_Y, 0))
    obj.visible_shadow = False
    mat, nb, out = new_material("VK_SkyMat")
    xs, zs = stage_coords(nb, SKY_Y)
    lo, hi = PAL["sky"][0][0], PAL["sky"][-1][0]
    stops = [((z - lo) / (hi - lo), lin(c, SKY_GAIN)) for z, c in PAL["sky"]]
    grad = nb.ramp(nb.maprange(zs, lo, hi), stops, interp="EASE")
    # dusk afterglow: a warmer, lighter violet low on the right-of-centre horizon (opposite the moon)
    gx = nb.math("DIVIDE", nb.math("SUBTRACT", xs, 9.0), 17.0)
    glow_x = nb.math("EXPONENT", nb.math("MULTIPLY", nb.math("MULTIPLY", gx, gx), -1.0))
    gz = nb.math("DIVIDE", nb.math("SUBTRACT", zs, 0.15), 0.85)
    glow_z = nb.math("EXPONENT", nb.math("MULTIPLY", nb.math("MAXIMUM", gz, 0.0), -1.0))
    glow = nb.math("MULTIPLY", glow_x, glow_z)
    col_glow = nb.mix(nb.math("MULTIPLY", glow, 0.55), grad, rgba(lin("#8a5cb8", SKY_GAIN * 1.15)))
    # very soft large-scale variation so the gradient is not perfectly uniform along 56 m
    vec = nb.comb(nb.math("MULTIPLY", xs, 0.11), 0.0, nb.math("MULTIPLY", zs, 0.35))
    cloud = nb.noise(vec, 1.0, 3.0, 0.55)
    col_var = nb.mix(nb.maprange(cloud, 0.35, 0.75, 0.0, 0.22), col_glow, rgba(lin("#3a4fb0", SKY_GAIN)))
    nb.l.new(nb.emission(col_var, 1.0), out.inputs["Surface"])
    obj.data.materials.append(mat)
    return obj


def _twinkle(nb, a_node, b_node):
    """1 + amp·(0.6·sin(2π(k1·Phase + p1)) + 0.4·sin(2π(k2·Phase + p2))), k integer → exact loop."""
    ph = nb.phase()
    a = nb.sep(a_node.outputs["Vector"])      # p1, p2, amp
    b = nb.sep(b_node.outputs["Vector"])      # k1, k2, brightness
    s1 = nb.math("SINE", nb.math("MULTIPLY", nb.math("MULTIPLY_ADD", ph, b[0], a[0]), TAU))
    s2 = nb.math("SINE", nb.math("MULTIPLY", nb.math("MULTIPLY_ADD", ph, b[1], a[1]), TAU))
    w = nb.math("ADD", nb.math("MULTIPLY", s1, 0.6), nb.math("MULTIPLY", s2, 0.4))
    tw = nb.math("MULTIPLY_ADD", w, a[2], 1.0)
    return nb.math("MULTIPLY", tw, b[2])


def build_stars(ctx, col):
    rng = np.random.default_rng(SEED + 1)
    n = STARS["count"]
    y = STARS["y"]
    k = float(kd(y))
    # candidates → rejection by a density field (slightly denser high up and along a faint band)
    m = n * 3
    xs = rng.uniform(C.PANO_X0 - 0.5, C.PANO_X1 + 0.5, m)
    zs = rng.uniform(0.2, C.FRAME_Z1 + 0.1, m)
    band = np.exp(-0.5 * (((zs - 2.3) - 0.055 * (xs - 14.0)) / 0.55) ** 2) * smooth(2.0, 12.0, xs)
    clump = fbm(xs * 0.6, zs * 0.6, 3, seed=5)
    dens = (0.45 + 0.35 * smooth(0.3, 2.2, zs) + 0.55 * band) * (0.55 + 0.9 * clump)
    keep = rng.uniform(0, dens.max(), m) < dens
    xs, zs = xs[keep][:n], zs[keep][:n]
    n = len(xs)
    u = rng.uniform(0, 1, n)
    size_px = 1.7 + 6.3 * u ** 6.0                     # most stars are tiny, a few reach ~8 px
    frac = (size_px - 1.7) / 6.3
    bright = (0.10 + 0.90 * rng.uniform(0, 1, n) ** 2.4) * (0.55 + 0.85 * frac)
    # extinction near the horizon, keep the moon's surroundings calmer
    bright *= 0.25 + 0.75 * smooth(0.35, 1.5, zs)
    dm = np.hypot(xs - MOON["xs"], zs - MOON["zs"])
    bright *= 0.15 + 0.85 * smooth(MOON["radius"] * 1.2, MOON["radius"] * 5.0, dm)
    temp = rng.uniform(0, 1, n)
    colr = np.where(temp[:, None] < 0.62, np.array([[0.80, 0.88, 1.0]]),
                    np.where(temp[:, None] < 0.86, np.array([[1.0, 1.0, 1.0]]), np.array([[1.0, 0.86, 0.68]])))
    amp = rng.uniform(0.15, 0.60, n) * (0.6 + 0.4 * (size_px < 4.0))
    a = np.stack([rng.uniform(0, 1, n), rng.uniform(0, 1, n), amp], axis=1)
    b = np.stack([rng.integers(3, 10, n).astype(float), rng.integers(11, 24, n).astype(float), bright], axis=1)
    X, Z = to_world(xs, zs, y)
    half = 1.5 * size_px / C.PPM * k
    cen = np.stack([X, np.zeros(n), Z], axis=1)
    obj = quads_mesh("VK_Stars", cen, half, half, col, attrs={"sa": a, "sb": b, "scol": np.c_[colr, np.ones(n)]}, loc=(0, y, 0))
    mat, nb, out = new_material("VK_StarMat", blended=True)
    _u, _v, r2 = card_uv(nb)
    core = nb.math("EXPONENT", nb.math("MULTIPLY", r2, -6.0))
    core = nb.math("MAXIMUM", nb.math("SUBTRACT", core, 0.0025), 0.0)
    tw = _twinkle(nb, nb.attr("sa"), nb.attr("sb"))
    additive(nb, out, nb.attr("scol").outputs["Color"], nb.math("MULTIPLY", nb.math("MULTIPLY", core, tw), STARS["strength"]))
    obj.data.materials.append(mat)

    # the brightest stars get a soft halo and a 4-point sparkle
    idx = np.argsort(-(size_px * bright))[:STARS["sparkles"]]
    half2 = 7.0 * size_px[idx] / C.PPM * k
    cen2 = cen[idx].copy()
    cen2[:, 1] -= 5.0
    obj2 = quads_mesh("VK_StarSparkles", cen2, half2, half2, col,
                      attrs={"sa": a[idx], "sb": b[idx], "scol": np.c_[colr[idx], np.ones(len(idx))]}, loc=(0, y, 0))
    mat2, nb, out = new_material("VK_SparkleMat", blended=True)
    u_, v_, r2 = card_uv(nb)
    au, av = nb.math("ABSOLUTE", u_), nb.math("ABSOLUTE", v_)
    sp_h = nb.math("MULTIPLY", nb.math("EXPONENT", nb.math("MULTIPLY", au, -5.0)), nb.math("EXPONENT", nb.math("MULTIPLY", nb.math("MULTIPLY", v_, v_), -900.0)))
    sp_v = nb.math("MULTIPLY", nb.math("EXPONENT", nb.math("MULTIPLY", av, -5.0)), nb.math("EXPONENT", nb.math("MULTIPLY", nb.math("MULTIPLY", u_, u_), -900.0)))
    halo = nb.math("MULTIPLY", nb.math("EXPONENT", nb.math("MULTIPLY", r2, -9.0)), 0.10)
    edge = nb.math("SUBTRACT", 1.0, nb.math("SMOOTHSTEP", 0.55, 1.0, nb.math("MAXIMUM", au, av)))
    shape = nb.math("MULTIPLY", nb.math("ADD", halo, nb.math("MULTIPLY", nb.math("ADD", sp_h, sp_v), 0.5)), edge)
    tw = _twinkle(nb, nb.attr("sa"), nb.attr("sb"))
    additive(nb, out, nb.attr("scol").outputs["Color"], nb.math("MULTIPLY", nb.math("MULTIPLY", shape, tw), STARS["strength"] * 0.8))
    obj2.data.materials.append(mat2)
    return [obj, obj2]


def build_moon(ctx, col):
    y = MOON["y"]
    k = float(kd(y))
    X, Z = to_world(MOON["xs"], MOON["zs"], y)
    R = MOON["radius"] * k
    seg = 96
    ang = np.linspace(0, TAU, seg, endpoint=False)
    verts = np.vstack([[0.0, 0.0, 0.0], np.stack([np.cos(ang) * R, np.zeros(seg), np.sin(ang) * R], axis=1)])
    faces = np.array([[0, 1 + i, 1 + (i + 1) % seg] for i in range(seg)])
    uv = np.vstack([[0.5, 0.5], np.stack([0.5 + 0.5 * np.cos(ang), 0.5 + 0.5 * np.sin(ang)], axis=1)])
    disc = mesh_from("VK_Moon", verts, faces, col, uv=uv, smooth_shade=False, loc=(float(X), y, float(Z)))
    disc.visible_shadow = False
    mat, nb, out = new_material("VK_MoonMat")
    u, v, r2 = card_uv(nb)
    vec = nb.comb(u, v, 0.0)
    maria = nb.noise(nb.vmath("ADD", vec, (3.2, 1.7, 0.0)), 1.55, 3.0, 0.55)
    craters = nb.node("ShaderNodeTexVoronoi", feature="F1")
    nb.set(craters.inputs["Vector"], vec)
    craters.inputs["Scale"].default_value = 4.2
    cr = nb.maprange(craters.outputs["Distance"], 0.08, 0.30, 0.0, 1.0)
    tone = nb.math("MULTIPLY", nb.maprange(maria, 0.42, 0.62, 0.72, 1.0), nb.math("MULTIPLY_ADD", cr, 0.10, 0.90))
    # stylised shading: soft terminator-less limb, slightly cooler / dimmer towards the lower right
    limb = nb.math("SUBTRACT", 1.0, nb.math("MULTIPLY", nb.math("POWER", r2, 2.0), 0.22))
    side = nb.maprange(nb.math("SUBTRACT", u, v), -1.6, 1.6, 1.0, 0.80)
    colr = nb.mix(nb.maprange(maria, 0.40, 0.66), rgba((0.72, 0.80, 1.0)), rgba((1.0, 0.97, 0.86)))
    strength = nb.math("MULTIPLY", nb.math("MULTIPLY", tone, nb.math("MULTIPLY", limb, side)), MOON["strength"])
    nb.l.new(nb.emission(colr, strength), out.inputs["Surface"])
    disc.data.materials.append(mat)

    # glow: three additive cards (tight corona, soft halo, very wide sky lift)
    glows = []
    for i, (rad, power, strength, tint) in enumerate([(2.2, 7.0, 1.10, (0.80, 0.88, 1.0)),
                                                       (6.0, 4.5, 0.20, (0.50, 0.62, 1.0)),
                                                       (17.0, 3.2, 0.060, (0.42, 0.50, 1.0))]):
        g = quads_mesh(f"VK_MoonGlow{i}", np.array([[0.0, 0.0, 0.0]]), R * rad, R * rad, col,
                       loc=(float(X), y - 20.0 - 10.0 * i, float(Z)))
        gm, nb, out = new_material(f"VK_MoonGlowMat{i}", blended=True)
        _u, _v, r2 = card_uv(nb)
        r = nb.math("SQRT", r2)
        fall = nb.math("EXPONENT", nb.math("MULTIPLY", r, -power))
        fall = nb.math("MULTIPLY", fall, nb.math("SUBTRACT", 1.0, nb.math("SMOOTHSTEP", 0.6, 1.0, r)))
        additive(nb, out, rgba(tint), nb.math("MULTIPLY", fall, strength))
        g.data.materials.append(gm)
        glows.append(g)
    return [disc] + glows


# ================================================================ far mountains + mist
def _peaks_profile(xs, rng, base, hmax, spacing, seed):
    """Skyline of a mountain range: a chain of sharp asymmetric peaks, lower behind the three characters."""
    top = np.full_like(xs, base)
    marks = [m[0] for m in C.MARKS.values()]
    x = xs[0] - 2.0
    while x < xs[-1] + 2.0:
        x += spacing * rng.uniform(0.5, 1.5)
        env = 0.42 + 0.58 * min(1.0, min(abs(x - m) for m in marks) / 6.0)
        big = 0.55 + 0.9 * float(fbm(np.array([x * 0.06]), np.array([seed * 1.7]), 2, seed=seed)[0])
        h = hmax * rng.uniform(0.30, 1.0) * env * min(big, 1.0)
        w = h * rng.uniform(1.5, 2.4) + 0.25
        skew = rng.uniform(-0.35, 0.35)
        d = (xs - x) / w
        d = np.where(d < 0.0, d / (1.0 + skew), d / (1.0 - skew))
        top = np.maximum(top, base + h * np.maximum(0.0, 1.0 - np.abs(d)) ** 1.2)
    rel = np.clip((top - base) / hmax, 0.0, 1.0)
    ridged = 1.0 - np.abs(2.0 * fbm(xs * 1.6, xs * 0.0 + 3.3, 4, seed=seed, gain=0.55) - 1.0)
    return top + hmax * (0.02 + 0.07 * rel) * (ridged - 0.5) + 0.03 * hmax * np.sin(xs * 0.9 + seed)


def build_mountains(ctx, col):
    """Two silhouette ranges far behind the meadow (emissive cards: they sit in the haze, not in the lights)."""
    objs = []
    layers = [
        # depth, foot zs, max peak height (stage m), mean peak spacing, colours (shadow side, moonlit side), snow line zs
        dict(y=1100.0, base=0.40, hmax=0.78, spacing=2.3, seed=41, shade="#3a3f96", lit="#6670cc", snow=0.80),
        dict(y=640.0, base=0.34, hmax=0.40, spacing=1.5, seed=77, shade="#2b3180", lit="#444ea6", snow=9.0),
    ]
    step = 0.03
    xs = np.arange(C.PANO_X0 - 2.0, C.PANO_X1 + 2.0 + step, step)
    for i, L in enumerate(layers):
        rng = np.random.default_rng(SEED + L["seed"])
        top = _peaks_profile(xs, rng, L["base"], L["hmax"], L["spacing"], L["seed"])
        slope = np.gradient(top, xs)
        ker = np.ones(9) / 9.0
        slope = np.convolve(slope, ker, mode="same")
        zb = -0.45
        rows = 6
        tt = np.linspace(0.0, 1.0, rows)
        XS = np.tile(xs, rows)
        ZS = (zb + (top[None, :] - zb) * tt[:, None]).ravel()
        X, Z = to_world(XS, ZS, L["y"])
        verts = np.stack([X, np.zeros_like(X), Z], axis=1)
        # the moon stands on the left: slopes that rise to the right face it
        info = np.stack([np.tile(np.clip(slope * 1.6, -1, 1), rows), np.repeat(tt, len(xs)), np.tile(top, rows)], axis=1)
        obj = mesh_from(f"VK_Mountains{i}", verts, grid_faces(len(xs), rows), col, smooth_shade=False,
                        attrs={"minfo": info}, loc=(0, L["y"], 0))
        obj.visible_shadow = False
        mat, nb, out = new_material(f"VK_MountainMat{i}")
        xs_n, zs_n = stage_coords(nb, L["y"])
        info_n = nb.sep(nb.attr("minfo").outputs["Vector"])        # lit (-1..1), rel height, top zs
        below = nb.math("SUBTRACT", info_n[2], zs_n)                # distance below the crest (stage m)
        # faces: the lit / shadow split follows the skyline slope, broken up by streaky rock noise
        vec = nb.comb(nb.math("MULTIPLY", xs_n, 4.0), 0.0, nb.math("MULTIPLY", zs_n, 1.6))
        rock = nb.noise(nb.vmath("ADD", vec, (L["seed"] * 1.0, 0.0, 0.0)), 1.0, 4.0, 0.6)
        lit = nb.math("ADD", info_n[0], nb.math("MULTIPLY_ADD", rock, 1.1, -0.55))
        lit = nb.math("SMOOTHSTEP", -0.25, 0.45, lit)
        lit = nb.math("MULTIPLY", lit, nb.math("SUBTRACT", 1.0, nb.math("MULTIPLY", nb.math("SMOOTHSTEP", 0.05, 0.6, below), 0.55)))
        colr = nb.mix(nb.math("MULTIPLY", lit, 0.75), rgba(lin(L["shade"], HAZE_GAIN)), rgba(lin(L["lit"], HAZE_GAIN)))
        # moonlit snow, on the highest peaks only
        line = nb.math("MULTIPLY_ADD", nb.noise(vec, 2.0, 3.0, 0.6), 0.16, L["snow"] - 0.08)
        snow = nb.math("SMOOTHSTEP", 0.0, 0.05, nb.math("SUBTRACT", zs_n, line))
        snow = nb.math("MULTIPLY", snow, nb.math("SUBTRACT", 1.0, nb.math("SMOOTHSTEP", 0.10, 0.30, below)))
        snow = nb.math("MULTIPLY", snow, nb.math("MULTIPLY_ADD", lit, 0.55, 0.30))
        colr = nb.mix(snow, colr, rgba(lin("#b4c0ff", HAZE_GAIN * 1.35)))
        # the feet dissolve into the valley haze
        foot = nb.math("SUBTRACT", 1.0, nb.maprange(zs_n, L["base"] - 0.12, L["base"] + 0.30, 0.0, 1.0, interp="SMOOTHSTEP"))
        colr = nb.mix(nb.math("MULTIPLY", foot, 0.85), colr, rgba(lin(PAL["haze"], HAZE_GAIN)))
        nb.l.new(nb.emission(colr, 1.0), out.inputs["Surface"])
        obj.data.materials.append(mat)
        objs.append(obj)
    return objs


def build_mist(ctx, col):
    """Soft alpha cards lying between the depth layers (valley mist).  Pure geometry."""
    objs = []
    mat, nb, out = new_material("VK_MistMat", blended=True)
    uvn = nb.node("ShaderNodeUVMap").outputs[0]
    s = nb.sep(uvn)
    info = nb.sep(nb.attr("mist").outputs["Vector"])       # alpha, noise offset, noise scale
    v = s[1]
    prof = nb.math("MULTIPLY", nb.math("SMOOTHSTEP", 0.0, 0.45, v), nb.math("SUBTRACT", 1.0, nb.math("SMOOTHSTEP", 0.45, 1.0, v)))
    vec = nb.comb(nb.math("MULTIPLY_ADD", s[0], info[2], info[1]), nb.math("MULTIPLY", v, 1.4), info[1])
    nz = nb.noise(vec, 1.0, 3.0, 0.55)
    edge = nb.math("MULTIPLY", nb.math("SMOOTHSTEP", 0.0, 0.04, s[0]), nb.math("SUBTRACT", 1.0, nb.math("SMOOTHSTEP", 0.96, 1.0, s[0])))
    alpha = nb.math("MULTIPLY", nb.math("MULTIPLY", prof, nb.maprange(nz, 0.25, 0.8, 0.15, 1.0)), nb.math("MULTIPLY", info[0], edge))
    mixs = nb.node("ShaderNodeMixShader")
    nb.l.new(alpha, mixs.inputs[0])
    nb.l.new(nb.node("ShaderNodeBsdfTransparent").outputs[0], mixs.inputs[1])
    nb.l.new(nb.emission(rgba(lin(PAL["mist"], HAZE_GAIN)), 1.0), mixs.inputs[2])
    nb.l.new(mixs.outputs[0], out.inputs["Surface"])
    # (depth y, centre zs, half height zs, alpha)
    cards = [(21.0, -0.52, 0.20, 0.16), (45.0, -0.26, 0.20, 0.22), (88.0, -0.03, 0.20, 0.30),
             (160.0, 0.20, 0.20, 0.34), (300.0, 0.38, 0.20, 0.34), (800.0, 0.46, 0.22, 0.42)]
    for i, (y, zc, hz, a) in enumerate(cards):
        x0, z0 = to_world(C.PANO_X0 - 3.0, zc - hz, y)
        x1, z1 = to_world(C.PANO_X1 + 3.0, zc + hz, y)
        nseg = 24
        xsv = np.linspace(x0, x1, nseg + 1)
        verts = np.vstack([np.stack([xsv, np.zeros(nseg + 1), np.full(nseg + 1, z0)], axis=1),
                           np.stack([xsv, np.zeros(nseg + 1), np.full(nseg + 1, z1)], axis=1)])
        uv = np.vstack([np.stack([np.linspace(0, 1, nseg + 1), np.zeros(nseg + 1)], axis=1),
                        np.stack([np.linspace(0, 1, nseg + 1), np.ones(nseg + 1)], axis=1)])
        attr = np.tile(np.array([[a, 3.7 * i + 1.0, 9.0]]), (len(verts), 1))
        obj = mesh_from(f"VK_Mist{i}", verts, grid_faces(nseg + 1, 2), col, uv=uv, smooth_shade=False,
                        attrs={"mist": attr}, loc=(0, y, 0))
        obj.visible_shadow = False
        obj.data.materials.append(mat)
        objs.append(obj)
    return objs


# ================================================================ procedural props (numpy mesh builders)
class MB:
    """Accumulates parts (verts, faces, per-vertex uv, material slot) into one mesh."""

    def __init__(self):
        self.v, self.f, self.uv, self.m = [], [], [], []
        self.n = 0

    def add(self, verts, faces, uv=None, mat=0):
        verts = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
        self.v.append(verts)
        self.uv.append(np.zeros((len(verts), 2)) if uv is None else np.asarray(uv, dtype=np.float64))
        for f in faces:
            self.f.append(tuple(int(i) + self.n for i in f))
            self.m.append(mat)
        self.n += len(verts)

    def mesh(self, name, materials, smooth_shade=True):
        me = mesh_from(name, np.vstack(self.v), self.f, uv=np.vstack(self.uv), smooth_shade=smooth_shade)
        for mat in materials:
            me.materials.append(mat)
        me.polygons.foreach_set("material_index", np.array(self.m, dtype=np.int32))
        return me


def blade_part(rng, base, h, w, lean, curl, segs, direction=None, twist=None):
    """One grass blade: tapered strip bending along `direction` (radians).  UV.y = 0 root … 1 tip."""
    a = rng.uniform(0, TAU) if direction is None else direction
    d = np.array([math.cos(a), math.sin(a), 0.0])
    tw = a + math.pi / 2 + (rng.uniform(-0.9, 0.9) if twist is None else twist)
    side = np.array([math.cos(tw), math.sin(tw), 0.0])
    verts, uv, faces = [], [], []
    for i in range(segs + 1):
        v = i / segs
        c = np.asarray(base) + d * (lean * v + curl * v * v) * h + np.array([0, 0, 1.0]) * h * v * (1.0 - 0.22 * curl * v)
        hw = 0.5 * w * (1.0 - v ** 1.7) ** 0.8 * (0.75 + 0.5 * math.sin(math.pi * min(v * 1.6, 1.0)) ** 0.5 * 0.5)
        if i < segs:
            verts += [c - side * hw, c + side * hw]
            uv += [(0.0, v), (1.0, v)]
        else:
            verts.append(c)
            uv.append((0.5, 1.0))
    for i in range(segs - 1):
        faces.append((2 * i, 2 * i + 1, 2 * i + 3, 2 * i + 2))
    faces.append((2 * (segs - 1), 2 * (segs - 1) + 1, 2 * segs))
    return np.array(verts), faces, np.array(uv)


def tuft_mesh(name, rng, mat, n_blades, segs, spread=0.22, width=0.11, hmin=0.45):
    mb = MB()
    for b in range(n_blades):
        r = spread * math.sqrt(rng.uniform(0, 1))
        a = rng.uniform(0, TAU)
        base = (r * math.cos(a), r * math.sin(a), 0.0)
        h = rng.uniform(hmin, 1.0) if b else 1.0
        v, f, uv = blade_part(rng, base, h, width * rng.uniform(0.75, 1.25), rng.uniform(0.05, 0.38), rng.uniform(0.0, 0.42),
                              segs, direction=a + rng.uniform(-0.8, 0.8))
        mb.add(v, f, uv)
    return mb.mesh(name, [mat])


def tube_part(path, radii, sides=5):
    path = np.asarray(path, dtype=np.float64)
    n = len(path)
    radii = np.broadcast_to(np.asarray(radii, dtype=np.float64), (n,))
    ang = np.linspace(0, TAU, sides, endpoint=False)
    ring = np.stack([np.cos(ang), np.sin(ang), np.zeros(sides)], axis=1)
    verts = (path[:, None, :] + ring[None, :, :] * radii[:, None, None]).reshape(-1, 3)
    faces = []
    for i in range(n - 1):
        for j in range(sides):
            a, b = i * sides + j, i * sides + (j + 1) % sides
            faces.append((a, b, b + sides, a + sides))
    uv = np.stack([np.tile(np.linspace(0, 1, sides), n), np.repeat(np.linspace(0, 1, n), sides)], axis=1)
    return verts, faces, uv


def lathe_part(profile, sides=12, cap_top=True):
    """profile: [(radius, z)] bottom → top."""
    prof = np.asarray(profile, dtype=np.float64)
    path = np.stack([np.zeros(len(prof)), np.zeros(len(prof)), prof[:, 1]], axis=1)
    verts, faces, uv = tube_part(path, prof[:, 0], sides)
    if cap_top:
        top = len(verts)
        verts = np.vstack([verts, [[0.0, 0.0, prof[-1, 1]]]])
        uv = np.vstack([uv, [[0.5, 1.0]]])
        base = (len(prof) - 1) * sides
        for j in range(sides):
            faces.append((base + j, base + (j + 1) % sides, top))
    return verts, faces, uv


_ICO = {}


def icosphere(subdiv=2):
    if subdiv not in _ICO:
        import bmesh
        bm = bmesh.new()
        bmesh.ops.create_icosphere(bm, subdivisions=subdiv, radius=1.0)
        v = np.array([vv.co[:] for vv in bm.verts])
        f = [tuple(vv.index for vv in ff.verts) for ff in bm.faces]
        bm.free()
        _ICO[subdiv] = (v / np.linalg.norm(v, axis=1, keepdims=True), f)
    return _ICO[subdiv]


def wobble(p, seed, freq=2.4):
    """Smooth pseudo-noise -1..1 for points on / near a unit sphere (cheap, deterministic)."""
    a = seed * 1.7
    return (np.sin(freq * p[:, 0] + a) * np.sin(freq * 1.13 * p[:, 1] + 2.1 * a) * np.sin(freq * 0.91 * p[:, 2] + 0.6 * a)
            + 0.5 * np.sin(freq * 2.1 * p[:, 0] + 1.3 * p[:, 2] + a) * np.sin(freq * 1.9 * p[:, 1] - a))


# ---------------------------------------------------------------- materials
def instancer_tint(nb):
    return nb.attr("tint", "INSTANCER")


def grass_material():
    mat, nb, out = new_material("VK_GrassMat")
    v = nb.sep(nb.node("ShaderNodeUVMap").outputs[0])[1]
    ti = nb.sep(instancer_tint(nb).outputs["Vector"])                 # patch, dry, brightness
    body = nb.ramp(ti[0], [(0.0, lin(PAL["grass_dark"])), (0.5, lin(PAL["grass_mid"])), (1.0, lin(PAL["grass_light"]))])
    tip = nb.mix(nb.math("MULTIPLY_ADD", ti[0], 0.5, 0.25), rgba(lin(PAL["grass_light"])), rgba(lin(PAL["grass_tip"])))
    colr = nb.mix(nb.math("POWER", v, 1.5), body, tip)
    dry_c = nb.mix(v, rgba(lin("#44602c")), rgba(lin("#93ad52")))
    colr = nb.mix(nb.math("MULTIPLY", ti[1], 0.55), colr, dry_c)
    # dark roots = cheap ambient occlusion between the blades
    ao = nb.maprange(v, 0.0, 0.55, 0.30, 1.0, interp="SMOOTHSTEP")
    colr = nb.mix(1.0, colr, nb.comb(ao, ao, ao), blend="MULTIPLY")
    colr = nb.mix(1.0, colr, nb.comb(ti[2], ti[2], ti[2]), blend="MULTIPLY")
    # soft shading: blades borrow most of the ground normal so the hills read as smooth forms
    geo = nb.node("ShaderNodeNewGeometry")
    gn = nb.attr("gnrm", "INSTANCER").outputs["Vector"]
    nrm = nb.vmath("NORMALIZE", nb.vmath("ADD", nb.vmath("SCALE", geo.outputs["Normal"], scale=0.28), nb.vmath("SCALE", gn, scale=0.72)))
    bsdf = nb.node("ShaderNodeBsdfPrincipled")
    nb.set(bsdf.inputs["Base Color"], colr)
    nb.set(bsdf.inputs["Normal"], nrm)
    bsdf.inputs["Roughness"].default_value = 0.70
    bsdf.inputs["Specular IOR Level"].default_value = 0.10
    # moonlight shining through the blades (back light) – tips only
    trans = nb.node("ShaderNodeBsdfTranslucent")
    nb.set(trans.inputs["Color"], nb.mix(0.5, colr, rgba(lin(PAL["grass_tip"]))))
    nb.set(trans.inputs["Normal"], nrm)
    mixs = nb.node("ShaderNodeMixShader")
    nb.set(mixs.inputs[0], nb.math("MULTIPLY", nb.math("POWER", v, 1.2), 0.30))
    nb.l.new(bsdf.outputs[0], mixs.inputs[1])
    nb.l.new(trans.outputs[0], mixs.inputs[2])
    nb.l.new(with_haze(nb, mixs.outputs[0]), out.inputs["Surface"])
    return mat


def simple_material(name, color, rough=0.6, spec=0.3, emission=None, emission_strength=0.0, instancer_color=None,
                    haze=True, soften_normal=0.0):
    """Principled + haze.  instancer_color: name of a per-instance colour attribute that replaces `color`."""
    mat, nb, out = new_material(name)
    bsdf = nb.node("ShaderNodeBsdfPrincipled")
    colr = rgba(color)
    if instancer_color:
        colr = nb.attr(instancer_color, "INSTANCER").outputs["Color"]
    nb.set(bsdf.inputs["Base Color"], colr)
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Specular IOR Level"].default_value = spec
    if emission_strength:
        nb.set(bsdf.inputs["Emission Color"], colr if emission is None else rgba(emission))
        bsdf.inputs["Emission Strength"].default_value = emission_strength
    if soften_normal:
        geo = nb.node("ShaderNodeNewGeometry")
        nrm = nb.vmath("NORMALIZE", nb.vmath("ADD", nb.vmath("SCALE", geo.outputs["Normal"], scale=1.0 - soften_normal),
                                             nb.comb(0.0, -0.25 * soften_normal, soften_normal)))
        nb.set(bsdf.inputs["Normal"], nrm)
    nb.l.new(with_haze(nb, bsdf.outputs[0]) if haze else bsdf.outputs[0], out.inputs["Surface"])
    return mat


# ---------------------------------------------------------------- scatter → Geometry Nodes instancing
def wind_tree(name, coll, amplitude):
    """Geometry Nodes group: instance the children of `coll` on the input points.
    Point attributes: variant (int), rot (euler), scale (vector), sway, wph (+ anything the shaders
    read through the Instancer attribute type).  Wind = small rotation about the root, travelling
    gusts with 3 / 5 / 4 / 11 cycles per loop → loops exactly.  Input "Time" is loop seconds."""
    ng = bpy.data.node_groups.new(name, "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Time", in_out="INPUT", socket_type="NodeSocketFloat")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    nb = NB(ng)
    gin = nb.node("NodeGroupInput")
    gout = nb.node("NodeGroupOutput")
    ci = nb.node("GeometryNodeCollectionInfo")
    ci.inputs["Collection"].default_value = coll
    ci.inputs["Separate Children"].default_value = True
    ci.inputs["Reset Children"].default_value = True

    def named(attr, kind):
        nd = nb.node("GeometryNodeInputNamedAttribute", data_type=kind)
        nd.inputs["Name"].default_value = attr
        return nd.outputs[0]

    ph = nb.math("MULTIPLY", gin.outputs["Time"], TAU / C.LOOP)
    pos = nb.sep(nb.node("GeometryNodeInputPosition").outputs[0])
    wph = named("wph", "FLOAT")
    sway = named("sway", "FLOAT")

    def wave(cycles, kx, ky, off, rnd=0.0):
        a = nb.math("MULTIPLY", ph, float(cycles))
        a = nb.math("SUBTRACT", a, nb.math("ADD", nb.math("MULTIPLY", pos[0], kx), nb.math("MULTIPLY", pos[1], ky)))
        a = nb.math("ADD", a, nb.math("MULTIPLY_ADD", wph, rnd * TAU, off))
        return nb.math("SINE", a)

    gust = nb.math("ADD", nb.math("MULTIPLY", wave(3, 0.42, 0.21, 0.0, 0.05), 0.55),
                   nb.math("ADD", nb.math("MULTIPLY", wave(5, 0.9, -0.5, 1.7, 0.08), 0.30),
                           nb.math("MULTIPLY", wave(11, 0.0, 0.0, 0.0, 1.0), 0.15)))
    ay = nb.math("MULTIPLY", nb.math("MULTIPLY", nb.math("ADD", gust, 0.30), sway), amplitude)
    ax = nb.math("MULTIPLY", nb.math("MULTIPLY", wave(4, 0.5, -0.8, 0.9, 0.3), sway), amplitude * 0.45)
    wind_rot = nb.node("FunctionNodeEulerToRotation")
    nb.l.new(nb.comb(ax, ay, 0.0), wind_rot.inputs[0])
    base_rot = nb.node("FunctionNodeEulerToRotation")
    nb.l.new(named("rot", "FLOAT_VECTOR"), base_rot.inputs[0])
    rr = nb.node("FunctionNodeRotateRotation", rotation_space="GLOBAL")
    nb.l.new(base_rot.outputs[0], rr.inputs[0])
    nb.l.new(wind_rot.outputs[0], rr.inputs[1])

    iop = nb.node("GeometryNodeInstanceOnPoints")
    nb.l.new(gin.outputs["Geometry"], iop.inputs["Points"])
    nb.l.new(ci.outputs[0], iop.inputs["Instance"])
    iop.inputs["Pick Instance"].default_value = True
    nb.l.new(named("variant", "INT"), iop.inputs["Instance Index"])
    nb.l.new(rr.outputs[0], iop.inputs["Rotation"])
    nb.l.new(named("scale", "FLOAT_VECTOR"), iop.inputs["Scale"])
    nb.l.new(iop.outputs[0], gout.inputs[0])
    return ng


def instancer(name, col, variants, pts, amplitude, state, shadow=True):
    """pts: dict with 'pos' (N,3) and per-point attribute arrays.  variants: list of meshes."""
    lib = bpy.data.collections.new(name + "_lib")              # not linked to the scene: source only
    for i, me in enumerate(variants):
        o = bpy.data.objects.new(f"{name}_v{i:02d}", me)
        lib.objects.link(o)
    n = len(pts["pos"])
    me = bpy.data.meshes.new(name)
    me.vertices.add(n)
    me.vertices.foreach_set("co", np.asarray(pts["pos"], dtype=np.float32).ravel())
    for k, v in pts.items():
        if k != "pos":
            set_attr(me, k, v)
    me.update()
    obj = bpy.data.objects.new(name, me)
    col.objects.link(obj)
    mod = obj.modifiers.new("scatter", "NODES")
    mod.node_group = wind_tree(name + "_gn", lib, amplitude)
    tid = next(it.identifier for it in mod.node_group.interface.items_tree if it.name == "Time")
    mod[tid] = 0.0
    obj.visible_shadow = shadow
    state["wind"].append((obj, mod, tid))
    print(f"[env] {name}: {n} instances")
    return obj


def candidates(rng, xs0, xs1, y0, y1, dmax):
    """Uniform random world points (area-uniform) in the fan region xs∈[xs0,xs1], y∈[y0,y1] at
    `dmax` points / m²."""
    k1 = float(kd(y1))
    n = int(dmax * (xs1 - xs0) * (y1 - y0) * k1)
    xs = rng.uniform(xs0, xs1, n)
    y = rng.uniform(y0, y1, n)
    keep = rng.uniform(0, k1, n) < kd(y)
    xs, y = xs[keep], y[keep]
    x, _ = to_world(xs, 0.0, y)
    return x, y


def common_attrs(rng, x, y, z, height, width=None, yaw=None, lean=0.12, sway=1.0):
    n = len(x)
    nrm = terrain.normal(x, y)
    patch, dry = tint_field(x, y)
    patch = np.clip(patch + rng.normal(0, 0.10, n), 0, 1)
    rot = np.stack([rng.normal(0, lean, n) - nrm[:, 1] * 0.5, rng.normal(0, lean, n) + nrm[:, 0] * 0.5,
                    rng.uniform(0, TAU, n) if yaw is None else yaw], axis=1)
    width = height if width is None else width
    return dict(pos=np.stack([x, y, z], axis=1), rot=rot, scale=np.stack([width, width, height], axis=1),
                tint=np.stack([patch, dry * (rng.uniform(0, 1, n) < 0.75), rng.uniform(0.82, 1.12, n), np.ones(n)], axis=1),
                gnrm=nrm, wph=rng.uniform(0, 1, n), sway=np.full(n, sway) * rng.uniform(0.7, 1.3, n))


def build_grass(ctx, col, vis, state, blockers):
    rng = np.random.default_rng(SEED + 10)
    # density multiplier: fewer tufts are made wider by 1/sqrt(q), so the meadow stays covered
    # (draft and small previews do not need blades that are thinner than a pixel)
    q = float(ctx.options.get("grass") or (0.10 if ctx.draft else 1.0))
    wk = min(q ** -0.5, 3.6)
    cast = bool(ctx.options.get("grass_shadow", False))
    gmat = state["grass_mat"]
    near_var = [tuft_mesh(f"VK_TuftN{i}", rng, gmat, int(rng.integers(7, 10)), 4) for i in range(6)]
    mid_var = [tuft_mesh(f"VK_TuftM{i}", rng, gmat, int(rng.integers(6, 9)), 3, spread=0.28, width=0.115) for i in range(5)]

    def blocked(x, y):
        m = np.zeros(len(x), dtype=bool)
        for bx, by, br in blockers:
            m |= np.hypot(x - bx, y - by) < br
        return m

    # ---- stage hills, foreground knolls and the first metres behind (hero grass)
    D0 = 560.0 * q
    x, y = candidates(rng, C.PANO_X0 - 1.0, C.PANO_X1 + 1.0, -20.0, 7.0, D0)
    z = terrain.height(x, y)
    md = mark_dist(x, y)
    mow = 0.42 + 0.58 * smooth(0.55, 1.5, md)                     # short grass around the marks (feet stay visible)
    patch_h = 0.75 + 0.5 * fbm(x * 0.9, y * 0.9, 3, seed=3)
    h = 0.17 * mow * patch_h * rng.uniform(0.7, 1.3, len(x))
    tall = (rng.uniform(0, 1, len(x)) < 0.04) & (md > 1.8)
    h = np.where(tall, h * rng.uniform(1.3, 1.75, len(x)), h)
    keep = vis.test(x, y, z, top=h) & ~blocked(x, y)
    x, y, z, h, md = x[keep], y[keep], z[keep], h[keep], md[keep]
    a = common_attrs(rng, x, y, z, h, width=np.maximum(h, 0.105) * wk, sway=1.0)
    a["variant"] = rng.integers(0, len(near_var), len(x))
    # grass receives shadows but does not cast them by default: 8 M blade triangles in every
    # shadow view of three suns cost far more than they add (the shader darkens the roots itself)
    instancer("VK_GrassNear", col, near_var, a, 0.085, state, shadow=cast)

    # ---- mid-ground: density falls off, tufts grow (same size on screen), simpler tufts
    x, y = candidates(rng, C.PANO_X0 - 1.0, C.PANO_X1 + 1.0, 5.0, 62.0, D0 * 0.80)
    fall = np.exp(-(y - 5.0) / 11.0)
    keep = rng.uniform(0, 1, len(x)) < fall
    x, y = x[keep], y[keep]
    z = terrain.height(x, y)
    grow = 1.0 + (y - 5.0) * 0.022
    patch_h = 0.75 + 0.5 * fbm(x * 0.9 / kd(y), y * 0.9 / kd(y), 3, seed=3)
    h = 0.155 * grow * patch_h * rng.uniform(0.7, 1.3, len(x))
    keep = vis.test(x, y, z, top=h) & ~blocked(x, y) & (hill_u(x, y) > 0.9)
    x, y, z, h = x[keep], y[keep], z[keep], h[keep]
    a = common_attrs(rng, x, y, z, h, width=h * (1.0 + (y - 5.0) * 0.025) * wk, sway=1.0)
    a["variant"] = rng.integers(0, len(mid_var), len(x))
    instancer("VK_GrassMid", col, mid_var, a, 0.085, state, shadow=False)


# ================================================================ flowers, clover, mushrooms
FLOWERS = [
    # name, petals, petal length, petal width, cup, stem height, centre radius, dark centre, palette (sRGB)
    dict(name="daisy", n=12, plen=0.046, pw=0.017, cup=0.12, stem=0.27, cr=0.015, dark=False,
         colors=["#f4f1e6", "#fbf7ee", "#eef0ff"]),
    dict(name="buttercup", n=5, plen=0.032, pw=0.032, cup=0.55, stem=0.22, cr=0.009, dark=False,
         colors=["#ffd23c", "#ffbf2e", "#ffe066"]),
    dict(name="cornflower", n=8, plen=0.036, pw=0.020, cup=0.30, stem=0.25, cr=0.010, dark=False,
         colors=["#7d7cff", "#a48cff", "#6f9bff", "#c0a0ff"]),
    dict(name="cosmos", n=8, plen=0.042, pw=0.026, cup=0.18, stem=0.29, cr=0.012, dark=False,
         colors=["#ff8fc0", "#ffa9d2", "#f76fa8"]),
    dict(name="poppy", n=5, plen=0.055, pw=0.062, cup=0.75, stem=0.32, cr=0.013, dark=True,
         colors=["#ff4a32", "#ff5f3a", "#f23a3a"]),
]


def flower_mesh(name, rng, spec, mats):
    """mats: (stem, petal, yellow centre, dark centre).  Head tilts towards the camera (-Y)."""
    mb = MB()
    sh = spec["stem"]
    ts = np.linspace(0, 1, 5)
    bend = rng.uniform(-0.05, 0.05, 2)
    path = np.stack([bend[0] * ts ** 2, bend[1] * ts ** 2 - 0.02 * ts ** 2, sh * ts], axis=1)
    v, f, uv = tube_part(path, np.linspace(0.0075, 0.005, 5), sides=4)
    uv[:, 1] = 0.45 + 0.3 * uv[:, 1]
    mb.add(v, f, uv, 0)
    for _ in range(2):
        v, f, uv = blade_part(rng, (0, 0, 0), sh * rng.uniform(0.35, 0.6), 0.04, rng.uniform(0.3, 0.6), rng.uniform(0.3, 0.6), 3)
        mb.add(v, f, uv, 0)
    tilt = rng.uniform(0.7, 1.05)
    az = rng.uniform(-0.5, 0.5)
    axis = np.array([math.sin(tilt) * math.sin(az), -math.sin(tilt) * math.cos(az), math.cos(tilt)])
    e1 = np.cross(axis, [0.0, 0.0, 1.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    head = path[-1]
    L, W, cup = spec["plen"], spec["pw"], spec["cup"]
    a0 = rng.uniform(0, TAU)
    for p in range(spec["n"]):
        a = a0 + TAU * p / spec["n"] + rng.uniform(-0.08, 0.08)
        dr = math.cos(a) * e1 + math.sin(a) * e2
        pp = -math.sin(a) * e1 + math.cos(a) * e2
        ll = L * rng.uniform(0.88, 1.08)
        pts, uvs = [], []
        for r, w in ((0.0, 0.25), (0.45, 1.0), (0.82, 0.8), (1.0, 0.0)):
            rr = spec["cr"] * 0.5 + r * ll
            lift = cup * rr - 0.25 * cup * rr * rr / ll
            c = head + dr * rr + axis * lift
            if w > 0.0:
                pts += [c - pp * W * 0.5 * w, c + pp * W * 0.5 * w]
                uvs += [(0.0, r), (1.0, r)]
            else:
                pts.append(c)
                uvs.append((0.5, 1.0))
        mb.add(pts, [(0, 1, 3, 2), (2, 3, 5, 4), (4, 5, 6)], uvs, 1)
    cr = spec["cr"]
    v, f, uv = lathe_part([(cr, 0.0), (cr * 0.85, cr * 0.45), (cr * 0.45, cr * 0.75)], sides=8)
    v = head[None, :] + v[:, 0:1] * e1[None, :] + v[:, 1:2] * e2[None, :] + v[:, 2:3] * axis[None, :]
    mb.add(v, f, uv, 3 if spec["dark"] else 2)
    return mb.mesh(name, list(mats))


def clover_mesh(name, rng, mat):
    mb = MB()
    for _ in range(int(rng.integers(6, 10))):
        a = rng.uniform(0, TAU)
        r = 0.9 * math.sqrt(rng.uniform(0, 1))
        base = np.array([r * math.cos(a), r * math.sin(a), 0.0])
        h = rng.uniform(0.55, 1.0)
        top = base + np.array([rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2), h])
        v, f, uv = tube_part(np.stack([base, top]), 0.03, sides=3)
        uv[:, 1] = 0.5
        mb.add(v, f, uv)
        a0 = rng.uniform(0, TAU)
        for kk in range(3):
            b = a0 + TAU * kk / 3
            dr = np.array([math.cos(b), math.sin(b), 0.12])
            pp = np.array([-math.sin(b), math.cos(b), 0.0])
            ll, ww = 0.34, 0.34
            pts = [top, top + dr * ll * 0.5 - pp * ww * 0.5, top + dr * ll * 0.5 + pp * ww * 0.5,
                   top + dr * ll - pp * ww * 0.32, top + dr * ll + pp * ww * 0.32]
            mb.add(pts, [(0, 1, 2), (1, 3, 4, 2)], [(0.5, 0.62), (0, 0.8), (1, 0.8), (0, 0.95), (1, 0.95)])
    return mb.mesh(name, [mat])


def mushroom_mesh(name, rng, mats, count):
    mb = MB()
    for i in range(count):
        s = rng.uniform(0.7, 1.15) * (1.0 if i == 0 else 0.7)
        off = np.array([0.0, 0.0, 0.0]) if i == 0 else np.array([rng.uniform(0.07, 0.11) * (1 if i == 1 else -1), rng.uniform(-0.05, 0.03), 0.0])
        lean = np.array([rng.uniform(-0.15, 0.15), rng.uniform(-0.15, 0.05)])
        v, f, uv = lathe_part([(0.024, -0.02), (0.019, 0.03), (0.015, 0.075), (0.017, 0.092)], sides=8, cap_top=False)
        v2, f2, uv2 = lathe_part([(0.02, 0.086), (0.064, 0.080), (0.070, 0.094), (0.060, 0.118), (0.040, 0.138), (0.016, 0.149)], sides=14)
        for vv, ff, uu, m in ((v, f, uv, 1), (v2, f2, uv2, 0)):
            vv = vv * s
            vv[:, 0] += lean[0] * vv[:, 2]
            vv[:, 1] += lean[1] * vv[:, 2]
            mb.add(vv + off, ff, uu, m)
    return mb.mesh(name, list(mats))


def mushroom_cap_material():
    mat, nb, out = new_material("VK_MushroomCap")
    tc = nb.node("ShaderNodeTexCoord").outputs["Object"]
    vor = nb.node("ShaderNodeTexVoronoi", feature="F1")
    nb.set(vor.inputs["Vector"], tc)
    vor.inputs["Scale"].default_value = 26.0
    dots = nb.maprange(vor.outputs["Distance"], 0.20, 0.27, 1.0, 0.0)
    colr = nb.mix(dots, rgba(lin("#d2432b")), rgba(lin("#f6ead2")))
    bsdf = nb.node("ShaderNodeBsdfPrincipled")
    nb.set(bsdf.inputs["Base Color"], colr)
    bsdf.inputs["Roughness"].default_value = 0.38
    bsdf.inputs["Specular IOR Level"].default_value = 0.5
    nb.set(bsdf.inputs["Emission Color"], colr)
    bsdf.inputs["Emission Strength"].default_value = 0.05
    nb.l.new(with_haze(nb, bsdf.outputs[0]), out.inputs["Surface"])
    return mat


def cluster_points(rng, vis, xs_range, y_range, cluster_density, members, spread, min_mark=0.95, blockers=(), on_hills=None):
    """Poisson-cluster process: → x, y, cluster id (only visible points outside the clear zones)."""
    cx, cy = candidates(rng, xs_range[0], xs_range[1], y_range[0], y_range[1], cluster_density)
    if on_hills is not None:
        hu = hill_u(cx, cy)
        keep = (hu < 1.15) if on_hills else (hu > 1.0)
        cx, cy = cx[keep], cy[keep]
    keep = vis.test(cx, cy, terrain.height(cx, cy), top=0.3)
    cx, cy = cx[keep], cy[keep]
    xs, ys, ids = [], [], []
    for i in range(len(cx)):
        n = int(rng.integers(members[0], members[1] + 1))
        sp = spread * float(kd(cy[i])) * rng.uniform(0.7, 1.3)
        xs.append(cx[i] + rng.normal(0, sp, n))
        ys.append(cy[i] + rng.normal(0, sp * 1.4, n))
        ids.append(np.full(n, i))
    if not xs:
        return np.zeros(0), np.zeros(0), np.zeros(0, dtype=int)
    x, y, ids = np.concatenate(xs), np.concatenate(ys), np.concatenate(ids)
    keep = (mark_dist(x, y) > min_mark) & vis.test(x, y, terrain.height(x, y), top=0.25)
    for bx, by, br in blockers:
        keep &= np.hypot(x - bx, y - by) > br
    return x[keep], y[keep], ids[keep]


def build_flowers(ctx, col, vis, state, blockers):
    rng = np.random.default_rng(SEED + 20)
    q = 0.5 if ctx.draft else 1.0
    stem = state["grass_mat"]
    petal = simple_material("VK_Petal", (1, 1, 1), rough=0.5, spec=0.2, emission_strength=0.05, instancer_color="pcol", soften_normal=0.6)
    cen_y = simple_material("VK_FlowerCentre", lin("#ffc21a"), rough=0.6, emission_strength=0.06)
    cen_d = simple_material("VK_FlowerCentreDark", lin("#1c1420"), rough=0.6)
    variants, vspec = [], []
    for si, spec in enumerate(FLOWERS):
        for j in range(3):
            variants.append(flower_mesh(f"VK_Flower_{spec['name']}{j}", rng, spec, (stem, petal, cen_y, cen_d)))
            vspec.append(si)
    vspec = np.array(vspec)
    parts = []
    # (xs range, y range, cluster density /m², members, spread, size, on_hills, flower-kind weights)
    zones = [
        ((C.PANO_X0, C.PANO_X1), (-5.5, 6.5), 0.26 * q, (5, 13), 0.30, 1.0, True, (3, 2.5, 2.5, 1.5, 0.8)),
        ((C.PANO_X0, C.PANO_X1), (-16.0, -6.0), 0.22 * q, (7, 15), 0.34, 1.15, None, (2.5, 2, 2, 2, 1.2)),
        ((C.PANO_X0 - 1, C.PANO_X1 + 1), (6.5, 20.0), 0.040 * q, (12, 30), 0.55, 1.0, False, (3, 2.5, 2, 1.5, 1.5)),
        ((C.PANO_X0 - 1, C.PANO_X1 + 1), (20.0, 44.0), 0.012 * q, (20, 46), 0.80, 1.2, False, (3, 3, 2, 1, 2)),
    ]
    for xr, yr, dens, members, spread, size, on_hills, weights in zones:
        x, y, ids = cluster_points(rng, vis, xr, yr, dens, members, spread, blockers=blockers, on_hills=on_hills)
        if not len(x):
            continue
        w = np.array(weights, dtype=float)
        kind_of_cluster = rng.choice(len(FLOWERS), size=int(ids.max()) + 1, p=w / w.sum())
        kind = kind_of_cluster[ids]
        stray = rng.uniform(0, 1, len(x)) < 0.12                     # a few strangers in every patch
        kind = np.where(stray, rng.integers(0, len(FLOWERS), len(x)), kind)
        z = terrain.height(x, y)
        s = size * rng.uniform(0.75, 1.3, len(x)) * (1.0 + np.clip(y - 5.0, 0, None) * 0.022)
        a = common_attrs(rng, x, y, z, s, yaw=rng.uniform(-0.6, 0.6, len(x)), lean=0.10, sway=1.0)
        a["variant"] = kind * 3 + rng.integers(0, 3, len(x))
        pc = np.zeros((len(x), 4))
        for i in range(len(x)):
            cols = FLOWERS[kind[i]]["colors"]
            pc[i, :3] = lin(cols[int(rng.integers(0, len(cols)))])
        pc[:, :3] *= rng.uniform(0.85, 1.05, (len(x), 1))
        pc[:, 3] = 1.0
        a["pcol"] = pc
        parts.append(a)
    pts = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    instancer("VK_Flowers", col, variants, pts, 0.11, state)

    # clover carpets on the hills (also inside the clear zone: it is only ankle-high)
    rng = np.random.default_rng(SEED + 21)
    cl = [clover_mesh(f"VK_Clover{i}", rng, stem) for i in range(4)]
    x, y = candidates(rng, C.PANO_X0, C.PANO_X1, -5.0, 6.0, 90.0 * q)
    mask = smooth(0.56, 0.66, fbm(x * 0.8, y * 0.8, 3, seed=77)) * (hill_u(x, y) < 1.1)
    keep = rng.uniform(0, 1, len(x)) < mask
    x, y = x[keep], y[keep]
    z = terrain.height(x, y)
    keep = vis.test(x, y, z, top=0.1)
    for bx, by, br in blockers:
        keep &= np.hypot(x - bx, y - by) > br
    x, y, z = x[keep], y[keep], z[keep]
    a = common_attrs(rng, x, y, z, rng.uniform(0.055, 0.085, len(x)), lean=0.05, sway=0.5)
    a["tint"][:, 0] = np.clip(a["tint"][:, 0] * 0.5 + 0.2, 0, 1)
    a["tint"][:, 1] = 0.0
    a["variant"] = rng.integers(0, len(cl), len(x))
    instancer("VK_Clover", col, cl, a, 0.05, state, shadow=False)


def build_mushrooms(ctx, col, vis, state, spots):
    rng = np.random.default_rng(SEED + 30)
    cap = mushroom_cap_material()
    stem = simple_material("VK_MushroomStem", lin("#eadfc6"), rough=0.6, emission_strength=0.03)
    variants = [mushroom_mesh(f"VK_Mushroom{i}", rng, (cap, stem), c) for i, c in enumerate((1, 2, 3, 3))]
    xs, ys, sc = [], [], []
    for x, y, s in spots:
        n = int(rng.integers(1, 3))
        for j in range(n):
            xs.append(x + rng.normal(0, 0.16) * j)
            ys.append(y + rng.normal(0, 0.2) * j)
            sc.append(s * rng.uniform(0.85, 1.25) * (1.0 if j == 0 else 0.75))
    x, y, sc = np.array(xs), np.array(ys), np.array(sc)
    z = terrain.height(x, y)
    a = common_attrs(rng, x, y, z, sc, yaw=rng.uniform(0, TAU, len(x)), lean=0.04, sway=0.0)
    a["variant"] = rng.integers(0, len(variants), len(x))
    instancer("VK_Mushrooms", col, variants, a, 0.0, state)


# ================================================================ molehills and stones (conform to the terrain)
def soil_material():
    mat, nb, out = new_material("VK_Soil")
    geo = nb.node("ShaderNodeNewGeometry")
    n1 = nb.noise(geo.outputs["Position"], 22.0, 3.0, 0.6)
    n2 = nb.noise(geo.outputs["Position"], 90.0, 2.0, 0.5)
    colr = nb.mix(n1, rgba(lin("#2a1c14")), rgba(lin("#5a4130")))
    colr = nb.mix(nb.math("MULTIPLY", n2, 0.35), colr, rgba(lin("#7a6048")))
    bump = nb.node("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.6
    bump.inputs["Distance"].default_value = 0.02
    nb.l.new(n2, bump.inputs["Height"])
    bsdf = nb.node("ShaderNodeBsdfPrincipled")
    nb.set(bsdf.inputs["Base Color"], colr)
    bsdf.inputs["Roughness"].default_value = 0.9
    bsdf.inputs["Specular IOR Level"].default_value = 0.25
    nb.l.new(bump.outputs[0], bsdf.inputs["Normal"])
    nb.l.new(with_haze(nb, bsdf.outputs[0]), out.inputs["Surface"])
    return mat


def stone_material():
    mat, nb, out = new_material("VK_Stone")
    geo = nb.node("ShaderNodeNewGeometry")
    n1 = nb.noise(geo.outputs["Position"], 9.0, 4.0, 0.6)
    colr = nb.mix(n1, rgba(lin("#4a5068")), rgba(lin("#8a90a6")))
    up = nb.sep(geo.outputs["Normal"])[2]
    moss = nb.math("MULTIPLY", nb.math("SMOOTHSTEP", 0.35, 0.8, nb.math("ADD", up, nb.math("MULTIPLY_ADD", n1, 0.6, -0.3))), 0.85)
    colr = nb.mix(moss, colr, rgba(lin(PAL["grass_mid"])))
    bsdf = nb.node("ShaderNodeBsdfPrincipled")
    nb.set(bsdf.inputs["Base Color"], colr)
    bsdf.inputs["Roughness"].default_value = 0.7
    nb.l.new(with_haze(nb, bsdf.outputs[0]), out.inputs["Surface"])
    return mat


def build_molehills(ctx, col, spots):
    """spots: [(x, y, radius, height)] world."""
    rng = np.random.default_rng(SEED + 40)
    soil = soil_material()
    ico, icof = icosphere(1)
    for i, (mx, my, R, H) in enumerate(spots):
        mb = MB()
        rings, sect = 12, 28
        rr = np.linspace(0, 1, rings + 1)[1:]
        ang = np.linspace(0, TAU, sect, endpoint=False)
        RR, AA = np.meshgrid(rr, ang, indexing="ij")
        edge = 1.0 + 0.12 * np.sin(3 * AA + i) + 0.07 * np.sin(7 * AA + 2.0 * i)
        x = mx + RR * R * edge * np.cos(AA)
        y = my + RR * R * edge * np.sin(AA)
        prof = (1.0 - smooth(0.0, 1.0, RR)) ** 1.15
        lumps = 0.10 * np.sin(5 * AA + 9 * RR + i) * np.sin(4 * RR * math.pi) + rng.normal(0, 0.025, RR.shape)
        z = terrain.height(x, y) + H * prof * (1.0 + lumps) - 0.012 * (1.0 - prof)
        verts = np.vstack([[mx, my, terrain.height1(mx, my) + H * 0.97], np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)])
        faces = [(0, 1 + j, 1 + (j + 1) % sect) for j in range(sect)]
        for r in range(rings - 1):
            for j in range(sect):
                a, b = 1 + r * sect + j, 1 + r * sect + (j + 1) % sect
                faces.append((a, a + sect, b + sect, b))
        mb.add(verts, faces)
        for _ in range(int(rng.integers(9, 15))):                # loose clods
            a, d = rng.uniform(0, TAU), R * rng.uniform(0.25, 1.25)
            cx, cy = mx + d * math.cos(a), my + d * math.sin(a)
            pr = float((1.0 - smooth(0.0, 1.0, min(d / R, 1.0))) ** 1.15)
            cz = terrain.height1(cx, cy) + H * pr
            r0 = rng.uniform(0.018, 0.05) * R / 0.33
            v = ico * (r0 * (1.0 + 0.25 * wobble(ico, rng.uniform(0, 9), 3.0)))[:, None] * np.array([1.0, 1.0, 0.75])
            mb.add(v + np.array([cx, cy, cz + r0 * 0.3]), icof)
        me = mb.mesh(f"VK_Molehill{i}", [soil])
        col.objects.link(bpy.data.objects.new(f"VK_Molehill{i}", me))


def build_stones(ctx, col, spots):
    rng = np.random.default_rng(SEED + 50)
    mat = stone_material()
    ico, icof = icosphere(2)
    for i, (sx, sy, r) in enumerate(spots):
        mb = MB()
        for j in range(int(rng.integers(1, 4))):
            rj = r * (1.0 if j == 0 else rng.uniform(0.35, 0.6))
            ox, oy = (0.0, 0.0) if j == 0 else (rng.uniform(-1, 1) * r * 1.5, rng.uniform(-0.8, 0.3) * r)
            px, py = sx + ox, sy + oy
            v = ico * (1.0 + 0.16 * wobble(ico, rng.uniform(0, 20), 1.7))[:, None] * np.array([rng.uniform(1.0, 1.4), rng.uniform(0.8, 1.1), rng.uniform(0.55, 0.8)]) * rj
            mb.add(v + np.array([px, py, terrain.height1(px, py) + rj * 0.22]), icof)
        me = mb.mesh(f"VK_Stone{i}", [mat])
        col.objects.link(bpy.data.objects.new(f"VK_Stone{i}", me))


# ================================================================ bushes and trees ("puffy" low-poly canopies)
def foliage_material():
    mat, nb, out = new_material("VK_Foliage")
    geo = nb.node("ShaderNodeNewGeometry")
    tc = nb.node("ShaderNodeTexCoord").outputs["Object"]
    info = nb.node("ShaderNodeObjectInfo")
    rnd = info.outputs["Random"]
    hz = nb.sep(tc)[2]                                             # unit meshes: 0 = ground, 1 = top
    n1 = nb.noise(nb.vmath("ADD", tc, nb.comb(rnd, rnd, rnd)), 3.2, 3.0, 0.6)
    t = nb.math("ADD", nb.math("MULTIPLY", hz, 0.55), nb.math("ADD", nb.math("MULTIPLY", n1, 0.5), nb.math("MULTIPLY_ADD", rnd, 0.3, -0.25)), clamp=True)
    colr = nb.ramp(t, [(0.0, lin("#082a22")), (0.45, lin("#14502c")), (0.8, lin("#2c7434")), (1.0, lin("#5a9a3e"))])
    # soft leafy clumps (bump) + fine leaf grain (colour only): no cell edges, the crowns must stay round
    clump = nb.noise(nb.vmath("ADD", tc, nb.comb(rnd, 0.0, rnd)), 7.0, 3.0, 0.55)
    grain = nb.noise(tc, 42.0, 2.0, 0.6)
    bump = nb.node("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.30
    bump.inputs["Distance"].default_value = 0.06
    nb.l.new(clump, bump.inputs["Height"])
    leaf = nb.math("MULTIPLY", nb.maprange(clump, 0.3, 0.7, 0.82, 1.12), nb.maprange(grain, 0.3, 0.7, 0.90, 1.10))
    colr = nb.mix(1.0, colr, nb.comb(leaf, leaf, leaf), blend="MULTIPLY")
    bsdf = nb.node("ShaderNodeBsdfPrincipled")
    nb.set(bsdf.inputs["Base Color"], colr)
    bsdf.inputs["Roughness"].default_value = 0.7
    bsdf.inputs["Specular IOR Level"].default_value = 0.15
    nb.l.new(bump.outputs[0], bsdf.inputs["Normal"])
    nb.l.new(with_haze(nb, bsdf.outputs[0]), out.inputs["Surface"])
    return mat


def puffy_mesh(name, rng, kind, mats, subdiv=2):
    """Unit-height plant (z 0..1): 'bush' | 'round' | 'poplar' | 'big'.  mats = (foliage, bark)."""
    ico, icof = icosphere(subdiv)
    blobs = []          # (centre, radius xyz)
    trunk = None
    if kind == "bush":
        n = int(rng.integers(5, 8))
        for i in range(n):
            a = rng.uniform(0, TAU)
            d = rng.uniform(0.0, 0.42) if i else 0.0
            r = rng.uniform(0.26, 0.40) if i else 0.46
            blobs.append((np.array([d * math.cos(a) * 1.3, d * math.sin(a) * 0.8, r * 0.75 + rng.uniform(0, 0.12) * (i > 0)]), np.array([r, r, r * 0.95])))
    elif kind in ("round", "big"):
        n = 7 if kind == "round" else 26
        trunk = (0.40, 0.055 if kind == "round" else 0.07)
        blobs.append((np.array([0.0, 0.0, 0.66]), np.array([0.30, 0.30, 0.27])))
        for i in range(n - 1):
            a = TAU * i / (n - 1) * (1.0 if kind == "round" else 2.618) + rng.uniform(-0.4, 0.4)
            d = rng.uniform(0.17, 0.27) if kind == "round" else rng.uniform(0.10, 0.33)
            r = rng.uniform(0.15, 0.23) if kind == "round" else rng.uniform(0.12, 0.19)
            zc = rng.uniform(0.50, 0.80) if kind == "round" else 0.66 + rng.uniform(-0.20, 0.20) * (1.0 - 0.5 * d / 0.33)
            blobs.append((np.array([d * math.cos(a), d * math.sin(a) * 0.9, zc]), np.array([r, r, r * 0.92])))
    else:  # poplar
        trunk = (0.20, 0.035)
        zs_ = [0.26, 0.42, 0.58, 0.73, 0.86, 0.94]
        rs_ = [0.105, 0.135, 0.13, 0.105, 0.075, 0.045]
        for z_, r_ in zip(zs_, rs_):
            blobs.append((np.array([rng.uniform(-0.02, 0.02), rng.uniform(-0.02, 0.02), z_]), np.array([r_, r_, r_ * 1.35])))
    mb = MB()
    normals = []
    top = max(c[2] + r[2] for c, r in blobs)
    cen = np.array([0.0, 0.0, 0.42 if kind != "bush" else 0.1])
    for bi, (c, r) in enumerate(blobs):
        d = 1.0 + 0.13 * wobble(ico, rng.uniform(0, 30), 2.6)
        v = c + ico * r * d[:, None]
        v[:, 2] /= top
        mb.add(v, icof, mat=0)
        out_dir = v - cen
        out_dir /= np.linalg.norm(out_dir, axis=1, keepdims=True)
        nn = 0.45 * ico + 0.55 * out_dir
        normals.append(nn / np.linalg.norm(nn, axis=1, keepdims=True))
    if trunk:
        th, tr = trunk
        ts = np.linspace(0, 1, 5)
        path = np.stack([0.03 * np.sin(ts * 2.5 + rng.uniform(0, 3)), 0.02 * np.sin(ts * 2.0), ts * (th + 0.12) / top], axis=1)
        v, f, uv = tube_part(path, tr * (1.25 - 0.55 * ts), sides=7)
        mb.add(v, f, uv, mat=1)
        nn = v - path.repeat(7, axis=0)
        nn[:, 2] = 0.0
        normals.append(nn / np.maximum(np.linalg.norm(nn, axis=1, keepdims=True), 1e-6))
    me = mb.mesh(name, list(mats))
    me.normals_split_custom_set_from_vertices([tuple(n) for n in np.vstack(normals)])
    return me


def tree_layout(rng):
    """→ [(xs, y, height on the stage plane, kind)] – everything is placed in picture terms."""
    L = []
    # bushes on the swell right behind the stage hills
    L += [(-13.7, 11.0, 0.40, "bush"), (-12.8, 11.8, 0.28, "bush"), (-7.3, 12.5, 0.34, "bush"), (6.2, 11.0, 0.38, "bush"),
          (7.1, 12.0, 0.26, "bush"), (13.5, 12.5, 0.36, "bush"), (-27.4, 11.0, 0.42, "bush"), (-16.4, 13.0, 0.30, "bush"),
          (16.0, 12.0, 0.30, "bush"), (25.2, 10.5, 0.34, "bush"), (-4.6, 14.0, 0.24, "bush"), (4.1, 15.0, 0.22, "bush")]
    # small trees on the rolling hills
    L += [(-14.9, 31.0, 0.80, "round"), (-9.4, 34.0, 0.60, "round"), (9.3, 30.0, 0.74, "round"), (15.4, 33.0, 0.60, "round"),
          (-27.7, 30.0, 0.72, "round"), (24.7, 31.0, 0.70, "round"), (-5.2, 60.0, 0.44, "round"), (4.7, 58.0, 0.46, "round"),
          (-24.3, 62.0, 0.52, "round"), (-17.2, 60.0, 0.42, "round"), (17.3, 63.0, 0.44, "round"), (11.7, 62.0, 0.40, "round"),
          (-11.9, 64.0, 0.38, "round"), (-15.6, 32.5, 0.34, "bush"), (-8.7, 35.0, 0.26, "bush"), (10.1, 31.0, 0.30, "bush"),
          (23.9, 32.0, 0.30, "bush"), (-26.8, 31.0, 0.30, "bush"), (-13.0, 58.0, 0.62, "poplar"), (-12.6, 59.5, 0.54, "poplar"),
          (13.6, 60.0, 0.60, "poplar"), (14.0, 61.0, 0.50, "poplar"), (-3.6, 61.0, 0.22, "bush"), (5.6, 59.0, 0.24, "bush")]
    # the hero tree that frames the right end of the panorama (the moon balances it on the left)
    L += [(26.35, 17.0, 2.05, "big"), (25.2, 18.5, 0.42, "bush"), (27.4, 16.0, 0.36, "bush")]
    # groves on the far hills: presence driven by noise, clear sky kept right behind the three characters
    for y0, step, hmin, hmax, pres, seed in ((118.0, 0.30, 0.20, 0.36, 0.52, 3), (215.0, 0.17, 0.10, 0.20, 0.47, 8)):
        xs = C.PANO_X0 - 1.0
        while xs < C.PANO_X1 + 1.0:
            xs += step * rng.uniform(0.6, 1.5)
            if fbm(np.array([xs * 0.33]), np.array([seed * 3.1]), 3, seed=seed)[0] < pres:
                continue
            if y0 < 150.0 and min(abs(xs - m[0]) for m in C.MARKS.values()) < 1.2:
                continue
            kind = "poplar" if rng.uniform() < 0.22 else "round"
            h = rng.uniform(hmin, hmax) * (1.55 if kind == "poplar" else 1.0)
            L.append((xs, y0 + rng.uniform(-0.12, 0.12) * y0, h, kind))
    return L


def build_trees(ctx, col, vis):
    rng = np.random.default_rng(SEED + 60)
    fol = foliage_material()
    bark = simple_material("VK_Bark", lin("#2a1d18"), rough=0.85, spec=0.2)
    lib = {k: [puffy_mesh(f"VK_{k}{i}", rng, k, (fol, bark), 5 if k == "big" else 3) for i in range(n)]
           for k, n in (("bush", 4), ("round", 5), ("poplar", 3), ("big", 1))}
    objs = []
    for i, (xs, y, hs, kind) in enumerate(tree_layout(rng)):
        k = float(kd(y))
        x = xs * k
        z = terrain.height1(x, y)
        if not vis.test(np.array([x]), np.array([y]), np.array([z]), top=hs * k)[0]:
            continue
        me = lib[kind][int(rng.integers(0, len(lib[kind])))]
        o = bpy.data.objects.new(f"VK_Plant{i:03d}", me)
        h = hs * k
        w = h * rng.uniform(0.9, 1.15) * (1.25 if kind == "bush" else 1.0)
        o.location = (x, y, z - 0.03 * h)
        o.scale = (w, w, h)
        o.rotation_euler = (0.0, 0.0, rng.uniform(0, TAU))
        o.visible_shadow = y < 80.0
        col.objects.link(o)
        objs.append(o)
    print(f"[env] plants: {len(objs)}")
    return objs


# ================================================================ fireflies
FIREFLIES = dict(count=150, strength=12.0, size=0.13)


def build_fireflies(ctx, col, vis):
    rng = np.random.default_rng(SEED + 70)
    n_want = FIREFLIES["count"]
    xs = rng.uniform(C.PANO_X0 + 0.3, C.PANO_X1 - 0.3, n_want * 6)
    y = rng.uniform(-4.0, 26.0, n_want * 6) ** 1.0
    x, _ = to_world(xs, 0.0, y)
    hgt = rng.uniform(0.18, 0.95, len(x)) * kd(y)
    z = terrain.height(x, y) + hgt
    _, zs = to_stage(x, y, z)
    keep = vis.test(x, y, z - hgt, top=hgt + 0.05) & (zs > C.FRAME_Z0 + 0.12) & (zs < 0.95)
    # never in front of a character (faces / costumes stay clean) and not right on the screen seams
    for mx, my in C.MARKS.values():
        keep &= ~((np.abs(xs - mx) < 1.9) & (y < 6.0))
    x, y, z = x[keep][:n_want], y[keep][:n_want], z[keep][:n_want]
    n = len(x)
    st = dict(
        n=n, p0=np.stack([x, y, z], axis=1), k=kd(y),
        amp=np.stack([rng.uniform(0.25, 0.7, n), rng.uniform(0.2, 0.5, n), rng.uniform(0.07, 0.2, n)], axis=1) * kd(y)[:, None],
        k1=np.stack([rng.integers(1, 4, n), rng.integers(1, 4, n), rng.integers(2, 6, n)], axis=1).astype(float),
        ph1=rng.uniform(0, TAU, (n, 3)), ph2=rng.uniform(0, TAU, (n, 3)),
        kb=rng.integers(3, 8, n).astype(float), phb=rng.uniform(0, TAU, n),
        kc=rng.integers(1, 3, n).astype(float), phc=rng.uniform(0, TAU, n),
        size=FIREFLIES["size"] * rng.uniform(0.75, 1.25, n) * kd(y),
    )
    obj = quads_mesh("VK_Fireflies", st["p0"], st["size"], st["size"], col, attrs={"fb": np.ones(n)})
    mat, nb, out = new_material("VK_FireflyMat", blended=True)
    _u, _v, r2 = card_uv(nb)
    core = nb.math("EXPONENT", nb.math("MULTIPLY", r2, -90.0))
    halo = nb.math("MULTIPLY", nb.math("EXPONENT", nb.math("MULTIPLY", nb.math("SQRT", r2), -5.5)), 0.16)
    shape = nb.math("MULTIPLY", nb.math("ADD", core, halo), nb.math("SUBTRACT", 1.0, nb.math("SMOOTHSTEP", 0.6, 1.0, nb.math("SQRT", r2))))
    colr = nb.mix(core, rgba((0.55, 1.0, 0.25)), rgba((1.0, 1.0, 0.65)))
    additive(nb, out, colr, nb.math("MULTIPLY", nb.math("MULTIPLY", shape, nb.attr("fb").outputs["Fac"]), FIREFLIES["strength"]))
    obj.data.materials.append(mat)
    st["obj"] = obj
    return st


def update_fireflies(st, t):
    w = TAU * t / C.LOOP
    p = st["p0"] + st["amp"] * (np.sin(st["k1"] * w + st["ph1"]) + 0.35 * np.sin((st["k1"] + 2.0) * w + st["ph2"]))
    blink = np.clip(np.sin(st["kb"] * w + st["phb"]), 0.0, 1.0) ** 2 * (0.65 + 0.35 * np.sin(st["kc"] * w + st["phc"]))
    cx = np.array([-1.0, 1.0, 1.0, -1.0])
    cz = np.array([-1.0, -1.0, 1.0, 1.0])
    v = np.repeat(p, 4, axis=0)
    v[:, 0] += (st["size"][:, None] * cx[None, :]).ravel()
    v[:, 2] += (st["size"][:, None] * cz[None, :]).ravel()
    me = st["obj"].data
    me.vertices.foreach_set("co", v.astype(np.float32).ravel())
    me.attributes["fb"].data.foreach_set("value", np.repeat(blink, 4).astype(np.float32))
    me.update()


# ================================================================ module entry points
PARTS = None        # None = everything; or a set of names (look-dev: build only some layers)

# props in world coordinates: molehills (x, y, radius, height) relative to a mark, stones, mushrooms
_E, _S, _X = C.MARKS["emo"][0], C.MARKS["singer"][0], C.MARKS["sax"][0]
MOLEHILLS = [(_E + 1.85, -1.45, 0.36, 0.20), (_E - 2.35, -0.3, 0.28, 0.15), (_S - 2.25, -1.35, 0.38, 0.21),
             (_S + 2.75, -0.55, 0.30, 0.16), (_X - 1.95, -1.25, 0.36, 0.20), (_X + 2.4, 0.2, 0.27, 0.15),
             (-10.3, 9.4, 0.50, 0.26), (-8.9, 10.3, 0.36, 0.18), (11.2, 9.6, 0.48, 0.25)]
STONES = [(_E - 1.5, -2.3, 0.17), (_E + 2.9, 0.5, 0.13), (_S + 1.7, -2.5, 0.19), (_S - 3.2, 0.1, 0.14),
          (_X + 1.4, -2.4, 0.17), (_X - 2.9, 0.4, 0.13), (-12.2, 9.0, 0.30), (8.6, 9.2, 0.26), (14.6, 9.8, 0.22),
          (-6.2, 10.0, 0.20)]
MUSHROOMS = [(_E - 1.25, -2.0, 1.0), (_E + 1.25, -2.55, 0.9), (_E + 2.55, -0.9, 0.8), (_S - 1.45, -2.35, 1.0),
             (_S + 1.3, -2.15, 0.9), (_S + 2.2, -1.5, 0.85), (_S - 2.8, -0.6, 0.8), (_X - 1.2, -2.3, 1.0),
             (_X + 1.75, -2.0, 0.9), (_X - 2.6, -0.5, 0.8), (-11.9, 9.2, 1.5), (8.3, 9.4, 1.4), (-9.6, 9.0, 1.3)]


def build(ctx):
    col = ctx.collection("Env")
    state = dict(wind=[], fireflies=None)
    parts = ctx.options.get("env") or PARTS
    want = (lambda name: parts is None or name in parts)
    vis = Visibility()
    state["ground"] = build_ground(ctx, col)
    if want("sky"):
        build_sky(ctx, col)
        build_stars(ctx, col)
        build_moon(ctx, col)
        build_mountains(ctx, col)
        build_mist(ctx, col)
    state["grass_mat"] = grass_material()
    blockers = [(x, y, r * 0.9) for x, y, r, _h in MOLEHILLS] + [(x, y, r * 0.9) for x, y, r in STONES]
    if want("props"):
        build_molehills(ctx, col, MOLEHILLS)
        build_stones(ctx, col, STONES)
        build_mushrooms(ctx, col, vis, state, MUSHROOMS)
    if want("grass"):
        build_grass(ctx, col, vis, state, blockers)
    if want("flowers"):
        build_flowers(ctx, col, vis, state, blockers)
    if want("trees"):
        build_trees(ctx, col, vis)
    if want("fireflies"):
        state["fireflies"] = build_fireflies(ctx, col, vis)
    return state


def update(state, ctx, t):
    """Pose the environment for loop time t (pure function of t, period = config.LOOP)."""
    t = float(t) % C.LOOP
    for obj, mod, tid in state["wind"]:
        mod[tid] = t
        obj.update_tag()
    if state.get("fireflies"):
        update_fireflies(state["fireflies"], t)
