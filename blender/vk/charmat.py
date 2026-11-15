"""Procedural materials for the characters (shader nodes only: no image textures, no screen-space
effects).  Everything that animates reads the loop `Phase` from util.time_node, so it loops.

Conventions shared with the modelling code (vk.meshkit):
  attribute 'rest'   float vector, point domain: rest position in character space (metres).  All
                     procedural patterns are evaluated in rest space so they stick to the cloth.
  attribute 'paint'  colour: three scalar fields (R, G, B).  A field is a smooth signed quantity
                     with its boundary at 0.5; the shader thresholds it, which gives crisp curved
                     outlines that are independent of the mesh resolution.
"""
import math

import bpy

from . import util

TAU = 2.0 * math.pi


class Graph:
    """Tiny node-graph builder: every method returns an output socket; inputs accept either a
    socket or a constant."""

    def __init__(self, name):
        self.mat, self.nodes, self.links, self.bsdf, self.out = util.material(name)
        self._rest = None
        self._paint = {}
        self._time = None

    # -- plumbing
    def put(self, sock, value):
        if isinstance(value, bpy.types.NodeSocket):
            self.links.new(value, sock)
        elif value is not None:
            if hasattr(sock.default_value, "__len__") and not hasattr(value, "__len__"):
                value = (value,) * len(sock.default_value)
            elif hasattr(sock.default_value, "__len__") and len(value) == 3 and len(sock.default_value) == 4:
                value = (*value, 1.0)
            sock.default_value = value

    def node(self, kind, **props):
        n = self.nodes.new(kind)
        for k, v in props.items():
            setattr(n, k, v)
        return n

    def set(self, **inputs):
        """Set Principled BSDF inputs: g.set(Base_Color=…, Roughness=…)."""
        for k, v in inputs.items():
            self.put(self.bsdf.inputs[k.replace("_", " ")], v)

    # -- inputs
    def rest(self):
        if self._rest is None:
            self._rest = self.node("ShaderNodeAttribute", attribute_name="rest").outputs["Vector"]
        return self._rest

    def paint(self, channel, name="paint"):
        """Raw field value of channel 0 / 1 / 2 of the colour attribute."""
        if name not in self._paint:
            a = self.node("ShaderNodeAttribute", attribute_name=name)
            sep = self.node("ShaderNodeSeparateColor")
            self.links.new(a.outputs["Color"], sep.inputs[0])
            self._paint[name] = sep
        return self._paint[name].outputs[channel]

    def mask(self, channel, width=0.04, name="paint", edge=0.5):
        """Thresholded field → 0 / 1 mask with an antialiased edge."""
        return self.smoothstep(self.paint(channel, name), edge - width, edge + width)

    def phase(self):
        if self._time is None:
            self._time = util.time_node(self.nodes)
        return self._time.outputs["Phase"]

    # -- maths
    def math(self, op, a, b=None, c=None, clamp=False):
        n = self.node("ShaderNodeMath", operation=op, use_clamp=clamp)
        for sock, v in zip(n.inputs, (a, b, c)):
            self.put(sock, v)
        return n.outputs[0]

    def vmath(self, op, a, b=None, scale=None):
        n = self.node("ShaderNodeVectorMath", operation=op)
        self.put(n.inputs[0], a)
        if b is not None:
            self.put(n.inputs[1], b)
        if scale is not None:
            self.put(n.inputs["Scale"], scale)
        return n.outputs["Value" if op in ("LENGTH", "DOT_PRODUCT", "DISTANCE") else "Vector"]

    def smoothstep(self, x, lo, hi):
        n = self.node("ShaderNodeMapRange", interpolation_type="SMOOTHSTEP")
        self.put(n.inputs["Value"], x)
        self.put(n.inputs["From Min"], lo)
        self.put(n.inputs["From Max"], hi)
        return n.outputs[0]

    def remap(self, x, lo, hi, a=0.0, b=1.0):
        n = self.node("ShaderNodeMapRange")
        self.put(n.inputs["Value"], x)
        for k, v in (("From Min", lo), ("From Max", hi), ("To Min", a), ("To Max", b)):
            self.put(n.inputs[k], v)
        return n.outputs[0]

    def mix(self, fac, a, b, blend="MIX"):
        """Colour mix a → b."""
        n = self.node("ShaderNodeMix", data_type="RGBA", blend_type=blend)
        self.put(n.inputs[0], fac)
        self.put(n.inputs[6], a)
        self.put(n.inputs[7], b)
        return n.outputs[2]

    def value(self, color, factor):
        """Scale the brightness of a colour."""
        n = self.node("ShaderNodeHueSaturation")
        self.put(n.inputs["Color"], color)
        self.put(n.inputs["Value"], factor)
        return n.outputs[0]

    def lerp(self, fac, a, b):
        n = self.node("ShaderNodeMix", data_type="FLOAT")
        self.put(n.inputs[0], fac)
        self.put(n.inputs[2], a)
        self.put(n.inputs[3], b)
        return n.outputs[0]

    def xyz(self, vec):
        n = self.node("ShaderNodeSeparateXYZ")
        self.put(n.inputs[0], vec)
        return n.outputs

    def combine(self, x=0.0, y=0.0, z=0.0):
        n = self.node("ShaderNodeCombineXYZ")
        for sock, v in zip(n.inputs, (x, y, z)):
            self.put(sock, v)
        return n.outputs[0]

    def mapping(self, vec, scale=(1.0, 1.0, 1.0), rotation=(0.0, 0.0, 0.0), location=(0.0, 0.0, 0.0)):
        n = self.node("ShaderNodeMapping")
        self.put(n.inputs["Vector"], vec)
        n.inputs["Scale"].default_value = scale
        n.inputs["Rotation"].default_value = rotation
        n.inputs["Location"].default_value = location
        return n.outputs[0]

    # -- textures
    def noise(self, vec, scale, detail=2.0, rough=0.5, out="Fac", distortion=0.0):
        n = self.node("ShaderNodeTexNoise")
        self.put(n.inputs["Vector"], vec)
        n.inputs["Scale"].default_value = scale
        n.inputs["Detail"].default_value = detail
        n.inputs["Roughness"].default_value = rough
        n.inputs["Distortion"].default_value = distortion
        return n.outputs[out]

    def voronoi(self, vec, scale, feature="F1", randomness=1.0):
        n = self.node("ShaderNodeTexVoronoi", feature=feature)
        self.put(n.inputs["Vector"], vec)
        n.inputs["Scale"].default_value = scale
        n.inputs["Randomness"].default_value = randomness
        return n.outputs      # Distance, Color, Position

    def wave(self, vec, scale, distortion=0.0, bands="X"):
        n = self.node("ShaderNodeTexWave", bands_direction=bands)
        self.put(n.inputs["Vector"], vec)
        n.inputs["Scale"].default_value = scale
        n.inputs["Distortion"].default_value = distortion
        return n.outputs["Fac"]

    def ramp(self, fac, stops, interpolation="LINEAR"):
        n = self.node("ShaderNodeValToRGB")
        cr = n.color_ramp
        cr.interpolation = interpolation
        while len(cr.elements) < len(stops):
            cr.elements.new(0.5)
        for e, (pos, col) in zip(cr.elements, stops):
            e.position = pos
            e.color = (*col[:3], 1.0) if len(col) == 3 else col
        self.put(n.inputs[0], fac)
        return n.outputs["Color"]

    def bump(self, height, strength=0.3, distance=0.002, normal=None):
        n = self.node("ShaderNodeBump")
        self.put(n.inputs["Height"], height)
        n.inputs["Strength"].default_value = strength
        n.inputs["Distance"].default_value = distance
        if normal is not None:
            self.links.new(normal, n.inputs["Normal"])
        return n.outputs[0]

    def fresnel(self, power=3.0):
        """View-dependent rim factor: 0 facing the camera … 1 at grazing angles."""
        n = self.node("ShaderNodeLayerWeight")
        n.inputs["Blend"].default_value = 0.5
        return self.math("POWER", n.outputs["Facing"], power)

    # -- ready-made effects
    def sparkle(self, scale=170.0, density=0.12, strength=7.0, color=(1.0, 1.0, 1.0), size=0.42):
        """Twinkling glitter flecks (emission colour socket, strength).  Each Voronoi cell of the
        rest-space pattern may hold a fleck; it flashes an integer number of times per loop with
        its own phase, so the loop closes exactly."""
        vor = self.voronoi(self.rest(), scale)
        rnd = self.node("ShaderNodeSeparateColor")
        self.links.new(vor["Color"], rnd.inputs[0])
        on = self.math("LESS_THAN", rnd.outputs[0], density)
        dot = self.math("LESS_THAN", vor["Distance"], size)
        k = self.math("FLOOR", self.math("MULTIPLY_ADD", rnd.outputs[1], 9.0, 6.0))       # 6 … 14 flashes / loop
        ph = self.math("MULTIPLY_ADD", self.phase(), k, rnd.outputs[2])
        tw = self.math("POWER", self.math("MAXIMUM", self.math("SINE", self.math("MULTIPLY", ph, TAU)), 0.0), 5.0)
        tw = self.math("ADD", self.math("MULTIPLY", tw, 0.955), 0.045)    # flecks never go completely dark
        fac = self.math("MULTIPLY", self.math("MULTIPLY", on, dot), tw)
        return self.math("MULTIPLY", fac, strength), color


_CACHE = {}


def cached(name, maker):
    """Materials are shared by the three characters."""
    mat = bpy.data.materials.get(name)
    if mat is None or _CACHE.get(name) != mat.as_pointer():
        mat = maker(name)
        _CACHE[name] = mat.as_pointer()
    return mat


# ---------------------------------------------------------------- palette (linear RGB)
S = util.srgb
FUR = S("#8e5c3c")
FUR_DARK = S("#6b4129")
TAN = S("#eab98c")
TAN_LIGHT = S("#f6d3ab")
NOSE = S("#7d4a40")
LID = S("#d9a07e")
BROW = S("#3f2416")
GLASSES = S("#1c4a30")
MOUTH = S("#5e1420")
TONGUE = S("#e0606e")
PINK = S("#ff3d9a")


# ---------------------------------------------------------------- organic
def fur(name="vk_fur"):
    """Short plush fur: brown coat with a painted tan mask (paint R), dark mouth lining (paint G),
    fine clumpy bump and a soft fuzzy rim (sheen)."""
    def make(name):
        g = Graph(name)
        p = g.rest()
        clumps = g.noise(p, 55.0, 3.0, 0.6)
        fine = g.noise(g.mapping(p, scale=(1.0, 1.0, 0.35)), 260.0, 2.0, 0.6)
        brown = g.mix(g.smoothstep(clumps, 0.3, 0.75), FUR_DARK, FUR)
        tan = g.mix(g.smoothstep(clumps, 0.3, 0.8), TAN, TAN_LIGHT)
        col = g.mix(g.mask(0, 0.10), brown, tan)
        inside = g.mask(1, 0.06)
        col = g.mix(inside, col, MOUTH)
        h = g.math("ADD", g.math("MULTIPLY", clumps, 0.6), g.math("MULTIPLY", fine, 0.4))
        g.set(Base_Color=col, Roughness=g.lerp(inside, 0.78, 0.45), Sheen_Weight=g.lerp(inside, 0.9, 0.0), Sheen_Roughness=0.45,
              Sheen_Tint=(1.0, 0.82, 0.62, 1.0), Normal=g.bump(h, 0.35, 0.004))
        g.set(Specular_IOR_Level=0.25)
        return g.mat
    return cached(name, make)


def skin_pad(name="vk_pad"):
    """Bare tan skin of the hands."""
    def make(name):
        g = Graph(name)
        n = g.noise(g.rest(), 90.0, 2.0)
        g.set(Base_Color=g.mix(g.smoothstep(n, 0.3, 0.8), TAN, TAN_LIGHT), Roughness=0.55, Sheen_Weight=0.5,
              Sheen_Tint=(1.0, 0.8, 0.7, 1.0), Normal=g.bump(n, 0.12, 0.002), Specular_IOR_Level=0.3)
        return g.mat
    return cached(name, make)


def hair(name, color, tint=(0.55, 0.55, 1.0), emission=0.0):
    """Glossy hair: fine strands running down the fall of the hair (rest-space Z) as a bump and
    as streaks of value, soft coloured sheen on the rim."""
    def make(name):
        g = Graph(name)
        p = g.rest()
        strand = g.noise(g.mapping(p, scale=(1.0, 1.0, 0.07)), 170.0, 2.0, 0.6)
        clump = g.noise(g.mapping(p, scale=(1.0, 1.0, 0.22)), 30.0, 2.0, 0.5)
        g.set(Base_Color=g.value((*color, 1.0), g.remap(strand, 0.25, 0.75, 0.65, 1.6)), Roughness=g.remap(strand, 0.3, 0.7, 0.30, 0.52),
              Sheen_Weight=0.8, Sheen_Tint=(*tint, 1.0), Sheen_Roughness=0.35, Coat_Weight=0.10, Coat_Roughness=0.25)
        g.set(Normal=g.bump(g.math("ADD", g.math("MULTIPLY", strand, 0.6), g.math("MULTIPLY", clump, 0.9)), 0.7, 0.004))
        if emission > 0:
            g.set(Emission_Color=(*color, 1.0), Emission_Strength=emission)
        return g.mat
    return cached(name, make)


def simple(name, color, roughness=0.5, metallic=0.0, coat=0.0, sheen=0.0, emission=0.0, sheen_tint=(1.0, 1.0, 1.0), spec=0.5,
           emission_color=None):
    def make(name):
        g = Graph(name)
        g.set(Base_Color=color, Roughness=roughness, Metallic=metallic, Coat_Weight=coat, Coat_Roughness=0.08, Sheen_Weight=sheen,
              Sheen_Tint=(*sheen_tint, 1.0), Specular_IOR_Level=spec)
        if emission > 0:
            g.set(Emission_Color=emission_color or color, Emission_Strength=emission)
        return g.mat
    return cached(name, make)


# ---------------------------------------------------------------- cloth
def cloth(name, layers, weave=0.0, sparkle=None, bump=0.25):
    """Painted cloth.  layers: list of up to 4 dicts(color, rough, sheen, tint, metallic, coat,
    emit): layer 0 is the base, layer i (1..3) shows where paint channel i-1 > 0.5.
    weave: scale of a fine woven bump (0 = off).  sparkle: dict(layers=(i, …), **Graph.sparkle
    arguments) adds twinkling glitter on the given layers."""
    def make(name):
        g = Graph(name)
        p = g.rest()

        def prop(key, default):
            v = layers[0].get(key, default)
            for i, lay in enumerate(layers[1:]):
                w = lay.get(key, default)
                if key in ("color", "tint"):
                    v = g.mix(g.mask(i), v, w)
                else:
                    v = g.lerp(g.mask(i), v, w)
            return v

        soft = g.noise(p, 22.0, 3.0, 0.55)
        col = prop("color", (0.5, 0.5, 0.5))
        col = g.value(col, g.remap(soft, 0.3, 0.8, 0.82, 1.18))        # soft cloudy variation of the dye
        rough = prop("rough", 0.6)
        g.set(Base_Color=col, Roughness=g.math("ADD", rough, g.math("MULTIPLY", g.math("SUBTRACT", soft, 0.5), 0.12)),
              Sheen_Weight=prop("sheen", 0.6), Sheen_Tint=prop("tint", (1.0, 1.0, 1.0)), Sheen_Roughness=0.4,
              Metallic=prop("metallic", 0.0), Coat_Weight=prop("coat", 0.0), Coat_Roughness=0.1)
        h = g.math("MULTIPLY", soft, 0.5)
        if weave:
            w1 = g.wave(g.mapping(p, rotation=(0.6, 0.5, 0.8)), weave)
            w2 = g.wave(g.mapping(p, rotation=(-0.6, 0.9, -0.7)), weave)
            h = g.math("ADD", h, g.math("MULTIPLY", g.math("ADD", w1, w2), 0.25))
        g.set(Normal=g.bump(h, bump, 0.003))
        emit_layers = [i for i, lay in enumerate(layers) if lay.get("emit", 0.0) > 0]
        strength, ecol = None, None
        if emit_layers:
            strength, ecol = prop("emit", 0.0), col
        if sparkle:
            sp = dict(sparkle)
            on_layers = sp.pop("layers", (0,))
            s_strength, s_col = g.sparkle(**sp)
            where = None
            for i in on_layers:      # coverage of layer i = its mask minus the masks painted over it
                m = g.mask(i - 1) if i > 0 else 1.0
                for j in range(i, len(layers) - 1):
                    m = g.math("MULTIPLY", m, g.math("SUBTRACT", 1.0, g.mask(j)))
                where = m if where is None else g.math("ADD", where, m)
            s_strength = g.math("MULTIPLY", s_strength, where)
            if strength is None:
                strength, ecol = s_strength, s_col
            else:
                ecol = g.mix(g.math("GREATER_THAN", s_strength, 0.01), ecol, s_col)
                strength = g.math("ADD", strength, s_strength)
        if strength is not None:
            g.set(Emission_Color=ecol, Emission_Strength=strength)
        return g.mat
    return cached(name, make)


BLACK_CLOTH = dict(color=(0.012, 0.012, 0.016), rough=0.48, sheen=1.0, tint=(0.55, 0.68, 1.0))
WHITE_CLOTH = dict(color=(0.82, 0.82, 0.80), rough=0.7, sheen=0.4, tint=(1.0, 1.0, 1.0))
SHADOW = dict(color=(0.004, 0.004, 0.005), rough=0.9, sheen=0.0)


def leather(name, layers, coat=0.6):
    """Polished leather / rubber for shoes: same painting scheme as cloth(), glossy clear coat."""
    def make(name):
        g = Graph(name)

        def prop(key, default):
            v = layers[0].get(key, default)
            for i, lay in enumerate(layers[1:]):
                w = lay.get(key, default)
                v = g.mix(g.mask(i), v, w) if key == "color" else g.lerp(g.mask(i), v, w)
            return v

        n = g.noise(g.rest(), 140.0, 2.0)
        g.set(Base_Color=prop("color", (0.02, 0.02, 0.02)), Roughness=prop("rough", 0.3), Coat_Weight=prop("coat", coat),
              Coat_Roughness=0.06, Normal=g.bump(n, 0.06, 0.001), Sheen_Weight=0.3, Sheen_Tint=(0.6, 0.7, 1.0, 1.0))
        return g.mat
    return cached(name, make)


# ---------------------------------------------------------------- metal
def metal(name, color, roughness=0.2, studio=1.0, tint2=None, bump_scale=0.0):
    """Polished metal with a built-in "studio" reflection: EEVEE without ray tracing only reflects
    the (dark night) world, so a soft striped reflection driven by the reflection vector is added
    as emission.  It slides over the surface when the object moves, like a real reflection."""
    def make(name):
        g = Graph(name)
        tc = g.node("ShaderNodeTexCoord")
        r = g.xyz(tc.outputs["Reflection"])
        up = g.math("MULTIPLY_ADD", r[2], 0.5, 0.5)
        side = g.math("MULTIPLY_ADD", r[0], 0.5, 0.5)
        bands = g.ramp(up, [(0.0, (0.02, 0.02, 0.02)), (0.30, (0.10, 0.10, 0.10)), (0.46, (0.03, 0.03, 0.03)), (0.58, (0.55, 0.55, 0.55)),
                            (0.70, (1.0, 1.0, 1.0)), (0.82, (0.25, 0.25, 0.25)), (1.0, (0.08, 0.08, 0.08))])
        soft = g.ramp(side, [(0.0, (0.35, 0.35, 0.35)), (0.35, (1.0, 1.0, 1.0)), (0.7, (0.5, 0.5, 0.5)), (1.0, (0.9, 0.9, 0.9))])
        refl = g.mix(1.0, bands, soft, "MULTIPLY")
        base = color if tint2 is None else g.mix(up, tint2, color)
        g.set(Base_Color=base, Metallic=1.0, Roughness=roughness, Emission_Color=g.mix(1.0, refl, base, "MULTIPLY"),
              Emission_Strength=0.9 * studio)
        if bump_scale:
            v = g.voronoi(g.rest(), bump_scale)
            g.set(Normal=g.bump(v["Distance"], 0.8, 0.001))
        return g.mat
    return cached(name, make)


GOLD = (1.0, 0.70, 0.22)
GOLD_DEEP = (0.85, 0.42, 0.08)
SILVER = (0.86, 0.87, 0.90)


# ---------------------------------------------------------------- face
def eye_white():
    return simple("vk_eye", (0.92, 0.92, 0.90), roughness=0.25, coat=0.6, emission=0.10, emission_color=(1.0, 1.0, 1.0))


def pupil():
    """Dark iris/pupil disc with a painted catch light (paint R)."""
    def make(name):
        g = Graph(name)
        hl = g.mask(0, 0.08)
        g.set(Base_Color=g.mix(hl, (0.012, 0.008, 0.006), (1.0, 1.0, 1.0)), Roughness=0.15, Coat_Weight=1.0,
              Emission_Color=(1.0, 1.0, 1.0), Emission_Strength=g.math("MULTIPLY", hl, 1.2))
        return g.mat
    return cached("vk_pupil", make)


def lens():
    """Spectacle lens: almost clear alpha-blended glass with diagonal painted glints (no
    refraction, so it is safe for slice rendering)."""
    def make(name):
        g = Graph(name)
        glint = g.mask(0, 0.06)
        rim = g.fresnel(2.0)
        alpha = g.math("ADD", g.math("MULTIPLY", glint, 0.16), g.math("MULTIPLY_ADD", rim, 0.22, 0.05), clamp=True)
        g.set(Base_Color=g.mix(glint, (0.55, 0.75, 0.62), (1.0, 1.0, 1.0)), Roughness=0.05, Alpha=alpha, Specular_IOR_Level=1.0,
              Emission_Color=(0.85, 1.0, 0.92), Emission_Strength=g.math("MULTIPLY", glint, 0.25))
        g.mat.surface_render_method = "BLENDED"
        g.mat.use_transparency_overlap = False
        return g.mat
    return cached("vk_lens", make)
