"""Base lighting of the show: a stylised bright moonlit night.

  world     soft blue ambient (sky above, darker green bounce below) – no visible background,
            the sky itself is an emissive card built by the environment module
  MoonKey   cool sun from behind-left (same side as the visible moon): rim light on the characters,
            real cast shadows that fall forward / to the right on the hill slopes
  Fill      soft, slightly warm sun from the camera side so faces and costumes read on LED walls
  Kicker    faint teal back light from the right for a second rim and to separate dark shapes

Everything is constant over the loop (spec 2.2 / 8.2); the fireworks module adds its own dynamic
lights.  Levels leave roughly 1.5 stops of headroom on the grass for the firework flashes.
"""
import math

import bpy
from mathutils import Vector

from . import util

# ---------------------------------------------------------------- tunables
AMBIENT_TOP = (0.060, 0.105, 0.300)      # linear RGB, sky hemisphere
AMBIENT_HORIZON = (0.085, 0.085, 0.230)
AMBIENT_GROUND = (0.012, 0.030, 0.035)   # bounce from the meadow
AMBIENT_STRENGTH = 0.75

# direction the light TRAVELS (world: +X right, +Y away from the camera, +Z up)
KEY = dict(direction=(0.80, -0.42, -0.62), color=(0.66, 0.80, 1.00), energy=4.4, angle=2.2)
FILL = dict(direction=(-0.22, 1.0, -0.30), color=(1.00, 0.93, 0.86), energy=0.75, angle=30.0)
KICK = dict(direction=(-0.86, -0.42, -0.28), color=(0.35, 0.80, 0.95), energy=0.9, angle=6.0)


def _sun(col, name, direction, color, energy, angle, shadow=True, specular=1.0):
    data = bpy.data.lights.new(name, "SUN")
    data.color = color
    data.energy = energy
    data.angle = math.radians(angle)
    data.specular_factor = specular
    data.use_shadow = shadow
    data.use_shadow_jitter = angle > 4.0          # wide suns: jittered soft shadows (accumulate over TAA samples)
    data.shadow_jitter_overblur = 10.0
    data.shadow_filter_radius = 1.5
    data.shadow_maximum_resolution = 0.004
    obj = bpy.data.objects.new(name, data)
    col.objects.link(obj)
    # a sun shines along its local -Z
    obj.rotation_euler = Vector(direction).normalized().to_track_quat("-Z", "Y").to_euler()
    return obj


def _world(scene):
    w = bpy.data.worlds.new("VK_World")
    w.use_nodes = True
    nt = w.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    mp = nt.nodes.new("ShaderNodeMapRange")
    nt.links.new(geo.outputs["Incoming"], sep.inputs[0])       # for the world: direction TO the camera
    mp.inputs["From Min"].default_value = -1.0
    mp.inputs["From Max"].default_value = 1.0
    nt.links.new(sep.outputs["Z"], mp.inputs["Value"])
    nt.links.new(mp.outputs[0], ramp.inputs["Fac"])
    cr = ramp.color_ramp
    # Incoming.z = -direction.z  →  0 = straight up, 0.5 = horizon, 1 = straight down
    cr.elements[0].position, cr.elements[0].color = 0.0, (*AMBIENT_TOP, 1.0)
    cr.elements[1].position, cr.elements[1].color = 1.0, (*AMBIENT_GROUND, 1.0)
    e = cr.elements.new(0.47)
    e.color = (*AMBIENT_HORIZON, 1.0)
    e = cr.elements.new(0.56)
    e.color = (*AMBIENT_GROUND, 1.0)
    nt.links.new(ramp.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = AMBIENT_STRENGTH
    nt.links.new(bg.outputs[0], out.inputs["Surface"])
    w.use_sun_shadow = False           # never extract a "sun" from the world
    scene.world = w
    return w


def build(ctx):
    col = ctx.collection("Lighting")
    scene = ctx.scene
    world = _world(scene)
    key = _sun(col, "VK_MoonKey", **KEY)
    fill = _sun(col, "VK_Fill", **FILL, specular=0.35)
    kick = _sun(col, "VK_Kicker", **KICK, specular=0.6)
    ee = scene.eevee
    ee.use_shadows = True
    ee.shadow_ray_count = 2 if ctx.draft else 3
    ee.shadow_step_count = 6
    ee.shadow_resolution_scale = 1.0
    ee.light_threshold = 0.01
    return dict(world=world, key=key, fill=fill, kick=kick)


def update(state, ctx, t):
    """Base lighting is constant over the loop."""
    return None
