"""Scene assembly.  Each scene module exposes

    build(ctx)            -> state        (create objects / materials once)
    update(state, ctx, t) -> None         (pose everything for loop time t; pure function of t)

Modules are built and updated in MODULES order.  A module can be skipped with --skip.
"""
import importlib
import time

import bpy

from . import camera as cam_mod
from . import config as C
from . import util

MODULES = ["environment", "lighting", "characters", "fireworks"]


class Ctx:
    """Shared build context handed to every module."""

    def __init__(self, scene, engine="eevee", quality="final", skip=(), options=None):
        self.scene = scene
        self.engine = engine            # 'eevee' | 'cycles'
        self.quality = quality          # 'draft' (fast previews) | 'final'
        self.skip = set(skip)
        self.options = options or {}
        self.camera = None
        self.states = {}                # module name → state returned by build()

    @property
    def draft(self):
        return self.quality == "draft"

    def collection(self, name):
        return util.collection("VK_" + name)


def reset_scene():
    """Empty file: no default cube / light / camera, no orphan data."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.name = "VANKING"
    return scene


def configure_render(scene, engine="eevee", samples=None, quality="final"):
    r = scene.render
    scene.frame_start, scene.frame_end = 0, C.FRAMES - 1
    r.fps = C.FPS
    r.image_settings.file_format = "PNG"
    r.image_settings.color_mode = "RGB"
    r.image_settings.color_depth = "8"
    r.image_settings.compression = 15
    r.dither_intensity = 1.0            # hides banding in the night-sky gradient on LED walls
    r.film_transparent = False
    r.use_file_extension = True
    r.threads_mode = "AUTO"
    scene.view_settings.view_transform = "AgX"
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
    except TypeError:
        pass
    scene.display_settings.display_device = "sRGB"
    if engine == "cycles":
        r.engine = "CYCLES"
        cy = scene.cycles
        cy.samples = samples or (24 if quality == "draft" else 96)
        cy.use_adaptive_sampling = True
        cy.use_denoising = True
        cy.max_bounces = 4
        cy.diffuse_bounces = 2
        cy.glossy_bounces = 3
        cy.transparent_max_bounces = 16
        cy.sample_clamp_indirect = 6.0
        try:  # GPU when available (Metal on Apple Silicon)
            prefs = bpy.context.preferences.addons["cycles"].preferences
            for kind in ("METAL", "OPTIX", "CUDA"):
                try:
                    prefs.compute_device_type = kind
                    prefs.get_devices()
                    if any(d.type == kind for d in prefs.devices):
                        for d in prefs.devices:
                            d.use = True
                        cy.device = "GPU"
                        break
                except Exception:
                    continue
        except Exception:
            pass
    else:
        r.engine = "BLENDER_EEVEE_NEXT"
        ee = scene.eevee
        ee.taa_render_samples = samples or (8 if quality == "draft" else 32)
        ee.use_shadows = True
        ee.use_raytracing = False       # no screen-space tracing → slices match exactly at the seams
        ee.use_volumetric_shadows = False


class Show:
    def __init__(self, engine="eevee", quality="final", samples=None, skip=(), options=None):
        self.scene = reset_scene()
        configure_render(self.scene, engine, samples, quality)
        self.ctx = Ctx(self.scene, engine, quality, skip, options)
        self.ctx.camera = cam_mod.create_camera(self.scene)
        self.modules = []

    def build(self):
        for name in MODULES:
            if name in self.ctx.skip:
                continue
            t0 = time.time()
            try:
                mod = importlib.import_module("vk." + name)
            except ModuleNotFoundError as e:
                if e.name != "vk." + name:
                    raise
                print(f"[vk] module '{name}' not present yet – skipped")
                continue
            state = mod.build(self.ctx)
            self.ctx.states[name] = state
            self.modules.append((name, mod, state))
            print(f"[vk] built {name} in {time.time() - t0:.1f}s")
        return self

    def update(self, t):
        """Pose the whole scene for loop time t (seconds)."""
        t = float(t) % C.LOOP
        util.set_time(t, C.LOOP)
        for _name, mod, state in self.modules:
            mod.update(state, self.ctx, t)

    def frame(self, slice_px, t, scale=1.0, overscan=0):
        """Set the camera slice and pose the scene; returns output size."""
        size = cam_mod.apply_slice(self.scene, self.ctx.camera, slice_px[0], slice_px[1], scale, overscan, t)
        self.update(t)
        return size
