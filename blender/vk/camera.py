"""One camera for the whole panorama.  Every screen (or any preview crop) is an off-axis slice of
the same perspective projection, so the three renders line up pixel-exactly at the seams.

The camera sits at (CAM_X, -CAM_DIST, CAM_Z), looks along +Y with no tilt, and uses lens shift to
frame the slice.  On the stage plane (y = 0) one metre is exactly config.PPM pixels.
"""
import math

import bpy

from . import config as C
from .mathx import TAU

F_PX = C.CAM_DIST * C.PPM          # focal length in full-resolution pixels

# named preview slices: (x0_px, x1_px) in panorama pixel columns
_HALF = 1500


def _around(x_world, half=_HALF):
    c = C.x_to_px(x_world)
    return (int(c - half), int(c + half))


SLICES = {name: (C.SCREEN_X0[name], C.SCREEN_X0[name] + C.SCREEN_W[name]) for name, _, _ in C.SCREENS}
SLICES.update({
    "pano": (0, C.PANO_W),
    "emo": _around(C.MARKS["emo"][0]),
    "singer": _around(C.MARKS["singer"][0]),
    "sax": _around(C.MARKS["sax"][0]),
    "text": _around(C.TEXT_CENTER[0], 3600),
})


def parse_slice(spec):
    """'left' | 'center' | 'right' | 'pano' | 'emo' | 'singer' | 'sax' | 'text' | 'x0:x1' (pixels)."""
    if spec in SLICES:
        return SLICES[spec]
    a, b = spec.split(":")
    return (int(a), int(b))


def camera_position(t=0.0):
    """Camera location at loop time t (closed path; static unless CAM_SWAY_* is set)."""
    ph = TAU * t / C.LOOP
    return (C.CAM_X + C.CAM_SWAY_X * math.sin(ph), -C.CAM_DIST, C.CAM_Z + C.CAM_SWAY_Z * math.sin(2 * ph))


def create_camera(scene):
    data = bpy.data.cameras.new("VK_Camera")
    cam = bpy.data.objects.new("VK_Camera", data)
    scene.collection.objects.link(cam)
    cam.location = camera_position(0.0)
    cam.rotation_euler = (math.pi / 2, 0.0, 0.0)      # look along +Y, up = +Z
    data.sensor_fit = "HORIZONTAL"
    data.sensor_width = 36.0
    data.clip_start, data.clip_end = C.CAM_CLIP
    scene.camera = cam
    return cam


def apply_slice(scene, cam, x0_px, x1_px, scale=1.0, overscan=0, t=0.0):
    """Frame panorama columns [x0_px, x1_px) (plus `overscan` px each side) at `scale`.
    Returns (width_px, height_px) of the output image."""
    w_full = (x1_px - x0_px) + 2 * overscan
    h_full = C.PANO_H
    r = scene.render
    r.resolution_x = max(4, int(round(w_full * scale)))
    r.resolution_y = max(4, int(round(h_full * scale)))
    r.resolution_percentage = 100
    r.pixel_aspect_x = r.pixel_aspect_y = 1.0
    px, py, pz = camera_position(t)
    cam.location = (px, py, pz)
    d = cam.data
    d.sensor_fit = "HORIZONTAL"
    d.lens = d.sensor_width * F_PX / w_full
    cx_px = 0.5 * (x0_px + x1_px)
    # shift is expressed in units of the sensor width (horizontal fit) for both axes
    d.shift_x = (cx_px - C.x_to_px(px)) / w_full
    zc = 0.5 * (C.FRAME_Z0 + C.FRAME_Z1)
    d.shift_y = (zc - pz) * C.PPM / w_full
    return r.resolution_x, r.resolution_y


def at_depth(x0, z0, y, t=0.0):
    """World point at depth y that projects onto the same pixel as stage-plane point (x0, 0, z0).
    Use it to lay out anything behind / in front of the stage in *screen* terms."""
    cx, cy, cz = camera_position(t)
    k = (y - cy) / (0.0 - cy)
    return (cx + (x0 - cx) * k, y, cz + (z0 - cz) * k)


def depth_scale(y):
    """Size multiplier for an object at depth y to appear as large as on the stage plane."""
    return (y + C.CAM_DIST) / C.CAM_DIST


def project(x, y, z, t=0.0):
    """World point → (panorama pixel column, pixel row from top)."""
    cx, cy, cz = camera_position(t)
    k = (0.0 - cy) / (y - cy)
    xs = cx + (x - cx) * k
    zs = cz + (z - cz) * k
    return (C.x_to_px(xs), C.z_to_py(zs))
