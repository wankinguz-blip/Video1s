"""Show constants: timing, screens, world layout.  No bpy imports here (usable from plain Python).

World convention (Blender units = metres):
  +X  screen right            +Z  up            +Y  away from the camera (depth)
  The stage plane is y = 0.  World x = 0 is the horizontal centre of the CENTER screen and
  z = 0 is the top of the central hill (where the lead singer's feet are).
  Characters face the camera, i.e. look along -Y.

Everything that animates must be a pure function of loop time t in [0, LOOP) with period LOOP,
so any frame can be rendered independently and frame 450 is identical to frame 0.
"""
import math

# ---------------------------------------------------------------- timing
LOOP = 15.0            # seconds, exact loop length
BPM = 120.0
BEAT = 60.0 / BPM      # 0.5 s
LOOP_BEATS = 30
FPS = 30
FRAMES = int(LOOP * FPS)   # 450 frames: 0 … 449, frame i shows t = i / FPS

# ---------------------------------------------------------------- screens (one panorama, left → right)
PANO_H = 1872
SCREENS = [            # physical order on site, left → right
    ("left", 6864, PANO_H),
    ("center", 11752, PANO_H),
    ("right", 6240, PANO_H),
]
PANO_W = sum(s[1] for s in SCREENS)          # 24856
SCREEN_X0 = {}                               # first panorama pixel column of each screen
_x = 0
for _n, _w, _h in SCREENS:
    SCREEN_X0[_n] = _x
    _x += _w
SCREEN_W = {n: w for n, w, _ in SCREENS}

# ---------------------------------------------------------------- panorama ↔ world mapping (on the stage plane y = 0)
FRAME_H_M = 4.2                    # metres visible vertically on the stage plane
PPM = PANO_H / FRAME_H_M           # pixels per metre on the stage plane (≈ 445.7)
FRAME_Z0 = -1.0                    # world z at the bottom edge of the frame (stage plane)
FRAME_Z1 = FRAME_Z0 + FRAME_H_M    # world z at the top edge (3.2)
ORIGIN_PX = SCREEN_X0["center"] + SCREEN_W["center"] / 2.0   # panorama column of world x = 0


def px_to_x(px):
    """Panorama pixel column → world x on the stage plane."""
    return (px - ORIGIN_PX) / PPM


def x_to_px(x):
    return x * PPM + ORIGIN_PX


def z_to_py(z):
    """World z on the stage plane → pixel row (0 = top)."""
    return (FRAME_Z1 - z) * PPM


def screen_center_x(name):
    return px_to_x(SCREEN_X0[name] + SCREEN_W[name] / 2.0)


PANO_X0 = px_to_x(0)               # ≈ -28.58 m
PANO_X1 = px_to_x(PANO_W)          # ≈ +27.19 m

# ---------------------------------------------------------------- camera (one position for the whole panorama)
CAM_DIST = 60.0                    # camera sits at y = -CAM_DIST, looks along +Y, no tilt
CAM_Z = 0.15                       # slightly low viewpoint: almost level with the hill top
CAM_X = 0.0
CAM_CLIP = (5.0, 3000.0)
# Optional very subtle periodic drift (metres); keep 0 for LED walls unless asked.
CAM_SWAY_X = 0.0
CAM_SWAY_Z = 0.0

# ---------------------------------------------------------------- stage marks
HILL_TOP_Z = {"center": 0.0, "left": -0.22, "right": -0.25}
MARKS = {              # character → (x, y) on the stage plane; z comes from terrain.height
    "emo": (screen_center_x("left"), 0.0),
    "singer": (0.0, 0.0),
    "sax": (screen_center_x("right"), 0.0),
}
HILLS = [              # the three stage hills: (role, x, y, top z, footprint radius, reserved)
    ("left", MARKS["emo"][0], 0.0, HILL_TOP_Z["left"], 4.0, 0.0),
    ("center", 0.0, 0.0, HILL_TOP_Z["center"], 4.6, 0.0),
    ("right", MARKS["sax"][0], 0.0, HILL_TOP_Z["right"], 4.0, 0.0),
]
CHAR_HEIGHT = 1.55                 # mole height to the top of the head (hats add ~0.2)

# ---------------------------------------------------------------- VANKING text + fireworks (stage-plane coordinates)
TEXT_CAP_H = 0.75                  # cap height in metres (as projected on the stage plane)
TEXT_CENTER = (0.0, 2.60)          # (x, z) of the word centre on the stage plane
FIREWORK_DEPTH = 14.0              # fireworks live on the plane y = FIREWORK_DEPTH (behind the hills)
FACE_SAFE_Z = 1.80                 # non-text sparks must fade out above this stage-plane height near the characters

FIREWORK_CYCLES = [
    dict(launch=0.0, burst=1.0, gather=(1.22, 2.0), hold=(2.0, 3.0), dissolve=(3.0, 3.86), speed=1.0),
    dict(launch=4.0, burst=5.0, gather=(5.22, 6.0), hold=(6.0, 7.0), dissolve=(7.0, 7.86), speed=1.0),
    dict(launch=8.0, burst=9.0, gather=(9.22, 10.0), hold=(10.0, 11.0), dissolve=(11.0, 11.86), speed=1.0),
    # fourth cycle is compressed so it is completely gone before the loop boundary
    dict(launch=12.0, burst=12.5, gather=(12.66, 13.25), hold=(13.25, 14.25), dissolve=(14.25, 14.93), speed=1.28),
]
COLORS = {             # linear-ish display RGB 0..1 (blue / red / green fireworks)
    "blue": (64 / 255, 150 / 255, 255 / 255),
    "red": (255 / 255, 62 / 255, 74 / 255),
    "green": (60 / 255, 235 / 255, 110 / 255),
}
CYCLE_COLORS = [       # left, centre, right burst colour per cycle
    ("blue", "red", "green"),
    ("green", "blue", "red"),
    ("red", "green", "blue"),
    ("blue", "red", "green"),
]

TAU = math.tau
