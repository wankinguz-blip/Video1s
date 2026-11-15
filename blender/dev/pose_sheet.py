"""Rig diagnostic: one character at the origin, free camera views, named test poses.
blender -b --factory-startup --python blender/dev/pose_sheet.py -- <variant> <out_dir> <pose1,pose2 | choreo:t1,t2> [views] [model_module]

variant: singer | sax | emo.  views: front,three,threeR,side,back,face,hands.  Env: RES (width px), SAMPLES.
Choreography mode: times in seconds, comma separated; a token a:b:step expands to a range
(b excluded), e.g. choreo:0:15:0.5 = one frame per beat.  Every frame is labelled with its
time and beat (LABEL=0 switches the label off).  File names sort by time, then view."""
import importlib, math, os, sys
import bpy
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vk import rig as R, propspec, util
from vk.rig import Arm, Leg, Pose, Attach, arm_fk, HEAD

argv = sys.argv[sys.argv.index("--") + 1:]
variant, out, what = argv[0], argv[1], argv[2]
views = (argv[3] if len(argv) > 3 else "front,three,side").split(",")
model = importlib.import_module("vk." + (argv[4] if len(argv) > 4 else "mole"))
RES = int(os.environ.get("RES", "560"))
SAMPLES = int(os.environ.get("SAMPLES", "8"))

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE_NEXT"
scene.eevee.taa_render_samples = SAMPLES
scene.render.resolution_x = RES
scene.render.resolution_y = int(RES * 1.25)
scene.view_settings.view_transform = "AgX"
try:
    scene.view_settings.look = "AgX - Medium High Contrast"
except Exception:
    pass
world = bpy.data.worlds.new("w"); scene.world = world; world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.25, 0.3, 0.4, 1)
world.node_tree.nodes["Background"].inputs[1].default_value = 0.6
col = util.collection("T")
bpy.ops.mesh.primitive_plane_add(size=6); g = bpy.context.object
g.data.materials.append(util.principled("g", (0.1, 0.25, 0.1)))
for rot, e, c in (((0.9, 0, -0.6), 3.0, (1, 0.95, 0.9)), ((1.2, 0, 2.6), 1.2, (0.6, 0.7, 1.0))):
    l = bpy.data.lights.new("s", "SUN"); l.energy = e; l.color = c
    o = bpy.data.objects.new("s", l); o.rotation_euler = rot; scene.collection.objects.link(o)
cam_d = bpy.data.cameras.new("c"); cam = bpy.data.objects.new("c", cam_d); scene.collection.objects.link(cam); scene.camera = cam
cam_d.lens = 85


class Ctx:
    engine = "eevee"; quality = "final"; draft = False; options = {}
    def __init__(s): s.scene = scene
    def collection(s, name): return util.collection("VK_" + name)


sk = R.Skeleton(extra=model.extra_bones(variant))
rig = R.Rig(variant, sk, np.eye(4), col)
state = model.build(Ctx(), rig, variant)


def base():
    p = Pose()
    p.arm_L = arm_fk("L", raise_=0.5, elbow=0.2)
    p.arm_R = arm_fk("R", raise_=0.5, elbow=0.2)
    return p


# face test poses (use with the 'face' view)
FACES = dict(
    neutral=dict(),
    smile=dict(smile=1.0),
    sing=dict(jaw=0.8, smile=0.7, brow_L=0.6, brow_R=0.6),
    ooh=dict(jaw=0.5, mouth_round=1.0, brow_L=0.4, brow_R=0.4),
    blink=dict(blink_L=1.0, blink_R=1.0, smile=0.8),
    wink=dict(blink_L=1.0, smile=1.0, brow_R=0.8, look=(0.6, 0.0)),
    frown=dict(brow_L=-1.0, brow_R=-1.0, smile=0.0, look=(0.0, -0.5)),
    puff=dict(cheeks=1.0, mouth_round=0.6, blink_L=0.9, blink_R=0.9, brow_L=0.5, brow_R=0.5),
    lookL=dict(look=(1.0, 0.4), smile=0.6, jaw=0.25),
    lookR=dict(look=(-1.0, -0.4), smile=0.6),
    shout=dict(jaw=1.0, smile=0.2, brow_L=1.0, brow_R=1.0, look=(0.0, 0.6)),
)


def named(name):
    p = base()
    if name == "rest":
        return sk.rest_pose
    if name == "curl0" or name == "curl5" or name == "curl1":
        c = {"curl0": 0.0, "curl5": 0.5, "curl1": 1.0}[name]
        p.arm_L = Arm(hand=(0.38, -0.30, 1.05), pole=(1, 0.5, -1), fingers=(0, 0, 1), palm=(0, -1, 0), curl=c)
        p.arm_R = Arm(hand=(-0.38, -0.30, 1.05), pole=(-1, 0.5, -1), fingers=(0, 0, 1), palm=(0, -1, 0), curl=c)
    elif name == "spread":
        p.arm_L = Arm(hand=(0.38, -0.30, 1.05), pole=(1, 0.5, -1), fingers=(0, 0, 1), palm=(0, -1, 0), curl=0.0, spread=1.0)
        p.arm_R = Arm(hand=(-0.38, -0.30, 1.05), pole=(-1, 0.5, -1), fingers=(0, 0, 1), palm=(1, 0, 0), curl=(0.0, 0.0, 1.0, 1.0))
    elif name == "tpose":
        p.arm_L = arm_fk("L", raise_=math.pi / 2, elbow=0.0)
        p.arm_R = arm_fk("R", raise_=math.pi / 2, elbow=1.57)
    elif name == "up":
        p.arm_L = arm_fk("L", raise_=2.8, elbow=0.1)
        p.arm_R = arm_fk("R", swing=1.2, raise_=1.4, elbow=1.9)
        p.jaw = 1.0
    elif name == "heel":
        p.leg_L.heel = 0.9; p.leg_R.toe = 0.6; p.hips_pos = (0, 0, -0.03)
    elif name == "squat":
        p.hips_pos = (0.0, 0.03, -0.20); p.hips_rot = (0.25, 0, 0); p.chest_rot = (0.2, 0.3, 0)
        p.leg_L = Leg((0.22, 0.0), yaw=0.4, knee=(0.5, -1, 0)); p.leg_R = Leg((-0.22, 0.0), yaw=-0.4, knee=(-0.5, -1, 0))
    elif name == "kick":
        p.hips_pos = (-0.04, 0, -0.03); p.leg_L = Leg((0.16, -0.30), lift=0.25, toe=0.0, knee=(0.2, -1, 0.3)); p.head_rot = (0.3, 0.5, 0.2)
        p.chest_rot = (0, -0.4, 0.15)
    elif name == "hat":
        b = HEAD["brim_front_L"]
        p.arm_L = Arm(hand=(b[0] + 0.02, b[1] - 0.03, b[2] - 0.05), pole=(1, 0.2, 0.2), space="head", fingers=(-0.5, 0.3, 0.8), palm=(-0.2, 0.9, 0.3), curl=0.5)
        p.head_rot = (0.25, -0.2, 0)
    elif name == "mic":
        m = HEAD["mouth"]
        p.arm_R = Arm(hand=(m[0] - 0.06, m[1] - 0.12, m[2] - 0.15), pole=(-0.8, 0.3, -0.6), space="head", fingers=(0.55, 0.15, 0.82), palm=(0.6, 0.6, -0.5), curl=0.85)
        p.props["mic"] = Attach("hand.R", matrix=propspec.MIC_IN_HAND["R"]); p.jaw = 0.7
    elif name == "saxhold":
        m = HEAD["mouth"]
        p.props["sax"] = Attach("head", pos=(m[0], m[1] - 0.01, m[2]), rot=(-0.25, 0.0, 0.12))
        ug, lg = propspec.SAX["upper_grip"], propspec.SAX["lower_grip"]
        p.arm_L = Arm(hand=(ug[0] + 0.075, ug[1] + 0.03, ug[2] - 0.01), pole=(0.8, 0.5, -0.4), space="prop:sax", fingers=(-1.0, -0.25, 0.1), palm=(-0.2, 0.0, 0.3), curl=0.55)
        p.arm_R = Arm(hand=(lg[0] - 0.075, lg[1] + 0.03, lg[2] - 0.01), pole=(-0.8, 0.5, -0.4), space="prop:sax", fingers=(1.0, -0.25, 0.1), palm=(0.2, 0.0, 0.3), curl=0.55)
    elif name in FACES:
        for k, v in FACES[name].items():
            setattr(p, k, v)
    else:
        raise SystemExit("unknown pose " + name)
    return p


VIEWS = {"front": (0, -5, 0.9), "three": (3.2, -4, 1.2), "side": (5.2, 0, 0.9), "back": (0, 5.2, 1.0), "threeR": (-3.2, -4, 1.2),
         "face": (0.5, -2.6, 1.25), "hands": (0, -2.4, 1.05)}
TARGET_Z = {"face": 1.27, "hands": 1.05}
os.makedirs(out, exist_ok=True)
jobs = []
if what.startswith("choreo:"):
    ch = importlib.import_module("vk.choreo." + variant)
    from vk import characters
    times = []
    for ts in what[7:].split(","):
        if ":" in ts:
            a, b, st = (float(x) for x in ts.split(":"))
            times += [a + i * st for i in range(int(round((b - a) / st)))]
        else:
            times.append(float(ts))
    for t in times:
        def fn(t=t):
            p = ch.pose(t); characters.follow_through(sk, rig.mark, ch.pose, p, t); return p
        jobs.append((f"t{t:06.3f}".replace(".", "_"), fn, t))
else:
    for n in what.split(","):
        jobs.append((n, lambda n=n: named(n), 0.0))
label = None
if what.startswith("choreo:") and os.environ.get("LABEL", "1") != "0":
    fc = bpy.data.curves.new("label", "FONT")
    fc.size = 0.10
    label = bpy.data.objects.new("label", fc)
    scene.collection.objects.link(label)
    lm = bpy.data.materials.new("label")
    lm.use_nodes = True
    nt = lm.node_tree
    nt.nodes.clear()
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs[0].default_value = (1, 1, 1, 1)
    em.inputs[1].default_value = 2.0
    out_n = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(em.outputs[0], out_n.inputs[0])
    fc.materials.append(lm)
    label.parent = cam
for name, fn, t in jobs:
    pose = fn()
    if label is not None:
        label.data.body = f"{t:.3f}s  b{t * 2:g}"
    frames = rig.apply(pose)
    model.update(state, rig, pose, frames, t)
    util.set_time(t, 15.0)
    for v in views:
        pos = np.array(VIEWS[v], dtype=float)
        tz = TARGET_Z.get(v, 0.82)
        cam.location = pos
        d = np.array([0, 0, tz]) - pos
        cam.rotation_euler = (math.atan2(math.hypot(d[0], d[1]), -d[2]), 0, math.atan2(-d[0], d[1]))
        cam_d.lens = 85 if v not in ("face", "hands") else 110
        if label is not None:
            label.location = (-0.48 * 85.0 / cam_d.lens, 0.52 * 85.0 / cam_d.lens, -3.0)
            label.scale = (85.0 / cam_d.lens,) * 3
        scene.render.filepath = os.path.join(out, f"{name}_{v}.png")
        bpy.ops.render.render(write_still=True)
        print("[rigtest]", scene.render.filepath)
