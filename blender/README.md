# Blender 3D pipeline

The show is generated from Python in Blender 4.5 LTS; there are no `.blend` source files. Every
scene element and every animation channel is a deterministic function of the loop time `t`
(0–15 s), which gives an exact loop and lets frames be rendered independently, in any order.

## Wall and camera

| Screen | Resolution | Character |
|---|---:|---|
| Left | 6864 × 1872 | Emo dancer |
| Center | 11752 × 1872 | Lead singer |
| Right | 6240 × 1872 | Saxophonist |
| Panorama | 24856 × 1872 | All three |

All screens are lens-shifted windows of the same camera, preserving continuity at the seams. World
axes: +X right, +Z up, +Y away from the camera. The stage plane is `y = 0`; `x = 0` is the middle of
the center screen; `z = 0` is the top of the center hill. Scale: 445.7 px per metre on the stage
plane; the frame shows `z` from −1.0 to 3.2 m. A character is 1.55 m tall (about 690 px).

## Key files

- `render.py`: stills, frame ranges, resume, parallel parts
- `vk/config.py`, `camera.py`, `show.py`: timing and layout, camera slices, scene assembly
  (modules `environment`, `lighting`, `characters`, `fireworks`, each with `build(ctx)` and
  `update(state, ctx, t)`)
- `vk/terrain.py`, `environment.py`, `lighting.py`: analytic terrain, meadow, sky, moon, trees; lights
- `vk/rig.py`, `skin.py`, `meshkit.py`: closed-form IK skeleton, mesh primitives, voxel "sculpt"
- `vk/mole.py`, `outfits.py`, `props.py`, `charmat.py`: the mascot, the three costumes, microphone
  and saxophone, materials
- `vk/characters.py`, `vk/choreo/`: the cast on its marks; routines `singer.py`, `sax.py`, `emo.py`
  built with the toolkit in `common.py` (periodic key tracks, pose-to-pose keys, arm vectors,
  scripted feet)
- `vk/fireworks.py`: rockets, bursts, the particle word VANKING, firework lights
- `vk/score.py`, `data/score.json`, `data/vanking_text.json`: music-driven lip-sync, fingering and
  accents; particle targets of the word

Limbs use closed-form IK with fixed bone lengths; one analytic weight function skins the body and
every garment layer, so limbs cannot detach and layers cannot slide apart. To avoid artefacts that
depend on the frame borders, the render uses no compositor, bloom, screen-space reflections or
volumetrics; glow is additive geometry. Time-dependent shaders read the `VK_Time` node group.

## Render driver

```bash
blender -b --factory-startup --python blender/render.py -- --slice center --scale 0.25 --time 2.5 --out out/test/center
blender -b --factory-startup --python blender/render.py -- --slice pano --scale 0.125 --time 0,1.3,2.5,9.2 --samples 4 --quality draft --out out/peek
blender -b --factory-startup --python blender/render.py -- --slice right --part 1/2
blender -b --factory-startup --python blender/render.py -- --save-blend out/show.blend
```

| Option | Meaning |
|---|---|
| `--slice` | `left`, `center`, `right`, `pano`, close-ups `emo`, `singer`, `sax`, `text`, or panorama columns `X0:X1` |
| `--scale` | resolution scale (1 = full size); use 1, 0.5, 0.25, 0.125 when screens must stay pixel-aligned |
| `--time T[,T…]` / `--frames A-B` | stills at loop times, or a frame range (default 0-449) |
| `--quality` | `final` (default) or `draft`: 10 % of the grass tufts (made wider), coarser ground, 8 samples |
| `--samples` | EEVEE samples (default 32 final, 8 draft) |
| `--grass Q`, `--grass-shadows` | grass density multiplier; let the hero grass cast shadows (much slower) |
| `--skip`, `--env`, `--cast` | leave out scene modules, build only some environment layers or characters |
| `--part I/N`, `--force` | parallel workers; re-render existing frames (default: skip them) |

Final quality draws 19 M triangles (367 k grass instances); the scene builds in about 8 s.

## `scripts/show.sh`

- `setup`: download Blender 4.5.16 LTS into `tools/`, check ffmpeg and the soundtrack
- `test [T]`: one still per screen at 1/4 size (default `T=2.5`) and the three side by side
- `bench`: two full-size frames of the center screen, time per frame and an estimate for `final`
- `preview`: draft panorama video at 1/8 size with sound
- `final`: all frames of the three screens, then the videos
- `frames SCREEN`: render or resume `left`, `center`, `right` or `pano`
- `encode`: encode the frames that exist
- `validate`: loop closure (`t=0` vs `t=15`, step 449→0 vs an ordinary step)
- `seams [T]`: the three screens next to each other vs one panorama render
- `clean`: remove rendered frames

Environment variables: `BLENDER`, `JOBS`, `SCALE`, `SAMPLES`, `ENGINE`, `CRF`, `EXTRA`, `FRAMES`
(see `scripts/show.sh help`). The complete Blender output goes to `out/logs/render.log`.

## Development tools (`blender/dev/`)

| Tool | Use |
|---|---|
| `check_choreo.py [singer] [sax] [emo] [-v] [-c]` | numeric acceptance test of the routines: loop closure, jumps and velocity kinks, reach, collisions (head, hat, fringe, torso, thighs, ground), planted-foot sliding, knees, prop contacts. Plain Python + numpy |
| `stick_sheet.py NAME OUT.png --from B0 --to B1 --step S --cols N [--view front\|side\|both] [--warn]` | stick-figure contact sheet of a routine, labelled in beats; no Blender needed |
| `pose_sheet.py` | one character on neutral light: named test poses or choreography times, several views (`RES`, `SAMPLES`) |
| `scene_stats.py [--quality final\|draft]` | build time, memory, objects, instances, triangles and lights per module, without rendering |
| `compare_frames.py`, `compare_seams.py` | used by `show.sh validate` and `show.sh seams` |
| `pano_strips.sh IN OUT [rows]`, `contact_sheet.sh DIR OUT COLS [H]` | cut a panorama into stacked rows; tile a folder of pictures |

numpy comes with Blender's own Python, e.g. on Linux:
`cd blender && ../tools/blender/4.5/python/bin/python3.11 dev/check_choreo.py`.

Headless rendering on Linux needs `libEGL` (`apt install libegl1`); without a GPU Blender falls back
to software OpenGL, which renders the same picture, only slowly.
