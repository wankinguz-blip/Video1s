# Project progress — VANKING.UZ Mole Music Show

_Every change is committed and pushed immediately (spec: development workflow). Working branch:
`vorflux/blender-3d-rework`. What to do next, in order: [`NEXT_STEPS.md`](NEXT_STEPS.md)._

## Current status (2026-11-14)

**Version 2 — real 3D in Blender: the scene and the pipeline are complete; the full-size render is
the remaining step and has to be made on the customer's Mac.**

| Part | State |
| --- | --- |
| Camera, three-screen slicing | Done, verified (seam check PASS on the complete scene) |
| Environment, lighting, fireworks, word VANKING | Done, reviewed on renders, accepted with the notes below |
| Mole model, three costumes, microphone, saxophone | Done, checked on pose sheets from four sides and in the scene |
| Choreography: singer, saxophonist, emo dancer | Done, numeric check PASS for all three, key moments checked on the final models |
| Loop closure | Verified: `scripts/show.sh validate` PASS |
| `scripts/show.sh` | Every command run on Linux at reduced size; **not yet run on macOS** |
| Full-size frames (6864 / 11752 / 6240 × 1872), EEVEE on Metal, render time | **Not verified** — impossible in the sandbox (no GPU) |
| Complete 450-frame preview video | **Not rendered** — about 4 hours in the sandbox; first thing to render on the Mac |
| Old 2.5D version | Moved to `legacy/2d/` |
| Pull request | Opened from `vorflux/blender-3d-rework` into `main` |

## What was verified, and how

| Check | Result |
| --- | --- |
| Scene build (`blender/dev/scene_stats.py`, final quality) | 7.7 s in total, characters 3.2 s (target was < 15 s); 19.1 M triangles, 367 k grass instances, 25 lights (3 suns with shadows) |
| Loop (`scripts/show.sh validate`, panorama 1/8, complete scene) | t = 0 vs t = 15: identical (0.0 of 255 levels); step 449→0: 0.650, ordinary step 0→1: 0.659, ratio 0.99 → **PASS** |
| Seams (`scripts/show.sh seams`, 1/8, t = 2.5 s) | screens vs panorama 0.68 levels (sample noise), seam columns 0.57 and 0.89 → **PASS** |
| Seam at full size, final quality (strips `6608:6864` + `6864:7120` vs one render of `6608:7120` at scale 1, the left / centre seam) | joined strips vs single render 0.19 levels, seam columns 0.22 → **PASS** ([picture](docs/preview/seam_full_size.jpg): joined strips left, single render right) |
| Choreography (`blender/dev/check_choreo.py`) | singer **PASS**, sax **PASS**, emo **PASS** (loop closure, jumps, velocity kinks, reach, collisions with head / hat / fringe / torso, planted-foot sliding, knees, microphone and saxophone contacts); [stick-figure sheet of the emo routine](docs/preview/emo_routine_sticks.png) |
| Costumes and props | pose sheets (rest, arms up, saxophone hold) from front, three-quarter, side and back; key moments of all three routines on the final models; close-ups of the hat pinch, the microphone at the mouth and the hands on the saxophone |
| Pipeline `scripts/show.sh` on Linux at 1/16 size | `setup`, `test`, `bench`, `preview` (`JOBS=2`), `frames` (resume), `final` (frames → three mp4 + panorama preview), `validate`, `seams`: all ran without errors |
| Final quality (full grass density) | close-ups of all three characters at 1/4 size (85–122 s each in the sandbox) and three full-size strips of 256–512 px: render correctly |
| Blender download | the three checksums in `scripts/show.sh` equal the official `blender-4.5.16.sha256` |

Sandbox render times (2 CPU cores, software OpenGL — **not** representative of a Mac): draft
panorama at 1/8 size about 8 s per sample, 33 s per frame with 4 samples; a 450-frame panorama
preview would take about 4 hours, which is why it was left for the Mac.

## Known limits and open points

* **EEVEE on Metal and the full-size frame have never been rendered.** `scripts/show.sh bench`
  measures two full-size frames of the widest screen and prints an estimate before `final` is started.
  If Blender prints `Shadow buffer full`, lower `shadow_resolution_scale` in `blender/vk/lighting.py`
  or raise `scene.eevee.shadow_pool_size`.
* **The motion was not watched as a video yet.** It was checked numerically (continuity, collisions,
  foot contact), on stick-figure sheets at quarter-beat steps and on key-pose renders. The preview
  video from the Mac is the first real look at the movement; choreography notes are expected.
* The central burst is cut by the top edge of the frame for about half a second after each burst:
  the wall is 13 : 1 and the burst is centred where the word forms. Lowering it would move it onto
  the singer.
* The hills are low and wide (the frame shows only 1 m below the feet); they read as three mounds
  with lighter crowns rather than as steep hills.
* Grass does not cast shadows (it only receives them): 8 M blade triangles in every shadow view cost
  far more than they add. `--grass-shadows` switches them on.
* `scripts/show.sh` was tested on Linux only. The macOS branch of `setup` (dmg mount, copy of
  Blender.app into `tools/`) is untested.
* Repository: about 34 MB of old mp4 previews are stored in git without LFS (`legacy/2d/deliverables/`).

## Session log

### 2026-11-14, session 3 (this one): from "never rendered" to a complete, verified scene

| # | Work | Verified by |
| --- | --- | --- |
| 1 | Blender 4.5.16 installed with `scripts/show.sh setup`; headless EEVEE needs `libEGL` on Linux (now reported by `setup`). | `setup` on Linux |
| 2 | First full-scene render. It did not finish at final quality (7.7 of 8 GB, 18 M grass triangles in every shadow view). Added `blender/dev/scene_stats.py`; grass no longer casts shadows; grass density option `--grass` with width compensation (draft = 10 %); `--env`, `--cast`; line-buffered progress output. | draft panoramas at t = 0, 1.3, 2.5, 9.2 s |
| 3 | Environment, lighting and fireworks reviewed on panoramas, character close-ups and a 15-frame timeline of the firework cycle (launch, burst, gather, hold, dissolve, compressed fourth cycle, empty sky at t = 0). Accepted as written. | renders |
| 4 | Costumes: saxophonist (burgundy suit, cream shirt, bow tie, pork-pie hat, spectator shoes, saxophone attached) and emo dancer (tee with heart, bare arms, jeans, studded belt, chain, wristbands, sneakers with laces, hair with fringe on the follow-through bone). New hair material. Fixed `skin.ring()` (the saxophone crashed). | pose sheets |
| 5 | Emo Tecktonik routine written (30 beats, closed loop); saxophonist's forearm intersections removed; `blender/dev/stick_sheet.py` (stick-figure sheets without Blender); checker's fringe volume matched to the model. | `check_choreo.py` PASS ×3, stick sheets, key poses on the final models |
| 6 | `scripts/show.sh`: all commands run; new `bench`, `seams`, render log, `FRAMES=n` trial runs; checksums verified. | runs at 1/16 size |
| 7 | Loop validation and seam check on the complete scene; the seam also at full size and final quality on a 512 px strip. | PASS / PASS / PASS |
| 8 | Old 2.5D renderer moved to `legacy/2d/` (still runs); `src/` keeps only the shared score, timing and text data; `scripts/build-text.mjs` fixed. | data export reproduces `blender/data/*.json` unchanged; a legacy still renders |
| 9 | Documentation rewritten: `README.md`, `PROGRESS.md`, `NEXT_STEPS.md`, `blender/README.md`. Pull request opened. | — |

The sandbox could not be enlarged (the platform refused every size above 2 CPU / 8 GB), so all
sequence renders were limited to short trial runs.

### 2026-11-14, sessions 1–2 (previous): foundation of the 3D pipeline

Config, panorama camera and screen slices, terrain, render driver; analytic IK skeleton, skinning,
proxy model, pose-sheet tool; environment, lighting, fireworks, mole model with the singer's
costume, singer choreography, `scripts/show.sh` (written, not run). See `git log`.

## Version 1 (2.5D canvas animation) — not accepted

The first version was a 2.5D vector animation rendered with Skia (Node + `@napi-rs/canvas`). It was
rejected: limbs were separate sprites, the picture was flat, background and fireworks were weak. Its
code, documents and previews are in [`legacy/2d/`](legacy/2d). Still used from it: the soundtrack
(`deliverables/soundtrack.wav`, built by `audio/build-soundtrack.mjs`), the score
(`src/music/score.js`) and the particle targets of the word (`src/assets/vanking-text.js`).

## Design decisions

* **One panorama, three slices**: 6864 | 11752 | 6240 × 1872 (left to right), 445.7 px per metre on
  the stage plane, camera 60 m away, 15 cm above the central hill top.
* **Tempo** 120 BPM (beat = 0.5 s, 30 beats per loop). Launches at 0 / 4 / 8 / 12 s fall on bar
  lines; bursts on beats 2, 10, 18 and 25; the fourth cycle is compressed and gone by 14.93 s.
* **Engine**: EEVEE (Blender 4.5 LTS) without ray tracing, compositor, bloom or volumetrics, so
  neighbouring screens match; 32 samples for the final render, 8 for drafts.
* **Characters**: custom closed-form IK instead of Blender constraints (frames are independent,
  limbs cannot detach); one analytic skin-weight function for the body and every garment layer.
* **Audio in MP4**: AAC adds priming samples; for gap-free looping use `soundtrack.wav`.
