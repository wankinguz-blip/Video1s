# Legacy: the 2.5D canvas version (not accepted)

This folder keeps the first version of the show: a 2.5D vector animation drawn with Skia
(`@napi-rs/canvas`). The customer did not accept it (limbs were separate sprites, flat look, weak
background and fireworks), and it was replaced by the Blender 3D pipeline in [`blender/`](../../blender).
Nothing here is used by the 3D show.

| Path | What it is |
| --- | --- |
| `src/` | scene, characters, fireworks and the frame renderer of the 2D version |
| `scripts/render.mjs`, `still.mjs`, `validate-loop.mjs` | its render pipeline and loop validator |
| `scripts/extract-mascot.mjs` | converted the brand SVG head into canvas draw operations |
| `preview/index.html` | browser live preview |
| `docs/` | progress pictures, loop-validation report and release notes of the 2D version |
| `deliverables/` | its low-resolution previews (the full-size videos were never committed) |

Still shared with the 3D show and therefore **not** in this folder:

* `src/music/score.js`, `src/config.js` — the score and timing; read by `audio/build-soundtrack.mjs`
  (soundtrack) and `scripts/export-blender-data.mjs` (data for Blender);
* `src/assets/vanking-text.js` — particle targets of the word VANKING (made by `scripts/build-text.mjs`).

`src/music/score.js`, `src/config.js` and `src/assets/vanking-text.js` in this folder only re-export
those shared files, so the old renderer still runs from the repository root:

```bash
npm install
node legacy/2d/scripts/still.mjs --screen center --scale 0.25 --t 2.5
```

Known inaccuracies of the old documents: the GitHub Release `v1.0` mentioned in
`docs/RELEASE_NOTES_v1.0.md` was never published; `docs/loop-validation.md` lists the spanned right
screen as `CHECK`, not `PASS`; `scripts/extract-mascot.mjs` still resolves its paths relative to the
old location.
