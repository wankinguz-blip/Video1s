## VANKING.UZ — Mole Music Show · v1.0

15-second seamless-loop animated concert for three LED screens (6240×1872, 11752×1872, 6864×1872).

### Files

**Standalone** — the full scene composed for each screen (use if every screen shows the whole show):
- `VANKING_MoleShow_15s_STANDALONE_left_6240x1872.mp4`
- `VANKING_MoleShow_15s_STANDALONE_center_11752x1872.mp4`
- `VANKING_MoleShow_15s_STANDALONE_right_6864x1872.mp4`

**Spanned** — one continuous panorama across Left │ Center │ Right (use if the three screens sit side by side as one canvas; play the three files frame-synchronised):
- `VANKING_MoleShow_15s_SPANNED_left_6240x1872.mp4`
- `VANKING_MoleShow_15s_SPANNED_center_11752x1872.mp4`
- `VANKING_MoleShow_15s_SPANNED_right_6864x1872.mp4`

**Audio** — `soundtrack.wav` (48 kHz / 16-bit, exactly 15.000 s, sample-exact loop; use this for gap-free looping on a media server).

**Previews** — downscaled overviews of both layouts and a 5× loop test (`*_preview_*.mp4`).

### Technical
- H.264 High @ L6.0, yuv420p, 30 fps, exactly 450 frames = 15.000 s, CRF 16 · AAC 320 kb/s 48 kHz stereo, −13 LUFS.
- Closed loop: frame 450 would be identical to frame 0; validated pixel-exactly (`scripts/validate-loop.mjs`).
- The 11752-px-wide stream needs an H.264 level-6 capable decoder (media server / software decoding).
