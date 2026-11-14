#!/bin/bash
# VANKING.UZ — Mole Music Show: render pipeline (macOS and Linux).
#
#   scripts/show.sh setup             download Blender 4.5.16 LTS into tools/ and check ffmpeg
#   scripts/show.sh test [T]          one still per screen at loop time T (default 2.5 s) -> out/test/
#   scripts/show.sh bench             one full-size frame of the widest screen: time and estimate for `final`
#   scripts/show.sh preview           low-resolution panorama video with sound -> out/preview/
#   scripts/show.sh final             full-resolution frames of the three screens + videos -> deliverables/
#   scripts/show.sh frames SCREEN     render (or resume) the frames of one screen: left | center | right | pano
#   scripts/show.sh encode            encode the frames that are already rendered
#   scripts/show.sh validate          check that the 15 s loop closes (frame 450 == frame 0)
#   scripts/show.sh seams [T]         check that the three screens join into one picture (compared with a panorama render)
#   scripts/show.sh clean             delete rendered frames (out/frames)
#
# Environment variables (all optional):
#   BLENDER   path of the Blender executable (default: tools/, /Applications, PATH)
#   JOBS      parallel Blender processes (default 1; 2-3 can help on a Mac with a lot of memory)
#   SCALE     resolution scale for `final`/`frames`/`encode` (default 1 = 6864/11752/6240 x 1872)
#   SAMPLES   render samples (default: scene default, 32)
#   ENGINE    eevee (default) | cycles
#   CRF       H.264 quality (default 16; lower = better and bigger)
#   EXTRA     extra arguments passed to blender/render.py, e.g. EXTRA="--quality draft" or "--grass 0.5"
#   FRAMES    number of frames (default 450 = the whole loop; a small number is only for trying the pipeline)
#
# Rendering is resumable: frames that already exist are skipped. Stop with Ctrl-C and start again.
# The complete Blender output of the last runs is kept in out/logs/render.log.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BLENDER_VERSION="4.5.16"
BLENDER_SERIES="4.5"
BLENDER_URL="https://download.blender.org/release/Blender${BLENDER_SERIES}"
SHA_MAC_ARM64="c5cd40979f30033bc55ca84fccd00f2c1ddebb1d2f41618f220ef1f596d8b0f6"
SHA_MAC_X64="009e437b33e5e131340b8d91eb9005d69ef6f1c67eff1402953a7d40cf94638f"
SHA_LINUX_X64="23432b064977908beb1ab4b7f32f024e7331105c875c187dce002aff84b10c11"

SCREENS="left center right"
FPS=30
LOOP_FRAMES=450
FRAMES="${FRAMES:-$LOOP_FRAMES}"
JOBS="${JOBS:-1}"
SCALE="${SCALE:-1}"
ENGINE="${ENGINE:-eevee}"
CRF="${CRF:-16}"
EXTRA="${EXTRA:-}"
AUDIO="deliverables/soundtrack.wav"
LOG="out/logs/render.log"

say() { printf '\n== %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

screen_width() {
  case "$1" in
    left) echo 6864 ;; center) echo 11752 ;; right) echo 6240 ;; pano) echo 24856 ;;
    *) die "unknown screen '$1'" ;;
  esac
}

find_blender() {
  if [ -n "${BLENDER:-}" ]; then echo "$BLENDER"; return; fi
  local c
  for c in \
    "$ROOT/tools/Blender.app/Contents/MacOS/Blender" \
    "$ROOT/tools/blender/blender" \
    "/Applications/Blender.app/Contents/MacOS/Blender" \
    "/code/tools/blender45/blender"; do
    if [ -x "$c" ]; then echo "$c"; return; fi
  done
  if command -v blender >/dev/null 2>&1; then command -v blender; return; fi
  echo ""
}

need_blender() {
  BLENDER_BIN="$(find_blender)"
  [ -n "$BLENDER_BIN" ] || die "Blender not found. Run: scripts/show.sh setup"
  local v
  v="$("$BLENDER_BIN" --version 2>/dev/null | head -1 || true)"
  case "$v" in
    *"Blender ${BLENDER_SERIES}."*) ;;
    *) printf 'WARNING: %s found, the scene is made for Blender %s LTS (%s)\n' "${v:-unknown version}" "$BLENDER_SERIES" "$BLENDER_BIN" >&2 ;;
  esac
}

need_ffmpeg() {
  command -v ffmpeg >/dev/null 2>&1 || die "ffmpeg not found. macOS: brew install ffmpeg   Linux: sudo apt install ffmpeg"
}

sha256_of() {
  if command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | cut -d' ' -f1; else sha256sum "$1" | cut -d' ' -f1; fi
}

cmd_setup() {
  mkdir -p tools
  local os arch file sha
  os="$(uname -s)"; arch="$(uname -m)"
  if [ -n "$(find_blender)" ] && "$(find_blender)" --version 2>/dev/null | head -1 | grep -q "Blender ${BLENDER_SERIES}\."; then
    say "Blender is ready: $("$(find_blender)" --version | head -1)  ($(find_blender))"
  else
    case "$os-$arch" in
      Darwin-arm64) file="blender-${BLENDER_VERSION}-macos-arm64.dmg"; sha="$SHA_MAC_ARM64" ;;
      Darwin-x86_64) file="blender-${BLENDER_VERSION}-macos-x64.dmg"; sha="$SHA_MAC_X64" ;;
      Linux-x86_64) file="blender-${BLENDER_VERSION}-linux-x64.tar.xz"; sha="$SHA_LINUX_X64" ;;
      *) die "no automatic Blender download for $os-$arch; install Blender ${BLENDER_VERSION} and set BLENDER=/path/to/blender" ;;
    esac
    say "Downloading $file (about 350 MB)"
    curl -L --fail --retry 3 -C - -o "tools/$file" "$BLENDER_URL/$file"
    [ "$(sha256_of "tools/$file")" = "$sha" ] || die "checksum mismatch for tools/$file (delete the file and run setup again)"
    if [ "$os" = "Darwin" ]; then
      say "Installing Blender.app into tools/"
      local mnt="tools/.blender-dmg"
      mkdir -p "$mnt"
      hdiutil attach -nobrowse -readonly -quiet -mountpoint "$mnt" "tools/$file"
      rm -rf tools/Blender.app
      cp -R "$mnt/Blender.app" tools/Blender.app
      hdiutil detach -quiet "$mnt" || true
      rmdir "$mnt" 2>/dev/null || true
      xattr -dr com.apple.quarantine tools/Blender.app 2>/dev/null || true
    else
      say "Unpacking Blender into tools/blender"
      rm -rf tools/blender && mkdir -p tools/blender
      tar -xJf "tools/$file" -C tools/blender --strip-components=1
    fi
    say "Blender is ready: $("$(find_blender)" --version | head -1)  ($(find_blender))"
  fi
  if command -v ffmpeg >/dev/null 2>&1; then
    say "ffmpeg is ready: $(ffmpeg -version | head -1)"
  else
    printf '\nffmpeg is missing. Install it:\n  macOS:  brew install ffmpeg\n  Linux:  sudo apt install ffmpeg\n'
    exit 1
  fi
  if [ "$os" = "Linux" ] && command -v ldconfig >/dev/null 2>&1 && ! ldconfig -p | grep -q 'libEGL\.so\.1'; then
    printf '\nlibEGL is missing (Blender needs it to render without a display). Install it:\n  sudo apt install libegl1\n'
    exit 1
  fi
  [ -f "$AUDIO" ] || die "$AUDIO is missing"
  printf '\nNext step:  scripts/show.sh test\n'
}

# blender_render ARGS…  — one Blender process; shows only the pipeline's own lines and errors,
# the complete output goes to $LOG
blender_render() {
  mkdir -p "$(dirname "$LOG")"
  printf '\n===== %s  blender/render.py --engine %s %s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$ENGINE" "$EXTRA" "$*" >> "$LOG"
  # shellcheck disable=SC2086
  "$BLENDER_BIN" -b --factory-startup --python blender/render.py -- --engine "$ENGINE" ${SAMPLES:+--samples "$SAMPLES"} $EXTRA "$@" 2>&1 \
    | tee -a "$LOG" \
    | grep --line-buffered -E '^\[vk\]|Traceback|^  File |Error|error:' | grep --line-buffered -v 'EGL Error' || true
}

frames_dir() { echo "out/frames/scale_${SCALE}/$1"; }

count_frames() { find "$1" -maxdepth 1 -name 'f[0-9][0-9][0-9][0-9].png' 2>/dev/null | wc -l | tr -d ' '; }

render_frames() {   # SCREEN
  local screen="$1" dir n pids="" i range="0-$((FRAMES - 1))"
  screen_width "$screen" >/dev/null
  dir="$(frames_dir "$screen")"
  mkdir -p "$dir"
  n="$(count_frames "$dir")"
  if [ "$n" -ge "$FRAMES" ]; then say "$screen: all $FRAMES frames exist in $dir"; return; fi
  say "$screen: rendering $((FRAMES - n)) of $FRAMES frames at scale $SCALE into $dir (jobs: $JOBS)"
  if [ "$JOBS" -le 1 ]; then
    blender_render --slice "$screen" --scale "$SCALE" --frames "$range" --out "$dir"
  else
    i=1
    while [ "$i" -le "$JOBS" ]; do
      blender_render --slice "$screen" --scale "$SCALE" --frames "$range" --out "$dir" --part "$i/$JOBS" &
      pids="$pids $!"
      i=$((i + 1))
    done
    # shellcheck disable=SC2086
    wait $pids
  fi
  n="$(count_frames "$dir")"
  [ "$n" -ge "$FRAMES" ] || die "$screen: only $n of $FRAMES frames were rendered — see the messages above and $LOG, then run the same command again (it resumes)"
}

# H.264 High, yuv420p (BT.709), 30 fps, exactly 450 frames, AAC sound.  Odd preview sizes are padded to even.
encode_sequence() {   # INPUT_PATTERN OUTPUT [EXTRA_FILTER]
  local filter="pad=ceil(iw/2)*2:ceil(ih/2)*2,scale=out_color_matrix=bt709:out_range=tv,format=yuv420p"
  [ -n "${3:-}" ] && filter="$3,$filter"
  mkdir -p "$(dirname "$2")"
  ffmpeg -hide_banner -loglevel error -stats -y \
    -framerate "$FPS" -start_number 0 -i "$1" -i "$AUDIO" \
    -frames:v "$FRAMES" -vf "$filter" \
    -c:v libx264 -preset slow -tune animation -crf "$CRF" -profile:v high -g "$FPS" \
    -colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv \
    -c:a aac -b:a 320k -t "$DURATION" -movflags +faststart "$2"
  printf '   -> %s\n' "$2"
}

DURATION="$(awk "BEGIN { printf \"%.6f\", $FRAMES / $FPS }")"

# finished full-size loops go to deliverables/, everything else (scaled or shortened trials) to out/video/
out_dir() {
  if [ "$SCALE" = "1" ] && [ "$FRAMES" = "$LOOP_FRAMES" ]; then echo "deliverables"; else echo "out/video/scale_${SCALE}"; fi
}

cmd_encode() {
  need_ffmpeg
  local screen dir w h n out have_all=1 inputs="" od
  od="$(out_dir)"
  for screen in $SCREENS; do
    dir="$(frames_dir "$screen")"
    n="$(count_frames "$dir")"
    if [ "$n" -lt "$FRAMES" ]; then printf 'skip %s: %s of %s frames in %s\n' "$screen" "$n" "$FRAMES" "$dir"; have_all=0; continue; fi
    w="$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 "$dir/f0000.png")"
    h="$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "$dir/f0000.png")"
    out="$od/VANKING_${screen}_${w}x${h}.mp4"
    say "encoding $screen (${w}x${h})"
    encode_sequence "$dir/f%04d.png" "$out"
    inputs="$inputs -framerate $FPS -start_number 0 -i $dir/f%04d.png"
  done
  if [ "$have_all" = "1" ]; then
    say "encoding the panorama preview (three screens side by side, 3840 px wide)"
    mkdir -p "$od"
    # shellcheck disable=SC2086
    ffmpeg -hide_banner -loglevel error -stats -y $inputs -i "$AUDIO" \
      -filter_complex "[0:v][1:v][2:v]hstack=inputs=3,scale=3840:-2:flags=lanczos,scale=out_color_matrix=bt709:out_range=tv,format=yuv420p[v]" \
      -map "[v]" -map 3:a -frames:v "$FRAMES" -c:v libx264 -preset slow -tune animation -crf 18 -profile:v high \
      -colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv \
      -c:a aac -b:a 320k -t "$DURATION" -movflags +faststart "$od/VANKING_panorama_preview.mp4"
    printf '   -> %s\n' "$od/VANKING_panorama_preview.mp4"
  fi
}

cmd_final() {
  need_blender; need_ffmpeg
  local screen
  for screen in $SCREENS; do render_frames "$screen"; done
  cmd_encode
  say "done — videos are in $(out_dir)/"
}

cmd_test() {
  need_blender
  local t="${1:-2.5}" screen
  mkdir -p out/test
  for screen in $SCREENS; do
    say "test still: $screen at t=$t s (scale ${TEST_SCALE:-0.25})"
    blender_render --slice "$screen" --scale "${TEST_SCALE:-0.25}" --time "$t" --out "out/test/$screen" --force
  done
  if command -v ffmpeg >/dev/null 2>&1; then
    local name
    name="$(cd out/test/left && ls -t t*.png | head -1)"
    ffmpeg -hide_banner -loglevel error -y -i "out/test/left/$name" -i "out/test/center/$name" -i "out/test/right/$name" \
      -filter_complex "hstack=inputs=3" "out/test/panorama_${name}"
    say "three screens side by side: out/test/panorama_${name}"
  fi
}

# Two consecutive frames of the widest screen; the second one (shaders are compiled by then) is the
# honest per-frame time.  The other two screens are assumed to scale with their width.
cmd_bench() {
  need_blender
  local dir="out/bench" secs
  rm -rf "$dir"; mkdir -p "$dir"
  say "bench: two frames of the centre screen at scale $SCALE (11752 x 1872 at scale 1)"
  secs="$(blender_render --slice center --scale "$SCALE" --time 2.5,2.5333333 --out "$dir" --force | tee /dev/stderr \
    | sed -n 's/.* \([0-9.][0-9.]*\)s\/frame.*/\1/p' | tail -1)"
  [ -n "$secs" ] || die "the frames were not rendered — see $LOG"
  awk -v s="$secs" -v f="$LOOP_FRAMES" 'BEGIN {
    h = s * f * 24856 / 11752 / 3600
    printf "\n== centre screen: %.1f s per frame  ->  `final` (3 screens x %d frames) needs about %.1f hours with JOBS=1\n", s, f, h }'
}

cmd_preview() {
  need_blender; need_ffmpeg
  local dir="out/preview/frames" scale="${PREVIEW_SCALE:-0.125}" n pids="" i range="0-$((FRAMES - 1))"
  mkdir -p "$dir"
  say "preview: panorama at scale $scale (jobs: $JOBS)"
  if [ "$JOBS" -le 1 ]; then
    blender_render --slice pano --scale "$scale" --frames "$range" --out "$dir" --quality draft
  else
    i=1
    while [ "$i" -le "$JOBS" ]; do
      blender_render --slice pano --scale "$scale" --frames "$range" --out "$dir" --quality draft --part "$i/$JOBS" &
      pids="$pids $!"
      i=$((i + 1))
    done
    # shellcheck disable=SC2086
    wait $pids
  fi
  n="$(count_frames "$dir")"
  [ "$n" -ge "$FRAMES" ] || die "preview: only $n of $FRAMES frames were rendered — see $LOG, then run the command again (it resumes)"
  encode_sequence "$dir/f%04d.png" "out/preview/VANKING_panorama_preview.mp4"
}

cmd_validate() {
  need_blender; need_ffmpeg
  local dir="out/validate" scale="${VALIDATE_SCALE:-0.125}"
  rm -rf "$dir"; mkdir -p "$dir"
  say "loop check: rendering t = 0, 1/30, 14.9667 and 15.0 s (panorama, scale $scale)"
  blender_render --slice pano --scale "$scale" --time "0,0.0333333,14.9666667,15" --out "$dir" --force
  "$BLENDER_BIN" -b --factory-startup --python blender/dev/compare_frames.py -- \
    "$dir/t00_000.png" "$dir/t15_000.png" "$dir/t14_967.png" "$dir/t00_033.png" 2>&1 | grep -E '^\[loop\]' || die "comparison failed"
}

# The three screens must be the same picture as one panorama render (no visible seam).
cmd_seams() {
  need_blender
  local dir="out/seams" scale="${VALIDATE_SCALE:-0.125}" t="${1:-2.5}" screen name
  rm -rf "$dir"; mkdir -p "$dir"
  say "seam check: left, center, right and the whole panorama at t=$t s (scale $scale)"
  for screen in $SCREENS pano; do
    blender_render --slice "$screen" --scale "$scale" --time "$t" --out "$dir/$screen" --force
  done
  name="$(cd "$dir/pano" && ls t*.png | head -1)"
  "$BLENDER_BIN" -b --factory-startup --python blender/dev/compare_seams.py -- \
    "$dir/left/$name" "$dir/center/$name" "$dir/right/$name" "$dir/pano/$name" 2>&1 | grep -E '^\[seams\]' || die "comparison failed"
}

cmd_clean() { rm -rf out/frames out/test out/validate out/seams out/preview out/bench; say "removed rendered frames"; }

case "${1:-help}" in
  setup) cmd_setup ;;
  test) shift; cmd_test "$@" ;;
  bench) cmd_bench ;;
  preview) cmd_preview ;;
  final) cmd_final ;;
  frames) need_blender; render_frames "${2:?usage: scripts/show.sh frames left|center|right|pano}" ;;
  encode) cmd_encode ;;
  validate) cmd_validate ;;
  seams) shift; cmd_seams "$@" ;;
  clean) cmd_clean ;;
  *) sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//' ;;
esac
