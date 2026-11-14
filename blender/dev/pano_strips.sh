#!/bin/bash
# pano_strips.sh <pano.png> <out.png> [rows=3] [scale=1]
# Cut a wide panorama into `rows` equal strips stacked top to bottom (left part first), so the whole
# 13:1 picture can be looked at on a normal screen.  `scale` enlarges or reduces the result.
set -euo pipefail
in=$1; out=$2; rows=${3:-3}; scale=${4:-1}
w=$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 "$in")
h=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "$in")
sw=$(( (w + rows - 1) / rows ))
f="[0:v]pad=$((sw * rows)):$h,split=$rows"
for i in $(seq 0 $((rows - 1))); do f="$f[s$i]"; done
f="$f;"
for i in $(seq 0 $((rows - 1))); do f="$f[s$i]crop=$sw:$h:$((sw * i)):0[c$i];"; done
for i in $(seq 0 $((rows - 1))); do f="$f[c$i]"; done
f="${f}vstack=inputs=$rows,scale=iw*$scale:ih*$scale:flags=lanczos"
ffmpeg -v error -y -i "$in" -filter_complex "$f" -frames:v 1 "$out"
