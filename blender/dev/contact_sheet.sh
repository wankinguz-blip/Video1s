#!/bin/bash
# sheet.sh <dir> <out.png> <cols> [height]  — tile all PNGs of a directory (sorted) into a contact sheet
d=$1; out=$2; cols=$3; h=${4:-500}
n=$(ls $d/*.png | wc -l); rows=$(( (n + cols - 1) / cols ))
ffmpeg -v error -y -pattern_type glob -i "$d/*.png" -vf "scale=-1:$h,tile=${cols}x${rows}" -frames:v 1 "$out"
