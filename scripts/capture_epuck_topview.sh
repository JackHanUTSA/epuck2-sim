#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 WORLD_FILE OUTPUT_MP4 [frame_count]" >&2
  exit 2
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORLD_INPUT="$1"
OUTPUT_MP4="$2"
FRAME_COUNT="${3:-220}"

if [[ "$WORLD_INPUT" != /* ]]; then
  WORLD_INPUT="$REPO_DIR/$WORLD_INPUT"
fi

if [[ ! -f "$WORLD_INPUT" ]]; then
  echo "world not found: $WORLD_INPUT" >&2
  exit 1
fi

world_base="$(basename "$WORLD_INPUT" .wbt)"
out_dir="$(mktemp -d /tmp/${world_base}_topview_XXXXXX)"
frames_dir="$out_dir/frames"
mkdir -p "$frames_dir"

capture_world="$WORLD_INPUT"
if [[ "$world_base" == "epuck2_obstacle_avoidance" ]]; then
  capture_world="$REPO_DIR/worlds/epuck2_obstacle_avoidance_capture.wbt"
fi

SWARM_CAPTURE_MODE=camera_sequence \
SWARM_CAPTURE_DIR="$frames_dir" \
SWARM_FRAME_COUNT="$FRAME_COUNT" \
SWARM_FRAME_STRIDE=2 \
SWARM_SETTLE_STEPS=60 \
SWARM_CAMERA_TRANSLATION='0,0,1.6' \
SWARM_CAMERA_ROTATION='0,1,0,1.5708' \
"$REPO_DIR/scripts/launch_webots_headless.sh" "$capture_world" --batch --mode=realtime --stdout --stderr >"$out_dir/webots_capture.log" 2>&1

mkdir -p "$(dirname "$OUTPUT_MP4")"
ffmpeg -y -framerate 25 -i "$frames_dir/frame_%04d.png" -c:v libx264 -pix_fmt yuv420p "$OUTPUT_MP4" >"$out_dir/ffmpeg_encode.log" 2>&1

sample_png="${OUTPUT_MP4%.mp4}-frame.png"
ffmpeg -y -i "$OUTPUT_MP4" -vf "select=eq(n\,160)" -vframes 1 "$sample_png" >"$out_dir/frame_extract.log" 2>&1

cat <<EOF
capture_world=$capture_world
video=$OUTPUT_MP4
sample_frame=$sample_png
logs=$out_dir
EOF
