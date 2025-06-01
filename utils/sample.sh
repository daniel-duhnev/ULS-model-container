#!/usr/bin/env bash
#
# sample.sh — subsample files from one dir into another
#
# Usage:
#   ./sample.sh SRC_DIR DST_DIR [STEP [ACTION]]
#
#   SRC_DIR   directory containing your images
#   DST_DIR   directory to receive the sampled images
#   STEP      take every STEP-th file (default: 7)
#   ACTION    "copy" or "move" (default: copy)
#

# --- parse arguments ---
SRC="$1"
DST="$2"
STEP="${3:-7}"
ACTION="${4:-copy}"

if [[ -z "$SRC" || -z "$DST" ]]; then
  echo "Usage: $0 SRC_DIR DST_DIR [STEP [ACTION]]"
  exit 1
fi

mkdir -p "$DST"

i=0
shopt -s nullglob
for file in "$SRC"/*; do
  (( i++ ))
  if (( i % STEP == 0 )); then
    if [[ "$ACTION" == "move" ]]; then
      mv -- "$file" "$DST/"
    else
      cp -- "$file" "$DST/"
    fi
  fi
done
