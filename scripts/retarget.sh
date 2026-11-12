#!/usr/bin/env bash
# Relation-preserving retargeting (Adam, 500 steps per batch of clips) of the generated hands.
# Usage: bash scripts/retarget.sh [validation] [test] [train]   (default: all three)
# Input:  data/generated/<split>.npz (scripts/generate_data.sh or scripts/download.sh train-data)
# Output: data/targets/relation/<split>.npz (robot FK landmarks as poses, normalized commands as actions)

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

generated=$(cfg paths.generated)
targets=$(cfg paths.targets)
splits=("$@")
((${#splits[@]})) || splits=(validation test train)
for split in "${splits[@]}"; do
    echo "== retarget $split"
    "${PY[@]}" -m robofs.retarget.retarget --dataset_path "$generated/$split.npz" --output_file "$targets/$split.npz"
done
