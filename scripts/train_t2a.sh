#!/usr/bin/env bash
# Train RoboFS-T2A on the RoboFS-Data shards (40k steps, batch 128, seed 42; 500 shard words held out for
# validation losses).
# Usage: bash scripts/train_t2a.sh [checkpoint_dir]   (default: checkpoints/t2a; it must not hold a model.pt yet)
# Input:  data/robofs_data/part_00.npz ... part_09.npz (scripts/build_robofs_data.sh or download.sh robofs-data)
# Output: <checkpoint_dir>/model.pt and the training history

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

checkpoint=${1:-$(dirname "$(cfg paths.t2a_checkpoint)")}
shards=$(cfg paths.robofs_data)
if [[ -f $checkpoint/model.pt ]]; then
    echo "$checkpoint/model.pt exists; remove it or pass another checkpoint_dir." >&2
    exit 1
fi
mkdir -p "$checkpoint"
# The shards are read in name order (part_00 first); the validation words are drawn from all of them.
"${PY[@]}" -m robofs.t2a.model train --shards "$shards"/part_*.npz --output_dir "$checkpoint"
