#!/usr/bin/env bash
# Train RoboFS-Tracker with PPO in Isaac Lab (1024 environments, 1000 iterations, seed 42) on the
# relation-retargeted train targets, and copy the final checkpoint with its saved settings.
# Usage: bash scripts/train_tracker.sh [checkpoint_dir] [extra train arguments...]
#   (default checkpoint_dir: checkpoints/tracker; it must not hold a model_1000.pt yet)
# Input:  data/targets/relation/train.npz
# Output: logs/rsl_rl/shadow_hand_landmark_tracking/<date>_robofs_tracker/ (full run),
#         <checkpoint_dir>/model_1000.pt and <checkpoint_dir>/params/{env,agent}.yaml

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

checkpoint=${1:-$(dirname "$(cfg paths.tracker_checkpoint)")}
shift $(($# > 0 ? 1 : 0))
targets=$(cfg paths.targets)
iterations=$(cfg tracker.max_iterations)
if [[ -f $checkpoint/model_$iterations.pt ]]; then
    echo "$checkpoint/model_$iterations.pt exists; remove it or pass another checkpoint_dir." >&2
    exit 1
fi
run_name=robofs_tracker
"${PY[@]}" -m robofs.tracker.train --dataset_path "$targets/train.npz" --run_name "$run_name" "$@"
run=$(ls -d logs/rsl_rl/shadow_hand_landmark_tracking/*_"$run_name" | tail -1)
mkdir -p "$checkpoint"
cp "$run/model_$iterations.pt" "$checkpoint/"
cp -r "$run/params" "$checkpoint/"
echo "Saved $checkpoint/model_$iterations.pt"
