#!/usr/bin/env bash
# Evaluate RoboFS-Tracker on the 2000 test words (relation-retargeted targets, 64 parallel environments,
# deterministic actions) and score the simulated motion with the OpenFS recognizer.
# Expected with the released checkpoint: letter accuracy 77.4%, word accuracy 17.0%.
# Usage: bash scripts/eval_tracker.sh [checkpoint] [output_dir]
#   (defaults: checkpoints/tracker/model_1000.pt, results/test/tracker)
# Input:  data/targets/relation/test.npz, the tracker checkpoint with params/ beside it
# Output: <output_dir>/metrics.json, recognition_summary.json, mean_metrics.json, recognition_per_word.csv,
#         word_<i>/ folders with per-word trajectories

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

checkpoint=${1:-$(cfg paths.tracker_checkpoint)}
output=${2:-$(cfg paths.results)/test/tracker}
test=$(cfg paths.targets)/test.npz
rm -rf "$output"
"${PY[@]}" -m robofs.evaluation.evaluate --method ppo --checkpoint "$checkpoint" --dataset_path "$test" \
    --recognition_source_dataset "$test" --num_envs "$(cfg evaluation.tracker_num_envs)" --save_trajectories \
    --output_dir "$output"
report "$output"
