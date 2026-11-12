#!/usr/bin/env bash
# Evaluate RoboFS-T2A on the 2000 test words: predict commands from text, execute them open loop in simulation
# (256 parallel environments, the tracker's environment settings) and score the motion with the OpenFS recognizer.
# Expected with the released checkpoint: letter accuracy 81.1%, word accuracy 22.6%.
# Usage: bash scripts/eval_t2a.sh [t2a_checkpoint] [output_dir]
#   (defaults: checkpoints/t2a/model.pt, results/test/t2a)
# Input:  data/targets/relation_net/test.npz (test words, and the generated hands the recognizer maps back to),
#         checkpoints/tracker/params/ (environment settings of the simulation)
# Output: <output_dir>/test_pred.npz (predicted commands), metrics.json, recognition_summary.json, ...

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

checkpoint=${1:-$(cfg paths.t2a_checkpoint)}
output=${2:-$(cfg paths.results)/test/t2a}
test=$(cfg paths.network_targets)/test.npz
rm -rf "$output"
mkdir -p "$output"
"${PY[@]}" -c 'import sys, numpy as np; print("\n".join(np.load(sys.argv[1])["words"].tolist()))' "$test" \
    > "$output/test_words.txt"
"${PY[@]}" -m robofs.t2a.model predict --checkpoint "$checkpoint" --words_file "$output/test_words.txt" \
    --output_file "$output/test_pred.npz"
"${PY[@]}" -m robofs.evaluation.evaluate --method replay --checkpoint "$(cfg paths.tracker_checkpoint)" \
    --replay_file "$output/test_pred.npz" --dataset_path "$test" --recognition_source_dataset "$test" \
    --num_envs "$(cfg evaluation.replay_num_envs)" --output_dir "$output"
report "$output"
