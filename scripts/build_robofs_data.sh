#!/usr/bin/env bash
# Build RoboFS-Data: (text, robot action) pairs for the 50k-word list, in 10 parts of 5000 consecutive words.
# Per part: OpenFS generator -> palm alignment -> retargeting network -> forward kinematics -> RoboFS-Tracker
# rollout in simulation (256 environments) -> one shard of executed commands.
# Usage: bash scripts/build_robofs_data.sh [part ...]   (parts 0-9, default: all; finished parts are skipped)
#   TRACKER_CHECKPOINT and RETARGET_NET_CHECKPOINT select other models (default: configs/robofs.yaml paths).
# Input:  data/word_lists/train_50k_words.txt, the tracker and retargeting-network checkpoints
# Output: data/robofs_data/part_XX.npz; intermediate files in data/robofs_data/work/part_XX/ and the rollouts
#         (per-word trajectories and recognition) in results/robofs_data/part_XX/

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

words=$(cfg paths.word_lists)/train_50k_words.txt
out=$(cfg paths.robofs_data)
results=$(cfg paths.results)/robofs_data
tracker=${TRACKER_CHECKPOINT:-$(cfg paths.tracker_checkpoint)}
retarget_net=${RETARGET_NET_CHECKPOINT:-$(cfg paths.retarget_net_checkpoint)}
chunk=$(cfg robofs_data.chunk_words)
num_envs=$(cfg robofs_data.num_envs)
batch_size=$(cfg generation.batch_size)
parts=("$@")
((${#parts[@]})) || parts=(0 1 2 3 4 5 6 7 8 9)
for part in "${parts[@]}"; do
    name=$(printf "part_%02d" "$part")
    [[ -f $out/$name.npz ]] && { echo "== $name exists, skipped"; continue; }
    work=$out/work/$name
    rollout=$results/$name/rollout
    mkdir -p "$work"
    sed -n "$((part * chunk + 1)),$(((part + 1) * chunk))p" "$words" > "$work/words.txt"
    echo "== $name: generate $(wc -l < "$work/words.txt") words"
    "${PY[@]}" -m robofs.generation.generate --words_file "$work/words.txt" --output_dir "$work/generation" \
        --batch_size "$batch_size" --resume
    rm -f "$work/hand.npz" "$work/pred_actions.npz" "$work/target.npz"
    "${PY[@]}" -m robofs.generation.pack --generation_dir "$work/generation" --split train \
        --output_file "$work/hand.npz"
    echo "== $name: retargeting network and forward kinematics"
    "${PY[@]}" -m robofs.retarget.network predict --checkpoint "$retarget_net" --dataset_path "$work/hand.npz" \
        --output_file "$work/pred_actions.npz"
    "${PY[@]}" -m robofs.retarget.retarget --dataset_path "$work/hand.npz" --init_file "$work/pred_actions.npz" \
        --steps 0 --output_file "$work/target.npz"
    echo "== $name: tracker rollout"
    rm -rf "$rollout"
    "${PY[@]}" -m robofs.evaluation.evaluate --method ppo --checkpoint "$tracker" \
        --dataset_path "$work/target.npz" --recognition_source_dataset "$work/target.npz" \
        --num_envs "$num_envs" --save_trajectories --output_dir "$rollout"
    report "$rollout"
    echo "== $name: pack"
    "${PY[@]}" -m robofs.t2a.pack --rollout_dir "$rollout" --output_file "$out/$name.npz"
done
