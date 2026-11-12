#!/usr/bin/env bash
# Train the retargeting network on the relation-retargeted train split, then turn its validation and test
# predictions into tracker targets (forward kinematics only).
# Usage: bash scripts/train_retarget_net.sh [checkpoint_dir]   (default: checkpoints/retarget_net)
#   Training is skipped when <checkpoint_dir>/model.pt exists (e.g. the downloaded model).
# Input:  data/generated/{train,validation,test}.npz, data/targets/relation/{train,validation}.npz
# Output: <checkpoint_dir>/{model.pt,history.json},
#         data/targets/relation_net/{validation,test}.npz (+ <split>_pred_actions.npz, the raw predictions)

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

generated=$(cfg paths.generated)
targets=$(cfg paths.targets)
network_targets=$(cfg paths.network_targets)
checkpoint=${1:-$(dirname "$(cfg paths.retarget_net_checkpoint)")}
mkdir -p "$checkpoint" "$network_targets"
if [[ ! -f $checkpoint/model.pt ]]; then
    echo "== train the retargeting network"
    "${PY[@]}" -m robofs.retarget.network train --dataset_path "$generated/train.npz" \
        --target_file "$targets/train.npz" --val_dataset_path "$generated/validation.npz" \
        --val_target_file "$targets/validation.npz" --output_dir "$checkpoint"
fi
for split in validation test; do
    echo "== network targets for $split"
    predictions=$network_targets/${split}_pred_actions.npz
    rm -f "$predictions" "$network_targets/$split.npz"
    "${PY[@]}" -m robofs.retarget.network predict --checkpoint "$checkpoint/model.pt" \
        --dataset_path "$generated/$split.npz" --output_file "$predictions"
    "${PY[@]}" -m robofs.retarget.retarget --dataset_path "$generated/$split.npz" --init_file "$predictions" \
        --steps 0 --output_file "$network_targets/$split.npz"
done
