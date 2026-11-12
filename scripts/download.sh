#!/usr/bin/env bash
# Download the released models and data.
# Usage: bash scripts/download.sh [openfs] [models] [eval-data] [train-data] [robofs-data]
#   (default: openfs models eval-data, everything the evaluation needs)
#
#   openfs       OpenFS generator and recognizer (FSboard) -> external/OpenFS/outputs_save/
#   models       Hugging Face model repository Jucha/RoboFS -> checkpoints/
#                  tracker/model_1000.pt, tracker/params/env.yaml, tracker/params/agent.yaml  (RoboFS-Tracker)
#                  retarget_net/model.pt                                                      (retargeting network)
#                  t2a/model.pt                                                               (RoboFS-T2A)
#   eval-data    Hugging Face dataset repository Jucha/RoboFS-Data -> data/
#                  word_lists/test_words.txt
#                  targets/relation/test.npz          (relation-retargeted test targets, tracker evaluation)
#                  targets/relation_net/test.npz      (retargeting-network test targets, T2A evaluation)
#   train-data   Jucha/RoboFS-Data -> data/
#                  word_lists/{train,validation,test,train_50k}_words.txt
#                  generated/{train,validation,test}.npz            (packed OpenFS hands)
#                  targets/relation/{train,validation,test}.npz
#                  targets/relation_net/{validation,test}.npz
#   robofs-data  Jucha/RoboFS-Data -> data/
#                  robofs_data/part_00.npz ... part_09.npz           (RoboFS-Data, 50k words)
# Files that already exist are kept.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

MODEL_REPO=Jucha/RoboFS
DATA_REPO=Jucha/RoboFS-Data

if command -v hf >/dev/null 2>&1; then
    hf_download=(hf download)
elif command -v huggingface-cli >/dev/null 2>&1; then
    hf_download=(huggingface-cli download)
else
    echo "Install the Hugging Face CLI first: pip install -U huggingface_hub" >&2
    exit 1
fi

# fetch <repo> <repo type> <local dir> <file ...>: download the files that are not there yet
fetch() {
    local repo=$1 type=$2 dir=$3 missing=()
    shift 3
    for file in "$@"; do
        [[ -f $dir/$file ]] || missing+=("$file")
    done
    if ((${#missing[@]})); then
        "${hf_download[@]}" "$repo" "${missing[@]}" --repo-type "$type" --local-dir "$dir"
    fi
}

openfs() {
    local out=external/OpenFS/outputs_save tmp model checkpoint
    declare -A ids=(
        [generator_df_fsboard]=1MM0kCtk40kvDHjoWnp2hChw_HrsIVFbs
        [recognizer_fsboard]=1iNdHOBO4HwYVAAs0V9baestX7z6q9Yh0
    )
    mkdir -p "$out"
    tmp=$(mktemp -d)
    for model in generator_df_fsboard recognizer_fsboard; do
        [[ -f $out/$model/best_model.pth ]] && continue
        "${PY[@]}" -m gdown "${ids[$model]}" -O "$tmp/$model.zip"
        unzip -q "$tmp/$model.zip" -d "$tmp/$model"
        checkpoint=$(find "$tmp/$model" -name best_model.pth -type f)
        # The recognizer's Hydra config (.hydra/config.yaml) stays beside its checkpoint.
        mv -T "$(dirname "$checkpoint")" "$out/$model"
    done
    rm -r "$tmp"
}

models() {
    fetch "$MODEL_REPO" model checkpoints \
        tracker/model_1000.pt tracker/params/env.yaml tracker/params/agent.yaml \
        retarget_net/model.pt \
        t2a/model.pt
}

eval_data() {
    fetch "$DATA_REPO" dataset data \
        word_lists/test_words.txt \
        targets/relation/test.npz \
        targets/relation_net/test.npz
}

train_data() {
    fetch "$DATA_REPO" dataset data \
        word_lists/train_words.txt word_lists/validation_words.txt word_lists/test_words.txt \
        word_lists/train_50k_words.txt \
        generated/train.npz generated/validation.npz generated/test.npz \
        targets/relation/train.npz targets/relation/validation.npz targets/relation/test.npz \
        targets/relation_net/validation.npz targets/relation_net/test.npz
}

robofs_data() {
    fetch "$DATA_REPO" dataset data $(printf "robofs_data/part_%02d.npz " $(seq 0 9))
}

groups=("$@")
((${#groups[@]})) || groups=(openfs models eval-data)
for group in "${groups[@]}"; do
    case $group in
        openfs) openfs ;;
        models) models ;;
        eval-data) eval_data ;;
        train-data) train_data ;;
        robofs-data) robofs_data ;;
        *) echo "Unknown group: $group (openfs, models, eval-data, train-data, robofs-data)" >&2; exit 1 ;;
    esac
done
echo "Downloaded: ${groups[*]}"
