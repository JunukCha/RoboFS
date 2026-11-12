#!/usr/bin/env bash
# Generate the train/validation/test hands with the OpenFS generator and pack them with palm alignment.
# Usage: bash scripts/generate_data.sh [train] [validation] [test]   (default: all three)
# Input:  data/word_lists/<split>_words.txt (bash scripts/download.sh train-data)
# Output: data/generation/<split>/ (per-word OpenFS outputs), data/generated/<split>.npz
# The released splits were sampled one word at a time (--batch_size 1); larger batches give the same samples up
# to float rounding (about 1e-6) and are much faster.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

words=$(cfg paths.word_lists)
generation=$(cfg paths.generation)
generated=$(cfg paths.generated)
splits=("$@")
((${#splits[@]})) || splits=(train validation test)
for split in "${splits[@]}"; do
    echo "== generate $split"
    "${PY[@]}" -m robofs.generation.generate --words_file "$words/${split}_words.txt" \
        --output_dir "$generation/$split" --batch_size "${BATCH_SIZE:-1}" --resume
    echo "== pack $split"
    "${PY[@]}" -m robofs.generation.pack --generation_dir "$generation/$split" --split "$split" \
        --output_file "$generated/$split.npz"
done
