# Shared helpers of the RoboFS scripts (sourced, not run).
# Python interpreter of the Isaac Lab environment; override with e.g. PYTHON="/path/to/IsaacLab/isaaclab.sh -p".

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
read -ra PY <<< "${PYTHON:-python}"

# cfg <dotted key>: print one value of configs/robofs.yaml, e.g. cfg paths.targets
cfg() {
    "${PY[@]}" -c 'import sys
from robofs.config import load_config
value = load_config()
for part in sys.argv[1].split("."):
    value = value[part]
print(value)' "$1"
}

# report <evaluation dir>: print letter and word accuracy of an evaluation
report() {
    "${PY[@]}" -c 'import json, sys
r = json.load(open(sys.argv[1] + "/metrics.json"))["physical_metrics"]["recognition"]
letter, word, count = r["pooled_character_accuracy_percent"], r["word_accuracy_percent"], r["words"]
print(f"{sys.argv[1]}: letter accuracy {letter:.1f}%, word accuracy {word:.1f}% over {count} words")' "$1"
}
