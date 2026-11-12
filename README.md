# RoboFS

Code for **RoboFS: Scalable Data Generation and Text-to-Action Learning for Robot Fingerspelling** (WACV 2027).

Paper: TODO | Project page: TODO | Models: Hugging Face `Jucha/RoboFS` (TODO: URL) | Data: Hugging Face
`Jucha/RoboFS-Data` (TODO: URL)

RoboFS turns written words into fingerspelling motions of a Shadow Hand with a fixed wrist, simulated in Isaac Lab.
The pipeline has four stages:

1. **Generation.** The OpenFS generator produces a hand landmark sequence for a word (8 frames per letter, 30 Hz).
   Palm alignment expresses each frame in a palm-local frame scaled to a 0.10 m wrist-to-middle-knuckle length.
2. **Relation-preserving retargeting.** An optimization (Adam, 500 steps) finds joint-limited robot commands whose
   forward-kinematics landmarks keep the generated hand's bone directions (weight 3 rho^2, rho = 25 mm) and
   fingertip-to-fingertip vectors (weight 2), with a frame-to-frame smoothness term (weight 1e-3). A retargeting
   network (a temporal 1D CNN) is trained on its outputs and replaces it when large data sets are built.
3. **RoboFS-Tracker.** A PPO policy trained in Isaac Lab tracks the retargeted landmarks in physics simulation. It
   observes the targets 0.1, 0.2 and 0.33 s ahead and is rewarded for landmark (weight 3, sigma 25 mm) and
   fingertip (weight 2, sigma 40 mm) tracking, with penalties on self-penetration (-3) and action changes (-0.005).
4. **RoboFS-Data and RoboFS-T2A.** The tracker's executed commands for 50,000 words form RoboFS-Data. RoboFS-T2A, a
   character-to-action Transformer trained on it, predicts robot commands directly from text; they are executed
   open loop in simulation.

Motions are scored with the OpenFS fingerspelling recognizer after mapping the robot landmarks back to the
generated hand's image frame. Letter accuracy follows OpenFS (HTK-cost alignment, errors pooled over all words);
word accuracy is the exact-match rate.

| Test set (2000 words) | Letter accuracy (%) | Word accuracy (%) |
|---|---|---|
| RoboFS-Tracker | 77.4 | 17.0 |
| RoboFS-T2A | 81.1 | 22.6 |

## Installation

RoboFS runs in Isaac Lab 3.0.0-rc1 (Isaac Sim, Newton physics, RSL-RL) on a Linux machine with an NVIDIA GPU. The
simplest setup is the Isaac Lab container `nvcr.io/nvidia/isaac-lab:3.0.0-rc1`.

```bash
git clone --recurse-submodules TODO_REPOSITORY_URL RoboFS
cd RoboFS
# Python of the Isaac Lab installation; in the container it is /workspace/isaaclab/_isaac_sim/python.sh.
export PYTHON=/workspace/isaaclab/_isaac_sim/python.sh
$PYTHON -m pip install -r requirements.txt
```

All scripts use `$PYTHON` (default `python`) and are run from the repository root. The OpenFS code is the
submodule `external/OpenFS` (if the repository was cloned without `--recurse-submodules`, run
`git submodule update --init`). All final settings are in `configs/robofs.yaml`; the Python entry points read their
defaults from it, so the scripts only pass paths.

The CPU tests need no simulator: `$PYTHON -m pytest tests`.

## Download

```bash
bash scripts/download.sh                 # OpenFS models, RoboFS models and the test data (enough for evaluation)
bash scripts/download.sh train-data      # word lists, generated hands and targets of all splits
bash scripts/download.sh robofs-data     # RoboFS-Data (10 shards, 50k words)
```

| Group | Source | Files |
|---|---|---|
| `openfs` | OpenFS (Google Drive) | `external/OpenFS/outputs_save/{generator_df_fsboard,recognizer_fsboard}/` |
| `models` | `Jucha/RoboFS` | `checkpoints/tracker/model_1000.pt`, `checkpoints/tracker/params/{env,agent}.yaml`, `checkpoints/retarget_net/model.pt`, `checkpoints/t2a/model.pt` |
| `eval-data` | `Jucha/RoboFS-Data` | `data/word_lists/test_words.txt`, `data/targets/relation/test.npz`, `data/targets/relation_net/test.npz` |
| `train-data` | `Jucha/RoboFS-Data` | `data/word_lists/*.txt`, `data/generated/*.npz`, `data/targets/relation/*.npz`, `data/targets/relation_net/*.npz` |
| `robofs-data` | `Jucha/RoboFS-Data` | `data/robofs_data/part_00.npz` ... `part_09.npz` |

The Hugging Face files are fetched with `hf download` (or `huggingface-cli download`) from `huggingface_hub`.

## Evaluation

```bash
bash scripts/eval_tracker.sh   # RoboFS-Tracker on the relation-retargeted test targets: 77.4 / 17.0
bash scripts/eval_t2a.sh       # RoboFS-T2A predictions executed open loop: 81.1 / 22.6
```

Each run executes all 2000 test words in parallel environments (64 for the tracker, 256 for open-loop replay):
a 1 s hold of the first pose, then the motion at 30 Hz. Frames of the hold are not scored. Results go to
`results/test/tracker/` and `results/test/t2a/`:

- `metrics.json`: pooled landmark and fingertip errors, self-penetration frame ratio and depth, fingertip jerk,
  and under `physical_metrics.recognition` the letter accuracy (`pooled_character_accuracy_percent`) and word
  accuracy (`word_accuracy_percent`); per-word reports under `per_word`.
- `recognition_summary.json` (also for the target and generated-hand streams), `mean_metrics.json` (word means)
  and `recognition_per_word.csv`.
- For the tracker, `word_<i>/` folders with the recorded trajectory of every word.

`eval_t2a.sh` takes the test words from `data/targets/relation_net/test.npz`, whose generated hands the recognizer
uses to map the robot motion back to the image frame, and runs the simulation with the tracker's saved environment
settings (`checkpoints/tracker/params/`). Both scripts accept another checkpoint and output folder as arguments.

## Training

The released models can be reproduced in this order. Each step reads the outputs of the previous ones, and each
can be started from the downloaded files instead (`bash scripts/download.sh train-data` provides the inputs of
steps 2 to 4, `robofs-data` those of step 6).

| Step | Command | Output |
|---|---|---|
| 1. Generate and align hands (train 10,000, validation 1,000, test 2,000 words) | `bash scripts/generate_data.sh` | `data/generated/{train,validation,test}.npz` |
| 2. Relation-preserving retargeting | `bash scripts/retarget.sh` | `data/targets/relation/{train,validation,test}.npz` |
| 3. Retargeting network, and its validation/test targets | `bash scripts/train_retarget_net.sh` | `checkpoints/retarget_net/model.pt`, `data/targets/relation_net/{validation,test}.npz` |
| 4. RoboFS-Tracker (PPO, 1024 environments, 1000 iterations, seed 42) | `bash scripts/train_tracker.sh` | `checkpoints/tracker/model_1000.pt`, `checkpoints/tracker/params/` |
| 5. RoboFS-Data (50k words, 10 parts) | `bash scripts/build_robofs_data.sh` | `data/robofs_data/part_{00..09}.npz` |
| 6. RoboFS-T2A (40,000 steps) | `bash scripts/train_t2a.sh` | `checkpoints/t2a/model.pt` |

Notes:

- Step 1 samples each word with its own seed (the base seed 0 plus a hash of the word), so the samples do not
  depend on the word order. The released splits were sampled one word at a time; `BATCH_SIZE=256 bash
  scripts/generate_data.sh` is much faster and gives the same samples up to float rounding (about 1e-6).
- Step 3 skips training when `checkpoints/retarget_net/model.pt` exists (e.g. the downloaded model) and only
  rebuilds the network targets. Steps 4 and 6 refuse to overwrite an existing checkpoint; pass another folder as
  the first argument (`bash scripts/train_tracker.sh checkpoints/my_tracker`).
- Step 4 writes the full Isaac Lab run (TensorBoard logs) to `logs/rsl_rl/shadow_hand_landmark_tracking/`.
- Step 5 runs per part: generation (batches of 256 words), palm alignment, retargeting-network prediction, forward
  kinematics, a tracker rollout in simulation (256 environments) and packing. Finished parts are skipped, so the
  script can be resumed, and parts can be built separately (`bash scripts/build_robofs_data.sh 0 1`). It uses
  `checkpoints/tracker/model_1000.pt` and `checkpoints/retarget_net/model.pt`; `TRACKER_CHECKPOINT` and
  `RETARGET_NET_CHECKPOINT` select others.
- Step 6 holds out 500 words of the shards for validation losses.

The entry points can also be called directly (`$PYTHON -m robofs.<module> --help`):

| Module | Purpose |
|---|---|
| `robofs.generation.generate`, `robofs.generation.pack` | OpenFS generation; palm alignment and packing |
| `robofs.retarget.retarget` | relation-preserving retargeting (also forward kinematics of given commands) |
| `robofs.retarget.network` | retargeting network (`train`, `predict`) |
| `robofs.tracker.train` | RoboFS-Tracker training |
| `robofs.evaluation.evaluate` | simulated evaluation of the tracker (`--method ppo`) or of stored commands (`--method replay`) |
| `robofs.t2a.pack`, `robofs.t2a.model` | RoboFS-Data packing; RoboFS-T2A (`train`, `predict`) |

## Data formats

All files are NumPy `.npz` archives; `metadata` is a JSON string. Landmarks use the MediaPipe order (0 wrist, then
four points per finger from thumb to little finger) and are given in meters in the body frame of the Shadow Hand
palm (`rh_palm`). Sequences are 30 Hz and padded to the longest one by repeating their last frame; `lengths` gives
the valid frames.

**Commands** (18 values per frame) are the policy's normalized actions in [-1, 1]: 14 joint position targets
(normalized to the soft joint limits) followed by 4 fixed-tendon position targets of the coupled finger joints.

**Generated hands** (`data/generated/<split>.npz`): `poses` [N, T, 21, 3] palm-aligned landmarks [m], `raw_poses`
[N, T, 21, 3] OpenFS output in its normalized image coordinates, `lengths` [N], `words` [N], `metadata`.

**Targets** (`data/targets/relation/<split>.npz`, `data/targets/relation_net/<split>.npz`): the generated-hand file
with `poses` replaced by the robot's forward-kinematics landmarks [m] and `actions` [N, T, 18] added (the
retargeted commands).

**RoboFS-Data** (`data/robofs_data/part_XX.npz`, 5,000 words each, 50,000 in total). Each word holds the motion
phase of a tracker rollout: frame `f` is the command sent for motion frame `f` (the 1 s hold before it is not
stored), so `actions` can be replayed open loop as they are.

| Key | Shape | Content |
|---|---|---|
| `words` | [N] | word (lower case) |
| `lengths` | [N] | frames of each word (8 per letter) |
| `actions` | [N, T, 18] | executed normalized commands |
| `joint_positions_rad` | [N, T, 24] | simulated joint positions [rad] |
| `landmarks` | [N, T, 21, 3] | simulated robot landmarks [m], palm frame |
| `character_accuracy_percent` | [N] | recognizer letter accuracy of the simulated motion (unit-cost edit distance) |
| `edit_distance` | [N] | edit distance of the recognized word |
| `penetration_frame_percent` | [N] | frames with self-penetration deeper than 1 mm [%] |
| `fingertip_jerk_rms_m_s3` | [N] | RMS fingertip jerk [m/s^3] |
| `metadata` | | frame rate, command layout and the source rollout |

The quality scores are stored for analysis; RoboFS-T2A is trained on all words.

## License

The code, models and data are released under the RoboFS Non-Commercial Research License (see `LICENSE`), for
non-commercial research and education only. Parts of the code are based on Isaac Lab (BSD-3-Clause; the notice is
included in `LICENSE`). The OpenFS generator and recognizer are under the OpenFS Non-Commercial Research License,
and the use of RoboFS grants no rights to OpenFS or to the source data sets listed below.

## Data sources and credits

The words of the experiments come from the label sets of FSboard (CC BY 4.0), ChicagoFSWild, ChicagoFSWild+ and
FSNeo: training, validation and test words were sampled from the corresponding splits with no word shared between
sets. The FSNeo (neobench) words were collected by the OpenFS authors from Reddit, news articles, the New York Times
and dictionaries. The human landmarks were generated with the OpenFS generator, and recognition uses the OpenFS
recognizer; no video or landmark data of the source data sets is redistributed.

Users are responsible for complying with the licenses and terms of these data sets. Please also cite OpenFS when
using RoboFS. The simulation builds on Isaac Lab, Newton and RSL-RL, and uses the Shadow Hand model of Isaac Lab.

## Citation

```bibtex
@inproceedings{kim2027robofs,
  title     = {{RoboFS}: Scalable Data Generation and Text-to-Action Learning for Robot Fingerspelling},
  author    = {Kim, Jihyeon and Cha, Junuk},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2027}
}
```
