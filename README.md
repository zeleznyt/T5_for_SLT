# T5_for_SLT

## Overview

This repository contains code for pose-based sign language translation with a T5-style encoder-decoder model. It includes training, evaluation, prediction, pose preprocessing, and configuration files for experiments with sign-language keypoint features.

The current default workflow trains from raw keypoint JSON files. The older H5 feature workflow is still supported for existing experiments.

## Citation

If you use this repository, please cite this paper:

```bibtex
@misc{zelezny2025exploring,
  title         = {Exploring Pose-based Sign Language Translation: Ablation Studies and Attention Insights},
  author        = {Zelezny, Tomas and Straka, Jakub and Javorek, Vaclav and Valach, Ondrej and Hruz, Marek and Gruber, Ivan},
  year          = {2025},
  eprint        = {2507.01532},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  doi           = {10.48550/arXiv.2507.01532}
}
```

## Used In

This codebase was used in:

- Exploring Pose-based Sign Language Translation: Ablation Studies and Attention Insights. arXiv:2507.01532. https://arxiv.org/abs/2507.01532
- Saudi Sign Language Translation Using T5. International Conference on Speech and Computer, 2025. https://arxiv.org/abs/2510.11183

## Installation

```bash
cd T5_for_SLT/
conda create -n t5slt python=3.12
conda activate t5slt
pip install -r requirements.txt
```

Run scripts from the repository root. If needed, add the project to `PYTHONPATH`:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

On PowerShell:

```powershell
$env:PYTHONPATH = "$env:PYTHONPATH;$PWD"
```

## Data

The training and prediction scripts expect an annotation file and visual features for each split. The recommended workflow uses raw keypoint JSON files directly.

We provide the YouTube-ASL Clip Keypoint Dataset used in our research. It contains keypoints extracted from video clips that were publicly available at the time of collection and that passed our preprocessing and filtering pipeline. In total, the dataset includes 390,547 clips. It is publicly available at: http://hdl.handle.net/11234/1-5898.

If you use this dataset, please cite this repo and:

```bibtex
@misc{11234/1-5898,
  title     = {{YouTube}-{ASL} Clip Keypoint Dataset},
  author    = {Zelezny, Tomas and Hruz, Marek and Straka, Jaub and Gueuwou, Shester},
  url       = {http://hdl.handle.net/11234/1-5898},
  note      = {{LINDAT}/{CLARIAH}-{CZ} digital library at the Institute of Formal and Applied Linguistics ({{\'U}FAL}), Faculty of Mathematics and Physics, Charles University},
  copyright = {Creative Commons - Attribution 4.0 International ({CC} {BY} 4.0)},
  year      = {2024}
}
```

### Recommended Raw JSON Layout

Example YouTube-ASL-style layout:

```text
YT-ASL/
|-- YT.annotations.train.json
|-- YT.annotations.dev.json
|-- YT.annotations.test.json
|-- raw_keypoints/
|   |-- clip_000001.json
|   |-- clip_000002.json
|   |-- ...
```

The config points to this data with:

```yaml
SignDataArguments:
  data_dir: /path/to/YT-ASL
  annotation_path:
    train: YT.annotations.train.json
    dev: YT.annotations.dev.json
    test: YT.annotations.test.json
  visual_features:
    pose:
      enable_input: True
      normalization:
        train_json_dir: raw_keypoints
        dev_json_dir: raw_keypoints
        test_json_dir: raw_keypoints
        normalization_method: sign_space
        data_key: cropped_keypoints
```

`train_json_dir`, `dev_json_dir`, and `test_json_dir` may be absolute paths or paths relative to `data_dir`. The loader reads all `*.json` files directly inside those directories. The directory name is not inferred from `data_key`.

Each raw keypoint JSON file is one clip. Its filename without `.json` must match a clip name in the annotation file. For example, `clip_000001.json` provides features for annotation clip `clip_000001`.

The `data_key` value selects the keypoint field inside each JSON file. It is commonly `cropped_keypoints`, but it can be changed if your JSON files use a different inner key.

Expected raw keypoint JSON structure:

```json
{
  "cropped_keypoints": [
    {
      "pose_landmarks": [[0.1, 0.2], [0.3, 0.4]],
      "right_hand_landmarks": [[0.1, 0.2], [0.3, 0.4]],
      "left_hand_landmarks": [[0.1, 0.2], [0.3, 0.4]],
      "face_landmarks": [[0.1, 0.2], [0.3, 0.4]]
    }
  ]
}
```

Empty landmark lists are allowed and are replaced with the configured `missing_values` behavior.

### Annotation Format

The annotation file groups clips by video and stores the translation for each clip:

```json
{
  "video_id": {
    "clip_order": ["clip_000001", "clip_000002"],
    "clip_000001": {
      "translation": "example translation"
    },
    "clip_000002": {
      "translation": "another translation"
    }
  }
}
```

For training, the code uses `annotation_path.train` and `annotation_path.dev`. For prediction, it uses the split selected by `EvaluationArguments.split`.

### Legacy H5 Layout

H5 features are still supported. If no raw JSON directory is configured or found for the pose split, `SignFeatureDataset` falls back to the existing metadata/H5 path when the corresponding modality metadata key is present.

Example H5 layout:

```text
YT-ASL/
|-- YouTubeASL.annotation.train.json
|-- YouTubeASL.annotation.dev.json
|-- keypoints/
|   |-- YouTubeASL.keypoints.train.json
|   |-- YouTubeASL.keypoints.train.0.h5
|   |-- YouTubeASL.keypoints.train.1.h5
|   |-- YouTubeASL.keypoints.dev.json
|   |-- YouTubeASL.keypoints.dev.0.h5
|   |-- YouTubeASL.keypoints.dev.1.h5
```

The metadata JSON stores a `{video_id: shard_id}` mapping. The metadata filename must match the shard filenames after replacing `.json` with `.{shard_id}.h5`.

Each H5 shard has this structure:

```text
{
  video_id: {
    clip_name: numpy.array,
    ...
  },
  ...
}
```

## Usage

### Training

Edit `configs/config.yaml` or create a copy with your dataset paths, then run:

```bash
python train/run_finetuning.py --config_file configs/config.yaml
```

For a short local smoke test, edit `SignDataArguments.data_dir` in `configs/config_for_testing.yaml`, then run:

```bash
python train/run_finetuning.py --config_file configs/config_for_testing.yaml --verbose
```

Training from scratch is controlled by:

```yaml
resume_from_checkpoint: none
load_only_weights: False
```

To load only model weights from a checkpoint while starting a new optimizer/scheduler state:

```bash
python train/run_finetuning.py \
  --config_file configs/config.yaml \
  --resume_from_checkpoint /path/to/checkpoint \
  --load_only_weights True
```

### Prediction

```bash
python eval/predict.py --config_file configs/predict_config.yaml
```

Predictions and scores are written under `EvaluationArguments.output_dir`.

Use `--verbose` for extended output. For full config details, see [configs/README.md](configs/README.md).
