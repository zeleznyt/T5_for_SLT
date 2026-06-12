# Config Reference

The scripts load YAML config files and then apply command-line overrides for matching runtime keys. Values written as `none` or `None` are converted to Python `None` by the training and prediction scripts.

## Training Config

Training configs use these top-level sections:

```yaml
ModelArguments:
TrainingArguments:
SignDataArguments:
SignModelArguments:
```

### ModelArguments

- `base_model_name`: Hugging Face T5/mT5 model name used by the tokenizer and base model.
- `hidden_dropout_prob`: Dropout probability in the sign input projection path.
- Generation defaults such as `num_beams`, `max_length`, `length_penalty`, `early_stopping`, and `no_repeat_ngram_size` are copied into the custom T5 config.

### TrainingArguments

- `project_name`: Weights & Biases project name when `report_to: wandb`.
- `model_name`: Run name and output checkpoint folder name.
- `output_dir`: Directory where checkpoints and validation outputs are saved.
- `seed`: Random seed.
- `resume_from_checkpoint`: Checkpoint path for resuming or loading weights. Use `none` to train from scratch.
- `load_only_weights`: If `True`, load model weights from `resume_from_checkpoint` but start a fresh optimizer and scheduler state.
- `freeze_t5`: If `True`, freeze the base T5 parameters.
- `report_to`: Use `wandb` for Weights & Biases logging. Any other value disables WandB in the training script.
- `logging_steps`, `eval_steps`, `save_steps`: Trainer logging/evaluation/checkpoint intervals.
- `push_to_hub`: Passed to Hugging Face trainer arguments.
- `max_train_samples`, `max_val_samples`: Optional dataset limits for debugging.
- `per_device_train_batch_size`, `per_device_eval_batch_size`: Batch sizes per device.
- `gradient_accumulation_steps`: Gradient accumulation factor.
- `learning_rate`, `lr_scheduler_type`, `max_training_steps`, `warmup_steps`, `weight_decay`: Optimizer and schedule settings.
- `fp16`: Enable mixed precision training.
- `dataloader_num_workers`: Number of DataLoader worker processes.
- `max_sequence_length`: Maximum number of input frames after cropping.
- `max_token_length`: Tokenizer max length for target translations.
- `skip_frames`: `False`, `True` for every second frame, or an integer stride.
- `float32`: If `False`, raw pose arrays are reduced to float16 before conversion back to torch float tensors.
- `decimal_points`: Optional rounding precision for pose features. Use `-1` to disable.
- `use_paraphrases`: If `True`, sample from `paraphrases` plus `translation` when available in annotations.
- `num_beams`, `early_stopping`, `no_repeat_ngram_size`, `bleu_effective_order`: Evaluation/generation settings used after training.

### SignDataArguments

`data_dir` is the data root. Relative paths inside `annotation_path`, visual-feature metadata keys, and raw JSON directory keys are resolved against it.

```yaml
SignDataArguments:
  data_dir: /path/to/data
  annotation_path:
    train: annotations.train.json
    dev: annotations.dev.json
    test: annotations.test.json
```

`annotation_path.train` and `annotation_path.dev` are required for training. `annotation_path.test` is optional and enables the post-training test evaluation block when present.

### Raw Pose JSON

Raw JSON is the recommended pose input format. Configure it under `visual_features.pose.normalization`:

```yaml
visual_features:
  pose:
    enable_input: True
    normalization:
      train_json_dir: raw_keypoints
      dev_json_dir: raw_keypoints
      test_json_dir: raw_keypoints
      normalization_method: sign_space
      data_key: cropped_keypoints
    augmentation_type: none
    missing_values: null
    interpolate: -1
```

- `train_json_dir`, `dev_json_dir`, `test_json_dir`: Directories containing raw `*.json` clip files. They can be absolute paths or relative to `data_dir`.
- `data_key`: Field inside each JSON file that contains the frame-level keypoints, for example `cropped_keypoints`. It is not used as a directory name.
- `normalization_method`: `sign_space`, `yasl`, `yasl2`, or an empty string for no normalization.
- `augmentation_type`: Name resolved by `utils/augmentation_config.py`.
- `missing_values`: Replacement value for missing landmarks. Use `null` for the default behavior.
- `interpolate`: Maximum missing-sequence length to interpolate. Use `-1` to disable.

The raw JSON filename without `.json` must match the clip name in the annotation file.

### Legacy H5 Features

H5-backed features are still supported. Keep the metadata keys under each modality:

```yaml
visual_features:
  pose:
    enable_input: True
    train: keypoints/metadata.train.json
    dev: keypoints/metadata.dev.json
    test: keypoints/metadata.test.json
```

For `mae`, `dino`, and `sign2vec`, the code always uses this metadata/H5 path when the modality is enabled. For `pose`, raw JSON is preferred when a configured JSON directory exists; otherwise the dataset falls back to metadata/H5 if `train`, `dev`, or `test` metadata is configured for the active split.

### SignModelArguments

`SignModelArguments.projectors` stores the input dimension for each enabled modality:

```yaml
SignModelArguments:
  projectors:
    pose:
      dim: 208
```

The total sign input dimension is calculated automatically from enabled modalities.

## Prediction Config

Prediction configs use:

```yaml
ModelArguments:
EvaluationArguments:
SignDataArguments:
SignModelArguments:
```

### EvaluationArguments

- `output_dir`: Directory where predictions and scores are saved.
- `model_name`: Name used for logging/output context.
- `skip_frames`: `False`, `True`, or integer stride.
- `split`: Annotation split to evaluate: `train`, `dev`, or `test`.
- `max_sequence_length`: Maximum number of input frames.
- `max_token_length`: Tokenizer max length.
- `float32`: Same preprocessing flag as training.
- `decimal_points`: Same rounding flag as training.
- `model_dir`: Fine-tuned checkpoint directory.
- `batch_size`: Evaluation batch size.
- `max_val_samples`: Optional dataset limit.
- `bleu_effective_order`: Passed to SacreBLEU.

The raw JSON and H5 data configuration under `SignDataArguments` has the same meaning as in training. Prediction uses `EvaluationArguments.split` to choose `train_json_dir`, `dev_json_dir`, or `test_json_dir`.

## Local Testing Config

`configs/config_for_testing.yaml` is a small local smoke-test config. Fill in `SignDataArguments.data_dir`, then run from the repository root:

```bash
python train/run_finetuning.py --config_file configs/config_for_testing.yaml --verbose
```
