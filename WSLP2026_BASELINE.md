# WSLP 2026 Shared Task Baseline

This branch contains a baseline solution for the [WSLP 2026 Shared Task](https://exploration-lab.github.io/WSLP-2026/). WSLP 2026 is the 2nd Workshop on Sign Language Processing, co-located with EMNLP 2026. The shared task focuses on Indian Sign Language and low-resource sign language processing.

The goal of this baseline is to provide a transparent starting point for participants. It uses pose/keypoint inputs and T5-style sequence-to-sequence models from this repository.

## Shared Tasks

The workshop shared task contains three related prediction settings:

- **SLT: Sign Language Translation**  
  Translate a signed sentence from Indian Sign Language keypoints into text.

- **ISLR: Isolated Sign Language Recognition**  
  Predict the label of an isolated sign clip.

- **WPP: Word Presence Prediction**  
  Given a signed sentence and a query sign, predict whether the query sign is present in the sentence.

## Preprocessing

The baseline assumes pose/keypoint JSON files as input. The videos were preprocessed with the pose extraction pipeline used in:

- Exploring Pose-based Sign Language Translation: Ablation Studies and Attention Insights  
  https://arxiv.org/abs/2507.01532

The preprocessing pipeline is available here:

- https://github.com/JSALT2024/PoseEstimation

After keypoint extraction, the workshop CSV annotations can be converted into the annotation JSON format expected by this repository with:

```bash
python utils/csv_to_context_json.py \
  --input path/to/annotations.csv \
  --output path/to/annotations.json \
  --uid-col uid
```

By default, the script auto-detects the text column from one of:

```text
translation, text, sentence, word, gloss
```

The generated JSON uses the same value for `video_id`, `clip_name`, and the CSV `uid`, which matches the workshop setting where each keypoint JSON file corresponds to one clip.

## Baseline Models

The baseline uses two separately trained models:

- **SLT model** for sentence-level translation.
- **ISLR model** for isolated sign recognition.

The ISLR model was trained for about 10 epochs on the provided ISLR training data from the workshop organizers.

The SLT dataset is much larger, so the baseline SLT model was trained for about 6 epochs on approximately 10% of the SLT training data. This is intended as a lightweight baseline rather than a fully optimized system.

Both models use the same general architecture from this repository: a T5-style model conditioned on normalized pose/keypoint sequences.

## CSV Prediction Helper

`eval/predict_csv.py` runs model inference for CSV-based workshop inputs. It supports three modes:

```text
translation
isolated
pair
```

### SLT Prediction

Use `--task translation` to produce a CSV with columns:

```text
uid,text
```

Example:

```bash
python eval/predict_csv.py \
  --task translation \
  --config_file path/to/slt_config.yaml \
  --model_dir path/to/slt_checkpoint \
  --input_csv path/to/slt_test.csv \
  --json_dir path/to/slt_keypoints \
  --output_csv SLT_predictions.csv
```

### ISLR Prediction

Use `--task isolated` to produce a CSV with columns:

```text
uid,pred1,pred2,pred3,pred4,pred5,pred6,pred7,pred8,pred9,pred10
```

The baseline writes the same top prediction into all ten prediction columns.

Example:

```bash
python eval/predict_csv.py \
  --task isolated \
  --config_file path/to/islr_config.yaml \
  --model_dir path/to/islr_checkpoint \
  --input_csv path/to/islr_test.csv \
  --json_dir path/to/islr_keypoints \
  --output_csv ISLR_predictions.csv
```

### WPP Intermediate Prediction

For WPP, the baseline first runs two models:

- the SLT model on the sentence clip
- the ISLR model on the query-sign clip

Use `--task pair` to produce an intermediate CSV with:

```text
sentence_id,query_id,prediction1,prediction2
```

`prediction1` is the generated sentence translation. `prediction2` is the predicted isolated-sign label.

Example:

```bash
python eval/predict_csv.py \
  --task pair \
  --config_file path/to/slt_config.yaml \
  --model_dir path/to/slt_checkpoint \
  --second_config_file path/to/islr_config.yaml \
  --second_model_dir path/to/islr_checkpoint \
  --input_csv path/to/wpp_test.csv \
  --json_dir path/to/sentence_keypoints \
  --second_json_dir path/to/query_keypoints \
  --first_column sentence_id \
  --second_column query_id \
  --output_csv WPP_pair_predictions.csv
```

## WPP Submission Helper

`eval/make_wpp_submission.py` converts the pair predictions into a final binary WPP submission.

The baseline rule is simple:

1. Normalize the generated sentence text and query-sign prediction.
2. Predict `is_present = 1` if the query-sign prediction appears as a substring of the generated sentence.
3. Otherwise predict `is_present = 0`.

The script currently expects:

```text
input:  WPP_pair_predictions.csv
output: WPP_pair_predictions_presence.csv
```

Run it from the directory containing `WPP_pair_predictions.csv`, or edit the constants at the top of the script:

```bash
python eval/make_wpp_submission.py
```

The output CSV contains:

```text
sentence_id,query_id,is_present
```

This WPP approach is intentionally simple. It is useful as a baseline because it decomposes WPP into two understandable model outputs: sentence translation and isolated sign recognition.

## Notes

- The baseline is pose-only.
- The keypoint JSON files should be produced before training or inference.
- Raw keypoint JSON loading is the default path in this branch.
- H5 feature loading is still supported by the core dataset code, but it is not the recommended path for the WSLP 2026 baseline.
