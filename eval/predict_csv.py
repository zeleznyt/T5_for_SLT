import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
import yaml
from transformers import T5Tokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.configuration_t5 import SignT5Config
from model.modeling_t5 import T5ModelForSLT
from utils.keypoint_dataset import KeypointDatasetJSON


POSE_NORMALIZATION_ORDER = (
    "global-pose_landmarks",
    "local-right_hand_landmarks",
    "local-left_hand_landmarks",
    "local-face_landmarks",
)
DEFAULT_MISSING_PREDICTION = "none"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict translations or isolated-sign labels for keypoint JSON files listed in a CSV."
    )
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--second_config_file", type=str, default=None)
    parser.add_argument("--second_model_dir", type=str, default=None)
    parser.add_argument("--second_json_dir", type=str, default=None)
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--json_dir", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--task", choices=["translation", "isolated", "pair"], default="translation")
    parser.add_argument("--uid_column", type=str, default="uid")
    parser.add_argument("--first_column", type=str, default="sentence_id")
    parser.add_argument("--second_column", type=str, default="query_id")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_sequence_length", type=int, default=None)
    parser.add_argument("--generation_max_length", type=int, default=None)
    parser.add_argument("--num_beams", type=int, default=None)
    parser.add_argument("--length_penalty", type=float, default=None)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=None)
    parser.add_argument("--early_stopping", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--missing_prediction",
        type=str,
        default=DEFAULT_MISSING_PREDICTION,
        help="Prediction text used when a CSV uid has no matching JSON file.",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    for section in ("TrainingArguments", "EvaluationArguments"):
        for key, value in config.get(section, {}).items():
            if value in ("none", "None"):
                config[section][key] = None
    return config


def str_to_bool(value):
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, str):
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
    return value


def skip_stride(skip_frames):
    skip_frames = str_to_bool(skip_frames)
    if skip_frames is True:
        return 2
    if skip_frames in (False, None):
        return None
    return int(skip_frames)


def get_runtime_config(config: dict) -> dict:
    return config.get("EvaluationArguments") or config.get("TrainingArguments") or {}


def get_sign_input_dim(config: dict) -> int:
    sign_input_dim = 0
    for modality, modality_config in config["SignDataArguments"]["visual_features"].items():
        if modality_config["enable_input"]:
            sign_input_dim += config["SignModelArguments"]["projectors"][modality]["dim"]
    return sign_input_dim


def build_model(config: dict, model_dir: str):
    model_config = dict(config["ModelArguments"])
    model_config["sign_input_dim"] = get_sign_input_dim(config)

    t5_config = SignT5Config()
    for param, value in model_config.items():
        setattr(t5_config, param, value)

    model = T5ModelForSLT.from_pretrained(model_dir, config=t5_config)
    for param in model.parameters():
        param.data = param.data.contiguous()
    tokenizer = T5Tokenizer.from_pretrained(model.config.base_model_name, clean_up_tokenization_spaces=True)
    return model, tokenizer


def build_keypoint_dataset(config: dict, json_dir: str):
    pose_config = config["SignDataArguments"]["visual_features"]["pose"]
    if not pose_config["enable_input"]:
        raise ValueError("This CSV inference script expects the pose modality to be enabled.")

    enabled_modalities = [
        modality
        for modality, modality_config in config["SignDataArguments"]["visual_features"].items()
        if modality_config["enable_input"]
    ]
    if enabled_modalities != ["pose"]:
        raise ValueError(
            "This CSV inference script only supports raw pose JSON input. "
            f"Enabled modalities in config: {enabled_modalities}"
        )

    normalization_config = pose_config.get("normalization", {})
    return KeypointDatasetJSON(
        json_folder=json_dir,
        kp_normalization=POSE_NORMALIZATION_ORDER,
        kp_normalization_method=normalization_config.get("normalization_method", "sign_space"),
        data_key=normalization_config.get("data_key", "cropped_keypoints"),
        missing_values=pose_config.get("missing_values"),
        augmentation_configs=[],
        interpolate=pose_config.get("interpolate", -1),
    )


class CsvKeypointDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        uids: List[str],
        keypoint_dataset: KeypointDatasetJSON,
        max_sequence_length: int,
        skip_frames=None,
        float32=False,
        decimal_points=-1,
    ):
        self.uids = uids
        self.keypoint_dataset = keypoint_dataset
        self.max_sequence_length = max_sequence_length
        self.skip_stride = skip_stride(skip_frames)
        self.float32 = str_to_bool(float32)
        self.decimal_points = int(decimal_points)

        missing_uids = [uid for uid in self.uids if uid not in keypoint_dataset.video_name_to_idx]
        if missing_uids:
            preview = ", ".join(missing_uids[:10])
            raise FileNotFoundError(
                f"{len(missing_uids)} CSV uid(s) do not have matching JSON files in the keypoint directory. "
                f"First missing uid(s): {preview}"
            )

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        uid = self.uids[idx]
        clip_data = self.keypoint_dataset.get_clip_data(uid)

        if self.skip_stride:
            clip_data = clip_data[:: self.skip_stride]
        if self.max_sequence_length:
            clip_data = clip_data[: self.max_sequence_length]
        if not self.float32:
            clip_data = clip_data.astype(np.dtype("float16"))
        if self.decimal_points > 0:
            clip_data = np.round(clip_data, self.decimal_points)

        sign_inputs = torch.tensor(clip_data).float()
        if sign_inputs.shape[0] == 0:
            raise ValueError(f"Clip {uid} has no frames after preprocessing.")

        return {
            "uid": uid,
            "sign_inputs": sign_inputs,
            "attention_mask": torch.ones(sign_inputs.shape[0]),
        }


def make_collate_fn(max_sequence_length: int, pose_dim: int):
    def collate_fn(batch):
        sign_inputs = torch.stack(
            [
                torch.cat(
                    (
                        sample["sign_inputs"],
                        torch.zeros(max_sequence_length - sample["sign_inputs"].shape[0], pose_dim),
                    ),
                    dim=0,
                )
                for sample in batch
            ]
        )
        attention_mask = torch.stack(
            [
                torch.cat(
                    (
                        sample["attention_mask"],
                        torch.zeros(max_sequence_length - sample["attention_mask"].shape[0]),
                    ),
                    dim=0,
                )
                for sample in batch
            ]
        )
        return {
            "uids": [sample["uid"] for sample in batch],
            "sign_inputs": sign_inputs,
            "attention_mask": attention_mask,
        }

    return collate_fn


def read_rows(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def unique_in_order(values: Iterable[str]) -> List[str]:
    seen = set()
    unique_values = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values


def predict_uids(
    model,
    tokenizer,
    keypoint_dataset,
    uids: List[str],
    max_sequence_length: int,
    batch_size: int,
    device: str,
    generation_kwargs: dict,
    runtime_config: dict,
    missing_prediction: str = DEFAULT_MISSING_PREDICTION,
) -> Dict[str, str]:
    predictions = {uid: missing_prediction for uid in uids}
    available_uids = [uid for uid in uids if uid in keypoint_dataset.video_name_to_idx]
    missing_uids = [uid for uid in uids if uid not in keypoint_dataset.video_name_to_idx]
    if missing_uids:
        preview = ", ".join(missing_uids[:10])
        print(
            f"Warning: {len(missing_uids)} CSV uid(s) do not have matching JSON files. "
            f"Using {missing_prediction!r}. First missing uid(s): {preview}"
        )
    if not available_uids:
        return predictions

    dataset = CsvKeypointDataset(
        uids=available_uids,
        keypoint_dataset=keypoint_dataset,
        max_sequence_length=max_sequence_length,
        skip_frames=runtime_config.get("skip_frames"),
        float32=runtime_config.get("float32", False),
        decimal_points=runtime_config.get("decimal_points", -1),
    )
    pose_dim = model.config.sign_input_dim
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=make_collate_fn(max_sequence_length, pose_dim),
    )

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            uids_batch = batch.pop("uids")
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model.generate(
                **batch,
                bos_token_id=tokenizer.pad_token_id,
                **generation_kwargs,
            )
            outputs[outputs > len(tokenizer) - 1] = tokenizer.unk_token_id
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            predictions.update(dict(zip(uids_batch, decoded)))
    return predictions


def generation_arguments(args, config: dict, runtime_config: dict) -> dict:
    model_config = config["ModelArguments"]
    return {
        "max_length": args.generation_max_length or model_config.get("max_length") or runtime_config.get("max_token_length"),
        "num_beams": args.num_beams or model_config.get("num_beams", 1),
        "length_penalty": args.length_penalty
        if args.length_penalty is not None
        else model_config.get("length_penalty", 1.0),
        "no_repeat_ngram_size": args.no_repeat_ngram_size
        if args.no_repeat_ngram_size is not None
        else model_config.get("no_repeat_ngram_size", 0),
        "early_stopping": str_to_bool(args.early_stopping)
        if args.early_stopping is not None
        else model_config.get("early_stopping", False),
    }


def write_translation(rows: List[dict], uid_column: str, predictions: Dict[str, str], output_csv: str):
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["uid", "text"])
        writer.writeheader()
        for row in rows:
            uid = row[uid_column]
            writer.writerow({"uid": uid, "text": predictions[uid]})


def write_isolated(rows: List[dict], uid_column: str, predictions: Dict[str, str], output_csv: str):
    fieldnames = ["uid"] + [f"pred{i}" for i in range(1, 11)]
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            uid = row[uid_column]
            prediction = predictions[uid]
            writer.writerow({"uid": uid, **{f"pred{i}": prediction for i in range(1, 11)}})


def write_pair(
    rows: List[dict],
    first_column: str,
    second_column: str,
    first_predictions: Dict[str, str],
    second_predictions: Dict[str, str],
    output_csv: str,
):
    fieldnames = [first_column, second_column, "prediction1", "prediction2"]
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            first_uid = row[first_column]
            second_uid = row[second_column]
            writer.writerow(
                {
                    first_column: first_uid,
                    second_column: second_uid,
                    "prediction1": first_predictions[first_uid],
                    "prediction2": second_predictions[second_uid],
                }
            )


def main():
    args = parse_args()
    config = load_config(args.config_file)
    runtime_config = get_runtime_config(config)

    model_dir = args.model_dir or runtime_config.get("model_dir")
    if not model_dir:
        raise ValueError("--model_dir must be provided or set in the config.")

    batch_size = args.batch_size or runtime_config.get("batch_size") or runtime_config.get("per_device_eval_batch_size") or 8
    max_sequence_length = args.max_sequence_length or runtime_config.get("max_sequence_length")
    if not max_sequence_length:
        raise ValueError("--max_sequence_length must be provided or set in the config.")

    rows = read_rows(args.input_csv)
    keypoint_dataset = build_keypoint_dataset(config, args.json_dir)
    model, tokenizer = build_model(config, model_dir)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    output_dir = os.path.dirname(os.path.abspath(args.output_csv))
    os.makedirs(output_dir, exist_ok=True)

    if args.task in ("translation", "isolated"):
        uids = unique_in_order(row[args.uid_column] for row in rows)
        predictions = predict_uids(
            model=model,
            tokenizer=tokenizer,
            keypoint_dataset=keypoint_dataset,
            uids=uids,
            max_sequence_length=max_sequence_length,
            batch_size=batch_size,
            device=device,
            generation_kwargs=generation_arguments(args, config, runtime_config),
            runtime_config=runtime_config,
            missing_prediction=args.missing_prediction,
        )

    if args.task == "translation":
        write_translation(rows, args.uid_column, predictions, args.output_csv)
    elif args.task == "isolated":
        write_isolated(rows, args.uid_column, predictions, args.output_csv)
    else:
        first_uids = unique_in_order(row[args.first_column] for row in rows)
        first_predictions = predict_uids(
            model=model,
            tokenizer=tokenizer,
            keypoint_dataset=keypoint_dataset,
            uids=first_uids,
            max_sequence_length=max_sequence_length,
            batch_size=batch_size,
            device=device,
            generation_kwargs=generation_arguments(args, config, runtime_config),
            runtime_config=runtime_config,
            missing_prediction=args.missing_prediction,
        )

        second_config = load_config(args.second_config_file) if args.second_config_file else config
        second_runtime_config = get_runtime_config(second_config)
        second_model_dir = args.second_model_dir or second_runtime_config.get("model_dir") or model_dir
        second_json_dir = args.second_json_dir or args.json_dir
        second_batch_size = (
            args.batch_size
            or second_runtime_config.get("batch_size")
            or second_runtime_config.get("per_device_eval_batch_size")
            or batch_size
        )
        second_max_sequence_length = args.max_sequence_length or second_runtime_config.get("max_sequence_length")
        if not second_max_sequence_length:
            raise ValueError("--max_sequence_length must be provided or set in the second config.")

        second_keypoint_dataset = build_keypoint_dataset(second_config, second_json_dir)
        second_model, second_tokenizer = build_model(second_config, second_model_dir)
        second_model.to(device)
        second_uids = unique_in_order(row[args.second_column] for row in rows)
        second_predictions = predict_uids(
            model=second_model,
            tokenizer=second_tokenizer,
            keypoint_dataset=second_keypoint_dataset,
            uids=second_uids,
            max_sequence_length=second_max_sequence_length,
            batch_size=second_batch_size,
            device=device,
            generation_kwargs=generation_arguments(args, second_config, second_runtime_config),
            runtime_config=second_runtime_config,
            missing_prediction=args.missing_prediction,
        )

        write_pair(
            rows,
            args.first_column,
            args.second_column,
            first_predictions,
            second_predictions,
            args.output_csv,
        )

    print(f"Predictions saved to {args.output_csv}")


if __name__ == "__main__":
    main()
