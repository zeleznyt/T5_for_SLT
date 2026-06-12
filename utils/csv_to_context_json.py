#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


TEXT_COLUMN_CANDIDATES = [
    "translation",
    "text",
    "sentence",
    "word",
    "gloss",
]


def detect_text_column(fieldnames):
    normalized = {name.strip(): name for name in fieldnames if name is not None}

    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate in normalized:
            return normalized[candidate]

    raise ValueError(
        "Could not auto-detect text column. "
        f"Tried: {TEXT_COLUMN_CANDIDATES}. "
        f"Available columns: {fieldnames}"
    )


def convert_csv_to_context_json(
    input_csv: Path,
    output_json: Path,
    uid_col: str = "uid",
    text_col: str | None = None,
):
    data = {}

    with input_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {input_csv}")

        if uid_col not in reader.fieldnames:
            raise ValueError(
                f"UID column '{uid_col}' not found. "
                f"Available columns: {reader.fieldnames}"
            )

        if text_col is None:
            text_col = detect_text_column(reader.fieldnames)

        if text_col not in reader.fieldnames:
            raise ValueError(
                f"Text column '{text_col}' not found. "
                f"Available columns: {reader.fieldnames}"
            )

        for row_idx, row in enumerate(reader, start=2):
            uid = (row.get(uid_col) or "").strip()
            text = (row.get(text_col) or "").strip()

            if not uid:
                print(f"Skipping row {row_idx}: empty uid")
                continue

            if uid in data:
                raise ValueError(
                    f"Duplicate uid '{uid}' found in row {row_idx}. "
                    "This mock structure assumes one clip per video."
                )

            # In this dataset, video_id == clip_name == uid.
            video_id = uid
            clip_name = uid

            data[video_id] = {
                "clip_order": [clip_name],
                clip_name: {
                    "translation": text
                }
            }

    output_json.parent.mkdir(parents=True, exist_ok=True)

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(data)} entries to {output_json}")
    print(f"Used uid column:  {uid_col}")
    print(f"Used text column: {text_col}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert annotation CSV to UniSign-style context JSON."
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        type=Path,
        help="Input CSV file."
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        type=Path,
        help="Output JSON file."
    )
    parser.add_argument(
        "--uid-col",
        default="uid",
        help="Column containing video/clip id. Default: uid"
    )
    parser.add_argument(
        "--text-col",
        default=None,
        help=(
            "Column containing translation text. "
            "If omitted, auto-detects one of: translation, text, sentence, word, gloss."
        )
    )

    args = parser.parse_args()

    convert_csv_to_context_json(
        input_csv=args.input,
        output_json=args.output,
        uid_col=args.uid_col,
        text_col=args.text_col,
    )


if __name__ == "__main__":
    main()
