import pandas as pd
import re

INPUT_CSV = "WPP_pair_predictions.csv"
OUTPUT_CSV = "WPP_pair_predictions_presence.csv"


def normalize(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Remove commas and periods
    text = text.replace(",", "")
    text = text.replace(".", "")

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


df = pd.read_csv(INPUT_CSV)

is_present = []

for _, row in df.iterrows():
    pred1 = normalize(row["prediction1"])
    pred2 = normalize(row["prediction2"])

    is_present.append(int(pred2 and pred2 in pred1))

output_df = df[["sentence_id", "query_id"]].copy()
output_df["is_present"] = is_present

output_df.to_csv(OUTPUT_CSV, index=False)

matches = sum(is_present)
total = len(is_present)

print(f"Matches: {matches}/{total} ({100 * matches / total:.2f}%)")
print(f"Saved to: {OUTPUT_CSV}")
