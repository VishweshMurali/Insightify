"""Measure how well the sentiment model agrees with the star ratings reviewers gave.

Usage:
    python evaluate.py reviews.csv                # needs review_text + rating columns
    python evaluate.py reviews.csv --json out.json

Ratings are mapped to the same 3 classes as the model output:
    1-2 stars -> NEGATIVE, 3 -> NEUTRAL, 4-5 -> POSITIVE.
AliExpress ratings on a 0-100 scale are converted to 1-5 automatically.

This measures agreement with reviewer ratings, not human-labelled sentiment:
a sarcastic 5-star review or a 1-star shipping complaint about a good product
will count as a model "error".
"""

import argparse
import json

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
)

from preprocessing import TextPreprocessor, auto_detect_review_column
from sentiment import stars_to_3way

LABELS = ["NEGATIVE", "NEUTRAL", "POSITIVE"]


def normalize_rating(series: pd.Series) -> pd.Series:
    """Return ratings as 1-5. Accepts 1-5 or 0-100 (AliExpress) scales."""
    r = pd.to_numeric(series, errors="coerce")
    if r.max() > 5:
        r = r / 20.0
    return r.round().clip(1, 5)


def compute_metrics(true_stars, pred_stars) -> dict:
    true_3 = stars_to_3way(true_stars)
    pred_3 = stars_to_3way(pred_stars)
    return {
        "n": len(true_3),
        "accuracy_3way": accuracy_score(true_3, pred_3),
        "macro_f1_3way": f1_score(true_3, pred_3, labels=LABELS, average="macro", zero_division=0),
        "stars_mae": mean_absolute_error(true_stars, pred_stars),
        "confusion_matrix": {
            "labels": LABELS,
            "rows_true_cols_pred": confusion_matrix(true_3, pred_3, labels=LABELS).tolist(),
        },
        "report": classification_report(
            true_3, pred_3, labels=LABELS, zero_division=0, output_dict=True
        ),
    }


def evaluate(df: pd.DataFrame, model, text_col=None, rating_col="rating") -> dict:
    text_col = text_col or auto_detect_review_column(df)
    if text_col is None:
        raise SystemExit("Could not find a review text column; pass --text-col.")
    if rating_col not in df.columns:
        raise SystemExit(f"No '{rating_col}' column found; pass --rating-col.")

    df = df.copy()
    df["_true_stars"] = normalize_rating(df[rating_col])
    df = df.dropna(subset=["_true_stars"])

    processed = TextPreprocessor().preprocess_df(df, text_col=text_col)
    scored = model.classify_df(processed)
    return compute_metrics(scored["_true_stars"].astype(int).tolist(), scored["stars"].tolist())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--text-col")
    ap.add_argument("--rating-col", default="rating")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--json", help="write full metrics to this file")
    args = ap.parse_args()

    from sentiment import SentimentModel

    df = pd.read_csv(args.csv)
    metrics = evaluate(df, SentimentModel(batch_size=args.batch_size), args.text_col, args.rating_col)

    print(f"Reviews evaluated : {metrics['n']}")
    print(f"3-way accuracy    : {metrics['accuracy_3way']:.3f}")
    print(f"3-way macro F1    : {metrics['macro_f1_3way']:.3f}")
    print(f"Star MAE          : {metrics['stars_mae']:.3f}")
    print("Confusion matrix (rows = rating-derived, cols = model):")
    print(pd.DataFrame(
        metrics["confusion_matrix"]["rows_true_cols_pred"], index=LABELS, columns=LABELS
    ))

    if args.json:
        with open(args.json, "w") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
