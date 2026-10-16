# preprocessing.py

import re
import pandas as pd


SUPPORTED_LANGS = {"en", "es", "pt", "fr", "de", "nl", "it"}


REVIEW_COL_CANDIDATES = [
    "review", "reviews", "review_text", "reviewtext", "reviewbody", "review_body",
    "text", "body", "content", "description",
    "comment", "comments",
    "feedback", "buyerfeedback", "customer_feedback",
    "message", "notes", "opinion"
]


def auto_detect_review_column(df: pd.DataFrame):
    """
    Detect review text column using candidate names + content inspection.
    Avoids selecting date/id columns.
    """
    lower_map = {c.lower(): c for c in df.columns}

    # Exact match by name
    for cand in REVIEW_COL_CANDIDATES:
        if cand in lower_map:
            col = lower_map[cand]
            if _looks_like_text_column(df[col]):
                return col

    # Partial name match
    for col in df.columns:
        lc = col.lower()
        if any(cand in lc for cand in REVIEW_COL_CANDIDATES):
            if _looks_like_text_column(df[col]):
                return col

    # evaluate all columns
    text_cols = []
    for col in df.columns:
        if _looks_like_text_column(df[col]):
            text_cols.append(col)

    if text_cols:
        return text_cols[0] 

    return None

def _looks_like_text_column(series):
    """Reject date/id/numeric columns and keep only text-like columns."""
    s = series.astype(str)

    if len(s) == 0:
        return False

    if (s.str.fullmatch(r"[0-9\-\:\/ ]+").sum() / len(s)) > 0.5:
        return False

    if s.nunique() < 20 and len(s) > 200:
        return False

    # Average length must be at least a few words
    avg_words = s.apply(lambda x: len(str(x).split())).mean()
    if avg_words < 3:
        return False

    return True


def _basic_clean(t: str) -> str:
    if not isinstance(t, str):
        return ""
    t = t.lower()
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = re.sub(r"[^a-záéíóúñçãõäëïöüß\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _detect_lang(t: str) -> str:
    if not t or len(t.split()) < 2:
        return "unknown"
    # Imported lazily so the rest of this module works without langdetect installed.
    from langdetect import DetectorFactory, detect

    DetectorFactory.seed = 0  # deterministic results
    try:
        return detect(t)
    except Exception:  # langdetect raises LangDetectException on undetectable text
        return "unknown"


class TextPreprocessor:

    def preprocess_df(self, df: pd.DataFrame, text_col: str):
        df = df.copy()

        df["cleaned_text"] = df[text_col].astype(str).apply(_basic_clean)
        df["language"] = df["cleaned_text"].apply(_detect_lang)

        # Used for sentiment analysis
        df["processed_text"] = df["cleaned_text"]

        df = df[df["processed_text"].str.len() > 0].reset_index(drop=True)
        return df
