# utils.py
"""Helpers for translation, word clouds and aspect extraction.

Heavy or UI-only dependencies (streamlit, matplotlib, wordcloud, deep-translator)
are imported lazily so the pure-logic parts can be imported and tested without them.
"""

import re
import logging
from functools import lru_cache

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

log = logging.getLogger(__name__)

# Same tokenisation as sklearn's TfidfVectorizer default, so aspect counts line up with it.
_TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")


@lru_cache(maxsize=8192)
def _translate_cached(text: str) -> str:
    from deep_translator import GoogleTranslator

    return GoogleTranslator(source="auto", target="en").translate(text)


def translate_to_en(text: str, lang: str = "auto") -> str:
    """Translate text to English. Results are cached; failures return the original text.

    NOTE: this sends the review text to Google Translate via deep-translator.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    if lang and lang.lower() == "en":
        return text

    try:
        return _translate_cached(text) or text
    except Exception as exc:  # network error, rate limit, unsupported language, ...
        log.warning("Translation failed, keeping original text: %s", exc)
        return text


def make_wordcloud(en_texts, title):
    """Render a word cloud from text that is already in English."""
    import matplotlib.pyplot as plt
    import streamlit as st
    from wordcloud import WordCloud

    joined = " ".join(pd.Series(en_texts).fillna("").astype(str).tolist())

    if len(joined.split()) < 5:
        st.info(f"Not enough text for {title}")
        return

    wc = WordCloud(
        width=450,
        height=250,
        background_color="white",
        max_words=100,
    ).generate(joined)

    fig, ax = plt.subplots(figsize=(4.5, 2.5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)


def _token_sets(texts):
    return [set(_TOKEN_RE.findall(str(t).lower())) for t in texts]


def compute_aspects(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Top TF-IDF terms with the number of positive / negative reviews that mention each.

    Uses the `translated_text` column when present (avoids re-translating);
    otherwise falls back to translating `processed_text`.
    Mentions are counted per review using whole-word matches.
    """
    columns = ["aspect", "positive_mentions", "negative_mentions"]

    if "translated_text" in df.columns:
        en_texts = df["translated_text"].fillna("").astype(str)
    else:
        en_texts = df["processed_text"].fillna("").astype(str).apply(translate_to_en)

    vec = TfidfVectorizer(stop_words="english", min_df=2, max_df=0.9)
    try:
        X = vec.fit_transform(en_texts)
    except ValueError:  # empty vocabulary, or min_df/max_df leave no terms
        return pd.DataFrame(columns=columns)

    tokens = vec.get_feature_names_out()
    scores = X.sum(axis=0).A1
    top_tokens = [tokens[i] for i in scores.argsort()[::-1][:top_n]]

    pos_sets = _token_sets(en_texts[df["sentiment_3way"] == "POSITIVE"])
    neg_sets = _token_sets(en_texts[df["sentiment_3way"] == "NEGATIVE"])

    rows = []
    for tok in top_tokens:
        pos_count = sum(tok in s for s in pos_sets)
        neg_count = sum(tok in s for s in neg_sets)
        if pos_count or neg_count:
            rows.append(
                {
                    "aspect": tok,
                    "positive_mentions": int(pos_count),
                    "negative_mentions": int(neg_count),
                }
            )

    return pd.DataFrame(rows, columns=columns)
