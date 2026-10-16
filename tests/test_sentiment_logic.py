import numpy as np
import pandas as pd

from sentiment import SentimentModel, stars_to_3way


def test_stars_to_3way_boundaries():
    assert stars_to_3way([1, 2, 3, 4, 5]) == [
        "NEGATIVE",
        "NEGATIVE",
        "NEUTRAL",
        "POSITIVE",
        "POSITIVE",
    ]


def _stub_model():
    """A SentimentModel with the neural nets replaced by fixed outputs (no torch needed)."""
    m = object.__new__(SentimentModel)
    m.calls = {"multi": [], "english": []}

    def multi(texts):
        m.calls["multi"].append(list(texts))
        return np.full(len(texts), 4), np.full(len(texts), 0.9)

    def english(texts):
        m.calls["english"].append(list(texts))
        return np.full(len(texts), 1), np.full(len(texts), 0.8)

    m._predict_multilingual = multi
    m._predict_english = english
    return m


def test_classify_routes_by_language_and_keeps_row_alignment():
    df = pd.DataFrame(
        {
            "processed_text": ["good one", "schlecht aber ok", "unknown lang text", "muy bueno"],
            "language": ["en", "ru", "unknown", "es"],
        },
        index=[10, 11, 12, 13],  # non-default index must not break alignment
    )
    m = _stub_model()
    out = m.classify_df(df)

    assert m.calls["multi"] == [["good one", "muy bueno"]]
    assert m.calls["english"] == [["schlecht aber ok", "unknown lang text"]]
    assert list(out["stars"]) == [4, 1, 1, 4]
    assert list(out["sentiment_3way"]) == ["POSITIVE", "NEGATIVE", "NEGATIVE", "POSITIVE"]
    assert list(out["confidence"]) == [0.9, 0.8, 0.8, 0.9]
    assert list(out.index) == [10, 11, 12, 13]


def test_classify_does_not_mutate_input():
    df = pd.DataFrame({"processed_text": ["fine"], "language": ["en"]})
    _stub_model().classify_df(df)
    assert "stars" not in df.columns
