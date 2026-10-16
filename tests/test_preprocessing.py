import pandas as pd
import pytest

import preprocessing
from preprocessing import (
    TextPreprocessor,
    _basic_clean,
    _detect_lang,
    auto_detect_review_column,
)


def test_basic_clean_strips_urls_punctuation_and_case():
    raw = "Great product!!! Visit https://spam.example/x NOW :) 5/5"
    assert _basic_clean(raw) == "great product visit now"


def test_basic_clean_keeps_accented_letters():
    assert _basic_clean("Excelente calidad, muy bueno ñandú") == "excelente calidad muy bueno ñandú"


def test_basic_clean_handles_non_strings():
    assert _basic_clean(None) == ""
    assert _basic_clean(123) == ""


def test_detect_lang_short_text_is_unknown():
    assert _detect_lang("") == "unknown"
    assert _detect_lang("ok") == "unknown"


def test_detect_lang_identifies_english_and_spanish():
    pytest.importorskip("langdetect")
    assert _detect_lang("this product works really well and arrived quickly") == "en"
    assert _detect_lang("el producto llegó muy rápido y funciona perfectamente") == "es"


def test_auto_detect_prefers_named_review_column_over_dates_and_ids():
    df = pd.DataFrame(
        {
            "id": [f"{i:06d}" for i in range(30)],
            "date": ["2024-01-01 10:00"] * 30,
            "review_text": [f"this is review number {i} and it is fine" for i in range(30)],
        }
    )
    assert auto_detect_review_column(df) == "review_text"


def test_auto_detect_skips_date_like_column_named_like_text():
    df = pd.DataFrame(
        {
            "comment": ["2024-01-01"] * 10,
            "feedback": [f"the item was okay and shipping was slow {i}" for i in range(10)],
        }
    )
    assert auto_detect_review_column(df) == "feedback"


def test_auto_detect_returns_none_when_no_text_column():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.5, 5.5, 6.5]})
    assert auto_detect_review_column(df) is None


def test_auto_detect_handles_empty_frame():
    assert auto_detect_review_column(pd.DataFrame({"review": []})) is None


def test_preprocess_df_drops_rows_that_clean_to_empty(monkeypatch):
    monkeypatch.setattr(preprocessing, "_detect_lang", lambda t: "en")  # no langdetect needed
    df = pd.DataFrame({"review": ["Nice item, works well", "!!! :) 123", None]})
    out = TextPreprocessor().preprocess_df(df, text_col="review")
    assert list(out["processed_text"]) == ["nice item works well"]
    assert list(out.index) == [0]
    assert {"cleaned_text", "language", "processed_text"} <= set(out.columns)
