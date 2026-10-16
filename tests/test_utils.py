import pandas as pd

import utils
from utils import compute_aspects


def _df(texts, sentiments):
    return pd.DataFrame(
        {
            "processed_text": texts,
            "translated_text": texts,
            "sentiment_3way": sentiments,
        }
    )


def test_aspects_use_whole_words_not_substrings(monkeypatch):
    # "use" must not match inside "because" / "house"
    texts = [
        "easy to use and great battery",
        "easy to use but poor battery",
        "because the house was cold the battery died",
        "because of the house battery was great",
        # filler so no term appears in >90% of documents (TF-IDF max_df)
        "fast shipping arrived early",
        "nice packaging arrived early",
        "color looks different than photo",
        "price was fair for quality",
    ]
    sentiments = ["POSITIVE", "NEGATIVE", "NEGATIVE", "POSITIVE"] + ["NEUTRAL"] * 4
    out = compute_aspects(_df(texts, sentiments), top_n=10).set_index("aspect")
    assert out.loc["use", "positive_mentions"] == 1
    assert out.loc["use", "negative_mentions"] == 1
    assert out.loc["battery", "positive_mentions"] == 2
    assert out.loc["battery", "negative_mentions"] == 2


def test_aspects_do_not_retranslate_when_translated_text_exists(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("translate_to_en should not be called")

    monkeypatch.setattr(utils, "translate_to_en", boom)
    texts = ["good battery life", "good battery life", "great screen", "slow shipping"]
    out = compute_aspects(_df(texts, ["POSITIVE", "POSITIVE", "NEGATIVE", "NEGATIVE"]))
    assert "battery" in set(out["aspect"])


def test_aspects_empty_when_vocabulary_is_empty():
    out = compute_aspects(_df(["a", "b"], ["POSITIVE", "NEGATIVE"]))
    assert out.empty
    assert list(out.columns) == ["aspect", "positive_mentions", "negative_mentions"]


def test_translate_returns_english_and_blank_input_unchanged():
    assert utils.translate_to_en("already english", "en") == "already english"
    assert utils.translate_to_en("   ") == ""
    assert utils.translate_to_en(None) == ""


def test_translate_falls_back_to_original_on_error(monkeypatch):
    def fail(_text):
        raise RuntimeError("network down")

    monkeypatch.setattr(utils, "_translate_cached", fail)
    assert utils.translate_to_en("hola mundo", "es") == "hola mundo"
