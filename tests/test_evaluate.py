import pandas as pd
import pytest

import evaluate
import preprocessing


def test_normalize_rating_handles_both_scales():
    assert list(evaluate.normalize_rating(pd.Series([1, 3, 5]))) == [1, 3, 5]
    assert list(evaluate.normalize_rating(pd.Series([20, 60, 100]))) == [1, 3, 5]
    assert evaluate.normalize_rating(pd.Series(["x", 4])).isna().tolist() == [True, False]


def test_compute_metrics_perfect_and_imperfect():
    perfect = evaluate.compute_metrics([1, 3, 5, 4], [1, 3, 5, 4])
    assert perfect["accuracy_3way"] == 1.0
    assert perfect["stars_mae"] == 0.0

    # one 5-star review predicted as 1-star -> 3-way accuracy 3/4
    m = evaluate.compute_metrics([1, 3, 5, 4], [1, 3, 1, 4])
    assert m["accuracy_3way"] == pytest.approx(0.75)
    assert m["stars_mae"] == pytest.approx(1.0)
    cm = m["confusion_matrix"]["rows_true_cols_pred"]
    assert cm[2][0] == 1  # true POSITIVE predicted NEGATIVE


def test_evaluate_end_to_end_with_stub_model(monkeypatch):
    monkeypatch.setattr(preprocessing, "_detect_lang", lambda t: "en")

    class Always5:
        def classify_df(self, df):
            df = df.copy()
            df["stars"] = 5
            return df

    df = pd.DataFrame(
        {
            "review_text": ["really great item works", "terrible broke fast", "okay item i guess"],
            "rating": [100, 20, 60],  # AliExpress 0-100 scale -> 5, 1, 3
        }
    )
    m = evaluate.evaluate(df, Always5())
    assert m["n"] == 3
    assert m["accuracy_3way"] == pytest.approx(1 / 3)
