# sentiment.py
"""Sentiment classification.

Reviews in a supported language go through a multilingual 1-5 star model.
Everything else (including undetected language) falls back to an English
binary model, mapped to 1 or 5 stars.
"""

import numpy as np

SUPPORTED_LANGS = {"en", "es", "pt", "fr", "de", "nl", "it"}

MULTILINGUAL_MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"
ENGLISH_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"


def stars_to_3way(stars):
    """Map 1-5 stars to NEGATIVE (<=2), NEUTRAL (3), POSITIVE (>=4)."""

    def conv(s):
        s = int(s)
        if s <= 2:
            return "NEGATIVE"
        if s == 3:
            return "NEUTRAL"
        return "POSITIVE"

    return [conv(s) for s in stars]


class SentimentModel:
    """Hybrid model: multilingual 1-5 stars, with an English binary fallback."""

    def __init__(self, batch_size=32):
        # Imported here so this module can be imported (and unit-tested) without torch.
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size

        self.m_tokenizer = AutoTokenizer.from_pretrained(MULTILINGUAL_MODEL)
        self.m_model = AutoModelForSequenceClassification.from_pretrained(
            MULTILINGUAL_MODEL
        ).to(self.device)

        self.e_tokenizer = AutoTokenizer.from_pretrained(ENGLISH_MODEL)
        self.e_model = AutoModelForSequenceClassification.from_pretrained(
            ENGLISH_MODEL
        ).to(self.device)

    def _probs(self, tokenizer, model, texts):
        """Softmax class probabilities for a list of texts, in batches."""
        torch = self._torch
        chunks = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                logits = model(**enc).logits
            chunks.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(chunks)

    def _predict_multilingual(self, texts):
        """Return (stars 1-5, confidence) arrays."""
        probs = self._probs(self.m_tokenizer, self.m_model, texts)
        return probs.argmax(axis=1) + 1, probs.max(axis=1)

    def _predict_english(self, texts):
        """Return (stars in {1, 5}, confidence) arrays."""
        probs = self._probs(self.e_tokenizer, self.e_model, texts)
        labels = probs.argmax(axis=1)  # 0 = NEG, 1 = POS
        return np.where(labels == 1, 5, 1), probs.max(axis=1)

    def classify_df(self, df):
        """Add `stars`, `confidence` and `sentiment_3way` columns.

        `confidence` is the model's top softmax probability. For the English
        fallback it is the binary probability, not a 5-class one, so values are
        not directly comparable across the two models.
        """
        df = df.copy()

        langs = df["language"].astype(str)
        texts = df["processed_text"].astype(str)

        use_multi = langs.isin(SUPPORTED_LANGS).to_numpy()

        stars = np.zeros(len(df), dtype=int)
        confidence = np.zeros(len(df), dtype=float)

        if use_multi.any():
            s, c = self._predict_multilingual(texts[use_multi].tolist())
            stars[use_multi] = s
            confidence[use_multi] = c

        if (~use_multi).any():
            s, c = self._predict_english(texts[~use_multi].tolist())
            stars[~use_multi] = s
            confidence[~use_multi] = c

        df["stars"] = stars
        df["confidence"] = confidence
        df["sentiment_3way"] = stars_to_3way(stars)
        return df
