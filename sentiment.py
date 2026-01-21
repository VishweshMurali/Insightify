# sentiment.py

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

SUPPORTED_LANGS = {"en", "es", "pt", "fr", "de", "nl", "it"}

MULTILINGUAL_MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"
ENGLISH_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"


class SentimentModel:
    """
    Hybrid model:
      - multilingual → 1–5 stars
      - fallback English → POS/NEG mapped to 5-star/1-star
    """

    def __init__(self, batch_size=32):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.batch_size = batch_size

        # multilingual star model
        self.m_tokenizer = AutoTokenizer.from_pretrained(MULTILINGUAL_MODEL)
        self.m_model = AutoModelForSequenceClassification.from_pretrained(
            MULTILINGUAL_MODEL
        ).to(self.device)

        # English fallback model
        self.e_tokenizer = AutoTokenizer.from_pretrained(ENGLISH_MODEL)
        self.e_model = AutoModelForSequenceClassification.from_pretrained(
            ENGLISH_MODEL
        ).to(self.device)

    def _predict_multilingual(self, texts):
        stars = np.zeros(len(texts), dtype=int)
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            enc = self.m_tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt").to(self.device)
            with torch.no_grad():
                probs = F.softmax(self.m_model(**enc).logits, dim=-1).cpu().numpy()
            stars[i:i+len(batch)] = np.argmax(probs, axis=1) + 1  # 0-4 → 1-5
        return stars

    def _predict_english(self, texts):
        stars = np.zeros(len(texts), dtype=int)
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            enc = self.e_tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt").to(self.device)
            with torch.no_grad():
                probs = F.softmax(self.e_model(**enc).logits, dim=-1).cpu().numpy()
            labels = np.argmax(probs, axis=1)  # 0 = NEG, 1 = POS
            stars[i:i+len(batch)] = np.where(labels == 1, 5, 1)
        return stars

    @staticmethod
    def _stars_to_3way(stars):
        def conv(s):
            if s <= 2:
                return "NEGATIVE"
            if s == 3:
                return "NEUTRAL"
            return "POSITIVE"
        return [conv(int(s)) for s in stars]

    def classify_df(self, df):
        df = df.copy()

        langs = df["language"].astype(str)
        texts = df["processed_text"].astype(str)

        idx_multi = df[langs.isin(SUPPORTED_LANGS)].index
        idx_fallback = df[~langs.isin(SUPPORTED_LANGS)].index

        stars = np.zeros(len(df), dtype=int)

        if len(idx_multi):
            stars[idx_multi] = self._predict_multilingual(texts.loc[idx_multi].tolist())

        if len(idx_fallback):
            stars[idx_fallback] = self._predict_english(texts.loc[idx_fallback].tolist())

        df["stars"] = stars
        df["sentiment_3way"] = self._stars_to_3way(stars)

        # Probabilities for trend
        df["prob_pos"] = (df["stars"] - 1) / 4
        df["prob_neg"] = 1 - df["prob_pos"]

        # Short-text adjustment
        def adjust(row):
            if len(row["processed_text"].split()) <= 2:
                if row["stars"] <= 2:
                    return "NEGATIVE"
                if row["stars"] >= 4:
                    return "POSITIVE"
                return "NEUTRAL"
            return row["sentiment_3way"]

        df["sentiment_3way"] = df.apply(adjust, axis=1)

        return df






