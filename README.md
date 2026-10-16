# Insightify

Review sentiment analysis dashboard. Load product reviews (scraped from AliExpress or uploaded as CSV), and Insightify classifies each one, then shows sentiment distribution, trends over time, word clouds, and the product aspects people praise or complain about, across languages.

**Stack:** Python · Streamlit · Hugging Face Transformers (PyTorch) · scikit-learn · langdetect · deep-translator

## How it works

```
CSV upload  ─┐
             ├─► preprocess ─► classify ─► translate to English ─► dashboard
AliExpress ──┘   clean text      1–5 stars    (for word clouds       distribution · trend ·
URL scrape       detect language              and aspects)           word clouds · aspects · CSV
```

1. **Input.** Upload a CSV (the review text column is auto-detected; if a product/ID column exists you pick one product), or paste an AliExpress product URL to scrape its reviews.
2. **Preprocess.** Lowercase, strip URLs and punctuation, detect language (`langdetect`).
3. **Classify.**
   - Languages `en, es, pt, fr, de, nl, it` use [`nlptown/bert-base-multilingual-uncased-sentiment`](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment), which outputs 1–5 stars.
   - Any other or undetected language falls back to [`distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english) (binary), mapped to 1 or 5 stars.
   - Stars map to labels: 1–2 = NEGATIVE, 3 = NEUTRAL, 4–5 = POSITIVE. The `confidence` column is the model's top softmax probability.
4. **Visualize.** Reviews are translated to English (Google Translate, cached) for word clouds and TF-IDF aspect extraction, where each aspect shows how many positive and negative reviews mention it.

## Quick start

```bash
git clone https://github.com/VishweshMurali/Insightify.git
cd Insightify
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The first run downloads both models from Hugging Face. To try it without scraping, upload [`examples/sample_reviews.csv`](examples/sample_reviews.csv). It contains **30 synthetic reviews** in several languages, written for demos and tests only.

Sidebar settings: **max pages to scrape** (default 10, about 20 reviews per page) and **inference batch size** (default 16).

## Evaluation

`evaluate.py` measures how often the model agrees with the star rating the reviewer gave:

```bash
python evaluate.py my_reviews.csv --json metrics.json
```

It reports 3-way accuracy, macro F1, star MAE, and a confusion matrix. Ratings on AliExpress's 0–100 scale are converted to 1–5.

This is agreement with reviewer ratings, not human-labelled sentiment, so sarcasm and mismatched ratings count as errors. No benchmark numbers are published here yet. Run it on your own data to get them.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests cover preprocessing, column detection, scraper pagination and de-duplication, aspect counting, label mapping, language routing, and the evaluation metrics. They use stubs in place of the neural models, so they run without downloading anything.

## Limitations and notes

- **Privacy.** Review text is sent to Google Translate (via `deep-translator`) to produce English text for visuals. Sentiment inference itself runs locally.
- **Scraping.** The scraper uses an unofficial AliExpress feedback endpoint that can change or block requests. Respect AliExpress's Terms of Service. Use the CSV path for anything serious.
- **Neutral class.** Only the multilingual model can output 3 stars. Reviews that fall back to the English binary model are never NEUTRAL.
- **Confidence** is not comparable between the two models (5-class vs. binary).
- **Aspects** are frequent TF-IDF terms, not true aspect-based sentiment. A term can appear in positive and negative reviews alike.
- No license has been chosen for this repository yet.

## Project layout

```
streamlit_app.py   Dashboard
scraper.py         AliExpress review scraper
preprocessing.py   Cleaning, language detection, review-column detection
sentiment.py       Hybrid multilingual / English classifier
utils.py           Translation (cached), word clouds, aspect extraction
evaluate.py        Agreement-with-ratings evaluation
examples/          Synthetic sample reviews
tests/             pytest suite
```
