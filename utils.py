import matplotlib.pyplot as plt
from wordcloud import WordCloud
from deep_translator import GoogleTranslator
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

# translate any text to English for visualization
def translate_to_en(text: str, lang: str = "auto") -> str:
    if not isinstance(text, str) or not text.strip():
        return ""

    # Skip if already English
    if lang and lang.lower() == "en":
        return text  

    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except:
        return text


def make_wordcloud(text_series, title):
    en_texts = [translate_to_en(t) for t in text_series.fillna("").astype(str).tolist()]
    joined = " ".join(en_texts)

    if len(joined.split()) < 5:
        st.info(f"Not enough text for {title}")
        return

    wc = WordCloud(
        width=450,        
        height=250,       
        background_color="white",
        max_words=100
    ).generate(joined)

    fig, ax = plt.subplots(figsize=(4.5, 2.5)) 
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)



def compute_aspects(df: pd.DataFrame, top_n=20):
    # translate processed_text for aspects
    en_texts = df["processed_text"].fillna("").astype(str).apply(translate_to_en)

    vec = TfidfVectorizer(stop_words="english", min_df=2, max_df=0.9)
    try:
        X = vec.fit_transform(en_texts)
    except:
        return pd.DataFrame()

    tokens = vec.get_feature_names_out()
    scores = X.sum(axis=0).A1

    top_idx = scores.argsort()[::-1][:top_n]
    top_tokens = [tokens[i] for i in top_idx]

    pos = df[df["sentiment_3way"]=="POSITIVE"]["processed_text"].astype(str).apply(translate_to_en)
    neg = df[df["sentiment_3way"]=="NEGATIVE"]["processed_text"].astype(str).apply(translate_to_en)

    aspects = []
    for tok in top_tokens:
        pos_count = sum(tok in t for t in pos)
        neg_count = sum(tok in t for t in neg)
        if pos_count or neg_count:
            aspects.append({
                "aspect": tok,
                "positive_mentions": int(pos_count),
                "negative_mentions": int(neg_count)
            })

    return pd.DataFrame(aspects)







