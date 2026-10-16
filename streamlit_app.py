# streamlit_app.py

import os
from datetime import datetime
import pandas as pd
import streamlit as st
from scraper import AliExpressReviewScraper  
from preprocessing import TextPreprocessor, auto_detect_review_column
from sentiment import SentimentModel
from utils import make_wordcloud, compute_aspects, translate_to_en


st.set_page_config(page_title="Insightify Dashboard", layout="wide")


@st.cache_resource(show_spinner="Loading sentiment models (first run downloads them)...")
def load_sentiment_model(batch_size: int) -> SentimentModel:
    """Load the models once per session instead of on every Streamlit rerun."""
    return SentimentModel(batch_size=batch_size)


st.sidebar.header("Pipeline settings")
max_pages = st.sidebar.number_input(
    "Max review pages to scrape", min_value=1, max_value=500, value=10, step=1
)
batch_size = st.sidebar.number_input(
    "Inference batch size", min_value=1, max_value=128, value=16, step=1
)

st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

/* Smooth background */
.stApp {
    background: linear-gradient(135deg, #f7fbff, #eef4ff);
}

/* Title */
.main-title {
    font-size: 40px !important;
    font-weight: 800 !important;
    color: #1F4E79 !important;
    text-align: center;
    margin-top: 40px !important;
    margin-bottom: 5px !important;
}

/* Subtitle */
.subtitle {
    font-size: 18px;
    text-align: center;
    color: #4A4A4A;
    margin-bottom: 25px !important;
}

/* Card styling */
.section-card {
    background: #ffffffcc;
    backdrop-filter: blur(6px);
    padding: 22px 28px;
    border-radius: 12px;
    border: 1px solid #dce6f5;
    margin-top: 18px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.05);
}



.section-card:empty {
    display: none !important;
    height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
    border: none !important;
    box-shadow: none !important;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    font-size: 16px;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    color: #1F4E79 !important;
}


/* Buttons */
.stButton > button {
    background-color: #2D6CDF !important;
    color: white !important;
    border-radius: 8px;
    padding: 8px 18px;
    border: none;
    font-weight: 600;
}
.stButton > button:hover {
    background-color: #1b4eb3 !important;
}


/* Footer */
.footer {
    text-align:center;
    margin-top:40px;
    color:#6c6c6c;
    font-size:14px;
}

</style>
""", unsafe_allow_html=True)



os.makedirs("data", exist_ok=True)

# HEADER UI

st.markdown('<h1 class="main-title">Insightify</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">E-Commerce Product Review Sentiment Dashboard</p>', unsafe_allow_html=True)
st.markdown("---")

os.makedirs("data", exist_ok=True)

#   INPUT SECTION
with st.container():
    st.markdown('<div class="section-card">', unsafe_allow_html=True)

    st.subheader("Upload CSV or enter an AliExpress URL")
    st.markdown("""
    **Supports:**
    - Alibaba Express Url
    - Multi-product CSV (user selects ONE product) 

    **Uses:**             
    - Translation only for visualization  
    - Hybrid multilingual + English sentiment model  

    **Gives:**             
    - Word clouds, aspects, distribution, trend  
    """)

    choice = st.radio(
        "Choose input method",
        ["Upload CSV", "Enter AliExpress URL"],
        horizontal=True
    )

    st.markdown("</div>", unsafe_allow_html=True)   


df_raw = None

# CSV INPUT
if choice == "Upload CSV":
    file = st.file_uploader("Upload your CSV", type=["csv"])

    if file:
        df_raw = pd.read_csv(file)
        st.success(f"Loaded {len(df_raw)} rows.")
        st.dataframe(df_raw.head())

        # detect review column
        text_col = auto_detect_review_column(df_raw)
        # fallback: any column containing "review"
        if not text_col:
            for c in df_raw.columns:
                if "review" in c.lower():
                    text_col = c
                    st.info(f"Fallback review column detected: **{text_col}**")
                    break

        if not text_col:
            st.error("Could not detect review text column.")
            st.stop()
        st.info(f"Detected review column: **{text_col}**")

        # detect product column
        PRODUCT_COLS = ["asin","product_id","productid","item_id","sku",
                        "product","title","name","id"]

        def detect_product_column(df):
            lower_map = {c.lower(): c for c in df.columns}
            for cand in PRODUCT_COLS:
                if cand in lower_map:
                    col = lower_map[cand]
                    if df[col].nunique() > 1:
                        return col
            return None

        product_col = detect_product_column(df_raw)
        if product_col:
            st.info(f"Detected product column: **{product_col}**")
            products = sorted(df_raw[product_col].astype(str).unique())
            selected_product = st.selectbox("Select product to analyze", products)
            df_raw = df_raw[df_raw[product_col].astype(str) == selected_product].reset_index(drop=True)
            st.success(f"Filtered to {len(df_raw)} rows for product '{selected_product}'")
        else:
            st.info("No product column detected — treating CSV as one product.")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        df_raw.to_csv(f"data/uploaded_{ts}.csv", index=False)

# ALIEXPRESS SCRAPER INPUT
elif choice == "Enter AliExpress URL":
    url = st.text_input("Enter AliExpress Product URL")

    if st.button("Scrape Reviews"):
        if not url or "aliexpress" not in url.lower():
            st.error("Invalid AliExpress URL.")
            st.stop()

        try:
            scraper = AliExpressReviewScraper()
            pid = scraper.extract_product_id(url)
            df_raw = scraper.scrape_all(pid, max_pages_cap=int(max_pages))
            st.success(f"Scraped {len(df_raw)} reviews for product {pid}.")
            df_raw.to_csv(f"data/scraped_{pid}.csv", index=False)
        except Exception as e:
            st.error(f"Scraping failed: {e}")
            st.stop()

# PROCESSING + SENTIMENT + VISUALIZATION
if df_raw is not None and not df_raw.empty:

    text_col = auto_detect_review_column(df_raw)
    if not text_col:
        for c in df_raw.columns:
            if "review" in c.lower():
                text_col = c
                st.info(f"Fallback review column detected: **{text_col}**")
                break

    if not text_col:
        st.error("No review column found.")
        st.stop()

    tp = TextPreprocessor()
    sm = load_sentiment_model(int(batch_size))

    # PREPROCESSING
    with st.container():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Step 1: Preprocessing")
        df = tp.preprocess_df(df_raw, text_col=text_col)
        st.write(df.head())
        st.markdown("</div>", unsafe_allow_html=True)

    # SENTIMENT
    with st.container():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Step 2: Sentiment Analysis")
        df = sm.classify_df(df)
        st.write(df.head())
        st.markdown("</div>", unsafe_allow_html=True)

    # TRANSLATIONS FOR VISUALS
    df["translated_text"] = df.apply(
        lambda row: translate_to_en(row["processed_text"], row["language"]),
        axis=1
    )

    # DATE CHECK
    has_trend = False
    parsed_col = None
    for c in ["review_date", "date", "created_at", "timestamp"]:
        if c in df.columns:
            parsed = pd.to_datetime(df[c], errors="coerce")
            if parsed.notna().sum() > 0:
                df["parsed_date"] = parsed.dt.date
                parsed_col = "parsed_date"
                has_trend = True
                break

    # TABS
    tab_names = ["Sentiment Distribution"]
    if has_trend:
        tab_names.append("Trend Over Time")
    tab_names += ["Word Clouds (English)", "Aspect Insights", "Full Data"]

    tabs = st.tabs(tab_names)

    # TAB 1
    with tabs[0]:
        st.subheader("Sentiment Distribution")
        order = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
        counts = df["sentiment_3way"].value_counts().reindex(order, fill_value=0)
        st.bar_chart(counts)

    # TAB 2
    idx = 1
    if has_trend:
        with tabs[idx]:
            st.subheader("Sentiment Trend Over Time")
            trend = df.groupby(parsed_col)["stars"].mean().reset_index()
            st.line_chart(trend.set_index(parsed_col))
        idx += 1

    # TAB 3
    with tabs[idx]:
        st.subheader("Word Clouds (Translated to English)")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Positive Reviews")
            make_wordcloud(
                df[df["sentiment_3way"] == "POSITIVE"]["translated_text"],
                "Positive Reviews",
            )

        with col2:
            st.markdown("### Negative Reviews")
            make_wordcloud(
                df[df["sentiment_3way"] == "NEGATIVE"]["translated_text"],
                "Negative Reviews",
            )

    idx += 1

    # TAB 4
    with tabs[idx]:
        st.subheader("Aspect Extraction (Translated Text)")
        st.dataframe(compute_aspects(df))
    idx += 1

    # TAB 5
    with tabs[idx]:
        st.subheader("Full Processed Dataset")
        st.dataframe(df)
        st.download_button(
            "Download CSV",
            df.to_csv(index=False).encode("utf-8"),
            "insightify_results.csv",
            "text/csv",
        )

else:
    st.info("Upload a CSV or enter a URL to begin.")

# FOOTER
st.markdown("""
<div class="footer">
    Team Insightify
</div>
""", unsafe_allow_html=True)
