"""
Sentiment Analysis Dashboard
============================
A Streamlit-powered web application for analyzing text sentiment using pretrained
Hugging Face transformer models. Supports single text input, bulk CSV analysis,
and comparison across categories/sources.

Features:
  - Single text sentiment analysis with confidence scores
  - CSV file upload and batch analysis
  - Synthesic sample dataset generation and analysis
  - Sentiment distribution bar charts (Plotly)
  - Time-series trend analysis for dated data
  - Word clouds per sentiment category
  - Cross-category and cross-source comparison
"""

import os
import sys
import io
import time
from collections import Counter, OrderedDict
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import streamlit as st
from transformers import pipeline, AutoTokenizer
from wordcloud import WordCloud
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
SAMPLE_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_reviews.csv")

SENTIMENT_COLORS = {
    "positive": "#2ecc71",
    "neutral": "#f39c12",
    "negative": "#e74c3c",
}

SENTIMENT_ORDER = ["positive", "neutral", "negative"]

# ---------------------------------------------------------------------------
# Cache the model pipeline so it loads only once
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading transformer model …")
def load_model():
    """Load the sentiment-analysis pipeline from HuggingFace."""
    pipe = pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        return_all_scores=True,
        truncation=True,
        max_length=512,
    )
    return pipe


# ---------------------------------------------------------------------------
# Helper: parse pipeline output into a tidy dict
# ---------------------------------------------------------------------------
def parse_scores(scores_list):
    """
    Convert the raw pipeline output for one text into
    {\"positive\": 0.xx, \"neutral\": 0.xx, \"negative\": 0.xx}.
    """
    # pipeline returns [{\"label\": \"positive\", \"score\": 0.98}, …]
    result = {}
    for item in scores_list:
        label = item["label"].lower()
        # Map roBERTa labels to our canonical names
        if "pos" in label:
            result["positive"] = round(item["score"], 4)
        elif "neg" in label:
            result["negative"] = round(item["score"], 4)
        elif "neu" in label:
            result["neutral"] = round(item["score"], 4)
    return result


def dominant_sentiment(scores: dict) -> str:
    """Return the label with the highest confidence."""
    return max(scores, key=scores.get)


# ---------------------------------------------------------------------------
# Helper: analyse a list of texts in batches
# ---------------------------------------------------------------------------
def analyze_texts(texts, pipe):
    """Run sentiment analysis on an iterable of strings, return list of dicts."""
    results = []
    # Process in small batches for progress tracking
    batch_size = 32
    texts_list = list(texts)

    for i in range(0, len(texts_list), batch_size):
        batch = texts_list[i : i + batch_size]
        outs = pipe(batch)
        for out in outs:
            scores = parse_scores(out)
            results.append(scores)
    return results


# ---------------------------------------------------------------------------
# Word‑cloud generator
# ---------------------------------------------------------------------------
COLOR_MAP = {
    "positive": "#2ecc71",
    "neutral": "#f39c12",
    "negative": "#e74c3c",
}


def build_wordcloud(text_series, sentiment_label):
    """Return a matplotlib Figure with a word cloud."""
    STOP_WORDS_EXTRA = {
        "product", "feature", "one", "also", "really", "would",
        "get", "got", "even", "just", "bit", "much", "many",
        "well", "still", "could", "make", "made", "use", "used",
    }
    wc = WordCloud(
        width=600,
        height=350,
        background_color="white",
        colormap="Greens" if sentiment_label == "positive" else
                 ("Oranges" if sentiment_label == "neutral" else "Reds"),
        max_words=80,
        stopwords=STOP_WORDS_EXTRA,
        min_font_size=8,
        prefer_horizontal=0.9,
        random_state=42,
    )
    wc.generate(" ".join(text_series.astype(str)))

    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=120)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"Word Cloud — {sentiment_label.capitalize()}", fontsize=12, fontweight="bold")
    plt.tight_layout(pad=0)
    return fig


# ---------------------------------------------------------------------------
# ── Sidebar ────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Settings")
st.sidebar.markdown(f"**Model:** `{MODEL_NAME}`")
st.sidebar.divider()

# Ensure sample data exists
if not os.path.exists(SAMPLE_DATA_PATH):
    with st.sidebar.status("Generating synthetic sample data …"):
        # Run the generator script
        import subprocess
        subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "generate_sample_data.py")],
                       check=True)

st.sidebar.success("✅ Sample data ready")


# ---------------------------------------------------------------------------
# ── App header ─────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------
st.title("🧠 Sentiment Analysis Dashboard")
st.markdown(
    "Powered by **RoBERTa-base** (fine-tuned on Twitter sentiment). "
    "Paste text, upload a CSV, or load a built-in synthetic dataset to explore sentiment patterns."
)

# ---------------------------------------------------------------------------
# ── Tab navigation ─────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------
tab_single, tab_csv, tab_dataset, tab_compare = st.tabs([
    "✏️ Single Text",
    "📁 CSV Upload",
    "📊 Sample Dataset",
    "⚖️ Compare Sources",
])

pipe = load_model()

# ===================================================================
# TAB 1 — Single Text Analysis
# ===================================================================
with tab_single:
    st.header("Single Text Sentiment Analysis")
    user_input = st.text_area(
        "Paste or type text below:",
        placeholder="e.g. I absolutely loved this product! The battery life is amazing and it works perfectly.",
        height=120,
    )

    if st.button("🔍 Analyze Sentiment", type="primary", disabled=not user_input.strip()):
        scores = parse_scores(pipe(user_input.strip())[0])
        sent = dominant_sentiment(scores)

        # KPI row
        col_a, col_b, col_c = st.columns(3)
        for i, (lbl, col) in enumerate(zip(SENTIMENT_ORDER, [col_a, col_b, col_c])):
            with col:
                st.metric(
                    label=lbl.capitalize(),
                    value=f"{scores[lbl]*100:.1f}%",
                    delta=None,
                )

        # Bar chart
        df_bar = pd.DataFrame({"sentiment": list(scores.keys()), "confidence": list(scores.values())})
        fig = px.bar(
            df_bar, x="sentiment", y="confidence", color="sentiment",
            color_discrete_map=SENTIMENT_COLORS,
            text_auto=".1%",
            labels={"confidence": "Confidence", "sentiment": ""},
        )
        fig.update_traces(width=0.5)
        fig.update_layout(showlegend=False, yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

        # Verdict
        emoji = {"positive": "😊", "neutral": "😐", "negative": "😞"}
        st.info(f"**Verdict →** {emoji[sent]} **{sent.upper()}** (confidence: {scores[sent]*100:.1f}%)")


# ===================================================================
# TAB 2 — CSV Upload & Batch Analysis
# ===================================================================
with tab_csv:
    st.header("CSV File Analysis")
    uploaded = st.file_uploader("Upload a CSV with a **text** column", type=["csv"])

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        # Detect text column
        text_col = None
        for c in ["text", "review", "comment", "content", "tweet", "feedback"]:
            if c in df.columns:
                text_col = c
                break
        if text_col is None:
            text_col = st.selectbox("Which column contains the text to analyze?", list(df.columns))

        date_col = None
        for c in ["date", "time", "timestamp", "created_at"]:
            if c in df.columns:
                date_col = c
                break

        st.write(f"Loaded **{len(df)}** rows. Analyzing …")

        texts = df[text_col].astype(str).tolist()
        results = analyze_texts(texts, pipe)

        df["sentiment"] = [dominant_sentiment(r) for r in results]
        df["pos_score"] = [r["positive"] for r in results]
        df["neu_score"] = [r["neutral"] for r in results]
        df["neg_score"] = [r["negative"] for r in results]

        # KPI cards
        counts = df["sentiment"].value_counts()
        k1, k2, k3 = st.columns(3)
        for lbl, col in zip(SENTIMENT_ORDER, [k1, k2, k3]):
            with col:
                pct = counts.get(lbl, 0) / len(df) * 100
                st.metric(lbl.capitalize(), f"{counts.get(lbl, 0)}", f"{pct:.1f}% of total")

        # Distribution chart
        fig_dist = px.bar(
            x=SENTIMENT_ORDER,
            y=[counts.get(s, 0) for s in SENTIMENT_ORDER],
            color=SENTIMENT_ORDER,
            color_discrete_map=SENTIMENT_COLORS,
            text_auto=True,
            labels={"x": "", "y": "Count"},
        )
        fig_dist.update_layout(showlegend=False)
        st.plotly_chart(fig_dist, use_container_width=True)

        # Time series (if date column detected)
        if date_col and date_col in df.columns:
            df_date = df.copy()
            df_date[date_col] = pd.to_datetime(df_date[date_col], errors="coerce")
            df_date = df_date.dropna(subset=[date_col]).sort_values(date_col)
            if len(df_date) > 1:
                df_date["month"] = df_date[date_col].dt.to_period("M").astype(str)
                ts = df_date.groupby(["month", "sentiment"]).size().reset_index(name="count")
                fig_ts = px.line(
                    ts, x="month", y="count", color="sentiment",
                    color_discrete_map=SENTIMENT_COLORS, markers=True,
                    labels={"month": "Month", "count": "Reviews"},
                )
                st.subheader("Sentiment Trend Over Time")
                st.plotly_chart(fig_ts, use_container_width=True)

        # Word clouds
        st.subheader("Word Clouds by Sentiment")
        wc1, wc2, wc3 = st.columns(3)
        for lbl, col in zip(SENTIMENT_ORDER, [wc1, wc2, wc3]):
            subset = df.loc[df["sentiment"] == lbl, text_col]
            if len(subset) > 0:
                with col:
                    fig_wc = build_wordcloud(subset, lbl)
                    st.pyplot(fig_wc)

        # Download enriched CSV
        csv_out = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download analysed CSV",
            csv_out,
            "analysed_sentiment.csv",
            "text/csv",
        )


# ===================================================================
# TAB 3 — Synthetic Sample Dataset
# ===================================================================
with tab_dataset:
    st.header("Synthetic Sample Dataset (500 Reviews)")

    if os.path.exists(SAMPLE_DATA_PATH):
        df_sample = pd.read_csv(SAMPLE_DATA_PATH)
        st.write(f"Loaded **{len(df_sample)}** generated reviews.")

        # User can re-generate
        if st.button("🔄 Regenerate Dataset"):
            import subprocess
            subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "generate_sample_data.py")],
                           check=True)
            st.rerun()

        # Run analysis
        texts = df_sample["text"].astype(str).tolist()
        with st.spinner("Analysing with transformer model (this may take ~30 s) …"):
            results = analyze_texts(texts, pipe)

        df_sample["sentiment"] = [dominant_sentiment(r) for r in results]
        df_sample["pos_score"] = [r["positive"] for r in results]
        df_sample["neu_score"] = [r["neutral"] for r in results]
        df_sample["neg_score"] = [r["negative"] for r in results]
        df_sample["date"] = pd.to_datetime(df_sample["date"])

        # ---- KPI row ----
        c1, c2, c3, c4 = st.columns(4)
        counts = df_sample["sentiment"].value_counts()
        total = len(df_sample)
        c1.metric("Total Reviews", total)
        c2.metric("😊 Positive", counts.get("positive", 0), f"{counts.get('positive',0)/total*100:.1f}%")
        c3.metric("😐 Neutral", counts.get("neutral", 0), f"{counts.get('neutral',0)/total*100:.1f}%")
        c4.metric("😞 Negative", counts.get("negative", 0), f"{counts.get('negative',0)/total*100:.1f}%")

        # ---- Distribution bar chart ----
        fig_dist = px.bar(
            x=SENTIMENT_ORDER,
            y=[counts.get(s, 0) for s in SENTIMENT_ORDER],
            color=SENTIMENT_ORDER,
            color_discrete_map=SENTIMENT_COLORS,
            text_auto=True,
            labels={"x": "Sentiment", "y": "Count"},
        )
        fig_dist.update_layout(showlegend=False)
        st.plotly_chart(fig_dist, use_container_width=True)

        # ---- Time series ----
        df_ts = df_sample.sort_values("date")
        df_ts["month"] = df_ts["date"].dt.to_period("M").astype(str)
        ts = df_ts.groupby(["month", "sentiment"]).size().reset_index(name="count")

        fig_ts = px.line(
            ts, x="month", y="count", color="sentiment",
            color_discrete_map=SENTIMENT_COLORS, markers=True,
            labels={"month": "Month", "count": "Number of Reviews"},
            title="Sentiment Trend Over Time (Monthly)",
        )
        st.plotly_chart(fig_ts, use_container_width=True)

        # ---- Confidence distribution ----
        fig_conf = go.Figure()
        for lbl in SENTIMENT_ORDER:
            fig_conf.add_trace(go.Violin(
                y=[r[lbl] for r in results],
                name=lbl.capitalize(),
                line_color=SENTIMENT_COLORS[lbl],
                box_visible=True,
                meanline_visible=True,
            ))
        fig_conf.update_layout(title="Confidence Score Distribution", yaxis_title="Confidence")
        st.plotly_chart(fig_conf, use_container_width=True)

        # ---- Word clouds ----
        st.subheader("Word Clouds by Sentiment Category")
        wc1, wc2, wc3 = st.columns(3)
        for lbl, col in zip(SENTIMENT_ORDER, [wc1, wc2, wc3]):
            subset = df_sample.loc[df_sample["sentiment"] == lbl, "text"]
            with col:
                fig_wc = build_wordcloud(subset, lbl)
                st.pyplot(fig_wc)

        # ---- Data preview ----
        with st.expander("📋 Preview first 20 rows"):
            st.dataframe(df_sample.head(20), use_container_width=True)

    else:
        st.warning("Sample data not found. Click the button in the sidebar to generate it.")


# ===================================================================
# TAB 4 — Compare Sources / Categories
# ===================================================================
with tab_compare:
    st.header("Sentiment Comparison Across Sources & Categories")

    if os.path.exists(SAMPLE_DATA_PATH):
        df_comp = pd.read_csv(SAMPLE_DATA_PATH)
        texts = df_comp["text"].astype(str).tolist()

        with st.spinner("Analysing dataset (this may take ~30 s) …"):
            results = analyze_texts(texts, pipe)

        df_comp["sentiment"] = [dominant_sentiment(r) for r in results]
        df_comp["date"] = pd.to_datetime(df_comp["date"])

        compare_by = st.radio("Group by:", ["source", "category"], horizontal=True)

        group_counts = df_comp.groupby([compare_by, "sentiment"]).size().unstack(fill_value=0)
        # Ensure all three columns exist
        for s in SENTIMENT_ORDER:
            if s not in group_counts.columns:
                group_counts[s] = 0
        group_counts = group_counts[SENTIMENT_ORDER]

        # Normalised stacked bar (100 %)
        group_pct = group_counts.div(group_counts.sum(axis=1), axis=0) * 100

        fig_comp = px.bar(
            group_pct.reset_index().melt(id_vars=compare_by, var_name="sentiment", value_name="pct"),
            x=compare_by, y="pct", color="sentiment",
            color_discrete_map=SENTIMENT_COLORS,
            barmode="stack",
            text_auto=".1f",
            labels={"pct": "% of reviews", compare_by: compare_by.capitalize()},
            title=f"Sentiment Distribution by {compare_by.capitalize()}",
        )
        fig_comp.update_layout(yaxis_range=[0, 105])
        st.plotly_chart(fig_comp, use_container_width=True)

        # Side-by-side count comparison
        col_left, col_right = st.columns(2)
        with col_left:
            fig_count = go.Figure()
            for lbl in SENTIMENT_ORDER:
                fig_count.add_trace(go.Bar(
                    name=lbl.capitalize(),
                    x=group_counts.index,
                    y=group_counts[lbl],
                    marker_color=SENTIMENT_COLORS[lbl],
                ))
            fig_count.update_layout(
                barmode="group",
                title=f"Raw Counts by {compare_by.capitalize()}",
                xaxis_title=compare_by.capitalize(),
                yaxis_title="Count",
            )
            st.plotly_chart(fig_count, use_container_width=True)

        with col_right:
            fig_avg = px.bar(
                df_comp.groupby([compare_by, "sentiment"])["pos_score"].mean().reset_index(),
                x=compare_by, y="pos_score", color="sentiment",
                color_discrete_map=SENTIMENT_COLORS,
                barmode="group",
                labels={"pos_score": "Avg Positive Score", compare_by: compare_by.capitalize()},
                title=f"Average Positive Confidence by {compare_by.capitalize()}",
            )
            st.plotly_chart(fig_avg, use_container_width=True)

        # Word clouds per group
        st.subheader(f"Word Clouds by {compare_by.capitalize()}")
        groups = list(df_comp[compare_by].unique())
        n_cols = min(3, len(groups))
        cols = st.columns(n_cols)
        for i, grp in enumerate(groups):
            with cols[i % n_cols]:
                subset = df_comp.loc[df_comp[compare_by] == grp, "text"]
                fig_wc = WordCloud(
                    width=500, height=300, background_color="white",
                    colormap="viridis", max_words=60, random_state=42,
                )
                fig_wc.generate(" ".join(subset.astype(str)))
                f, ax = plt.subplots(figsize=(5, 3), dpi=110)
                ax.imshow(fig_wc, interpolation="bilinear")
                ax.axis("off")
                ax.set_title(grp, fontsize=11, fontweight="bold")
                plt.tight_layout(pad=0)
                st.pyplot(f)
                plt.close(f)
    else:
        st.warning("Sample data not found. Generate it from the sample dataset tab first.")


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    f"Sentiment Analysis Dashboard  •  Model: {MODEL_NAME}  •  Built with Streamlit + HuggingFace Transformers"
)
