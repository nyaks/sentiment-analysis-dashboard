# 🧠 Sentiment Analysis Dashboard

A production-ready, interactive web application for analyzing text sentiment using state-of-the-art transformer models from HuggingFace.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-orange)

---

## 🚀 Quick Start

```bash
# 1. Clone / navigate to the project
cd ~/ai-portfolio/sentiment-dashboard

# 2. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Pre-generate the sample dataset
python generate_sample_data.py

# 5. Launch the dashboard
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`. The transformer model (~500 MB) downloads automatically on first run.

---

## 🏗️ Architecture Overview

```
┌──────────────────────────────────────────────────────┐
│                    Streamlit Frontend                 │
│  ┌──────────┐ ┌──────────┐ ┌───────────┐ ┌────────┐ │
│  │SingleText│ │CSVUpload │ │SampleData │ │Compare │ │
│  │   Tab    │ │   Tab    │ │   Tab     │ │  Tab   │ │
│  └────┬─────┘ └────┬─────┘ └─────┬─────┘ └───┬────┘ │
│       └─────────────┴─────────────┴───────────┘      │
│                          │                            │
│              ┌───────────▼───────────┐               │
│              │  Transformers Pipeline │               │
│              │  (RoBERTa-base-sent)   │               │
│              └───────────┬───────────┘               │
│                          │                            │
│        ┌─────────────────┼─────────────────┐         │
│        ▼                 ▼                 ▼         │
│   ┌─────────┐     ┌────────────┐    ┌──────────┐    │
│   │ Plotly  │     │  WordCloud │    │  Pandas  │    │
│   │ Charts  │     │  (mpl)     │    │  DataWr  │    │
│   └─────────┘     └────────────┘    └──────────┘    │
└──────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Model**: `cardiffnlp/twitter-roberta-base-sentiment-latest` | Fine-tuned RoBERTa on Twitter data; handles informal language, emojis, and short texts well. Returns probability scores for all three classes. |
| **Batch processing** (32 per call) | Balances GPU memory with throughput. The pipeline API handles tokenization + inference in one call. |
| **`@st.cache_resource`** | Prevents re-loading the 500 MB model on every user interaction. Loaded once per session. |
| **Plotly for charts** | Interactive, zoomable, hover-tooltips — much more impressive in demos than static matplotlib. |
| **WordCloud + matplotlib** | WordCloud doesn't support Plotly natively; renders via matplotlib and `st.pyplot()`. |

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web UI framework |
| `transformers` + `torch` | HuggingFace transformer model loading & inference |
| `pandas` | Data manipulation, CSV handling, time-series grouping |
| `plotly` | Interactive bar charts, line charts, violin plots |
| `wordcloud` + `matplotlib` | Word cloud visualization per sentiment category |
| `numpy` | Numerical operations |
| `sentencepiece` | Tokenizer backend for RoBERTa |

---

## 🧩 Features

### 1. ✏️ Single Text Analysis
Paste any text and get instant sentiment classification with confidence scores, a bar chart, and a verdict label.

### 2. 📁 CSV Upload
Upload any CSV — the app auto-detects the text and date columns, runs batch analysis, then shows:
- Sentiment distribution bar chart
- Time-series trends (if dates are present)
- Word clouds per sentiment category
- Downloadable enriched CSV

### 3. 📊 Sample Dataset (500 Reviews)
A synthetic dataset of product reviews across 5 categories and 5 sources, with realistic ratings and dates. Includes:
- KPI summary cards
- Distribution chart + time-series
- Confidence violin plots
- Per-sentiment word clouds

### 4. ⚖️ Compare Sources
Group sentiment results by **source** (Twitter, Amazon, Reddit, etc.) or **category** (Electronics, Books, etc.) and view:
- 100% stacked bar chart (normalised)
- Grouped raw count comparison
- Average positive confidence comparison
- Word clouds per group

---

## 🎤 Interview Talking Points

### 1. Model Selection & Trade-offs
> "I chose `cardiffnlp/twitter-roberta-base-sentiment-latest` — a RoBERTa-base model fine-tuned on ~124M tweets. It's a good trade-off between accuracy (state-of-the-art for social media text) and runtime (~500 MB, runs on CPU in ~1-2 sec per batch). For a production system, I'd evaluate whether the domain matches our data, and potentially fine-tune on our own labeled examples."

### 2. Caching Strategy
> "The transformer model is cached using `@st.cache_resource`, which means Streamlit loads the 500 MB model once and reuses it across rerenders. Without this, every user interaction would trigger a full model reload, making the app unusable. I also batch predictions in groups of 32 to balance memory and throughput."

### 3. Data Pipeline Design
> "The app handles three input modes: free text, CSV upload, and a built-in synthetic dataset. All three converge on the same analysis pipeline — `analyze_texts()` returns standardized score dicts, which downstream code maps to dominant sentiment labels, confidence columns, and aggregation tables. This single-source-of-truth approach avoids code duplication."

### 4. Visualization Choices
> "I used Plotly for all interactive charts because zoom, hover tooltips, and legend toggling let users explore the data during a live demo. Word clouds give an at-a-glance view of language differences between sentiment categories. The violin plot for confidence scores shows not just the mean but the full distribution — useful for understanding model certainty."

### 5. Handling Edge Cases
> "The app handles several edge cases: missing date columns gracefully skip time-series charts; unknown CSV column names let the user select the right one via a dropdown; the model pipeline truncates inputs at 512 tokens to avoid OOM errors; and sentiment labels are normalized from the model's output format to a canonical set."

### 6. Production Readiness & Next Steps
> "This is a strong demo/prototype. For production, I'd add: (1) async batch processing with a task queue (Celery/RQ) for large CSVs, (2) model monitoring and drift detection, (3) a proper database instead of in-memory pandas, (4) authentication and rate limiting, and (5) an A/B framework to compare multiple models. I'd also containerize with Docker for deployment."

---

## 📁 File Structure

```
sentiment-dashboard/
├── app.py                  # Main Streamlit application (~400 lines)
├── generate_sample_data.py # Synthetic dataset generator (500 reviews)
├── sample_reviews.csv      # Generated dataset (auto-created)
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## 📜 License

MIT License — free to use, modify, and distribute.
