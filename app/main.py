import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re
import os
from PIL import Image
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Jindal Steel NLP Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# COLOR PALETTE  (from user's NexaDash images)
# Black:     #070A0D   (deep bg)
# Card:      #0E1418   (secondary bg)
# Teal:      #0E6243   (dark teal)
# Cyan:      #00E5CC   (neon accent)
# SlateGray: #7E8D8E   (muted text)
# White:     #FFFFFF
# ─────────────────────────────────────────────
CYAN   = "#00E5CC"
TEAL   = "#0E6243"
BG     = "#070A0D"
CARD   = "#0E1418"
BORDER = "#1A2A2A"
GRAY   = "#7E8D8E"
WHITE  = "#F0F4F8"

PLOTLY_TEMPLATE = "plotly_dark"
TEAL_SCALE = [[0, "#0E1418"], [0.5, "#0E6243"], [1.0, "#00E5CC"]]

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
base_dir           = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cleaned_data_path  = os.path.join(base_dir, "data", "processed", "reviews_cleaned.csv")
labeled_data_path  = os.path.join(base_dir, "outputs", "labeled_dataset.csv")
conf_matrix_path   = os.path.join(base_dir, "outputs", "figures", "confusion_matrix.png")
comp_chart_path    = os.path.join(base_dir, "outputs", "figures", "sentiment_comparison_chart.png")
wc_neg_path        = os.path.join(base_dir, "outputs", "wordclouds", "wordcloud_negative.png")
wc_pos_path        = os.path.join(base_dir, "outputs", "wordclouds", "wordcloud_positive.png")

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    background-color: {BG};
}}

/* Sidebar */
[data-testid="stSidebar"] {{
    background: {CARD} !important;
    border-right: 1px solid {BORDER};
}}
[data-testid="stSidebar"] .stRadio label {{
    color: {GRAY};
    font-size: 0.9rem;
}}
[data-testid="stSidebar"] .stRadio label:hover {{
    color: {CYAN};
}}

/* Metric cards */
.kpi-card {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-top: 3px solid {CYAN};
    border-radius: 10px;
    padding: 18px 14px;
    text-align: center;
    margin-bottom: 12px;
}}
.kpi-value {{
    font-size: 1.9rem;
    font-weight: 700;
    color: {CYAN};
    line-height: 1.2;
}}
.kpi-label {{
    font-size: 0.75rem;
    color: {GRAY};
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
}}
.kpi-sub {{
    font-size: 0.78rem;
    color: {GRAY};
    margin-top: 6px;
}}

/* Pipeline steps */
.pipe-box {{
    background: {CARD};
    border: 1px solid {TEAL};
    border-radius: 8px;
    padding: 12px 8px;
    text-align: center;
    flex: 1;
    margin: 0 4px;
}}
.pipe-title {{
    font-size: 0.82rem;
    font-weight: 600;
    color: {CYAN};
    margin-bottom: 4px;
}}
.pipe-desc {{
    font-size: 0.70rem;
    color: {GRAY};
}}
.pipe-arrow {{
    font-size: 1.4rem;
    color: {TEAL};
    padding: 0 2px;
    flex-shrink: 0;
}}

/* Section divider */
.section-divider {{
    border: none;
    border-top: 1px solid {BORDER};
    margin: 24px 0;
}}

/* Insight boxes */
.insight-box {{
    background: {CARD};
    border-left: 4px solid {CYAN};
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin-bottom: 12px;
}}
.insight-title {{
    font-size: 0.9rem;
    font-weight: 600;
    color: {CYAN};
    margin-bottom: 6px;
}}
.insight-body {{
    font-size: 0.85rem;
    color: {WHITE};
    line-height: 1.6;
}}

/* Live analyser result cards */
.result-card {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 16px;
    text-align: center;
    margin-top: 8px;
    height: 160px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
}}
.result-model {{
    font-size: 0.75rem;
    color: {GRAY};
    text-transform: uppercase;
    letter-spacing: 0.08em;
}}
.result-label {{
    font-size: 2rem;
    font-weight: 700;
    margin: 6px 0;
}}
.result-conf {{
    font-size: 0.8rem;
    color: {GRAY};
}}
.result-positive  {{ color: {CYAN}; }}
.result-neutral   {{ color: #FBBF24; }}
.result-negative  {{ color: #EF4444; }}
.agree-badge  {{ background:#0E6243; color:{CYAN}; border-radius:6px; padding:6px 14px; font-size:0.85rem; font-weight:600; }}
.disagree-badge {{ background:#3A1212; color:#EF4444; border-radius:6px; padding:6px 14px; font-size:0.85rem; font-weight:600; }}

/* Explorer table tweaks */
[data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 8px; }}

/* Buttons — dark teal so text is readable */
.stButton > button {{
    background-color: {TEAL} !important;
    color: {WHITE} !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
}}
.stButton > button:hover {{
    background-color: #0A4D34 !important;
}}

/* Multiselect pills & selectbox — keep text visible */
[data-baseweb="select"] {{
    background-color: {CARD} !important;
}}
[data-baseweb="tag"] {{
    background-color: {TEAL} !important;
}}
[data-baseweb="tag"] span {{
    color: {WHITE} !important;
}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data
def load_data(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_resource
def load_vader():
    try:
        sia = SentimentIntensityAnalyzer()
        sia.polarity_scores("test")
        return sia
    except Exception:
        nltk.download("vader_lexicon", quiet=True)
        return SentimentIntensityAnalyzer()


df_cleaned = load_data(cleaned_data_path)
df_labeled = load_data(labeled_data_path)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.markdown(f"""
<div style='padding:10px 0 16px;'>
  <div style='font-size:1.1rem;font-weight:700;color:{CYAN};'>Jindal Steel NLP</div>
  <div style='font-size:0.75rem;color:{GRAY};margin-top:2px;'>Quality Feedback Analyzer</div>
</div>
""", unsafe_allow_html=True)

navigation = st.sidebar.radio(
    "Navigation",
    [
        "Overview & Pipeline",
        "Week 1 · Exploration",
        "Week 2 · Model Analysis",
        "Live Analyser",
        "Data Explorer",
    ]
)

st.sidebar.markdown("<hr style='border-color:#1A2A2A;margin:16px 0;'>", unsafe_allow_html=True)
if df_cleaned is not None:
    st.sidebar.markdown(f"""
    <div style='font-size:0.75rem;color:{GRAY};'>
      <b style='color:{WHITE};'>Dataset</b><br>
      {len(df_cleaned):,} feedback records<br>
      {df_cleaned['product_category'].nunique()} product lines<br>
      {df_cleaned['customer_region'].nunique()} regions
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def kpi(value, label, sub="", accent=CYAN):
    return f"""
    <div class="kpi-card">
      <div class="kpi-value" style="color:{accent};">{value}</div>
      <div class="kpi-label">{label}</div>
      <div class="kpi-sub">{sub}</div>
    </div>"""

def insight_box(title, body):
    return f"""
    <div class="insight-box">
      <div class="insight-title">{title}</div>
      <div class="insight-body">{body}</div>
    </div>"""

def teal_bar(df_col, title, orientation="v", top_n=10):
    counts = df_col.value_counts().head(top_n).reset_index()
    counts.columns = ["label", "count"]
    if orientation == "h":
        fig = px.bar(counts, x="count", y="label", orientation="h",
                     color="count", color_continuous_scale=TEAL_SCALE,
                     template=PLOTLY_TEMPLATE)
        fig.update_layout(yaxis=dict(autorange="reversed"))
    else:
        fig = px.bar(counts, x="label", y="count",
                     color="count", color_continuous_scale=TEAL_SCALE,
                     template=PLOTLY_TEMPLATE)
    fig.update_layout(
        title=title, margin=dict(t=40, b=20, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False, showlegend=False,
        font_color=WHITE
    )
    fig.update_traces(marker_line_width=0)
    return fig

def sentiment_color(label):
    if label == "positive": return "result-positive"
    if label == "negative": return "result-negative"
    return "result-neutral"

# ─────────────────────────────────────────────
# PAGE 1: OVERVIEW & PIPELINE
# ─────────────────────────────────────────────
if navigation == "Overview & Pipeline":
    st.markdown(f"<h1 style='color:{WHITE};margin-bottom:4px;'>Jindal Steel Limited</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{GRAY};font-size:1rem;margin-top:0;'>B2B Quality Feedback NLP Analyzer · Weeks 1 & 2</p>", unsafe_allow_html=True)
    st.markdown("<hr style='border-color:#1A2A2A;'>", unsafe_allow_html=True)

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(kpi("10,000", "Feedback Records", "Full dataset cleaned"), unsafe_allow_html=True)
    c2.markdown(kpi("10,000", "VADER Evaluated", "100% lexicon run"), unsafe_allow_html=True)
    c3.markdown(kpi("2,500", "DistilBERT Sample", "Representative sample"), unsafe_allow_html=True)
    c4.markdown(kpi("78.48%", "DistilBERT Overall Acc", "vs VADER 68.35%"), unsafe_allow_html=True)
    c5.markdown(kpi("96.60%", "DistilBERT Binary Acc", "Polar reviews only", accent=CYAN), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Pipeline flowchart
    st.markdown(f"<h3 style='color:{WHITE};'> End-to-End NLP Pipeline</h3>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="display:flex;align-items:center;background:{CARD};border:1px solid {BORDER};padding:20px;border-radius:10px;overflow-x:auto;">
      <div class="pipe-box"><div class="pipe-title">① Data Ingestion</div><div class="pipe-desc">10,000 Jindal Steel B2B feedback rows loaded from CSV</div></div>
      <div class="pipe-arrow">→</div>
      <div class="pipe-box"><div class="pipe-title">② Preprocessing</div><div class="pipe-desc">Lowercase, punctuation stripping, metadata mapping</div></div>
      <div class="pipe-arrow">→</div>
      <div class="pipe-box"><div class="pipe-title">③ Tokenization</div><div class="pipe-desc">NLTK word tokenization & stopword removal</div></div>
      <div class="pipe-arrow">→</div>
      <div class="pipe-box"><div class="pipe-title">④ Lemmatization</div><div class="pipe-desc">WordNet lemmatizer reduces to root forms</div></div>
      <div class="pipe-arrow">→</div>
      <div class="pipe-box"><div class="pipe-title">⑤ VADER Analysis</div><div class="pipe-desc">Rule-based lexical classifier on raw text</div></div>
      <div class="pipe-arrow">→</div>
      <div class="pipe-box"><div class="pipe-title">⑥DistilBERT</div><div class="pipe-desc">Deep learning transformer on 2,500 sample</div></div>
      <div class="pipe-arrow">→</div>
      <div class="pipe-box"><div class="pipe-title">⑦ Evaluation</div><div class="pipe-desc">Accuracy, Confusion Matrix, Speed Benchmarks</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Pipeline step expanders (Before → After)
    st.markdown(f"<h3 style='color:{WHITE};'> Pipeline Step Inspector</h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{GRAY};'>Expand each step to see exactly how text is transformed through the pipeline.</p>", unsafe_allow_html=True)

    sample_raw = "We appreciate the prompt engineer visit, but the Rebar TMT bars from Raigarh have inconsistent tensile strength! IS 1786 compliance is failing."

    # Pre-compute all transformations — guarded so any NLTK issue never crashes the page
    _cleaned = re.sub(r'[^\w\s]', '', sample_raw.lower())
    try:
        for _pkg in ["stopwords", "punkt_tab", "punkt", "wordnet"]:
            nltk.download(_pkg, quiet=True)
        from nltk.corpus import stopwords as _sw
        from nltk.tokenize import word_tokenize as _wt
        from nltk.stem import WordNetLemmatizer as _WNL
        _stop_words = set(_sw.words("english"))
        _tokens = [t for t in _wt(_cleaned) if t not in _stop_words and len(t) > 1]
        _lem = _WNL()
        _lemmatized = [_lem.lemmatize(t) for t in _tokens]
        _pipeline_ok = True
    except Exception as _e:
        _tokens = [w for w in _cleaned.split() if len(w) > 1]
        _lemmatized = _tokens
        _pipeline_ok = False
        _pipeline_err = str(_e)

    _sia = load_vader()
    _scores = _sia.polarity_scores(sample_raw)
    _vader_label = "POSITIVE" if _scores['compound'] >= 0.05 else "NEGATIVE" if _scores['compound'] <= -0.05 else "NEUTRAL"

    if not _pipeline_ok:
        st.warning(f"NLTK pipeline error (using basic fallback): {_pipeline_err}")

    with st.expander("① Raw Text (Original Input)"):
        st.code(sample_raw, language="text")
    with st.expander("② After Lowercasing & Punctuation Removal"):
        st.code(_cleaned, language="text")
    with st.expander("③ After Tokenization & Stopword Removal"):
        st.code(" | ".join(_tokens), language="text")
    with st.expander("④ After WordNet Lemmatization"):
        st.code(" | ".join(_lemmatized), language="text")
    with st.expander("⑤ VADER Compound Score"):
        col_a, col_b = st.columns(2)
        col_a.json(_scores)
        col_b.markdown(f"**Compound**: `{_scores['compound']:.4f}`  \n**Label**: `{_vader_label}`")

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown(f"<h3 style='color:{WHITE};'> Technology Stack</h3>", unsafe_allow_html=True)
        tech = {
            "Streamlit": "Interactive dashboard & UI",
            "Pandas / NumPy": "Data wrangling and metrics",
            "NLTK": "VADER, tokenization, stopwords, lemmatization",
            "HuggingFace Transformers": "distilbert-base-uncased-finetuned-sst-2-english",
            "Plotly Express": "All interactive charts",
            "Scikit-Learn": "Accuracy, precision, recall, confusion matrix",
        }
        for tool, desc in tech.items():
            st.markdown(f"""
            <div style='background:{CARD};border-left:3px solid {TEAL};border-radius:0 6px 6px 0;padding:8px 12px;margin-bottom:8px;'>
              <span style='color:{CYAN};font-weight:600;'>{tool}</span>
              <span style='color:{GRAY};font-size:0.82rem;'> — {desc}</span>
            </div>""", unsafe_allow_html=True)

    with col_r:
        st.markdown(f"<h3 style='color:{WHITE};'> Project Checklist</h3>", unsafe_allow_html=True)
        items = [
            ("Week 1", "Raw Jindal dataset ingestion & column mapping"),
            ("Week 1", "Text preprocessing (lowercasing, punctuation removal)"),
            ("Week 1", "NLTK tokenization & WordNet lemmatization"),
            ("Week 1", "Polarity word clouds (Negative vs Positive)"),
            ("Week 2", "VADER analysis on raw uncleaned feedback text"),
            ("Week 2", "Rating-based ground truth categorisation (1–2 Neg, 3 Neu, 4–5 Pos)"),
            ("Week 2", "DistilBERT transformer on 2,500-row sample with timing"),
            ("Week 2", "Accuracy, Classification Report, Confusion Matrix"),
        ]
        for week, task in items:
            color = CYAN if week == "Week 2" else TEAL
            st.markdown(f"""
            <div style='display:flex;align-items:flex-start;gap:8px;margin-bottom:6px;'>
              <span style='color:{CYAN};font-size:1rem;'>•</span>
              <span style='color:{GRAY};font-size:0.85rem;'><b style='color:{color};'>[{week}]</b> {task}</span>
            </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE 2: WEEK 1 EXPLORATION
# ─────────────────────────────────────────────
elif navigation == "Week 1 · Exploration":
    st.markdown(f"<h1 style='color:{WHITE};'> Week 1 · Preprocessing & Exploration</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{GRAY};'>10,000 Jindal Steel B2B feedback records · Post-cleaning analysis</p>", unsafe_allow_html=True)
    st.markdown("<hr style='border-color:#1A2A2A;'>", unsafe_allow_html=True)

    if df_cleaned is None:
        st.error("Dataset not found at data/processed/reviews_cleaned.csv")
        st.stop()

    # Dynamic Filter Sidebar
    with st.sidebar:
        st.markdown(f"<b style='color:{CYAN};'> Page Filters</b>", unsafe_allow_html=True)
        region_filter = st.multiselect("Customer Region:", df_cleaned["customer_region"].unique(), default=list(df_cleaned["customer_region"].unique()))
        seg_filter    = st.multiselect("Customer Type:",   df_cleaned["customer_type"].unique(),   default=list(df_cleaned["customer_type"].unique()))

    df_f = df_cleaned[df_cleaned["customer_region"].isin(region_filter) & df_cleaned["customer_type"].isin(seg_filter)]

    if df_f.empty:
        st.warning("No data matches the selected filters.")
        st.stop()

    # Token stats KPIs
    if "tokens" in df_f.columns:
        raw_words   = int(df_f["feedback_text"].dropna().apply(lambda x: len(str(x).split())).sum())
        token_words = int(df_f["tokens"].dropna().apply(lambda x: len(str(x).split())).sum())
        reduction   = round((1 - token_words / raw_words) * 100, 1) if raw_words > 0 else 0
    else:
        raw_words = token_words = reduction = "N/A"

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi(f"{len(df_f):,}", "Filtered Records", f"of {len(df_cleaned):,} total"), unsafe_allow_html=True)
    c2.markdown(kpi(f"{raw_words:,}", "Total Raw Words", "Before preprocessing"), unsafe_allow_html=True)
    c3.markdown(kpi(f"{token_words:,}", "Cleaned Tokens", "After stopword removal"), unsafe_allow_html=True)
    c4.markdown(kpi(f"{reduction}%", "Token Reduction", "Preprocessing efficiency"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 1: Product & Customer charts
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(teal_bar(df_f["product_category"], "Product Category Distribution"), use_container_width=True)
    with col2:
        cust_counts = df_f["customer_type"].value_counts().reset_index()
        cust_counts.columns = ["label", "count"]
        fig_pie = px.pie(cust_counts, names="label", values="count",
                         color_discrete_sequence=[CYAN, TEAL, "#00A896", "#028090", "#05668D", "#02C39A", "#F0F3BD"],
                         hole=0.45, template=PLOTLY_TEMPLATE, title="Customer Segment Mix")
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color=WHITE, margin=dict(t=40, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)

    # Row 2: Region & Severity
    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(teal_bar(df_f["customer_region"], "Customer Region Distribution", orientation="h"), use_container_width=True)
    with col4:
        rating_counts = df_f["severity_rating"].value_counts().reset_index()
        rating_counts.columns = ["rating", "count"]
        rating_counts = rating_counts.sort_values("rating")
        colors_rating = ["#EF4444", "#F97316", "#FBBF24", "#4ADE80", CYAN]
        fig_rate = px.bar(rating_counts, x="rating", y="count",
                          color="rating", color_discrete_sequence=colors_rating,
                          template=PLOTLY_TEMPLATE, title="Severity Rating Distribution",
                          labels={"rating": "Severity Rating (1=Worst, 5=Best)", "count": "Count"})
        fig_rate.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color=WHITE,
                               margin=dict(t=40, b=20), showlegend=False)
        st.plotly_chart(fig_rate, use_container_width=True)

    # Row 3: Feedback Topic Distribution
    if "feedback_topic" in df_f.columns:
        st.plotly_chart(teal_bar(df_f["feedback_topic"], "Feedback Topic Distribution", orientation="h", top_n=12), use_container_width=True)

    # N-gram Analysis
    # st.markdown("<hr style='border-color:#1A2A2A;'>", unsafe_allow_html=True)
    # st.markdown(f"<h3 style='color:{WHITE};'>📌 Top Bigram Analysis (N-Gram)</h3>", unsafe_allow_html=True)
    # st.markdown(f"<p style='color:{GRAY};'>Bigrams (two-word phrases) reveal richer industrial signals than single tokens.</p>", unsafe_allow_html=True)

    # def get_bigrams(texts, n=15):
    #     all_tokens = []
    #     for text in texts.dropna():
    #         words = re.sub(r'[^\w\s]', '', str(text).lower()).split()
    #         all_tokens.extend(zip(words[:-1], words[1:]))
    #     counts = Counter(all_tokens).most_common(n)
    #     return pd.DataFrame(counts, columns=["bigram", "count"]).assign(bigram=lambda d: d["bigram"].apply(lambda t: " ".join(t)))

    # neg_texts = df_f[df_f["severity_rating"] <= 2]["feedback_text"] if "severity_rating" in df_f.columns and "feedback_text" in df_f.columns else pd.Series()
    # pos_texts = df_f[df_f["severity_rating"] >= 4]["feedback_text"] if "severity_rating" in df_f.columns and "feedback_text" in df_f.columns else pd.Series()

    # bg_col1, bg_col2 = st.columns(2)
    # with bg_col1:
    #     if not neg_texts.empty:
    #         bg_neg = get_bigrams(neg_texts)
    #         fig_bg_neg = px.bar(bg_neg, x="count", y="bigram", orientation="h",
    #                             template=PLOTLY_TEMPLATE, title="Top Bigrams — Negative (1–2 ★)",
    #                             color="count", color_continuous_scale=[[0, CARD], [1, "#EF4444"]])
    #         fig_bg_neg.update_layout(yaxis=dict(autorange="reversed"),
    #                                  paper_bgcolor="rgba(0,0,0,0)", font_color=WHITE,
    #                                  margin=dict(t=40, b=10), coloraxis_showscale=False)
    #         st.plotly_chart(fig_bg_neg, use_container_width=True)
    # with bg_col2:
    #     if not pos_texts.empty:
    #         bg_pos = get_bigrams(pos_texts)
    #         fig_bg_pos = px.bar(bg_pos, x="count", y="bigram", orientation="h",
    #                             template=PLOTLY_TEMPLATE, title="Top Bigrams — Positive (4–5 ★)",
    #                             color="count", color_continuous_scale=[[0, CARD], [1, CYAN]])
    #         fig_bg_pos.update_layout(yaxis=dict(autorange="reversed"),
    #                                  paper_bgcolor="rgba(0,0,0,0)", font_color=WHITE,
    #                                  margin=dict(t=40, b=10), coloraxis_showscale=False)
    #         st.plotly_chart(fig_bg_pos, use_container_width=True)

    # Word Clouds
    st.markdown("<hr style='border-color:#1A2A2A;'>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='color:{WHITE};'> Polarity Word Clouds</h3>", unsafe_allow_html=True)
    wc_c1, wc_c2 = st.columns(2)
    with wc_c1:
        st.markdown(f"<div style='text-align:center;color:#EF4444;font-weight:600;margin-bottom:8px;'>Negative Feedback (1–2 Stars)</div>", unsafe_allow_html=True)
        if os.path.exists(wc_neg_path):
            st.image(Image.open(wc_neg_path), use_container_width=True)
        else:
            st.warning("Negative word cloud not found.")
    with wc_c2:
        st.markdown(f"<div style='text-align:center;color:{CYAN};font-weight:600;margin-bottom:8px;'>Positive Feedback (4–5 Stars)</div>", unsafe_allow_html=True)
        if os.path.exists(wc_pos_path):
            st.image(Image.open(wc_pos_path), use_container_width=True)
        else:
            st.warning("Positive word cloud not found.")

# ─────────────────────────────────────────────
# PAGE 3: WEEK 2 MODEL ANALYSIS
# ─────────────────────────────────────────────
elif navigation == "Week 2 · Model Analysis":
    st.markdown(f"<h1 style='color:{WHITE};'> Week 2 · Sentiment Model Benchmarks</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{GRAY};'>VADER (rule-based) vs.DistilBERT (transformer) evaluated on 2,500 Jindal Steel B2B reviews.</p>", unsafe_allow_html=True)
    st.markdown("<hr style='border-color:#1A2A2A;'>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs([" Benchmarks & Results", " Model Insights"])

    with tab1:
        # KPI row
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(kpi("78.48%", "DistilBERT Overall Acc", "VADER: 68.35%"), unsafe_allow_html=True)
        c2.markdown(kpi("96.60%", "DistilBERT Binary Acc", "Excl. neutral reviews"), unsafe_allow_html=True)
        c3.markdown(kpi("85.38%", "VADER Binary Acc", "Excl. neutral reviews"), unsafe_allow_html=True)
        c4.markdown(kpi("73.28%", "Model Agreement", "VADER–DistilBERT"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Performance charts
        chart_c1, chart_c2 = st.columns(2)
        with chart_c1:
            st.markdown(f"<h4 style='color:{WHITE};'>Model Comparison Chart</h4>", unsafe_allow_html=True)
            if os.path.exists(comp_chart_path):
                st.image(Image.open(comp_chart_path), use_container_width=True)
            else:
                st.warning("Comparison chart not found in outputs/figures/")
        with chart_c2:
            st.markdown(f"<h4 style='color:{WHITE};'>DistilBERT Confusion Matrix</h4>", unsafe_allow_html=True)
            if os.path.exists(conf_matrix_path):
                st.image(Image.open(conf_matrix_path), use_container_width=True)
            else:
                st.warning("Confusion matrix not found in outputs/figures/")

        st.markdown("<hr style='border-color:#1A2A2A;'>", unsafe_allow_html=True)

        # Accuracy formulas
        st.markdown(f"<h4 style='color:{WHITE};'> How Accuracy is Calculated</h4>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='background:{CARD};border:1px solid {BORDER};border-radius:8px;padding:18px;margin-bottom:16px;'>
          <p style='color:{GRAY};margin:0 0 10px;'>Standard formula:</p>
          <code style='color:{CYAN};font-size:0.95rem;'>Accuracy = Correct Predictions / Total Sample Size</code>
          <div style='display:flex;gap:16px;margin-top:16px;flex-wrap:wrap;'>
            <div style='flex:1;min-width:200px;background:#070A0D;border-radius:6px;padding:12px;'>
              <div style='color:{GRAY};font-size:0.75rem;'>OVERALL (3-Class)</div>
              <div style='color:{CYAN};font-size:1.2rem;font-weight:600;'>1,962 / 2,500 = 78.48%</div>
              <div style='color:{GRAY};font-size:0.78rem;margin-top:4px;'>Includes 469 neutral reviews thatDistilBERT cannot classify (SST-2 binary model → always 0 neutral predictions)</div>
            </div>
            <div style='flex:1;min-width:200px;background:#070A0D;border-radius:6px;padding:12px;'>
              <div style='color:{GRAY};font-size:0.75rem;'>BINARY (2-Class, Fair Eval)</div>
              <div style='color:{CYAN};font-size:1.2rem;font-weight:600;'>1,962 / 2,031 = 96.60%</div>
              <div style='color:{GRAY};font-size:0.78rem;margin-top:4px;'>Neutral reviews removed. Only pos/neg evaluations whereDistilBERT was designed to operate.</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        #DistilBERT confidence distribution
        # if df_labeled is not None and "distilbert_confidence" in df_labeled.columns:
        #     st.markdown(f"<h4 style='color:{WHITE};'>DistilBERT Confidence Distribution</h4>", unsafe_allow_html=True)
        #     fig_conf = px.histogram(df_labeled, x="distilbert_confidence",
        #                             nbins=40, color_discrete_sequence=[CYAN],
        #                             template=PLOTLY_TEMPLATE,
        #                             labels={"distilbert_confidence": "Confidence Score", "count": "Count"})
        #     fig_conf.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color=WHITE,
        #                            margin=dict(t=20, b=20))
        #     st.plotly_chart(fig_conf, use_container_width=True)

        # Model error explorer
        if df_labeled is not None:
            st.markdown("<hr style='border-color:#1A2A2A;'>", unsafe_allow_html=True)
            st.markdown(f"<h4 style='color:{WHITE};'> Model Misclassification Explorer</h4>", unsafe_allow_html=True)
            st.markdown(f"<p style='color:{GRAY};font-size:0.85rem;'>Filter for cases where a model got it wrong to understand failure patterns.</p>", unsafe_allow_html=True)

            err_c1, err_c2 = st.columns(2)
            with err_c1:
                err_model = st.selectbox("Model:", ["DistilBERT", "VADER"])
            with err_c2:
                err_type = st.selectbox("Error Type:", ["False Positive (predicted positive, truth negative)",
                                                         "False Negative (predicted negative, truth positive)",
                                                         "Missed Neutral (truth neutral, predicted non-neutral)"])

            col_pred = "distilbert_label" if err_model == "DistilBERT" else "vader_label"
            if "False Positive" in err_type:
                err_df = df_labeled[(df_labeled[col_pred] == "positive") & (df_labeled["rating_based_truth"] == "negative")]
            elif "False Negative" in err_type:
                err_df = df_labeled[(df_labeled[col_pred] == "negative") & (df_labeled["rating_based_truth"] == "positive")]
            else:
                err_df = df_labeled[(df_labeled["rating_based_truth"] == "neutral") & (df_labeled[col_pred] != "neutral")]

            st.markdown(f"<p style='color:{CYAN};font-weight:600;'>{len(err_df)} matching misclassifications</p>", unsafe_allow_html=True)
            if not err_df.empty:
                st.dataframe(err_df[["review text", "rating", col_pred, "rating_based_truth"]].head(20),
                             use_container_width=True, hide_index=True)

    with tab2:
        st.markdown(f"<br>", unsafe_allow_html=True)
        st.markdown(insight_box(
            " SST-2 Binary Constraint (0% Neutral Accuracy)",
            "The <code>distilbert-base-uncased-finetuned-sst-2-english</code> model is pre-trained on the Stanford Sentiment Treebank — a <b>binary</b> movie-review dataset. It has no third output node for 'neutral'. All 469 neutral B2B reviews (3-star ratings) in our sample were force-classified as either positive (143) or negative (326), guaranteeing 0% neutral recall. This is <b>not a model failure</b> — it is a pre-training dataset mismatch."
        ), unsafe_allow_html=True)

        st.markdown(insight_box(
            " Binary Accuracy: 96.60%",
            "When the 469 neutral reviews are excluded from the evaluation — making it a fair binary test —DistilBERT correctly classifies <b>1,962 out of 2,031 polar reviews (96.60%)</b>. By comparison, VADER achieves 85.38% on the same binary subset. This proves the transformer architecture's deep semantic understanding is exceptional for polar B2B sentiment."
        ), unsafe_allow_html=True)

        st.markdown(insight_box(
            " VADER Corporate Politeness Skew (73.15% Positive Predicted)",
            "VADER is a lexical scorer — it evaluates word by word. B2B feedback contains polite corporate phrasing like <i>\"We appreciate the prompt engineer visit, but the rebar tensile strength is inconsistent with IS 1786...\"</i>. VADER scores 'appreciate' as strongly positive and misses the critical conjunction. Result: VADER predicts <b>73.15% positive</b> overall vs. the true <b>55.22%</b> distribution.DistilBERT reads context, negation, and sentence structure."
        ), unsafe_allow_html=True)

        st.markdown(insight_box(
            "🚀 Speed vs. Accuracy Trade-off",
            "VADER processes all 2,500 reviews in <b>&lt;1 second</b> (rule lookup).DistilBERT takes <b>~28 seconds on CPU</b>. VADER is ideal for real-time streams;DistilBERT for overnight batch jobs."
        ), unsafe_allow_html=True)

        # st.markdown(insight_box(
        #     "🔧 Recommendation for Week 4",
        #     "Fine-tune aDistilBERT head directly on the Jindal Steel 3-class labelled dataset (ratings 1–2 Negative, 3 Neutral, 4–5 Positive). Expected outcomes: (1) Neutral class recognition will emerge; (2) Domain vocabulary like 'tensile strength', 'IS 1786', 'rebar', and 'dispatch deviation' will be embeddings-aware; (3) Overall 3-class accuracy should rise from 78.48% towards ~92–95%."
        # ), unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE 4: LIVE ANALYSER
# ─────────────────────────────────────────────
elif navigation == "Live Analyser":
    st.markdown(f"<h1 style='color:{WHITE};'> Live Sentiment Analyser</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{GRAY};'>Type or paste a B2B feedback message and get instant analysis from both models.</p>", unsafe_allow_html=True)
    st.markdown("<hr style='border-color:#1A2A2A;'>", unsafe_allow_html=True)

    st.markdown(f"<p style='color:{GRAY};font-size:0.82rem;'> <i>DistilBERT loads on first use — may take a few seconds on CPU.</i></p>", unsafe_allow_html=True)

    @st.cache_resource
    def load_distilbert():
        from transformers import pipeline as hf_pipeline
        return hf_pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )

    user_input = st.text_area(
        "Enter feedback text:",
        placeholder="e.g. We appreciate the prompt delivery, but the HR plates from Angul showed surface pitting that failed our IS 2062 acceptance criteria.",
        height=130,
        key="live_input"
    )

    analyse_btn = st.button("Analyse", type="primary")

    if analyse_btn and user_input.strip():
        with st.spinner("Running models..."):
            sia = load_vader()
            vader_scores = sia.polarity_scores(user_input)
            compound = vader_scores["compound"]
            if compound >= 0.05:
                vader_label = "positive"
            elif compound <= -0.05:
                vader_label = "negative"
            else:
                vader_label = "neutral"

            classifier = load_distilbert()
            db_result  = classifier(user_input[:512])[0]
            db_label   = db_result["label"].lower()
            db_conf    = db_result["score"]

        st.markdown("<br>", unsafe_allow_html=True)

        # Results row
        res_c1, res_c2, res_c3 = st.columns(3)

        with res_c1:
            vc = sentiment_color(vader_label)
            st.markdown(f"""
            <div class="result-card">
              <div class="result-model">NLTK VADER</div>
              <div class="result-label {vc}">{vader_label.upper()}</div>
              <div class="result-conf">Compound score: <b>{compound:+.4f}</b></div>
              <div style='margin-top:10px;font-size:0.78rem;color:{GRAY};'>
                Pos: {vader_scores['pos']:.3f} &nbsp;|&nbsp; Neu: {vader_scores['neu']:.3f} &nbsp;|&nbsp; Neg: {vader_scores['neg']:.3f}
              </div>
            </div>""", unsafe_allow_html=True)

        with res_c2:
            dc = sentiment_color(db_label)
            st.markdown(f"""
            <div class="result-card">
              <div class="result-model">DistilBERT</div>
              <div class="result-label {dc}">{db_label.upper()}</div>
              <div class="result-conf">Confidence: <b>{db_conf*100:.2f}%</b></div>
              <div style='margin-top:10px;font-size:0.78rem;color:{GRAY};'>
                SST-2 binary model (no neutral output)
              </div>
            </div>""", unsafe_allow_html=True)

        with res_c3:
            agree = vader_label == db_label
            badge_class = "agree-badge" if agree else "disagree-badge"
            badge_text  = "AGREE" if agree else "DISAGREE"
            st.markdown(f"""
            <div class="result-card">
              <div class="result-model">Model Consensus</div>
              <div style='margin-top:12px;'>
                <span class="{badge_class}">{badge_text}</span>
              </div>
              <div style='margin-top:10px;font-size:0.78rem;color:{GRAY};'>
                {'Both models agree' if agree else 'Predictions differ — review context'}
              </div>
            </div>""", unsafe_allow_html=True)

        # # VADER breakdown bar chart
        # st.markdown("<br>", unsafe_allow_html=True)
        # st.markdown(f"<h4 style='color:{WHITE};'>VADER Score Breakdown</h4>", unsafe_allow_html=True)
        # vader_df = pd.DataFrame({
        #     "Category": ["Positive", "Neutral", "Negative"],
        #     "Score": [vader_scores["pos"], vader_scores["neu"], vader_scores["neg"]],
        #     "Color": [CYAN, "#FBBF24", "#EF4444"]
        # })
        # fig_vader = go.Figure(go.Bar(
        #     x=vader_df["Category"], y=vader_df["Score"],
        #     marker_color=vader_df["Color"],
        #     text=[f"{v:.3f}" for v in vader_df["Score"]],
        #     textposition="outside"
        # ))
        # fig_vader.update_layout(template=PLOTLY_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)",
        #                         font_color=WHITE, margin=dict(t=10, b=10),
        #                         yaxis=dict(range=[0, 1]))
        # st.plotly_chart(fig_vader, use_container_width=True)

    elif analyse_btn:
        st.warning("Please enter some feedback text before clicking Analyse.")

# ─────────────────────────────────────────────
# PAGE 5: DATA EXPLORER
# ─────────────────────────────────────────────
elif navigation == "Data Explorer":
    st.markdown(f"<h1 style='color:{WHITE};'> Interactive Data Explorer</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{GRAY};'>Filter, search, and download the 2,500 model-evaluated feedback records.</p>", unsafe_allow_html=True)
    st.markdown("<hr style='border-color:#1A2A2A;'>", unsafe_allow_html=True)

    if df_labeled is None:
        st.error("Labeled dataset not found at outputs/labeled_dataset.csv")
        st.stop()

    # Filters
    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        rating_ch = st.multiselect("Severity Rating:", sorted(df_labeled["rating"].unique()), default=sorted(df_labeled["rating"].unique()))
    with fc2:
        vader_ch = st.multiselect("VADER Label:", sorted(df_labeled["vader_label"].unique()), default=sorted(df_labeled["vader_label"].unique()))
    with fc3:
        db_ch = st.multiselect("DistilBERT Label:", sorted(df_labeled["distilbert_label"].unique()), default=sorted(df_labeled["distilbert_label"].unique()))
    with fc4:
        agree_only = st.checkbox("Show only model disagreements")
        kw = st.text_input("Keyword Search:")

    filt = df_labeled[
        (df_labeled["rating"].isin(rating_ch)) &
        (df_labeled["vader_label"].isin(vader_ch)) &
        (df_labeled["distilbert_label"].isin(db_ch))
    ]
    if kw:
        filt = filt[filt["review text"].str.contains(kw, case=False, na=False)]
    if agree_only:
        filt = filt[filt["vader_label"] != filt["distilbert_label"]]

    st.markdown(f"<p style='color:{CYAN};font-weight:600;margin-bottom:12px;'>Showing {len(filt):,} records</p>", unsafe_allow_html=True)

    # Random review sandbox
    if st.button("Random Review"):
        if not filt.empty:
            row = filt.sample(1).iloc[0]
            agree_str = " Both models agree" if row["vader_label"] == row["distilbert_label"] else " Models disagree"
            vc = sentiment_color(row["vader_label"])
            dc = sentiment_color(row["distilbert_label"])
            st.markdown(f"""
            <div style='background:{CARD};border:1px solid {BORDER};border-radius:10px;padding:20px;margin-bottom:16px;'>
              <div style='color:{GRAY};font-size:0.8rem;margin-bottom:10px;'> {'★'*int(row['rating'])}{'☆'*(5-int(row['rating']))} · Ground Truth: <b style='color:{CYAN};'>{row['rating_based_truth'].upper()}</b></div>
              <div style='color:{WHITE};font-size:0.9rem;line-height:1.7;margin-bottom:14px;'>{row['review text']}</div>
              <div style='display:flex;gap:12px;flex-wrap:wrap;'>
                <span style='background:#0E1418;border:1px solid #1A2A2A;border-radius:6px;padding:6px 12px;font-size:0.82rem;'>VADER: <b class="{vc}" style="color:{'#00E5CC' if row['vader_label']=='positive' else '#EF4444' if row['vader_label']=='negative' else '#FBBF24'};">{row['vader_label'].upper()}</b></span>
                <span style='background:#0E1418;border:1px solid #1A2A2A;border-radius:6px;padding:6px 12px;font-size:0.82rem;'>DistilBERT: <b style="color:{'#00E5CC' if row['distilbert_label']=='positive' else '#EF4444' if row['distilbert_label']=='negative' else '#FBBF24'};">{row['distilbert_label'].upper()}</b> ({row['distilbert_confidence']*100:.1f}%)</span>
                <span style='background:#0E1418;border:1px solid #1A2A2A;border-radius:6px;padding:6px 12px;font-size:0.82rem;color:{CYAN};'>{agree_str}</span>
              </div>
            </div>""", unsafe_allow_html=True)

    # Main table
    st.dataframe(
        filt,
        column_config={
            "review text":             st.column_config.TextColumn("Feedback Text", width="large"),
            "rating":                  st.column_config.NumberColumn("Rating", width="small"),
            "vader_label":             st.column_config.TextColumn("VADER", width="small"),
            "distilbert_label":        st.column_config.TextColumn("DistilBERT", width="small"),
            "distilbert_confidence":   st.column_config.NumberColumn("Confidence", format="%.3f", width="small"),
            "rating_based_truth":      st.column_config.TextColumn("Ground Truth", width="small"),
        },
        use_container_width=True,
        hide_index=True
    )

    # Downloadbutton
    csv_data = filt.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f" Download{len(filt):,} filtered records as CSV",
        data=csv_data,
        file_name="jindal_feedback_filtered.csv",
        mime="text/csv"
    )
