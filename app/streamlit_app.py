import os
import sys
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from collections import Counter
import plotly.express as px

# Ensure the project root is on the path so `from utils import ...` works
# regardless of whether the app is run as `streamlit run app/streamlit_app.py`
# (from project root) or from inside the app/ directory.
_this_dir  = os.path.dirname(os.path.abspath(__file__))
_proj_root = os.path.dirname(_this_dir)
for _p in [_proj_root, _this_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from utils import analyze_review          # when run from project root
except ModuleNotFoundError:
    from app.utils import analyze_review      # fallback

# ─────────────────────────────────────────────
# VISUAL SYSTEM & COLOR PALETTE
# ─────────────────────────────────────────────
CYAN            = "#00E5CC"
TEAL            = "#0E6243"
BG              = "#070A0D"
CARD            = "#0E1418"
BORDER          = "#1A2A2A"
GRAY            = "#7E8D8E"
WHITE           = "#F0F4F8"
PLOTLY_TEMPLATE = "plotly_dark"

# ─────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="B2B Quality Feedback Analyzer",
    page_icon="⚙️",
    layout="wide"
)

# ─────────────────────────────────────────────
# GLOBAL CSS  (matches main.py theme)
# ─────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    background-color: {BG};
}}

/* Result cards */
.result-card {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 18px 14px;
    text-align: center;
    margin-top: 8px;
    height: 165px;
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
    margin-bottom: 6px;
}}
.result-label {{
    font-size: 1.5rem;
    font-weight: 700;
    line-height: 1.2;
}}
.result-conf {{
    font-size: 0.85rem;
    color: {WHITE};
    margin-top: 8px;
}}
.pos-label {{ color: {CYAN}; }}
.neg-label {{ color: #EF4444; }}
.neu-label {{ color: #FBBF24; }}

.agree-badge {{
    background: #0E6243;
    color: #00E5CC;
    padding: 4px 12px;
    border-radius: 4px;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 0.06em;
}}
.disagree-badge {{
    background: #4C1D95;
    color: #C084FC;
    padding: 4px 12px;
    border-radius: 4px;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 0.06em;
}}

/* Sample buttons — grey, not green */
.sample-btn button {{
    background-color: #121820 !important;
    color: {GRAY} !important;
    border: 1px solid {BORDER} !important;
    font-size: 0.82rem !important;
}}
.sample-btn button:hover {{
    border-color: {CYAN} !important;
    color: {CYAN} !important;
}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown(
    f"<h1 style='color:{CYAN};font-weight:700;margin-bottom:2px;'>⚙️ B2B Quality Feedback Analyzer</h1>",
    unsafe_allow_html=True
)
st.markdown(
    f"<p style='color:{GRAY};font-size:1.1rem;margin-top:0;'>"
    "Instant NLP sentiment analysis, topic modeling, and entity extraction on B2B customer feedback.</p>",
    unsafe_allow_html=True
)
st.markdown("<hr style='border-color:#1A2A2A;margin-bottom:24px;'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍 Live Analyzer", "📊 Aggregate Insights"])

# ═════════════════════════════════════════════
# TAB 1 — LIVE ANALYZER
# ═════════════════════════════════════════════
with tab1:
    st.markdown(
        f"<p style='color:{WHITE};font-size:0.95rem;margin-bottom:12px;'>"
        "Paste any B2B feedback review below. The app runs VADER, DistilBERT, LDA, and spaCy NER simultaneously.</p>",
        unsafe_allow_html=True
    )

    SAMPLE_REVIEWS = [
        (
            "Sample 1: Quality Reject",
            "The HR plates from Angul showed severe surface pitting and dimensional inconsistencies "
            "that failed our IS 2062 acceptance criteria. Tensile strength was below the specified "
            "grade, forcing a full consignment rejection."
        ),
        (
            "Sample 2: Delivery & Logistics",
            "The TMT rebar consignment was delayed by 12 days beyond the committed date. "
            "The packaging was completely degraded on arrival, with strapping broken and coils "
            "exposed to moisture during freight."
        ),
        (
            "Sample 3: Positive Pricing",
            "We appreciate the prompt delivery and competitive pricing on the Beams and Columns "
            "sections from Raigarh. The accounts team processed our payment and issued credit notes "
            "efficiently. Looking forward to expanding this partnership."
        ),
    ]

    def _load_sample(idx: int) -> None:
        st.session_state["analyzer_input"] = SAMPLE_REVIEWS[idx][1]

    inp_col, btn_col = st.columns([3, 1])

    with btn_col:
        st.markdown(
            f"<div style='color:{GRAY};font-size:0.75rem;margin-bottom:6px;"
            "text-transform:uppercase;letter-spacing:0.06em;'>Try a sample</div>",
            unsafe_allow_html=True
        )
        st.markdown('<div class="sample-btn">', unsafe_allow_html=True)
        for i, (label, _) in enumerate(SAMPLE_REVIEWS):
            st.button(label, key=f"samp{i}", on_click=_load_sample, args=(i,), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with inp_col:
        user_input = st.text_area(
            "Enter feedback text:",
            placeholder="Type or paste B2B feedback text here...",
            height=148,
            key="analyzer_input"
        )

    analyse_btn = st.button("Analyse Feedback", type="primary")

    # ─── Run analysis ───────────────────────────────────────────────────────
    if analyse_btn and user_input.strip():
        with st.spinner("Running models — this may take a few seconds on first load…"):
            vader_label, db_label, db_conf, predicted_topic, entities, probs_dict = analyze_review(user_input)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Row 1: Sentiment Cards ─────────────────────────────────────────
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:12px;'>"
            f"<div style='width:8px;height:8px;border-radius:50%;background:{CYAN};'></div>"
            f"<h3 style='color:{WHITE};margin:0;font-size:1.1rem;font-weight:600;'>Sentiment Analysis</h3></div>",
            unsafe_allow_html=True
        )

        def _sentiment_class(label: str) -> str:
            return {"positive": "pos-label", "negative": "neg-label"}.get(label, "neu-label")

        res_c1, res_c2, res_c3 = st.columns(3)

        with res_c1:
            vc = _sentiment_class(vader_label)
            st.markdown(f"""
            <div class="result-card">
              <div class="result-model">NLTK VADER</div>
              <div class="result-label {vc}">{vader_label.upper()}</div>
              <div style='margin-top:10px;font-size:0.78rem;color:{GRAY};'>Rule-based lexicon</div>
            </div>""", unsafe_allow_html=True)

        with res_c2:
            dc = _sentiment_class(db_label)
            st.markdown(f"""
            <div class="result-card">
              <div class="result-model">DistilBERT SST-2</div>
              <div class="result-label {dc}">{db_label.upper()}</div>
              <div class="result-conf">Confidence: <b>{db_conf*100:.2f}%</b></div>
            </div>""", unsafe_allow_html=True)

        with res_c3:
            agree      = (vader_label == db_label)
            badge_cls  = "agree-badge" if agree else "disagree-badge"
            badge_txt  = "AGREE" if agree else "DISAGREE"
            note       = "Both models agree" if agree else "Labels differ — review context"
            st.markdown(f"""
            <div class="result-card">
              <div class="result-model">Model Consensus</div>
              <div style='margin-top:12px;'><span class="{badge_cls}">{badge_txt}</span></div>
              <div style='margin-top:10px;font-size:0.78rem;color:{GRAY};'>{note}</div>
            </div>""", unsafe_allow_html=True)

        # ── Row 2: LDA Topic ───────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:12px;'>"
            f"<div style='width:8px;height:8px;border-radius:50%;background:{CYAN};'></div>"
            f"<h3 style='color:{WHITE};margin:0;font-size:1.1rem;font-weight:600;'>Topic Prediction · LDA</h3></div>",
            unsafe_allow_html=True
        )

        lda_c1, lda_c2 = st.columns([1, 2])
        dominant_prob  = probs_dict[predicted_topic]

        with lda_c1:
            dash_offset = int(201 * (1 - dominant_prob))
            st.markdown(f"""
            <div class="result-card" style="height:240px;padding:20px 12px;">
              <div style="text-align:center;">
                <div style="font-size:1.5rem;margin-bottom:2px;">📊</div>
                <div class="result-model">DOMINANT TOPIC</div>
                <div style="font-size:1.2rem;font-weight:700;color:{CYAN};line-height:1.3;margin-top:4px;">{predicted_topic.title()}</div>
              </div>
              <div style="display:flex;flex-direction:column;align-items:center;margin-top:14px;">
                <svg width="76" height="76" viewBox="0 0 80 80">
                  <circle cx="40" cy="40" r="32" stroke="#1A2A2A" stroke-width="6" fill="transparent"/>
                  <circle cx="40" cy="40" r="32" stroke="{CYAN}" stroke-width="6" fill="transparent"
                          stroke-dasharray="201" stroke-dashoffset="{dash_offset}"
                          stroke-linecap="round" transform="rotate(-90 40 40)"/>
                  <text x="50%" y="54%" text-anchor="middle" fill="#F0F4F8"
                        font-weight="700" font-size="13px" dy=".3em">{dominant_prob*100:.1f}%</text>
                </svg>
                <div style="font-size:0.68rem;color:{GRAY};text-transform:uppercase;letter-spacing:0.08em;margin-top:8px;">CONFIDENCE</div>
              </div>
            </div>""", unsafe_allow_html=True)

        with lda_c2:
            sorted_items  = sorted(probs_dict.items(), key=lambda x: x[1])
            topics_sorted = [t[0].title() for t in sorted_items]
            probs_sorted  = [t[1] * 100 for t in sorted_items]

            fig = go.Figure(go.Bar(
                x=probs_sorted, y=topics_sorted, orientation='h',
                marker=dict(color=probs_sorted,
                            colorscale=[[0.0, TEAL], [1.0, CYAN]],
                            line=dict(width=0)),
                text=[f"{p:.1f}%" for p in probs_sorted],
                textposition='outside',
                textfont=dict(color=WHITE, size=10)
            ))
            fig.update_layout(
                template=PLOTLY_TEMPLATE,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=10, b=10, l=10, r=45), height=240,
                xaxis=dict(
                    title=dict(text="Probability (%)", font=dict(color=GRAY, size=11)),
                    tickfont=dict(color=GRAY, size=10),
                    showgrid=True, gridcolor="#1A2A2A", zeroline=False,
                    range=[0, max(probs_sorted) * 1.25]
                ),
                yaxis=dict(title="", tickfont=dict(color=WHITE, size=11), showgrid=False),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── Row 3: Named Entities ──────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:12px;'>"
            f"<div style='width:8px;height:8px;border-radius:50%;background:#D946EF;'></div>"
            f"<h3 style='color:{WHITE};margin:0;font-size:1.1rem;font-weight:600;'>Extracted Named Entities · spaCy NER</h3></div>",
            unsafe_allow_html=True
        )

        if entities:
            tags_html = "".join(
                f"<span style='background:#121820;border:1px solid #1A2A2A;color:#F0F4F8;"
                f"border-radius:6px;padding:6px 12px;font-size:0.82rem;"
                f"margin-right:8px;margin-bottom:8px;display:inline-block;'>🏷️ {ent}</span>"
                for ent in entities
            )
            st.markdown(f"<div style='display:flex;flex-wrap:wrap;'>{tags_html}</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                f"<p style='color:{GRAY};font-size:0.85rem;font-style:italic;'>"
                "No named entities detected in this feedback review.</p>",
                unsafe_allow_html=True
            )

    elif analyse_btn:
        st.warning("Please enter some feedback text before clicking Analyse.")

# ═════════════════════════════════════════════
# TAB 2 — AGGREGATE INSIGHTS  (Person 2 placeholder)
# ═════════════════════════════════════════════
with tab2:

    st.title("📊 Aggregate Insights Dashboard")

    df = pd.read_csv("outputs/extracted_topics.csv")
    total_reviews = len(df)

positive = (df["distilbert_label"] == "POSITIVE").sum()
negative = (df["distilbert_label"] == "NEGATIVE").sum()

col1, col2, col3 = st.columns(3)

col1.metric("Total Reviews", total_reviews)
col2.metric("Positive Reviews", positive)
col3.metric("Negative Reviews", negative)
st.subheader("Sentiment Distribution")

sentiment_counts = df["distilbert_label"].value_counts()

fig = px.bar(
    x=sentiment_counts.index,
    y=sentiment_counts.values,
    labels={
        "x":"Sentiment",
        "y":"Number of Reviews"
    },
    title="Review Sentiment Distribution"
)

st.plotly_chart(fig, use_container_width=True)
st.subheader("Top Feedback Topics")

topic_counts = df["topic_label"].value_counts().head(10)

fig = px.bar(
    x=topic_counts.values,
    y=topic_counts.index,
    orientation="h",
    title="Top 10 Topics"
)

st.plotly_chart(fig, use_container_width=True)
st.sidebar.header("Filters")

topic = st.sidebar.selectbox(
    "Topic",
    ["All"] + sorted(df["topic_label"].dropna().unique().tolist())
)

sentiment = st.sidebar.selectbox(
    "Sentiment",
    ["All"] + sorted(df["distilbert_label"].dropna().unique().tolist())
)
filtered_df = df.copy()

if topic != "All":
    filtered_df = filtered_df[
        filtered_df["topic_label"] == topic
    ]

if sentiment != "All":
    filtered_df = filtered_df[
        filtered_df["distilbert_label"] == sentiment
    ]
    st.subheader("Filtered Reviews")

st.dataframe(
    filtered_df[
        [
            "review text",
            "topic_label",
            "distilbert_label",
            "entities"
        ]
    ],
    use_container_width=True
)
st.subheader("Most Common Named Entities")

entity_counter = Counter()

for entity in df["entities"].dropna():
    if entity.strip():
        entity_counter.update(
            [e.strip() for e in entity.split(",")]
        )

entity_df = pd.DataFrame(
    entity_counter.most_common(10),
    columns=["Entity","Frequency"]
)

st.dataframe(entity_df)
st.subheader("Topic Distribution")

fig = px.pie(
    df,
    names="topic_label",
    title="Feedback Topic Distribution"
)

st.plotly_chart(fig, use_container_width=True)
st.subheader("Overall Project Summary")

st.success(f"""
✔ Total Reviews Analysed : {total_reviews}

✔ Positive Reviews : {positive}

✔ Negative Reviews : {negative}

✔ Total Topics Identified : {df['topic_label'].nunique()}
""")