import os
import sys
import ast as _ast
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from collections import Counter

# ─── Path resolution ────────────────────────────────────────────────────────
_this_dir  = os.path.dirname(os.path.abspath(__file__))
_proj_root = os.path.dirname(_this_dir)
for _p in [_proj_root, _this_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from utils import analyze_review, load_extracted_topics
except ModuleNotFoundError:
    from app.utils import analyze_review, load_extracted_topics

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be the first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="B2B Quality Feedback Analyzer",
    page_icon="⚙️",
    layout="wide"
)

# ─────────────────────────────────────────────
# DESIGN SYSTEM
# ─────────────────────────────────────────────
CYAN            = "#00E5CC"
TEAL            = "#0E6243"
BG              = "#070A0D"
CARD            = "#0E1418"
BORDER          = "#1A2A2A"
GRAY            = "#7E8D8E"
WHITE           = "#F0F4F8"
RED             = "#EF4444"
YELLOW          = "#FBBF24"
PURPLE          = "#D946EF"
PLOTLY_TEMPLATE = "plotly_dark"
TEAL_SCALE      = [[0, TEAL], [1, CYAN]]
RED_SCALE       = [[0, "#7f1d1d"], [1, RED]]

# ─────────────────────────────────────────────
# SESSION STATE INITIALISATION
# (must happen before any widget with the same key)
# ─────────────────────────────────────────────
if "analyzer_input" not in st.session_state:
    st.session_state["analyzer_input"] = ""

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; background-color: {BG}; }}
.result-card {{
    background: {CARD}; border: 1px solid {BORDER}; border-radius: 10px;
    padding: 18px 14px; text-align: center; margin-top: 8px; height: 165px;
    display: flex; flex-direction: column; justify-content: center; align-items: center;
}}
.result-model {{ font-size:0.75rem; color:{GRAY}; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px; }}
.result-label {{ font-size:1.5rem; font-weight:700; line-height:1.2; }}
.result-conf  {{ font-size:0.85rem; color:{WHITE}; margin-top:8px; }}
.pos-label    {{ color:{CYAN}; }}
.neg-label    {{ color:#EF4444; }}
.neu-label    {{ color:#FBBF24; }}
.agree-badge    {{ background:#0E6243; color:#00E5CC; padding:4px 12px; border-radius:4px; font-weight:700; font-size:0.85rem; }}
.disagree-badge {{ background:#4C1D95; color:#C084FC; padding:4px 12px; border-radius:4px; font-weight:700; font-size:0.85rem; }}
.sample-btn button {{ background-color:#121820 !important; color:{GRAY} !important; border:1px solid {BORDER} !important; font-size:0.82rem !important; }}
.sample-btn button:hover {{ border-color:{CYAN} !important; color:{CYAN} !important; }}
div[data-testid="stRadio"] > div {{ flex-direction: row !important; gap: 12px; }}
div[data-testid="stRadio"] label {{
    background:{CARD}; border:1px solid {BORDER}; border-radius:8px;
    padding:8px 20px; cursor:pointer; font-size:0.9rem; color:{GRAY}; font-weight:500;
    transition: all 0.15s ease;
}}
div[data-testid="stRadio"] label:has(input:checked) {{
    border-color:{CYAN}; color:{CYAN}; background: rgba(0,229,204,0.06);
}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HELPER FUNCTIONS  (pure Python — no st.* calls inside cached helpers)
# ─────────────────────────────────────────────
def _kpi(val, label, sub="", accent=CYAN):
    return (
        f"<div style='background:{CARD};border:1px solid {BORDER};"
        f"border-top:3px solid {accent};border-radius:10px;padding:18px 14px;"
        f"text-align:center;min-height:130px;display:flex;flex-direction:column;"
        f"justify-content:center;align-items:center;'>"
        f"<div style='font-size:1.9rem;font-weight:700;color:{accent};line-height:1.2;'>{val}</div>"
        f"<div style='font-size:0.75rem;color:{GRAY};text-transform:uppercase;"
        f"letter-spacing:0.08em;margin-top:4px;'>{label}</div>"
        f"<div style='font-size:0.78rem;color:{GRAY};margin-top:6px;'>{sub}</div>"
        f"</div>"
    )

def _section(title, color=CYAN):
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:8px;margin:20px 0 12px;'>"
        f"<div style='width:8px;height:8px;border-radius:50%;background:{color};flex-shrink:0;'></div>"
        f"<h3 style='color:{WHITE};margin:0;font-size:1.1rem;font-weight:600;'>{title}</h3></div>",
        unsafe_allow_html=True
    )

def _top_words(subset_df, n=12, token_col="tokens"):
    words = []
    for tok in subset_df[token_col].dropna():
        try:
            lst = _ast.literal_eval(tok)
        except Exception:
            lst = str(tok).split()
        words.extend([w for w in lst if isinstance(w, str) and len(w) > 2])
    return Counter(words).most_common(n)

def _verbatim_feed(df_sub, key):
    """Render a filterable verbatim review feed."""
    sent_opt = st.selectbox(
        "Filter by sentiment",
        ["All", "Positive only", "Negative only"],
        key=key
    )
    if sent_opt == "Positive only":
        df_sub = df_sub[df_sub["distilbert_label"] == "positive"]
    elif sent_opt == "Negative only":
        df_sub = df_sub[df_sub["distilbert_label"] == "negative"]

    sample = df_sub[["review text", "rating", "distilbert_label"]].dropna().head(8)
    for _, row in sample.iterrows():
        color = CYAN if row["distilbert_label"] == "positive" else RED
        badge = "✅ POSITIVE" if row["distilbert_label"] == "positive" else "❌ NEGATIVE"
        st.markdown(
            f"<div style='background:{CARD};border:1px solid {BORDER};border-left:3px solid {color};"
            f"border-radius:8px;padding:12px 16px;margin-bottom:10px;'>"
            f"<div style='display:flex;justify-content:space-between;margin-bottom:6px;'>"
            f"<span style='font-size:0.75rem;color:{color};font-weight:600;'>{badge}</span>"
            f"<span style='font-size:0.75rem;color:{GRAY};'>⭐ {int(row['rating'])}/5</span></div>"
            f"<p style='color:{WHITE};font-size:0.85rem;line-height:1.6;margin:0;'>{row['review text']}</p>"
            f"</div>",
            unsafe_allow_html=True
        )

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown(
    f"<h1 style='color:{CYAN};font-weight:700;margin-bottom:2px;'>⚙️ B2B Quality Feedback Analyzer</h1>"
    f"<p style='color:{GRAY};font-size:1.05rem;margin-top:0;'>"
    "Instant NLP sentiment analysis, topic modeling, and entity extraction on B2B customer feedback.</p>",
    unsafe_allow_html=True
)
st.markdown("<hr style='border-color:#1A2A2A;margin-bottom:24px;'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Live Analyzer", "📊 Aggregate Insights", "🔬 Deep-Dive"])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE ANALYZER
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown(
        f"<p style='color:{WHITE};font-size:0.95rem;margin-bottom:12px;'>"
        "Paste any B2B feedback review below. The app runs VADER, DistilBERT, LDA, and spaCy NER simultaneously.</p>",
        unsafe_allow_html=True
    )

    SAMPLE_REVIEWS = [
        ("Sample 1: Quality Reject",
         "The HR plates from Angul showed severe surface pitting and dimensional inconsistencies "
         "that failed our IS 2062 acceptance criteria. Tensile strength was below the specified "
         "grade, forcing a full consignment rejection."),
        ("Sample 2: Delivery & Logistics",
         "The TMT rebar consignment was delayed by 12 days beyond the committed date. "
         "The packaging was completely degraded on arrival, with strapping broken and coils "
         "exposed to moisture during freight."),
        ("Sample 3: Positive Pricing",
         "We appreciate the prompt delivery and competitive pricing on the Beams and Columns "
         "sections from Raigarh. The accounts team processed our payment and issued credit notes "
         "efficiently. Looking forward to expanding this partnership."),
    ]

    def _load_sample(idx):
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
            st.button(label, key=f"samp{i}", on_click=_load_sample, args=(i,), width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)

    with inp_col:
        # key is the single source of truth; no 'value=' argument to avoid conflict
        user_input = st.text_area(
            "Enter feedback text:", placeholder="Type or paste B2B feedback text here...",
            height=148, key="analyzer_input"
        )

    analyse_btn = st.button("Analyse Feedback", type="primary")

    if analyse_btn and user_input.strip():
        with st.spinner("Running models — this may take a few seconds on first load…"):
            vader_label, db_label, db_conf, predicted_topic, entities, probs_dict = analyze_review(user_input)

        st.markdown("<br>", unsafe_allow_html=True)

        # Sentiment cards
        _section("Sentiment Analysis")
        def _sc(lbl): return {"positive": "pos-label", "negative": "neg-label"}.get(lbl, "neu-label")
        c1, c2, c3 = st.columns(3)
        with c1:
            vc = _sc(vader_label)
            st.markdown(f"""<div class="result-card"><div class="result-model">NLTK VADER</div>
              <div class="result-label {vc}">{vader_label.upper()}</div>
              <div style='margin-top:10px;font-size:0.78rem;color:{GRAY};'>Rule-based lexicon</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            dc = _sc(db_label)
            st.markdown(f"""<div class="result-card"><div class="result-model">DistilBERT SST-2</div>
              <div class="result-label {dc}">{db_label.upper()}</div>
              <div class="result-conf">Confidence: <b>{db_conf*100:.2f}%</b></div>
            </div>""", unsafe_allow_html=True)
        with c3:
            agree = (vader_label == db_label)
            bc    = "agree-badge" if agree else "disagree-badge"
            bt    = "AGREE" if agree else "DISAGREE"
            note  = "Both models agree" if agree else "Labels differ — review context"
            st.markdown(f"""<div class="result-card"><div class="result-model">Model Consensus</div>
              <div style='margin-top:12px;'><span class="{bc}">{bt}</span></div>
              <div style='margin-top:10px;font-size:0.78rem;color:{GRAY};'>{note}</div>
            </div>""", unsafe_allow_html=True)

        # LDA Topic
        st.markdown("<br>", unsafe_allow_html=True)
        _section("Topic Prediction · LDA")
        lda_c1, lda_c2 = st.columns([1, 2])
        dominant_prob  = probs_dict[predicted_topic]
        with lda_c1:
            dash_offset = int(201 * (1 - dominant_prob))
            st.markdown(f"""<div class="result-card" style="height:240px;padding:20px 12px;">
              <div style="text-align:center;">
                <div style="font-size:1.5rem;">📊</div>
                <div class="result-model">DOMINANT TOPIC</div>
                <div style="font-size:1.2rem;font-weight:700;color:{CYAN};margin-top:4px;">{predicted_topic.title()}</div>
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
            si = sorted(probs_dict.items(), key=lambda x: x[1])
            fig = go.Figure(go.Bar(
                x=[t[1]*100 for t in si], y=[t[0].title() for t in si], orientation='h',
                marker=dict(color=[t[1]*100 for t in si], colorscale=TEAL_SCALE, line=dict(width=0)),
                text=[f"{t[1]*100:.1f}%" for t in si], textposition='outside',
                textfont=dict(color=WHITE, size=10)
            ))
            fig.update_layout(
                template=PLOTLY_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=10, b=10, l=10, r=45), height=240,
                xaxis=dict(title=dict(text="Probability (%)", font=dict(color=GRAY, size=11)),
                           tickfont=dict(color=GRAY, size=10), showgrid=True, gridcolor="#1A2A2A",
                           zeroline=False, range=[0, max(t[1]*100 for t in si)*1.25]),
                yaxis=dict(title="", tickfont=dict(color=WHITE, size=11), showgrid=False),
                showlegend=False
            )
            st.plotly_chart(fig, width='stretch')

        # Named Entities
        st.markdown("<br>", unsafe_allow_html=True)
        _section("Extracted Named Entities · spaCy NER", color=PURPLE)
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
                "No named entities detected in this feedback review.</p>", unsafe_allow_html=True
            )

    elif analyse_btn:
        st.warning("Please enter some feedback text before clicking Analyse.")

# ═════════════════════════════════════════════════════════════════════════════
# SHARED DATA LOAD (used by Tab 2 & 3) — UI spinner wraps the pure cached fn
# ═════════════════════════════════════════════════════════════════════════════
with st.spinner("Loading dataset…"):
    df_all = load_extracted_topics()

# Sidebar filters
st.sidebar.markdown(
    f"<div style='color:{CYAN};font-size:0.78rem;text-transform:uppercase;"
    f"letter-spacing:0.1em;font-weight:600;margin-bottom:8px;margin-top:20px;'>"
    f"Aggregate Filters</div>", unsafe_allow_html=True
)
all_topics    = sorted(df_all["topic_label"].dropna().unique().tolist())
sel_topic     = st.sidebar.selectbox("Topic", ["All"] + all_topics, key="agg_topic")
sel_sentiment = st.sidebar.selectbox("Sentiment", ["All", "positive", "negative"], key="agg_sent")

df_f = df_all.copy()
if sel_topic != "All":
    df_f = df_f[df_f["topic_label"] == sel_topic]
if sel_sentiment != "All":
    df_f = df_f[df_f["distilbert_label"] == sel_sentiment]

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — AGGREGATE INSIGHTS
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    total   = len(df_f)
    pos_n   = (df_f["distilbert_label"] == "positive").sum()
    neg_n   = (df_f["distilbert_label"] == "negative").sum()
    avg_rat = df_f["rating"].mean() if total else 0

    st.markdown(
        f"<h2 style='color:{CYAN};font-weight:700;margin-bottom:4px;'>📊 Aggregate Insights</h2>"
        f"<p style='color:{GRAY};font-size:0.95rem;margin-top:0;margin-bottom:20px;'>"
        f"Dataset-wide analytics across {len(df_all):,} B2B feedback reviews — "
        f"filtered to <b style='color:{WHITE};'>{total:,}</b> records.</p>",
        unsafe_allow_html=True
    )
    st.markdown("<hr style='border-color:#1A2A2A;margin-bottom:24px;'>", unsafe_allow_html=True)

    # ── KPI row ────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(_kpi(f"{total:,}", "Reviews", "Filtered set"),          unsafe_allow_html=True)
    k2.markdown(_kpi(f"{pos_n:,}", "Positive", "DistilBERT", CYAN),    unsafe_allow_html=True)
    k3.markdown(_kpi(f"{neg_n:,}", "Negative", "DistilBERT", RED),     unsafe_allow_html=True)
    k4.markdown(_kpi(f"{avg_rat:.2f} ★", "Avg Rating", "1–5 scale", YELLOW), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Top Complaint Topics ───────────────────────────────────────────────
    _section("Top Complaint Topics (Negative-Weighted)", color=RED)
    neg_t = df_f[df_f["distilbert_label"] == "negative"]["topic_label"].value_counts().reset_index()
    neg_t.columns = ["topic", "negative"]
    pos_t = df_f[df_f["distilbert_label"] == "positive"]["topic_label"].value_counts().reset_index()
    pos_t.columns = ["topic", "positive"]
    cmp   = neg_t.merge(pos_t, on="topic", how="left").fillna(0).sort_values("negative", ascending=True)

    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Bar(y=cmp["topic"].str.title(), x=cmp["negative"], name="Negative",
        orientation="h", marker_color=RED,
        text=cmp["negative"].astype(int), textposition="outside", textfont=dict(color=WHITE, size=10)))
    fig_cmp.add_trace(go.Bar(y=cmp["topic"].str.title(), x=cmp["positive"], name="Positive",
        orientation="h", marker_color=TEAL,
        text=cmp["positive"].astype(int), textposition="outside", textfont=dict(color=WHITE, size=10)))
    fig_cmp.update_layout(
        barmode="group", template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=10, b=10, l=10, r=60), height=300,
        xaxis=dict(title=dict(text="Count", font=dict(color=GRAY, size=11)),
                   tickfont=dict(color=GRAY, size=10), showgrid=True, gridcolor="#1A2A2A"),
        yaxis=dict(tickfont=dict(color=WHITE, size=11), showgrid=False),
        legend=dict(font=dict(color=WHITE), bgcolor="rgba(0,0,0,0)",
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_cmp, width='stretch')

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Top Keywords side-by-side ──────────────────────────────────────────
    _section("Top Keywords: Positive vs Negative Reviews", color=PURPLE)
    kw_c1, kw_c2 = st.columns(2)

    with kw_c1:
        pos_words = _top_words(df_f[df_f["distilbert_label"] == "positive"])
        if pos_words:
            pw = pd.DataFrame(pos_words, columns=["word", "freq"]).sort_values("freq")
            fig_pw = go.Figure(go.Bar(
                x=pw["freq"], y=pw["word"], orientation="h",
                marker=dict(color=pw["freq"], colorscale=TEAL_SCALE, showscale=False),
                text=pw["freq"], textposition="outside", textfont=dict(color=WHITE, size=9),
            ))
            fig_pw.update_layout(
                title=dict(text="✅ Positive", font=dict(color=CYAN, size=13)),
                template=PLOTLY_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=40, b=10, l=10, r=40), height=340,
                xaxis=dict(showgrid=True, gridcolor="#1A2A2A", tickfont=dict(color=GRAY, size=9)),
                yaxis=dict(tickfont=dict(color=WHITE, size=10), showgrid=False),
                showlegend=False,
            )
            st.plotly_chart(fig_pw, width='stretch')

    with kw_c2:
        neg_words = _top_words(df_f[df_f["distilbert_label"] == "negative"])
        if neg_words:
            nw = pd.DataFrame(neg_words, columns=["word", "freq"]).sort_values("freq")
            fig_nw = go.Figure(go.Bar(
                x=nw["freq"], y=nw["word"], orientation="h",
                marker=dict(color=nw["freq"], colorscale=RED_SCALE, showscale=False),
                text=nw["freq"], textposition="outside", textfont=dict(color=WHITE, size=9),
            ))
            fig_nw.update_layout(
                title=dict(text="❌ Negative", font=dict(color=RED, size=13)),
                template=PLOTLY_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=40, b=10, l=10, r=40), height=340,
                xaxis=dict(showgrid=True, gridcolor="#1A2A2A", tickfont=dict(color=GRAY, size=9)),
                yaxis=dict(tickfont=dict(color=WHITE, size=10), showgrid=False),
                showlegend=False,
            )
            st.plotly_chart(fig_nw, width='stretch')

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Filtered data table ────────────────────────────────────────────────
    _section(f"Filtered Review Data ({total:,} records)", color=GRAY)
    has_prod  = "product_category" in df_f.columns
    base_cols = ["review text", "topic_label", "distilbert_label", "vader_label", "rating"]
    extra     = ["product_category", "customer_region"] if has_prod else []
    display_df = df_f[base_cols + extra].rename(columns={
        "review text":      "Review",
        "topic_label":      "Topic",
        "distilbert_label": "DistilBERT",
        "vader_label":      "VADER",
        "rating":           "Rating",
        "product_category": "Product",
        "customer_region":  "Region",
    }).reset_index(drop=True)
    st.dataframe(display_df, width='stretch', height=320)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — DEEP-DIVE  (Topic | Product toggle)
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(
        f"<h2 style='color:{CYAN};font-weight:700;margin-bottom:4px;'>🔬 Deep-Dive Explorer</h2>"
        f"<p style='color:{GRAY};font-size:0.95rem;margin-top:0;margin-bottom:16px;'>"
        "Select a lens to drill into granular metrics, regional distribution, key terms, and verbatim customer voices.</p>",
        unsafe_allow_html=True
    )

    # ── Top-level toggle ──────────────────────────────────────────────────
    explore_mode = st.radio(
        "Explore by:", ["🏷️  Topic", "🏭  Product Category"],
        horizontal=True, key="dd_mode"
    )
    st.markdown("<hr style='border-color:#1A2A2A;margin:16px 0 20px;'>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────
    # MODE A: TOPIC DEEP-DIVE
    # ─────────────────────────────────────────────────────────────────────
    if explore_mode == "🏷️  Topic":
        topics_available = sorted(df_all["topic_label"].dropna().unique().tolist())
        sel = st.selectbox(
            "Select a topic to explore:",
            options=topics_available,
            format_func=str.title,
            key="dd_topic_sel"
        )

        df_t = df_all[df_all["topic_label"] == sel]

        # KPIs
        t_total   = len(df_t)
        t_pct     = t_total / len(df_all) * 100
        t_avg_r   = df_t["rating"].mean()
        t_neg_pct = (df_t["distilbert_label"] == "negative").mean() * 100
        t_dom_reg = (df_t["customer_region"].value_counts().idxmax()
                     if "customer_region" in df_t.columns and len(df_t) else "N/A")

        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(_kpi(f"{t_total:,}", "Reviews", f"{t_pct:.1f}% of total"),         unsafe_allow_html=True)
        k2.markdown(_kpi(f"{t_avg_r:.2f} ★", "Avg Rating", "1–5 scale", YELLOW),      unsafe_allow_html=True)
        k3.markdown(_kpi(f"{t_neg_pct:.1f}%", "Negative Rate", "DistilBERT", RED),     unsafe_allow_html=True)
        k4.markdown(_kpi(t_dom_reg, "Top Region", "Most feedback from", CYAN),          unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Row 1: Top keywords + Geographic distribution
        row1_c1, row1_c2 = st.columns(2)
        with row1_c1:
            _section("Top Keywords in This Topic")
            words = _top_words(df_t, n=10)
            if words:
                wd = pd.DataFrame(words, columns=["word", "freq"]).sort_values("freq")
                fig_w = go.Figure(go.Bar(
                    x=wd["freq"], y=wd["word"], orientation="h",
                    marker=dict(color=wd["freq"], colorscale=TEAL_SCALE, showscale=False),
                    text=wd["freq"], textposition="outside", textfont=dict(color=WHITE, size=9),
                ))
                fig_w.update_layout(
                    template=PLOTLY_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(t=10, b=10, l=10, r=40), height=300,
                    xaxis=dict(showgrid=True, gridcolor="#1A2A2A", tickfont=dict(color=GRAY, size=9)),
                    yaxis=dict(tickfont=dict(color=WHITE, size=10), showgrid=False),
                    showlegend=False,
                )
                st.plotly_chart(fig_w, width='stretch')

        with row1_c2:
            _section("Geographic Distribution")
            if "customer_region" in df_t.columns:
                reg = df_t["customer_region"].value_counts().reset_index()
                reg.columns = ["region", "count"]
                fig_reg = go.Figure(go.Bar(
                    x=reg["region"], y=reg["count"],
                    marker=dict(color=reg["count"], colorscale=TEAL_SCALE, showscale=False),
                    text=reg["count"], textposition="outside", textfont=dict(color=WHITE, size=10),
                ))
                fig_reg.update_layout(
                    template=PLOTLY_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(t=10, b=10, l=10, r=10), height=300,
                    xaxis=dict(tickfont=dict(color=WHITE, size=10), showgrid=False),
                    yaxis=dict(tickfont=dict(color=GRAY, size=9), gridcolor="#1A2A2A", title="Count"),
                    showlegend=False,
                )
                st.plotly_chart(fig_reg, width='stretch')

        # Row 2: Avg rating by region + Product breakdown
        row2_c1, row2_c2 = st.columns(2)
        with row2_c1:
            _section("Avg Rating by Region")
            if "customer_region" in df_t.columns:
                avg_reg = df_t.groupby("customer_region")["rating"].mean().sort_values().reset_index()
                avg_reg.columns = ["region", "avg"]
                fig_ar = go.Figure(go.Bar(
                    x=avg_reg["avg"], y=avg_reg["region"], orientation="h",
                    marker=dict(color=avg_reg["avg"],
                                colorscale=[[0, RED], [0.5, YELLOW], [1, CYAN]], showscale=False),
                    text=[f"{v:.2f}" for v in avg_reg["avg"]], textposition="outside",
                    textfont=dict(color=WHITE, size=10),
                ))
                fig_ar.update_layout(
                    template=PLOTLY_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(t=10, b=10, l=10, r=50), height=280,
                    xaxis=dict(range=[0, 5.5], tickfont=dict(color=GRAY, size=9), showgrid=True, gridcolor="#1A2A2A"),
                    yaxis=dict(tickfont=dict(color=WHITE, size=10), showgrid=False),
                    showlegend=False,
                )
                st.plotly_chart(fig_ar, width='stretch')

        with row2_c2:
            _section("Product Category Breakdown")
            if "product_category" in df_t.columns:
                prod = df_t["product_category"].value_counts().head(8).reset_index()
                prod.columns = ["product", "count"]
                prod = prod.sort_values("count")
                fig_pr = go.Figure(go.Bar(
                    x=prod["count"], y=prod["product"], orientation="h",
                    marker_color=CYAN,
                    text=prod["count"], textposition="outside", textfont=dict(color=WHITE, size=10),
                ))
                fig_pr.update_layout(
                    template=PLOTLY_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(t=10, b=10, l=10, r=40), height=280,
                    xaxis=dict(showgrid=True, gridcolor="#1A2A2A", tickfont=dict(color=GRAY, size=9)),
                    yaxis=dict(tickfont=dict(color=WHITE, size=10), showgrid=False),
                    showlegend=False,
                )
                st.plotly_chart(fig_pr, width='stretch')
            else:
                st.markdown(
                    f"<p style='color:{GRAY};font-style:italic;'>Product category data not in CSV — re-run the Week 3 notebook.</p>",
                    unsafe_allow_html=True
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # Verbatim feed
        _section("Customer Voices: Verbatim Reviews", color=PURPLE)
        _verbatim_feed(df_t, key="topic_vfeed")

    # ─────────────────────────────────────────────────────────────────────
    # MODE B: PRODUCT CATEGORY DEEP-DIVE
    # ─────────────────────────────────────────────────────────────────────
    else:
        if "product_category" not in df_all.columns:
            st.warning("⚠️  `product_category` column not found. Please re-run the updated Week 3 notebook first.")
        else:
            prods_available = sorted(df_all["product_category"].dropna().unique().tolist())
            sel_prod = st.selectbox(
                "Select a product category to explore:",
                options=prods_available,
                key="dd_prod_sel"
            )

            df_p = df_all[df_all["product_category"] == sel_prod]

            # KPIs
            p_total   = len(df_p)
            p_pct     = p_total / len(df_all) * 100
            p_avg_r   = df_p["rating"].mean()
            p_neg_pct = (df_p["distilbert_label"] == "negative").mean() * 100
            p_dom_top = df_p["topic_label"].value_counts().idxmax() if len(df_p) else "N/A"

            k1, k2, k3, k4 = st.columns(4)
            k1.markdown(_kpi(f"{p_total:,}", "Reviews", f"{p_pct:.1f}% of total"),                    unsafe_allow_html=True)
            k2.markdown(_kpi(f"{p_avg_r:.2f} ★", "Avg Rating", "1–5 scale", YELLOW),                 unsafe_allow_html=True)
            k3.markdown(_kpi(f"{p_neg_pct:.1f}%", "Negative Rate", "DistilBERT", RED),                unsafe_allow_html=True)
            k4.markdown(_kpi(p_dom_top.title()[:22], "Top Complaint Topic", "Most common", CYAN),      unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Row 1: Sentiment donut + Topic breakdown
            row1_c1, row1_c2 = st.columns(2)
            with row1_c1:
                _section("Sentiment Split")
                s_ct   = df_p["distilbert_label"].value_counts()
                colors = [CYAN if l == "positive" else RED for l in s_ct.index]
                fig_sd = go.Figure(go.Pie(
                    labels=s_ct.index.str.title(), values=s_ct.values, hole=0.5,
                    marker_colors=colors, textinfo="percent+label", textposition="outside",
                ))
                fig_sd.update_layout(
                    template=PLOTLY_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(t=20, b=20, l=20, r=20), height=300,
                    legend=dict(font=dict(color=WHITE), bgcolor="rgba(0,0,0,0)"),
                    showlegend=False,
                )
                st.plotly_chart(fig_sd, width='stretch')

            with row1_c2:
                _section("Top Complaint Topics")
                neg_top = df_p[df_p["distilbert_label"] == "negative"]["topic_label"].value_counts().reset_index()
                neg_top.columns = ["topic", "count"]
                neg_top = neg_top.sort_values("count")
                fig_nt = go.Figure(go.Bar(
                    x=neg_top["count"], y=neg_top["topic"].str.title(), orientation="h",
                    marker=dict(color=neg_top["count"], colorscale=[[0, "#7f1d1d"], [1, RED]], showscale=False),
                    text=neg_top["count"], textposition="outside", textfont=dict(color=WHITE, size=10),
                ))
                fig_nt.update_layout(
                    template=PLOTLY_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(t=10, b=10, l=10, r=40), height=300,
                    xaxis=dict(showgrid=True, gridcolor="#1A2A2A", tickfont=dict(color=GRAY, size=9)),
                    yaxis=dict(tickfont=dict(color=WHITE, size=10), showgrid=False),
                    showlegend=False,
                )
                st.plotly_chart(fig_nt, width='stretch')

            # Row 2: Regional demand + Top keywords
            row2_c1, row2_c2 = st.columns(2)
            with row2_c1:
                _section("Regional Demand Split")
                if "customer_region" in df_p.columns:
                    rdem = df_p["customer_region"].value_counts().reset_index()
                    rdem.columns = ["region", "count"]
                    fig_rd = go.Figure(go.Bar(
                        x=rdem["region"], y=rdem["count"],
                        marker=dict(color=rdem["count"], colorscale=TEAL_SCALE, showscale=False),
                        text=rdem["count"], textposition="outside", textfont=dict(color=WHITE, size=10),
                    ))
                    fig_rd.update_layout(
                        template=PLOTLY_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(t=10, b=10, l=10, r=10), height=280,
                        xaxis=dict(tickfont=dict(color=WHITE, size=10), showgrid=False),
                        yaxis=dict(tickfont=dict(color=GRAY, size=9), gridcolor="#1A2A2A", title="Reviews"),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_rd, width='stretch')

            with row2_c2:
                _section("Top Keywords for This Product")
                neg_prod_words = _top_words(df_p[df_p["distilbert_label"] == "negative"], n=10)
                if neg_prod_words:
                    npw = pd.DataFrame(neg_prod_words, columns=["word", "freq"]).sort_values("freq")
                    fig_npw = go.Figure(go.Bar(
                        x=npw["freq"], y=npw["word"], orientation="h",
                        marker=dict(color=npw["freq"], colorscale=RED_SCALE, showscale=False),
                        text=npw["freq"], textposition="outside", textfont=dict(color=WHITE, size=9),
                    ))
                    fig_npw.update_layout(
                        title=dict(text="From negative reviews", font=dict(color=RED, size=11)),
                        template=PLOTLY_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(t=30, b=10, l=10, r=40), height=280,
                        xaxis=dict(showgrid=True, gridcolor="#1A2A2A", tickfont=dict(color=GRAY, size=9)),
                        yaxis=dict(tickfont=dict(color=WHITE, size=10), showgrid=False),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_npw, width='stretch')

            st.markdown("<br>", unsafe_allow_html=True)

            # Verbatim feed
            _section("Customer Voices: Verbatim Reviews", color=PURPLE)
            _verbatim_feed(df_p, key="prod_vfeed")
