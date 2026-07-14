import os
import re
import requests
import pandas as pd
import streamlit as st
import nltk

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────────
# CACHED MODEL LOADERS
# ─────────────────────────────────────────────

@st.cache_resource
def load_vader():
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    try:
        sia = SentimentIntensityAnalyzer()
        sia.polarity_scores("test")
        return sia
    except Exception:
        nltk.download("vader_lexicon", quiet=True)
        nltk.download("punkt", quiet=True)
        nltk.download("stopwords", quiet=True)
        nltk.download("wordnet", quiet=True)
        return SentimentIntensityAnalyzer()

# ──────────────────────────────────────────────────────────────────────────────
# DistilBERT via HuggingFace Inference API
#
# WHY: Loading DistilBERT locally requires ~300 MB PyTorch + ~268 MB model
# weights, which exceeds Streamlit Cloud's 1 GB free-tier RAM limit and
# causes a silent container crash (appears as infinite loading).
#
# SOLUTION: Call the HuggingFace Serverless Inference API instead.
# - Same model, identical results, zero local memory cost.
# - No API key required for public models (free tier: ~30K req/month).
# - Falls back to VADER label if the API call fails.
# ──────────────────────────────────────────────────────────────────────────────
HF_API_URL = (
    "https://api-inference.huggingface.co/models/"
    "distilbert-base-uncased-finetuned-sst-2-english"
)
HF_HEADERS = {"Content-Type": "application/json"}

def _call_distilbert_api(text: str):
    """
    POST text to HuggingFace Serverless Inference API.
    Returns (label: str, score: float) or raises on failure.
    """
    payload = {"inputs": text[:512], "options": {"wait_for_model": True}}
    resp = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload, timeout=30)
    resp.raise_for_status()
    result = resp.json()
    # API returns [[{label, score}, ...]] — pick the top prediction
    if isinstance(result, list) and isinstance(result[0], list):
        top = result[0][0]
    else:
        top = result[0]
    return top["label"].lower(), float(top["score"])

@st.cache_resource
def load_lda_model_and_dict():
    try:
        from gensim.models import LdaModel
        from gensim.corpora import Dictionary
    except ImportError:
        return None, None

    lda_path  = os.path.join(BASE_DIR, "models", "lda_model", "lda_model")
    dict_path = os.path.join(BASE_DIR, "models", "lda_model", "lda_dict")
    try:
        model = LdaModel.load(lda_path)
        dictionary = Dictionary.load(dict_path)
        return model, dictionary
    except Exception as e:
        # Fallback retraining mechanism on version mismatch or pickle corruption
        try:
            import ast
            csv_path = os.path.join(BASE_DIR, "outputs", "extracted_topics.csv")
            df = pd.read_csv(csv_path)

            tokens_list = []
            for t in df["tokens"].dropna():
                try:
                    tokens_list.append(ast.literal_eval(t))
                except Exception:
                    tokens_list.append(str(t).split())

            dictionary = Dictionary(tokens_list)
            dictionary.filter_extremes(no_below=5, no_above=0.5)
            corpus = [dictionary.doc2bow(tok) for tok in tokens_list]

            model = LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=6,
                random_state=42,
                passes=10
            )
            model.random_state = None  # prevent numpy BitGenerator pickle issues

            os.makedirs(os.path.dirname(lda_path), exist_ok=True)
            model.save(lda_path)
            dictionary.save(dict_path)
            return model, dictionary
        except Exception as retrain_err:
            raise RuntimeError(
                f"Failed to load or retrain LDA model: {retrain_err}"
            ) from e

@st.cache_resource
def load_spacy_model():
    import spacy
    return spacy.load("en_core_web_sm")

@st.cache_data
def load_extracted_topics():
    """
    Load outputs/extracted_topics.csv and normalise all string columns
    so downstream code never has to repeat .str.lower() / .str.strip().
    Returns a clean DataFrame.
    """
    path = os.path.join(BASE_DIR, "outputs", "extracted_topics.csv")
    df = pd.read_csv(path)

    for col in ("distilbert_label", "vader_label", "rating_based_truth"):
        if col in df.columns:
            df[col] = df[col].str.lower().str.strip()

    for col in ("topic_label", "product_category", "customer_type",
                "customer_region", "plant_location", "resolution_status"):
        if col in df.columns:
            df[col] = df[col].str.strip()

    if "feedback_date" in df.columns:
        df["feedback_date"] = pd.to_datetime(df["feedback_date"], errors="coerce")

    return df

# ─────────────────────────────────────────────
# HELPER INFERENCE FUNCTION
# ─────────────────────────────────────────────

def analyze_review(text):
    """
    Takes a raw review string and runs:
    - VADER Sentiment
    - DistilBERT Sentiment (via HuggingFace Inference API)
    - LDA Topic Prediction
    - spaCy NER Entities extraction

    Returns:
    - vader_label (str)
    - db_label (str)
    - db_conf (float)
    - predicted_topic (str)
    - entities (list of str)
    - probs_dict (dict mapping topic names to float probabilities)
    """
    # 1. NLTK VADER Sentiment
    sia = load_vader()
    compound = sia.polarity_scores(text)["compound"]
    if compound >= 0.05:
        vader_label = "positive"
    elif compound <= -0.05:
        vader_label = "negative"
    else:
        vader_label = "neutral"

    # 2. DistilBERT Sentiment — via HuggingFace Inference API
    try:
        db_label, db_conf = _call_distilbert_api(text)
    except Exception:
        # Graceful fallback: use VADER result if API is unreachable
        db_label = vader_label if vader_label != "neutral" else "positive"
        db_conf  = abs(compound) if compound != 0 else 0.5

    # 3. LDA Topic Prediction
    lda_model, dictionary = load_lda_model_and_dict()

    topic_mapping = {
        0: 'packaging and handling',
        1: 'delivery and logistics',
        2: 'product and quality',
        3: 'customer service',
        4: 'pricing and commercial',
        5: 'technical compliance',
    }

    if lda_model is None or dictionary is None:
        probs_dict      = {topic_mapping[i]: 1.0 / 6.0 for i in range(6)}
        probs_dict['product and quality'] = 0.5
        predicted_topic = 'product and quality'
    else:
        cleaned_text = re.sub(r'[^\w\s]', '', text.lower())

        try:
            from nltk.corpus import stopwords
            from nltk.tokenize import word_tokenize
            from nltk.stem import WordNetLemmatizer
            stop_words = set(stopwords.words("english"))
            tokens     = [t for t in word_tokenize(cleaned_text)
                          if t not in stop_words and len(t) > 1]
            lem        = WordNetLemmatizer()
            lemmatized = [lem.lemmatize(t) for t in tokens]
        except Exception:
            lemmatized = [w for w in cleaned_text.split() if len(w) > 1]

        bow         = dictionary.doc2bow(lemmatized)
        topic_probs = lda_model.get_document_topics(bow, minimum_probability=0.0)

        probs_dict      = {topic_mapping[t_idx]: float(prob)
                           for t_idx, prob in topic_probs}
        predicted_topic = max(probs_dict, key=probs_dict.get)

    # 4. spaCy NER Entities
    nlp      = load_spacy_model()
    doc      = nlp(text)
    entities = sorted(list(set(
        [ent.text.strip() for ent in doc.ents if len(ent.text.strip()) > 1]
    )))

    return vader_label, db_label, db_conf, predicted_topic, entities, probs_dict
