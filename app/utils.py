import os
import re
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

@st.cache_resource
def load_distilbert():
    from transformers import pipeline as hf_pipeline
    return hf_pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

@st.cache_resource
def load_lda_model_and_dict():
    from gensim.models import LdaModel
    from gensim.corpora import Dictionary
    lda_path  = os.path.join(BASE_DIR, "models", "lda_model", "lda_model")
    dict_path = os.path.join(BASE_DIR, "models", "lda_model", "lda_dict")
    return LdaModel.load(lda_path), Dictionary.load(dict_path)

@st.cache_resource
def load_spacy_model():
    import spacy
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
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

    # Normalise label columns
    for col in ("distilbert_label", "vader_label", "rating_based_truth"):
        if col in df.columns:
            df[col] = df[col].str.lower().str.strip()

    # Normalise categorical columns if present (added in updated notebook)
    for col in ("topic_label", "product_category", "customer_type",
                "customer_region", "plant_location", "resolution_status"):
        if col in df.columns:
            df[col] = df[col].str.strip()

    # Parse feedback_date if present
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
    - DistilBERT Sentiment
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

    # 2. DistilBERT Sentiment
    classifier = load_distilbert()
    db_result  = classifier(text[:512])[0]
    db_label   = db_result["label"].lower()
    db_conf    = db_result["score"]

    # 3. LDA Topic Prediction
    lda_model, dictionary = load_lda_model_and_dict()
    cleaned_text = re.sub(r'[^\w\s]', '', text.lower())

    try:
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        from nltk.stem import WordNetLemmatizer
        stop_words = set(stopwords.words("english"))
        tokens     = [t for t in word_tokenize(cleaned_text) if t not in stop_words and len(t) > 1]
        lem        = WordNetLemmatizer()
        lemmatized = [lem.lemmatize(t) for t in tokens]
    except Exception:
        lemmatized = [w for w in cleaned_text.split() if len(w) > 1]

    bow         = dictionary.doc2bow(lemmatized)
    topic_probs = lda_model.get_document_topics(bow, minimum_probability=0.0)

    topic_mapping = {
        0: 'packaging and handling',
        1: 'delivery and logistics',
        2: 'product and quality',
        3: 'customer service',
        4: 'pricing and commercial',
        5: 'technical compliance',
    }

    probs_dict      = {topic_mapping[t_idx]: float(prob) for t_idx, prob in topic_probs}
    predicted_topic = max(probs_dict, key=probs_dict.get)

    # 4. spaCy NER Entities
    nlp      = load_spacy_model()
    doc      = nlp(text)
    entities = sorted(list(set([ent.text.strip() for ent in doc.ents if len(ent.text.strip()) > 1])))

    return vader_label, db_label, db_conf, predicted_topic, entities, probs_dict
