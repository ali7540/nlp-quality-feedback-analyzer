# NLP-Based Quality Feedback Analyzer

The **NLP-Based Quality Feedback Analyzer** is an end-to-end natural language processing pipeline designed to ingest, clean, and analyze product reviews from Amazon. By employing advanced text preprocessing (tokenization, stopword filtering, lemmatization), sentiment classification, and topic modeling, the project identifies key drivers of customer satisfaction and exposes product quality issues. A dashboard interface allows stakeholder interaction with findings to facilitate quality control and customer experience optimization.

---

## 🛠️ Tech Stack
* **Language**: Python
* **NLP Libraries**: NLTK, spaCy, Hugging Face Transformers
* **Visualization**: Matplotlib, WordCloud
* **Interface**: Streamlit
* **Development**: Jupyter Notebook, Pandas, NumPy

---

## 📂 Project Structure
```text
nlp-quality-feedback-analyzer/
├── app/                     # Streamlit dashboard application files
├── data/
│   ├── processed/           # Processed datasets (reviews_cleaned.csv)
│   └── raw/                 # Unmodified raw source dataset (Reviews.csv)
├── models/                  # Saved models (classifiers, topic models)
├── notebooks/               # Jupyter notebooks for data analysis & pipelines
│   └── week1_preprocessing.ipynb
├── outputs/
│   ├── figures/             # Data charts, graphs, and analysis figures
│   └── wordclouds/          # Generated word cloud visualization exports
├── requirements.txt         # Project package dependencies
└── README.md                # Project documentation and guide
```

---

## 🔍 Preprocessing Pipeline & Details
During the initial preprocessing phase, the raw review dataset was cleaned and curated:
* **Dataset Auditing & Deduplication**:
  * Ingested a raw corpus of **568,454** review records.
  * Identified and removed **174,875** exact text duplicates.
  * Developed a normalization algorithm (standardizing case, stripping punctuation, and collapsing spacing) to identify and remove an additional **475** near-duplicate reviews.
  * Total duplicates removed: **175,350** reviews.
* **Downsampling**: Sampled **exactly 10,000 unique reviews** (using `random_state=42` for reproducibility) to create a manageable development set.
* **Cleaning Pipeline**:
  * Lowercased text and removed all punctuation and special characters.
  * Segmented text into words (Tokenization) and excluded standard English stopwords using NLTK.
  * Standardized terms using NLTK's `WordNetLemmatizer` to preserve grammatical roots (e.g., "waffles" $\rightarrow$ "waffle").
* **Output Artifacts**:
  * Cleaned review data stored in `data/processed/reviews_cleaned.csv`.
  * Generated 1-star and 5-star comparative word clouds inside `outputs/wordclouds/`.

---

## 🚦 Progress
- [x] **Week 1** — Text Data & Preprocessing
- [ ] **Week 2** — Sentiment Analysis
- [ ] **Week 3** — Topic Modeling & Entity Extraction
- [ ] **Week 4** — Dashboard & Presentation

---

## 🚀 Setup

Follow these instructions to set up the project locally:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ali7540/nlp-quality-feedback-analyzer.git
   cd nlp-quality-feedback-analyzer
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK Corpora**:
   Run the following command to download the necessary NLTK data models:
   ```bash
   python3 -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"
   ```

4. **Run the Preprocessing Pipeline**:
   Open and execute the Jupyter notebook to see the data loading, cleaning, and word cloud generation:
   ```bash
   jupyter notebook notebooks/week1_preprocessing.ipynb
   ```
