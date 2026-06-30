# NLP-Based Quality Feedback Analyzer

The **NLP-Based Quality Feedback Analyzer** is an end-to-end natural language processing pipeline designed to ingest, clean, and analyze B2B customer feedback for Jindal Steel Limited (synthetic). By employing advanced text preprocessing (tokenization, stopword filtering, lemmatization), sentiment classification, and topic modeling, the project identifies key drivers of B2B customer satisfaction and exposes product quality and logistics issues. A dashboard interface allows stakeholder interaction with findings to facilitate quality control and customer experience optimization.

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
│   └── raw/                 # Unmodified raw source dataset (jindal_feedback.csv)
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
During the initial preprocessing phase, the raw dataset was cleaned and curated:
* **B2B Steel Quality Feedback Dataset (Synthetic)**:
  * Ingested a raw corpus of **10,000** B2B quality feedback records from Jindal Steel Limited (pre-validated with zero duplicates, zero null values, and consistent severity-to-sentiment alignment).
  * Spans **10 product lines** (TMT Rebars, Steel Rails, HR Plates and Coils, Beams and Columns, Wire Rods, Angles and Channels, Billets, Cement, and Jindal Speedfloor) and **7 customer segments** across India.
  * Carried forward metadata columns: `feedback_id`, `feedback_date`, `customer_type`, `customer_region`, `product_category`, `order_quantity_mt`, `severity_rating`, `feedback_text`, `feedback_topic`, `feedback_source`, `plant_location`, `resolution_status`.
* **Cleaning Pipeline**:
  * Lowercased text and removed all punctuation and special characters from `feedback_text` to generate the `cleaned_text` column.
  * Segmented text into words (Tokenization) and excluded standard English stopwords using NLTK.
  * Standardized terms using NLTK's `WordNetLemmatizer` to preserve grammatical roots, saving the result to `tokens`.
* **Output Artifacts**:
  * Cleaned review data stored in `data/processed/reviews_cleaned.csv`.
  * Generated negative and positive comparative word clouds inside `outputs/wordclouds/` (`wordcloud_negative.png` and `wordcloud_positive.png`).

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
