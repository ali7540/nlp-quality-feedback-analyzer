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
│   ├── week1_preprocessing.ipynb
│   └── week2_sentiment.ipynb
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

 ## 🧠 Sentiment Analysis Pipeline & Details
During the sentiment analysis phase, lexical rule-based scoring and deep learning transformers were evaluated:
* **Ground-Truth Label Derivation**:
  * Categorized true ratings into three classes: 1-2 stars $\rightarrow$ `negative`, 3 stars $\rightarrow$ `neutral`, and 4-5 stars $\rightarrow$ `positive`.
* **VADER Classifier (Rule-Based)**:
  * Applied VADER's `SentimentIntensityAnalyzer` to raw (uncleaned) text to preserve stylistic cues (punctuation, capitalization, intensifiers).
  * Mapped compound scores to classes ($\ge 0.05$ positive, $\le -0.05$ negative, neutral otherwise).
  * Uncovered a corporate politeness bias (VADER predicted **73.15% positive** feedback vs. **55.22%** true due to polite syntax in complaints).
* **DistilBERT Classifier (Deep Learning)**:
  * Loaded pre-trained `distilbert-base-uncased-finetuned-sst-2-english` transformer pipeline to perform inference on a 2,500-review sample.
  * Evaluated 3-class overall accuracy (**78.48%** vs. VADER's **68.35%**) and isolated 2-class binary accuracy (**96.60%** vs. VADER's **85.38%**, excluding neutral reviews due to the model's binary constraint).
* **Output Artifacts**:
  * Intermediate dataset containing VADER predictions stored in `outputs/sentiment_stage1.csv` (10,000 rows).
  * Inference dataset containing DistilBERT predictions stored in `outputs/distilbert_stage2.csv` (2,500 rows).
  * Final comparative validation dataset stored in `outputs/labeled_dataset.csv` (2,500 rows).
  * Performance charts saved under `outputs/figures/`: Confusion Matrix Heatmap (`confusion_matrix.png`) and Clustered Accuracy/Speed comparison chart (`sentiment_comparison_chart.png`).


---

## 🚦 Progress
- [x] **Week 1** — Text Data & Preprocessing
- [x] **Week 2** — Sentiment Analysis & Benchmarking
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

5. **Run the Sentiment Pipeline**:
   Open and execute the Jupyter notebook to run VADER and DistilBERT model evaluations:
   ```bash
   jupyter notebook notebooks/week2_sentiment.ipynb
   ```

6. **Launch the Streamlit Dashboard**:
   Start the interactive dashboard locally to explore the pipeline flow, EDA visualizations, model benchmarks, and live sentiment analyzer:
   ```bash
   streamlit run app/main.py
   ```