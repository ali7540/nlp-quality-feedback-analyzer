# NLP-Based Quality Feedback Analyzer

A comprehensive, end-to-end NLP project designed to analyze customer reviews, detect product quality issues, extract key themes, and classify sentiments. 

This repository is set up for collaborative work between a two-person team.

---

## 📂 Project Structure

```text
nlp-quality-feedback-analyzer/
├── data/
│   ├── raw/                 # Contains the original, unmodified dataset (Reviews.csv)
│   └── processed/           # Contains intermediate/processed datasets (reviews_stage1.csv)
├── notebooks/               # Jupyter notebooks for analysis and preprocessing
│   └── week1_preprocessing.ipynb
├── outputs/
│   ├── wordclouds/          # Generated word cloud visualization exports
│   └── figures/             # Data charts, graphs, and analysis figures
├── models/                  # Saved models (sentiment classifiers, topic modeling inputs)
├── app/                     # Source files for the user interface / application
├── requirements.txt         # List of project dependencies
└── README.md                # Project documentation and guide
```

---

## 🚦 Project Status (Week 1 - First Half Completed)

### Person 1 Completion Summary (First Half of Week 1)
- **Directory Structure**: Initialized project skeleton directories (`data/raw/`, `data/processed/`, `notebooks/`, `outputs/wordclouds/`, `outputs/figures/`, `models/`, `app/`).
- **Data Ingestion**: Moved the raw `Reviews.csv` file into `data/raw/` without modification.
- **Exploratory Data Analysis (EDA)**: Created `notebooks/week1_preprocessing.ipynb` and explored the dataset columns, shape, missing values, and score/rating distributions.
- **Identified Targets**:
  - Review text column: `Text`
  - Rating/Score column: `Score`
- **Product Category Check**: Identified that the dataset has no explicit category columns and proceeded with the complete dataset.
- **Deduplication**:
  - Dropped **174,875** exact duplicates on raw review text (leaving **393,579** rows).
  - Dropped **475** near-duplicates by normalizing the text (lowercased, punctuation/special characters removed, whitespaces collapsed) and deduplicating based on the normalized representation (leaving **393,104** rows).
  - Sampled down to **exactly 10,000 unique reviews** (using `random_state=42` for reproducibility).
- **Text Cleaning**:
  - Lowercased and removed punctuation/special characters from the 10,000 sampled reviews.
- **Data Export**: Exported the intermediate dataset containing `Text` (original text), `cleaned_text` (preprocessed text), and `Score` (rating) to [reviews_stage1.csv](file:///Users/faizali1/Documents/internship%20project/nlp-quality-feedback-analyzer/data/processed/reviews_stage1.csv).

---

## 🤝 Handover Instructions for Person 2 (Second Half of Week 1)

Person 2 should continue in the same Jupyter notebook ([week1_preprocessing.ipynb](file:///Users/faizali1/Documents/internship%20project/nlp-quality-feedback-analyzer/notebooks/week1_preprocessing.ipynb)).

### Expected Next Steps:
1. **Tokenization**: Segment the cleaned review text (`cleaned_text`) into individual tokens.
2. **Stopwords Removal**: Remove common, non-informative English stopwords.
3. **Lemmatization/Stemming**: Reduce words to their base or dictionary form.
4. **Visualizations**:
   - Generate word clouds using the processed reviews and save outputs in `outputs/wordclouds/`.
   - Build charts for rating distribution and other features, exporting them to `outputs/figures/`.

---

## 🚀 Getting Started

### Prerequisites
Make sure you have python 3.8+ installed.

### Setup and Installation
1. Clone the repository and navigate to the project root:
   ```bash
   cd nlp-quality-feedback-analyzer
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open and run the Jupyter notebook to continue development:
   ```bash
   jupyter notebook notebooks/week1_preprocessing.ipynb
   ```
   ## 🚦 Project Status (Week 1 - Second Half Completed)

### Person 2 Completion Summary (Second Half of Week 1)

* **Loaded Intermediate Dataset**: Imported `data/processed/reviews_stage1.csv` containing 10,000 unique reviews with columns:

  * `Text` (original review text)
  * `cleaned_text` (lowercased and punctuation-stripped text)
  * `Score` (review rating)

* **Tokenization**:

  * Tokenized the `cleaned_text` column into individual words using NLTK's `word_tokenize()` function.
  * Stored tokenized output in a separate column for traceability.

* **Stopword Removal**:

  * Removed common English stopwords using NLTK's stopword corpus.
  * Generated a cleaned token list containing only informative terms.

* **Lemmatization**:

  * Applied NLTK's `WordNetLemmatizer` to reduce words to their base forms.
  * Preserved semantic meaning while reducing vocabulary size.

* **Final Text Preparation**:

  * Combined lemmatized tokens into a final processed text column named `final_cleaned_text`.
  * Retained all intermediate columns (`Text`, `cleaned_text`, tokens, and final cleaned text) for transparency and future analysis.

* **Processed Dataset Export**:

  * Saved the fully processed dataset as:

    * `data/processed/reviews_cleaned.csv`
  * This file serves as the final Week 1 output and will be used in subsequent project phases.

* **Word Cloud Visualization**:

  * Generated a word cloud for **1-star reviews** and saved it as:

    * `outputs/wordclouds/wordcloud_1star.png`
  * Generated a word cloud for **5-star reviews** and saved it as:

    * `outputs/wordclouds/wordcloud_5star.png`

* **Insights & Observations**:

  * Positive reviews frequently contained words such as *great*, *best*, *love*, and *delicious*, indicating customer satisfaction.
  * Negative reviews contained more purchase- and experience-related terms such as *received*, *bought*, *tried*, and *time*.
  * Product-related terms including *taste*, *tea*, *coffee*, and *flavor* appeared in both positive and negative reviews, highlighting the primary discussion themes.
  * Word clouds provided a visual comparison of vocabulary differences between positive and negative customer feedback.

* **Dependencies Updated**:

  * Added required NLP and visualization libraries to `requirements.txt`:

    * `nltk`
    * `wordcloud`
    * `matplotlib`
    * `pandas`
    * `numpy`
    ### Output Files
- data/processed/reviews_cleaned.csv
- outputs/wordclouds/wordcloud_1star.png
- outputs/wordclouds/wordcloud_5star.png

* **Week 1 Status**:

  * ✅ Text preprocessing completed
  * ✅ Tokenization completed
  * ✅ Stopword removal completed
  * ✅ Lemmatization completed
  * ✅ Word cloud generation completed
  * ✅ Final cleaned dataset exported
  * ✅ Week 1 deliverables completed
