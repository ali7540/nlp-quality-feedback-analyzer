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