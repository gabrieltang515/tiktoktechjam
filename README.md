# ML-Based Google Location Review Quality Detection System

## Problem Statement
Design and implement an ML-based system to evaluate the quality and relevancy of Google location reviews.

## System Requirements
- **Gauge review quality**: Detect spam, advertisements, irrelevant content, and rants from users who have likely never visited the location
- **Assess relevancy**: Determine whether the content of a review is genuinely related to the location being reviewed
- **Enforce policies**: Automatically flag or filter out reviews that violate policies:
  - No advertisements or promotional content
  - No irrelevant content (e.g., reviews about unrelated topics)
  - No rants or complaints from users who have not visited the place

## Project Structure
```
├── data/                   # Dataset storage
├── src/                    # Source code
│   ├── preprocessing/      # Data preprocessing modules
│   ├── feature_engineering/ # Feature engineering modules
│   ├── models/            # ML model implementations
│   ├── evaluation/        # Model evaluation scripts
│   └── utils/             # Utility functions
├── notebooks/             # Jupyter notebooks for exploration
├── results/               # Model results and outputs
├── config/                # Configuration files
└── tests/                 # Unit tests
```

## Task Requirements Implementation

### 1. Preprocess and Clean Data
- Remove noise from text data
- Handle missing values
- Standardize formats

### 2. Feature Engineering
- **Textual features (NLP)**: Sentiment analysis, topic modeling, keyword extraction
- **Metadata features**: Review length, posting time, user history, GPS proximity

### 3. Policy Enforcement Module
- Implement logic to detect policy violations

### 4. Model Development
- Train/prompt engineering ML/NLP models for classification and relevancy scoring
- Support for transformers, LSTM, and classical ML approaches
- Optional ensemble or multi-task learning approach

### 5. Evaluation and Reporting
- Evaluate models on provided and internal datasets
- Report key metrics: precision, recall, F1-score
- Provide summary of findings and recommendations

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download the dataset:
```bash
python src/data/download_dataset.py
```

3. Run the complete pipeline:
```bash
python src/main.py
```

## Dataset
Using Google Maps Restaurant Reviews dataset from Kaggle: https://www.kaggle.com/datasets/denizbilginn/google-maps-restaurant-reviews

## Technologies Used
- Python 3.13
- Transformers (Hugging Face)
- PyTorch
- Scikit-learn
- NLTK & spaCy
- XGBoost & LightGBM
- Pandas & NumPy
- Matplotlib & Seaborn
