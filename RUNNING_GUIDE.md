# How to Run the Machine Learning Files on Your Computer

This guide will walk you through setting up and running the review quality detection system on your computer.

## 🛠️ Prerequisites

### 1. Python Requirements

- **Python 3.8 or higher** (Python 3.13 recommended)
- **pip** (Python package installer)

### 2. System Requirements

- **macOS** (you're on macOS 22.6.0, which is perfect)
- **At least 4GB RAM** (8GB+ recommended for training)
- **2GB free disk space**

## 📦 Installation Steps

### Step 1: Clone/Download the Project

You already have the project at `/Users/Jove/tiktokjam/`, so you can skip this step.

### Step 2: Navigate to Project Directory

```bash
cd /Users/Jove/tiktokjam
```

### Step 3: Set Up Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Your prompt should now show (venv) at the beginning
```

### Step 4: Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# Install additional missing dependencies
pip install lightgbm pyyaml
```

**Note**: The `requirements.txt` was missing `lightgbm` and `pyyaml`, so we install them separately.

### Step 5: Download NLTK Data (Required for text processing)

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
```

## 🚀 Running the Machine Learning Files

### Option 1: Quick Demo (Recommended to start)

```bash
# Make sure you're in the project directory and virtual environment is activated
python scripts/demo/demo_review_quality_detection.py
```

This will:

- Load the pre-trained models
- Analyze sample reviews
- Show how the system detects quality vs. policy violations
- Demonstrate the difference between rating and quality

### Option 2: Train New Models

```bash
# Train all models from scratch
python scripts/training/train_review_quality_model.py
```

This will:

- Load the dataset from `data/reviews.csv`
- Preprocess the text data
- Extract features (NO rating data used)
- Train multiple models (Random Forest, XGBoost, Ensemble)
- Save new models to `models/review_quality/`
- Generate performance reports

### Option 3: Run Main Application

```bash
# Run the main application with different options
python src/main.py --help

# Example: Run with specific model type
python src/main.py --model_type ensemble --action evaluate
```

### Option 4: Test with Your Own Data

```bash
# Test specific reviews
python test_dumpling_darlings.py
```

## 📊 Understanding the Output

### Demo Output Example:

```
REVIEW QUALITY DETECTION DEMO
Focus: Text Quality and Policy Compliance (NO RATING DATA)
================================================================================

ANALYZING SAMPLE REVIEWS...
================================================================================

REVIEW 1: High rating but contains advertisements
Text: "This restaurant is AMAZING! Best food ever! Call now for special offers!!! Visit our website www.example.com"
Rating: 5 stars
------------------------------------------------------------
Quality Score: 0.15 (Low Quality)
Policy Decision: REJECTED
Violations: advertisements, spam_content
```

### Key Points:

- **Quality Score**: 0.0-1.0 (higher = better quality)
- **Policy Decision**: APPROVED, APPROVED_WITH_WARNING, UNDER_REVIEW, REJECTED
- **Rating vs Quality**: Independent concepts (5-star review can be low quality)

## 🔧 Troubleshooting

### Common Issues:

#### 1. Import Errors

```bash
# If you get import errors, make sure you're in the right directory
cd /Users/Jove/tiktokjam
source venv/bin/activate
```

#### 2. Missing Dependencies

```bash
# Install missing packages
pip install lightgbm pyyaml requests
```

#### 3. NLTK Data Not Found

```bash
# Download required NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

#### 4. Memory Issues (during training)

```bash
# Reduce model complexity in the training script
# Edit scripts/training/train_review_quality_model.py
# Change max_features=1000 (instead of 2000)
# Change n_topics=5 (instead of 10)
```

#### 5. Permission Errors

```bash
# Make scripts executable
chmod +x scripts/demo/demo_review_quality_detection.py
chmod +x scripts/training/train_review_quality_model.py
```

## 📁 File Structure Explanation

### Key Files You'll Use:

- `scripts/demo/demo_review_quality_detection.py` - **Start here!**
- `scripts/training/train_review_quality_model.py` - Train new models
- `src/main.py` - Main application
- `models/review_quality/*.pkl` - Pre-trained models

### Data Files:

- `data/reviews.csv` - Main dataset (1100 reviews)
- `config/config.yaml` - Configuration settings

## 🎯 Quick Start Commands

```bash
# 1. Navigate to project
cd /Users/Jove/tiktokjam

# 2. Activate virtual environment
source venv/bin/activate

# 3. Install dependencies (if not done already)
pip install -r requirements.txt lightgbm pyyaml

# 4. Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# 5. Run demo
python scripts/demo/demo_review_quality_detection.py
```

## 🔍 What Each Model Does

### 1. **Random Forest** (`review_quality_random_forest_*.pkl`)

- Uses multiple decision trees
- Good for complex patterns
- 2.0MB file size

### 2. **XGBoost** (`review_quality_xgboost_*.pkl`)

- Gradient boosting algorithm
- Fast and efficient
- 293KB file size

### 3. **Ensemble** (`review_quality_ensemble_*.pkl`)

- Combines all models for best accuracy
- Most accurate but slowest
- 5.3MB file size

## 📈 Performance Metrics

The models achieve:

- **Ensemble**: 98.6% accuracy
- **XGBoost**: 97.3% accuracy
- **Random Forest**: 88.6% accuracy

## 🎉 Success!

Once you run the demo successfully, you'll see:

- Sample reviews being analyzed
- Quality scores calculated
- Policy violations detected
- Clear explanation of how rating ≠ quality

The system demonstrates that a 1-star review can be high quality (well-written criticism) and a 5-star review can be low quality (spam, ads, irrelevant content).
