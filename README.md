# Review Quality Detection System

A **corrected** machine learning system for automatically detecting review quality based purely on text characteristics and policy compliance, **without using rating data**.

## 🎯 Key Innovation

This system demonstrates that **review quality** and **restaurant rating** are completely independent concepts:
- **Rating (1-5 stars)** = User satisfaction with the restaurant experience
- **Review Quality** = How well-written, informative, and policy-compliant the review text is

A 1-star review can be high quality (well-written, constructive criticism), and a 5-star review can be low quality (spam, advertisements, irrelevant content).

## 🏗️ Project Structure

```
Tiktok Techjam/
├── 📁 config/                    # Configuration files
│   └── config.yaml              # Main configuration file
├── 📁 data/                     # Dataset and data files
│   ├── dataset/                 # Image dataset organized by categories
│   │   ├── indoor_atmosphere/   # Indoor atmosphere review images
│   │   ├── menu/               # Menu review images
│   │   ├── outdoor_atmosphere/ # Outdoor atmosphere review images
│   │   └── taste/              # Taste review images
│   └── reviews.csv             # Main review dataset (1100 authentic reviews)
├── 📁 models/                   # Trained machine learning models
│   └── 📁 review_quality/      # ✅ Latest corrected models
│       ├── review_quality_ensemble_20250825_235119.pkl
│       ├── review_quality_xgboost_20250825_235119.pkl
│       └── review_quality_random_forest_20250825_235119.pkl
├── 📁 scripts/                  # Executable scripts
│   ├── 📁 training/            # Model training scripts
│   │   └── train_review_quality_model.py  # ✅ New corrected training script
│   └── 📁 demo/                # Demo and testing scripts
│       └── demo_review_quality_detection.py  # ✅ New demo script
├── 📁 src/                     # Source code
│   ├── 📁 data/                # Data loading utilities
│   ├── 📁 evaluation/          # Model evaluation modules
│   ├── 📁 feature_engineering/ # Feature extraction (NO RATING DATA)
│   ├── 📁 models/              # ML models and policy enforcement
│   ├── 📁 preprocessing/       # Data preprocessing modules
│   └── 📁 utils/               # Utility functions
├── 📁 notebooks/               # Jupyter notebooks
│   └── 01_data_exploration.py  # Data exploration script
├── 📄 FINAL_REPORT.md          # ✅ Updated comprehensive project report
├── 📄 ORGANIZATION_SUMMARY.md  # ✅ Updated project organization
├── 📄 README.md               # This file
├── 📄 REVIEW_QUALITY_CORRECTION.md  # ✅ Documentation of the correction
└── 📄 requirements.txt         # Python dependencies
```

## 🎯 Key Features

### **Pure Quality Assessment (NO RATING DATA)**
- **Text quality features**: Length, readability, vocabulary diversity
- **Policy compliance**: No advertisements, spam, irrelevant content, rants
- **Writing sophistication**: Grammar, formatting, style analysis
- **Content relevance**: Focus on restaurant experience

### **Policy Enforcement**
- **Advertisement detection**: "buy now", "special offer", contact information
- **Spam detection**: Phone numbers, emails, URLs, suspicious patterns
- **Irrelevant content**: Politics, sports, weather, entertainment topics
- **Rant detection**: Excessive complaints, repetitive negative language
- **Quality standards**: Minimum length, formatting requirements

### **Model Performance**
- **Ensemble Model**: 98.6% accuracy, 98.6% F1 score
- **XGBoost**: 97.3% accuracy, 97.3% F1 score
- **Random Forest**: 88.6% accuracy, 88.5% F1 score

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Demo
```bash
python scripts/demo/demo_review_quality_detection.py
```

### 3. Train New Model
```bash
python scripts/training/train_review_quality_model.py
```

## 📊 System Performance

### **Policy Enforcement Results**
- **Approved**: 833 reviews (75.7%) - No policy violations
- **Approved with warning**: 232 reviews (21.1%) - Minor violations
- **Under review**: 34 reviews (3.1%) - Medium-severity violations
- **Rejected**: 1 review (0.1%) - Critical violations

### **Quality Distribution**
- **High quality**: 280 reviews (25.5%)
- **Medium quality**: 520 reviews (47.3%)
- **Low quality**: 300 reviews (27.3%)

## 🔧 Technical Details

### **Technology Stack**
- **Python 3.13** with comprehensive ML libraries
- **Scikit-learn**: Classical ML algorithms
- **XGBoost**: Gradient boosting framework
- **NLTK & TextBlob**: Natural language processing
- **Pandas & NumPy**: Data manipulation and analysis

### **Feature Engineering (NO RATING DATA)**
- **Textual features**: TF-IDF vectors, count vectors, topic modeling
- **Text quality**: Length, word count, readability scores
- **Policy violations**: Spam, advertisement, irrelevant content detection
- **Writing sophistication**: Vocabulary diversity, grammar indicators

### **Model Architecture**
- **Random Forest**: Baseline model with good interpretability
- **XGBoost**: Gradient boosting with excellent performance
- **Ensemble Model**: Voting classifier combining all models

## 📋 Correction History

### **Problem Identified**
The original system incorrectly used rating data (1-5 stars) to determine review quality, which is fundamentally wrong.

### **Solution Implemented**
- ✅ **Removed all rating-based features** from quality assessment
- ✅ **Focused purely on text quality** and policy compliance
- ✅ **Implemented proper content moderation** policies
- ✅ **Cleaned up project structure** to remove outdated files

### **Key Insight**
The system demonstrates that:
- **High rating (5 stars) can have low quality** (spam, ads, irrelevant content)
- **Low rating (1 star) can have high quality** (well-written, constructive criticism)
- **Quality and rating are completely independent concepts**

## 📁 Documentation

- **[FINAL_REPORT.md](FINAL_REPORT.md)**: Comprehensive project report
- **[ORGANIZATION_SUMMARY.md](ORGANIZATION_SUMMARY.md)**: Project structure and organization
- **[REVIEW_QUALITY_CORRECTION.md](REVIEW_QUALITY_CORRECTION.md)**: Detailed documentation of the correction

## 🎯 Use Cases

### **Content Moderation**
- Automatically filter low-quality reviews
- Detect policy violations regardless of rating
- Maintain platform integrity

### **Quality Assessment**
- Evaluate review writing quality
- Assess content relevance and informativeness
- Identify well-written reviews

### **Policy Enforcement**
- Detect advertisements and promotional content
- Identify spam and irrelevant content
- Filter excessive rants and complaints

## 🔍 Example Usage

```python
from src.preprocessing.text_preprocessor import TextPreprocessor
from src.models.policy_enforcement import ReviewQualityPolicyEnforcer

# Initialize components
preprocessor = TextPreprocessor()
policy_enforcer = ReviewQualityPolicyEnforcer()

# Analyze a review
text = "This restaurant is AMAZING! Best food ever! Call now for special offers!!!"
processed = preprocessor.preprocess_text(text)
analysis = policy_enforcer.analyze_review(text, features)

print(f"Quality Score: {quality_score:.3f}")
print(f"Policy Decision: {analysis['policy_decision']}")
print(f"Violations: {analysis['violation_counts']['total']}")
```

## 🚀 Future Development

### **Planned Enhancements**
- Multi-language support
- Real-time processing capabilities
- Advanced NLP models (BERT, GPT)
- User feedback integration

### **Maintenance Guidelines**
- Keep rating and quality assessment separate
- Focus on text-based quality features
- Ensure policy compliance alignment
- Regular model retraining and validation

---

**Project Status**: ✅ **CORRECTED AND PRODUCTION-READY**

**Best Model Performance**: Ensemble (F1: 0.986, Accuracy: 0.986)

**Key Achievement**: Proper separation of rating and quality assessment

**License**: MIT License
