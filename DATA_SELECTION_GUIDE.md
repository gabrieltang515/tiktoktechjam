# How to Cherry-Pick Data for Your Review Quality Model

This guide shows you how to selectively choose which reviews to train your model on from the `webscrapper/reviews.csv` file.

## 🎯 Why Cherry-Pick Data?

### **Benefits:**

- **Better Model Performance**: Focus on relevant examples
- **Faster Training**: Use smaller, curated datasets
- **Targeted Learning**: Train on specific types of reviews
- **Quality Control**: Remove irrelevant or problematic data
- **Balanced Classes**: Ensure equal representation of different review types

### **Your Dataset:**

- **Total Reviews**: 49,879 reviews
- **Rating Distribution**:
  - 1-star: 1,453 reviews
  - 2-star: 704 reviews
  - 3-star: 1,526 reviews
  - 4-star: 3,957 reviews
  - 5-star: 11,720 reviews

## 🛠️ Data Selection Methods

### **1. Select by Rating**

```python
# Select only 1-star and 5-star reviews
selected = selector.select_by_rating([1, 5], max_per_rating=500)
```

### **2. Select by Business**

```python
# Select reviews from specific restaurants
businesses = ["OCEAN Restaurant", "Hong Kong Kitchen"]
selected = selector.select_by_business(businesses)
```

### **3. Select by Text Length**

```python
# Select reviews between 50-500 characters
selected = selector.select_by_text_length(min_length=50, max_length=500)
```

### **4. Select by Keywords**

```python
# Select reviews containing promotional keywords
keywords = ["buy now", "special offer", "call", "website"]
selected = selector.select_by_keywords(keywords, include=True)

# Select reviews NOT containing spam keywords
selected = selector.select_by_keywords(keywords, include=False)
```

### **5. Select Balanced Sample**

```python
# Select equal number of reviews per rating
selected = selector.select_balanced_sample(n_per_rating=200)
```

### **6. Select by Quality Indicators**

```python
# Select high-quality reviews
selected = selector.select_by_quality_indicators(
    min_length=50,
    max_exclamation=2,
    exclude_keywords=['buy now', 'special offer', 'call']
)
```

## 📊 Pre-Made Data Selections

I've created several pre-made data selections for you:

### **1. Balanced Training Sample** (`data/balanced_training_sample.csv`)

- **1,001 reviews** (200 per rating)
- **Purpose**: Balanced training data
- **Use case**: General model training

### **2. High-Quality Reviews** (`data/high_quality_reviews.csv`)

- **16,424 reviews**
- **Purpose**: Clean, well-written reviews
- **Use case**: Train model to recognize good quality

### **3. Problematic Reviews** (`data/problematic_reviews.csv`)

- **1,430 reviews**
- **Purpose**: Spam, ads, promotional content
- **Use case**: Train model to detect violations

### **4. Specific Business Reviews** (`data/specific_business_reviews.csv`)

- **300 reviews**
- **Purpose**: Reviews from specific restaurants
- **Use case**: Domain-specific training

### **5. Custom Combination** (`data/custom_training_set.csv`)

- **501 reviews**
- **Purpose**: Mix of high-quality and problematic
- **Use case**: Balanced training with examples of both

## 🚀 How to Use Custom Data

### **Method 1: Interactive Selection**

```bash
python train_with_custom_data.py
```

This will show you a menu to choose which data to use.

### **Method 2: Direct File Replacement**

```bash
# Backup original data
cp data/reviews.csv data/reviews_backup.csv

# Replace with custom data
cp data/balanced_training_sample.csv data/reviews.csv

# Train the model
python scripts/training/train_review_quality_model.py

# Restore original data
cp data/reviews_backup.csv data/reviews.csv
```

### **Method 3: Create Your Own Selection**

```python
from data_selection_guide import DataSelector

# Initialize selector
selector = DataSelector("webscrapper/reviews.csv")

# Create your custom selection
my_selection = selector.select_by_rating([4, 5], max_per_rating=300)
my_selection = selector.select_by_text_length(min_length=100, max_length=300)

# Save your selection
selector.save_selection(my_selection, "data/my_custom_data.csv")
```

## 🎯 Recommended Strategies

### **For Detecting Spam/Ads:**

```python
# Use problematic reviews + some good reviews
spam_reviews = selector.select_by_keywords(['buy now', 'special offer', 'call'])
good_reviews = selector.select_by_quality_indicators(min_length=50)
combined = selector.combine_selections([spam_reviews, good_reviews])
```

### **For Balanced Training:**

```python
# Use balanced sample
balanced = selector.select_balanced_sample(n_per_rating=200)
```

### **For High-Quality Detection:**

```python
# Use only high-quality reviews
quality = selector.select_by_quality_indicators(
    min_length=50,
    max_exclamation=2,
    exclude_keywords=['buy now', 'special offer', 'call', 'website']
)
```

## 📈 Training Results Comparison

### **Original Data (1,100 reviews):**

- Ensemble: 98.6% accuracy
- XGBoost: 97.3% accuracy
- Random Forest: 88.6% accuracy

### **Balanced Sample (1,001 reviews):**

- More balanced class distribution
- Better generalization
- Potentially higher accuracy on minority classes

### **High-Quality Only (16,424 reviews):**

- Larger training set
- Focus on quality patterns
- May miss problematic examples

## 🔧 Custom Selection Examples

### **Example 1: Focus on Short Reviews**

```python
short_reviews = selector.select_by_text_length(min_length=10, max_length=100)
```

### **Example 2: Focus on Specific Rating Range**

```python
mid_ratings = selector.select_by_rating([3, 4], max_per_rating=500)
```

### **Example 3: Exclude Specific Authors**

```python
# Get all authors
all_authors = selector.df['author_name'].unique()
# Remove specific authors
exclude_authors = ['spam_user1', 'spam_user2']
filtered_authors = [a for a in all_authors if a not in exclude_authors]
filtered_reviews = selector.df[selector.df['author_name'].isin(filtered_authors)]
```

### **Example 4: Combine Multiple Criteria**

```python
# Reviews that are:
# 1. 4-5 stars
# 2. 50-300 characters
# 3. Don't contain promotional keywords
good_reviews = selector.select_by_rating([4, 5])
good_reviews = selector.select_by_text_length(min_length=50, max_length=300)
good_reviews = selector.select_by_keywords(['buy now', 'special offer'], include=False)
```

## 💡 Tips for Effective Data Selection

### **1. Start Small**

- Begin with 1,000-2,000 reviews
- Test model performance
- Gradually increase dataset size

### **2. Maintain Balance**

- Include examples of all review types
- Don't over-represent any single category
- Consider class imbalance

### **3. Focus on Quality**

- Remove obvious spam/ads
- Include diverse writing styles
- Ensure relevance to restaurant reviews

### **4. Iterate and Improve**

- Train model with selection A
- Test performance
- Adjust selection criteria
- Retrain with selection B

### **5. Document Your Choices**

- Keep track of selection criteria
- Note performance changes
- Document reasoning for changes

## 🎉 Quick Start Commands

```bash
# 1. Generate all data selections
python data_selection_guide.py

# 2. Train with balanced sample
python train_with_custom_data.py
# Choose option 1

# 3. Train with high-quality reviews
python train_with_custom_data.py
# Choose option 2

# 4. Train with problematic reviews
python train_with_custom_data.py
# Choose option 3
```

## 📁 File Structure After Selection

```
data/
├── reviews.csv                    # Original data
├── reviews_backup.csv            # Backup of original
├── balanced_training_sample.csv  # 1,001 balanced reviews
├── high_quality_reviews.csv      # 16,424 quality reviews
├── problematic_reviews.csv       # 1,430 spam/ads
├── specific_business_reviews.csv # 300 business-specific
└── custom_training_set.csv       # 501 custom combination

models/review_quality/
├── review_quality_ensemble_balanced.pkl
├── review_quality_xgboost_balanced.pkl
├── review_quality_random_forest_balanced.pkl
└── ... (other model variations)
```

This approach gives you complete control over your training data and allows you to create models specifically tailored to detect the types of reviews you're most interested in!
