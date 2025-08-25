#!/usr/bin/env python3
"""
Data Exploration - Google Maps Restaurant Reviews

This script explores the Google Maps restaurant reviews dataset to understand 
the data structure, quality, and characteristics.
"""

import sys
sys.path.append('../src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from preprocessing.data_loader import ReviewDataLoader
from preprocessing.text_preprocessor import TextPreprocessor
from feature_engineering.feature_extractor import FeatureExtractor

# Set up plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 100)

def main():
    print("="*60)
    print("DATA EXPLORATION - GOOGLE MAPS RESTAURANT REVIEWS")
    print("="*60)
    
    # 1. Load and examine data
    print("\n1. Loading and examining data...")
    loader = ReviewDataLoader()
    reviews_df, restaurant_df = loader.load_data()
    
    print(f"Reviews dataset shape: {reviews_df.shape}")
    print(f"Restaurant dataset shape: {restaurant_df.shape}")
    print(f"Reviews dataset columns: {reviews_df.columns.tolist()}")
    
    # Display basic information
    info = loader.get_data_info()
    print("\nDataset Information:")
    for key, value in info.items():
        if key != 'sample_reviews':
            print(f"{key}: {value}")
    
    # Display sample reviews
    loader.display_sample_reviews(3)
    
    # 2. Data quality analysis
    print("\n2. Analyzing data quality...")
    
    # Check for missing values
    missing_data = reviews_df.isnull().sum()
    missing_percent = (missing_data / len(reviews_df)) * 100
    
    missing_df = pd.DataFrame({
        'Missing Values': missing_data,
        'Percentage': missing_percent
    })
    
    print("Missing Values Analysis:")
    print(missing_df[missing_df['Missing Values'] > 0])
    
    # Analyze text length distribution
    reviews_df['text_length'] = reviews_df['text'].str.len()
    reviews_df['word_count'] = reviews_df['text'].str.split().str.len()
    
    print(f"\nText length statistics:")
    print(reviews_df['text_length'].describe())
    print(f"\nWord count statistics:")
    print(reviews_df['word_count'].describe())
    
    # 3. Rating analysis
    print("\n3. Analyzing ratings...")
    
    rating_counts = reviews_df['rating'].value_counts().sort_index()
    category_counts = reviews_df['rating_category'].value_counts()
    
    print("Rating distribution:")
    print(rating_counts)
    print("\nRating category distribution:")
    print(category_counts)
    
    # 4. Business and user analysis
    print("\n4. Analyzing businesses and users...")
    
    # Business analysis
    business_stats = reviews_df.groupby('business_name').agg({
        'rating': ['count', 'mean', 'std'],
        'author_name': 'nunique'
    }).round(2)
    
    business_stats.columns = ['review_count', 'avg_rating', 'rating_std', 'unique_authors']
    business_stats = business_stats.sort_values('review_count', ascending=False)
    
    print("Top 10 businesses by review count:")
    print(business_stats.head(10))
    
    # User analysis
    user_stats = reviews_df.groupby('author_name').agg({
        'rating': ['count', 'mean', 'std'],
        'business_name': 'nunique'
    }).round(2)
    
    user_stats.columns = ['review_count', 'avg_rating', 'rating_std', 'unique_businesses']
    user_stats = user_stats.sort_values('review_count', ascending=False)
    
    print("\nTop 10 users by review count:")
    print(user_stats.head(10))
    
    # 5. Text preprocessing and analysis
    print("\n5. Preprocessing text data...")
    
    preprocessor = TextPreprocessor()
    processed_df = preprocessor.preprocess_dataframe(reviews_df)
    
    print(f"Text preprocessing completed!")
    print(f"New features added: {len(processed_df.columns) - len(reviews_df.columns)}")
    
    # Analyze sentiment
    print(f"\nSentiment statistics:")
    print(processed_df[['sentiment_polarity', 'sentiment_subjectivity']].describe())
    
    # Analyze spam and irrelevant content
    spam_analysis = processed_df['is_spam'].value_counts()
    irrelevant_analysis = processed_df['is_irrelevant'].value_counts()
    
    print(f"\nSpam detected: {spam_analysis.get(True, 0)} reviews ({spam_analysis.get(True, 0)/len(processed_df)*100:.1f}%)")
    print(f"Irrelevant content detected: {irrelevant_analysis.get(True, 0)} reviews ({irrelevant_analysis.get(True, 0)/len(processed_df)*100:.1f}%)")
    
    # 6. Feature engineering
    print("\n6. Extracting features...")
    
    extractor = FeatureExtractor(max_features=1000, n_topics=5)
    features, labels = extractor.extract_all_features(processed_df)
    
    print(f"Feature matrix shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Quality score range: {labels['quality_score'].min():.3f} - {labels['quality_score'].max():.3f}")
    print(f"High quality reviews: {labels['is_high_quality'].sum()} / {len(labels)} ({labels['is_high_quality'].sum()/len(labels)*100:.1f}%)")
    
    # Analyze feature importance
    importance_analysis = extractor.get_feature_importance_analysis(processed_df)
    
    print("\nTop 10 features by correlation:")
    top_features = list(importance_analysis['top_correlated_features'].items())[:10]
    for feature, corr in top_features:
        print(f"  {feature}: {corr:.3f}")
    
    # 7. Quality score analysis
    print("\n7. Analyzing quality scores...")
    
    print("Quality score statistics:")
    print(labels['quality_score'].describe())
    
    quality_categories = labels['quality_category'].value_counts()
    print("\nQuality category distribution:")
    print(quality_categories)
    
    # Analyze relationship between quality and rating
    correlation = processed_df['rating'].corr(labels['quality_score'])
    print(f"\nCorrelation between rating and quality score: {correlation:.3f}")
    
    # 8. Summary
    print("\n8. Summary and insights...")
    
    summary_stats = {
        'Total Reviews': len(reviews_df),
        'Unique Businesses': reviews_df['business_name'].nunique(),
        'Unique Users': reviews_df['author_name'].nunique(),
        'Average Rating': reviews_df['rating'].mean(),
        'Average Text Length': reviews_df['text'].str.len().mean(),
        'Average Word Count': reviews_df['text'].str.split().str.len().mean(),
        'Spam Detected (%)': (processed_df['is_spam'].sum() / len(processed_df)) * 100,
        'Irrelevant Content (%)': (processed_df['is_irrelevant'].sum() / len(processed_df)) * 100,
        'High Quality Reviews (%)': (labels['is_high_quality'].sum() / len(labels)) * 100,
        'Average Quality Score': labels['quality_score'].mean()
    }
    
    print("Dataset Summary:")
    for key, value in summary_stats.items():
        if 'Average' in key or 'Score' in key:
            print(f"  {key}: {value:.2f}")
        elif '%' in key:
            print(f"  {key}: {value:.1f}%")
        else:
            print(f"  {key}: {value}")
    
    # Save processed data
    print("\nSaving processed data...")
    processed_df.to_csv('../data/processed_reviews.csv', index=False)
    features.to_csv('../data/features.csv', index=False)
    labels.to_csv('../data/labels.csv', index=False)
    
    print("Processed data saved to data/ directory")
    print("\nData exploration completed!")

if __name__ == "__main__":
    main()
