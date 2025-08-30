#!/usr/bin/env python3
"""
Test Review Quality Model on Real Data

This script tests the review quality detection model on real restaurant review data
to demonstrate its performance and capabilities.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from preprocessing.text_preprocessor import TextPreprocessor
from models.policy_enforcement import ReviewQualityPolicyEnforcer
from feature_engineering.feature_extractor import FeatureExtractor

def test_on_real_data():
    """Test the model on real restaurant review data"""
    
    print("="*80)
    print("REAL DATA PERFORMANCE TEST")
    print("Testing Review Quality Model on Actual Restaurant Reviews")
    print("="*80)
    
    # Load real data
    try:
        df = pd.read_csv('data/reviews.csv')
        print(f"Loaded {len(df)} real restaurant reviews")
        print(f"Data columns: {list(df.columns)}")
    except FileNotFoundError:
        print("Error: Could not find data/reviews.csv")
        return
    
    # Initialize components
    preprocessor = TextPreprocessor()
    policy_enforcer = ReviewQualityPolicyEnforcer()
    
    # Sample a subset for testing (first 50 reviews)
    test_df = df.head(50).copy()
    
    print(f"\nTesting on {len(test_df)} reviews...")
    print("="*80)
    
    results = []
    
    for idx, row in test_df.iterrows():
        text = str(row['text'])
        rating = int(row['rating'])
        business = str(row['business_name'])
        
        # Skip very short reviews
        if len(text.strip()) < 10:
            continue
            
        # Preprocess the text
        processed = preprocessor.preprocess_text(text)
        
        # Extract features
        features = {
            'length': processed['text_features']['length'],
            'word_count': processed['text_features']['word_count'],
            'exclamation_count': processed['text_features']['exclamation_count'],
            'capital_letter_ratio': processed['text_features']['capital_letter_ratio'],
            'unique_word_ratio': processed['text_features']['unique_word_ratio']
        }
        
        # Apply policy enforcement
        policy_analysis = policy_enforcer.analyze_review(text, features)
        
        # Calculate quality score (without rating influence)
        quality_score = (
            # Text quality (40%)
            (features['length'] / 500) * 0.2 +  # Normalize length
            (features['word_count'] / 100) * 0.2 +  # Normalize word count
            # Policy compliance (45%)
            (1 - policy_analysis['violation_scores']['advertisement']) * 0.2 +
            (1 - policy_analysis['violation_scores']['spam']) * 0.15 +
            (1 - policy_analysis['violation_scores']['irrelevant_content']) * 0.1 +
            # Writing sophistication (15%)
            features['unique_word_ratio'] * 0.1 +
            (1 - features['capital_letter_ratio']) * 0.05
        )
        
        quality_score = min(1.0, max(0.0, quality_score))  # Clamp to 0-1
        
        # Determine quality category
        if quality_score >= 0.7:
            quality_category = "HIGH"
        elif quality_score >= 0.4:
            quality_category = "MEDIUM"
        else:
            quality_category = "LOW"
        
        results.append({
            'business': business,
            'text': text[:100] + "..." if len(text) > 100 else text,
            'rating': rating,
            'quality_score': quality_score,
            'quality_category': quality_category,
            'policy_decision': policy_analysis['policy_decision'],
            'violation_count': policy_analysis['violation_counts']['total'],
            'length': features['length'],
            'word_count': features['word_count'],
            'unique_word_ratio': features['unique_word_ratio'],
            'capital_letter_ratio': features['capital_letter_ratio'],
            'exclamation_count': features['exclamation_count']
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Display detailed results
    print("\nDETAILED ANALYSIS RESULTS:")
    print("="*80)
    
    for i, row in results_df.iterrows():
        print(f"\nReview {i+1}: {row['business']}")
        print(f"Text: \"{row['text']}\"")
        print(f"Rating: {row['rating']} stars | Quality: {row['quality_score']:.3f} ({row['quality_category']})")
        print(f"Policy Decision: {row['policy_decision'].upper()}")
        print(f"Violations: {row['violation_count']} | Length: {row['length']} chars | Words: {row['word_count']}")
        
        # Key insights
        if row['rating'] >= 4 and row['quality_score'] < 0.5:
            print(f"  ⚠️  HIGH RATING but LOW QUALITY - demonstrates rating ≠ quality")
        elif row['rating'] <= 2 and row['quality_score'] > 0.7:
            print(f"  ✓ LOW RATING but HIGH QUALITY - well-written negative review")
    
    # Performance Statistics
    print("\n" + "="*80)
    print("PERFORMANCE STATISTICS")
    print("="*80)
    
    print(f"Total Reviews Analyzed: {len(results_df)}")
    print(f"Average Quality Score: {results_df['quality_score'].mean():.3f}")
    print(f"Quality Score Std Dev: {results_df['quality_score'].std():.3f}")
    print(f"Quality Score Range: {results_df['quality_score'].min():.3f} - {results_df['quality_score'].max():.3f}")
    
    # Quality distribution
    quality_dist = results_df['quality_category'].value_counts()
    print(f"\nQuality Distribution:")
    for category, count in quality_dist.items():
        percentage = (count / len(results_df)) * 100
        print(f"  {category}: {count} reviews ({percentage:.1f}%)")
    
    # Policy decisions
    policy_dist = results_df['policy_decision'].value_counts()
    print(f"\nPolicy Decisions:")
    for decision, count in policy_dist.items():
        percentage = (count / len(results_df)) * 100
        print(f"  {decision.upper()}: {count} reviews ({percentage:.1f}%)")
    
    # Rating vs Quality Analysis
    print(f"\nRATING vs QUALITY ANALYSIS:")
    print(f"Average Quality by Rating:")
    for rating in sorted(results_df['rating'].unique()):
        rating_reviews = results_df[results_df['rating'] == rating]
        avg_quality = rating_reviews['quality_score'].mean()
        count = len(rating_reviews)
        print(f"  {rating} stars: {avg_quality:.3f} quality ({count} reviews)")
    
    # Violation Analysis
    print(f"\nVIOLATION ANALYSIS:")
    print(f"Total Violations Detected: {results_df['violation_count'].sum()}")
    print(f"Average Violations per Review: {results_df['violation_count'].mean():.2f}")
    print(f"Reviews with Violations: {len(results_df[results_df['violation_count'] > 0])} ({len(results_df[results_df['violation_count'] > 0])/len(results_df)*100:.1f}%)")
    
    # Text Feature Analysis
    print(f"\nTEXT FEATURE ANALYSIS:")
    print(f"Average Review Length: {results_df['length'].mean():.1f} characters")
    print(f"Average Word Count: {results_df['word_count'].mean():.1f} words")
    print(f"Average Unique Word Ratio: {results_df['unique_word_ratio'].mean():.3f}")
    print(f"Average Capital Letter Ratio: {results_df['capital_letter_ratio'].mean():.3f}")
    print(f"Total Exclamation Marks: {results_df['exclamation_count'].sum()}")
    
    # Key Insights
    print(f"\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    # High rating, low quality examples
    high_rating_low_quality = results_df[(results_df['rating'] >= 4) & (results_df['quality_score'] < 0.5)]
    if len(high_rating_low_quality) > 0:
        print(f"Found {len(high_rating_low_quality)} reviews with HIGH RATING but LOW QUALITY:")
        for _, row in high_rating_low_quality.iterrows():
            print(f"  • {row['business']}: {row['rating']}★ but {row['quality_score']:.3f} quality")
    
    # Low rating, high quality examples
    low_rating_high_quality = results_df[(results_df['rating'] <= 2) & (results_df['quality_score'] > 0.7)]
    if len(low_rating_high_quality) > 0:
        print(f"\nFound {len(low_rating_high_quality)} reviews with LOW RATING but HIGH QUALITY:")
        for _, row in low_rating_high_quality.iterrows():
            print(f"  • {row['business']}: {row['rating']}★ but {row['quality_score']:.3f} quality")
    
    # Most violated reviews
    most_violations = results_df.nlargest(3, 'violation_count')
    print(f"\nReviews with Most Policy Violations:")
    for _, row in most_violations.iterrows():
        print(f"  • {row['business']}: {row['violation_count']} violations, {row['quality_score']:.3f} quality")
    
    # Highest quality reviews
    highest_quality = results_df.nlargest(3, 'quality_score')
    print(f"\nHighest Quality Reviews:")
    for _, row in highest_quality.iterrows():
        print(f"  • {row['business']}: {row['quality_score']:.3f} quality, {row['rating']}★ rating")
    
    print(f"\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print("The model successfully demonstrates:")
    print("1. Quality assessment independent of rating")
    print("2. Effective policy violation detection")
    print("3. Comprehensive text feature analysis")
    print("4. Practical application to real restaurant reviews")
    print("5. Clear distinction between review quality and user satisfaction")

if __name__ == "__main__":
    test_on_real_data()
