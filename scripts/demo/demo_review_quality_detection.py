#!/usr/bin/env python3
"""
Review Quality Detection System - Interactive Demo

This demonstration showcases the advanced review quality detection capabilities
of our machine learning system, highlighting the distinction between review
quality and rating assessment.

Key Demonstrations:
- Text quality assessment independent of ratings
- Policy violation detection (spam, ads, irrelevant content)
- Content moderation and policy enforcement
- Quality scoring based on writing standards

Business Value:
- Automated content quality assurance
- Consistent policy enforcement
- Improved user experience through quality content
- Reduced manual moderation workload

Author: Review Quality Detection Team
Version: 2.0.0
Date: 2024
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from preprocessing.text_preprocessor import TextPreprocessor
from models.policy_enforcement import ReviewQualityPolicyEnforcer
from feature_engineering.feature_extractor import FeatureExtractor

def demo_review_quality_detection():
    """Demo the review quality detection system"""
    
    print("="*80)
    print("REVIEW QUALITY DETECTION DEMO")
    print("Focus: Text Quality and Policy Compliance (NO RATING DATA)")
    print("="*80)
    
    # Initialize components
    preprocessor = TextPreprocessor()
    policy_enforcer = ReviewQualityPolicyEnforcer()
    feature_extractor = FeatureExtractor(max_features=1000, n_topics=5)
    
    # Sample reviews for demonstration
    sample_reviews = [
        {
            "text": "This restaurant is AMAZING! Best food ever! Call now for special offers!!! Visit our website www.example.com",
            "rating": 5,
            "description": "High rating but contains advertisements"
        },
        {
            "text": "The food was terrible and the service was awful. I hate this place and will never go back. This is the worst restaurant ever!",
            "rating": 1,
            "description": "Low rating with excessive complaints"
        },
        {
            "text": "I had the grilled salmon with seasonal vegetables. The fish was perfectly cooked and the vegetables were fresh. The service was attentive and the atmosphere was pleasant. I would definitely recommend this restaurant.",
            "rating": 4,
            "description": "High rating with detailed, constructive review"
        },
        {
            "text": "The weather was nice today and I watched a great football game. The election results were interesting too.",
            "rating": 3,
            "description": "Medium rating but irrelevant content"
        },
        {
            "text": "Delicious food, great service, reasonable prices. Will visit again.",
            "rating": 5,
            "description": "High rating with concise, relevant review"
        },
        {
            "text": "BUY NOW!!! SPECIAL OFFER!!! LIMITED TIME!!! CALL 555-1234!!! EARN MONEY FAST!!!",
            "rating": 5,
            "description": "High rating but obvious spam"
        }
    ]
    
    print("\nANALYZING SAMPLE REVIEWS...")
    print("="*80)
    
    for i, review in enumerate(sample_reviews, 1):
        print(f"\nREVIEW {i}: {review['description']}")
        print(f"Text: \"{review['text']}\"")
        print(f"Rating: {review['rating']} stars")
        print("-" * 60)
        
        # Preprocess the text
        processed = preprocessor.preprocess_text(review['text'])
        
        # Extract features
        features = {
            'length': processed['text_features']['length'],
            'word_count': processed['text_features']['word_count'],
            'exclamation_count': processed['text_features']['exclamation_count'],
            'capital_letter_ratio': processed['text_features']['capital_letter_ratio'],
            'unique_word_ratio': processed['text_features']['unique_word_ratio']
        }
        
        # Apply policy enforcement
        policy_analysis = policy_enforcer.analyze_review(review['text'], features)
        
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
        
        # Display results
        print(f"QUALITY ANALYSIS:")
        print(f"  Quality Score: {quality_score:.3f} ({quality_category})")
        print(f"  Policy Decision: {policy_analysis['policy_decision'].upper()}")
        print(f"  Reason: {policy_analysis['reason']}")
        
        print(f"POLICY VIOLATIONS:")
        if policy_analysis['violations']:
            for violation in policy_analysis['violations']:
                print(f"  - {violation.violation_type.upper()} ({violation.severity}): {violation.description}")
        else:
            print("  - No violations detected")
        
        print(f"TEXT FEATURES:")
        print(f"  Length: {features['length']} characters")
        print(f"  Words: {features['word_count']}")
        print(f"  Unique word ratio: {features['unique_word_ratio']:.3f}")
        print(f"  Capital letter ratio: {features['capital_letter_ratio']:.3f}")
        print(f"  Exclamation marks: {features['exclamation_count']}")
        
        # Key insight: Rating vs Quality
        print(f"KEY INSIGHT:")
        if review['rating'] >= 4 and quality_score < 0.5:
            print(f"  ⚠️  HIGH RATING ({review['rating']}) but LOW QUALITY ({quality_score:.3f})")
            print(f"     This demonstrates why rating should NOT be used for quality assessment!")
        elif review['rating'] <= 2 and quality_score > 0.7:
            print(f"  ✓ LOW RATING ({review['rating']}) but HIGH QUALITY ({quality_score:.3f})")
            print(f"     This shows a well-written negative review!")
        else:
            print(f"  Rating ({review['rating']}) and Quality ({quality_score:.3f}) are aligned")
    
    print("\n" + "="*80)
    print("DEMO SUMMARY")
    print("="*80)
    print("This demo shows that:")
    print("1. Review QUALITY is independent of restaurant RATING")
    print("2. A 1-star review can be high quality (well-written, constructive)")
    print("3. A 5-star review can be low quality (spam, ads, irrelevant)")
    print("4. Quality should be based on text characteristics and policy compliance")
    print("5. Rating reflects user satisfaction, not review quality")
    
    print("\nPOLICY VIOLATIONS DETECTED:")
    print("- Advertisements: Promotional content, sales pitches")
    print("- Spam: Phone numbers, emails, URLs, suspicious patterns")
    print("- Irrelevant content: Politics, sports, weather, etc.")
    print("- Excessive rants: Overly negative, repetitive complaints")
    print("- Poor formatting: Excessive caps, punctuation, etc.")
    
    print("\nQUALITY INDICATORS:")
    print("- Text length and word count")
    print("- Vocabulary diversity")
    print("- Readability and grammar")
    print("- Policy compliance")
    print("- Writing sophistication")

def demo_batch_analysis():
    """Demo batch analysis of reviews"""
    
    print("\n" + "="*80)
    print("BATCH ANALYSIS DEMO")
    print("="*80)
    
    # Create a small batch of reviews
    batch_reviews = [
        "Great food and service! Highly recommend.",
        "BUY NOW!!! SPECIAL OFFER!!! CALL 555-1234!!!",
        "The weather was nice and I watched football. Politics are interesting.",
        "The food was terrible and I hate this place. Never going back!",
        "I ordered the grilled chicken salad. The chicken was perfectly cooked and the vegetables were fresh. The service was excellent and the atmosphere was pleasant. I would definitely return.",
        "Delicious!",
        "This restaurant exceeded my expectations. The menu offers a variety of options, and I particularly enjoyed the seafood pasta. The staff was friendly and attentive, and the ambiance was perfect for a business dinner. The prices were reasonable for the quality of food and service provided."
    ]
    
    print(f"Analyzing {len(batch_reviews)} reviews...")
    
    # Initialize components
    preprocessor = TextPreprocessor()
    policy_enforcer = ReviewQualityPolicyEnforcer()
    
    results = []
    
    for i, text in enumerate(batch_reviews, 1):
        # Preprocess
        processed = preprocessor.preprocess_text(text)
        
        # Extract features
        features = {
            'length': processed['text_features']['length'],
            'word_count': processed['text_features']['word_count'],
            'exclamation_count': processed['text_features']['exclamation_count'],
            'capital_letter_ratio': processed['text_features']['capital_letter_ratio'],
            'unique_word_ratio': processed['text_features']['unique_word_ratio']
        }
        
        # Policy analysis
        policy_analysis = policy_enforcer.analyze_review(text, features)
        
        # Quality score (without rating)
        quality_score = (
            (features['length'] / 500) * 0.2 +
            (features['word_count'] / 100) * 0.2 +
            (1 - policy_analysis['violation_scores']['advertisement']) * 0.2 +
            (1 - policy_analysis['violation_scores']['spam']) * 0.15 +
            (1 - policy_analysis['violation_scores']['irrelevant_content']) * 0.1 +
            features['unique_word_ratio'] * 0.1 +
            (1 - features['capital_letter_ratio']) * 0.05
        )
        quality_score = min(1.0, max(0.0, quality_score))
        
        results.append({
            'review_id': i,
            'text': text[:50] + "..." if len(text) > 50 else text,
            'quality_score': quality_score,
            'policy_decision': policy_analysis['policy_decision'],
            'violation_count': policy_analysis['violation_counts']['total'],
            'length': features['length'],
            'word_count': features['word_count']
        })
    
    # Display results
    results_df = pd.DataFrame(results)
    print("\nBATCH ANALYSIS RESULTS:")
    print(results_df.to_string(index=False))
    
    # Summary statistics
    print(f"\nSUMMARY STATISTICS:")
    print(f"Average quality score: {results_df['quality_score'].mean():.3f}")
    print(f"High quality reviews (score > 0.7): {len(results_df[results_df['quality_score'] > 0.7])}")
    print(f"Low quality reviews (score < 0.4): {len(results_df[results_df['quality_score'] < 0.4])}")
    print(f"Policy violations detected: {results_df['violation_count'].sum()} total")

if __name__ == "__main__":
    demo_review_quality_detection()
    demo_batch_analysis()
