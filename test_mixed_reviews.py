#!/usr/bin/env python3
"""
Test Mixed Reviews for Review Quality Detection Model

This script contains 10 carefully selected mixed reviews to test the model's ability
to distinguish between high-quality and low-quality reviews, regardless of rating.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing.text_preprocessor import TextPreprocessor
from models.policy_enforcement import ReviewQualityPolicyEnforcer
from feature_engineering.feature_extractor import FeatureExtractor

def test_mixed_reviews():
    """Test the model with 10 mixed reviews"""
    
    print("="*80)
    print("TESTING REVIEW QUALITY DETECTION MODEL")
    print("10 Mixed Reviews (Good vs Bad Quality)")
    print("="*80)
    
    # Initialize components
    preprocessor = TextPreprocessor()
    policy_enforcer = ReviewQualityPolicyEnforcer()
    feature_extractor = FeatureExtractor(max_features=1000, n_topics=5)
    
    # 10 carefully selected mixed reviews
    test_reviews = [
        {
            "id": 1,
            "text": "The food was absolutely delicious! The service was impeccable and the atmosphere was perfect. I had the grilled salmon with seasonal vegetables and it was cooked to perfection. The staff was attentive and friendly. I would definitely recommend this restaurant to anyone looking for a great dining experience.",
            "rating": 5,
            "expected_quality": "HIGH",
            "description": "High rating, high quality - detailed, constructive review"
        },
        {
            "id": 2,
            "text": "BUY NOW!!! SPECIAL OFFER!!! LIMITED TIME!!! CALL 555-1234!!! EARN MONEY FAST!!! VISIT OUR WEBSITE www.example.com!!!",
            "rating": 5,
            "expected_quality": "LOW",
            "description": "High rating, low quality - obvious spam/advertisement"
        },
        {
            "id": 3,
            "text": "The food was terrible and the service was awful. I hate this place and will never go back. This is the worst restaurant ever! The staff was rude and the food was cold. I want my money back!",
            "rating": 1,
            "expected_quality": "LOW",
            "description": "Low rating, low quality - excessive ranting and complaints"
        },
        {
            "id": 4,
            "text": "While the food wasn't quite what I expected, I appreciate the effort. The service was slow but the staff was trying their best. The atmosphere was pleasant. I think with some improvements, this place could be really good.",
            "rating": 2,
            "expected_quality": "HIGH",
            "description": "Low rating, high quality - constructive criticism"
        },
        {
            "id": 5,
            "text": "The weather was nice today and I watched a great football game. The election results were interesting too. Also, I went to this restaurant.",
            "rating": 3,
            "expected_quality": "LOW",
            "description": "Medium rating, low quality - irrelevant content"
        },
        {
            "id": 6,
            "text": "Delicious food, great service, reasonable prices. Will visit again.",
            "rating": 5,
            "expected_quality": "MEDIUM",
            "description": "High rating, medium quality - concise but relevant"
        },
        {
            "id": 7,
            "text": "I ordered the grilled chicken salad and it was perfectly cooked. The vegetables were fresh and the dressing was flavorful. The service was quick and friendly. The portion size was generous for the price. Highly recommend!",
            "rating": 4,
            "expected_quality": "HIGH",
            "description": "Medium rating, high quality - detailed and specific"
        },
        {
            "id": 8,
            "text": "FREE!!! FREE!!! FREE!!! JOIN NOW!!! MAKE MONEY!!! SPECIAL DEAL!!! LIMITED TIME!!! CALL NOW!!!",
            "rating": 5,
            "expected_quality": "LOW",
            "description": "High rating, low quality - promotional spam"
        },
        {
            "id": 9,
            "text": "The restaurant exceeded my expectations. The menu offered a great variety of options, and everything I tried was excellent. The staff was knowledgeable about the dishes and made great recommendations. The ambiance was perfect for both casual and special occasions.",
            "rating": 5,
            "expected_quality": "HIGH",
            "description": "High rating, high quality - comprehensive review"
        },
        {
            "id": 10,
            "text": "This place sucks. Worst food ever. Never coming back. Terrible service. Awful atmosphere. Hate it.",
            "rating": 1,
            "expected_quality": "LOW",
            "description": "Low rating, low quality - brief, unconstructive rant"
        }
    ]
    
    print("\nANALYZING 10 MIXED REVIEWS...")
    print("="*80)
    
    results = []
    
    for review in test_reviews:
        print(f"\nREVIEW {review['id']}: {review['description']}")
        print(f"Text: \"{review['text']}\"")
        print(f"Rating: {review['rating']} stars")
        print(f"Expected Quality: {review['expected_quality']}")
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
        
        # Calculate quality score (simplified version)
        quality_score = calculate_quality_score(policy_analysis, features)
        
        # Determine quality category
        if quality_score >= 0.7:
            quality_category = "HIGH"
        elif quality_score >= 0.4:
            quality_category = "MEDIUM"
        else:
            quality_category = "LOW"
        
        # Store results
        result = {
            'id': review['id'],
            'text': review['text'],
            'rating': review['rating'],
            'expected_quality': review['expected_quality'],
            'predicted_quality': quality_category,
            'quality_score': quality_score,
            'policy_decision': policy_analysis['policy_decision'],
            'violations': [v.violation_type for v in policy_analysis['violations']] if policy_analysis['violations'] else []
        }
        results.append(result)
        
        # Display analysis
        print(f"QUALITY ANALYSIS:")
        print(f"  Quality Score: {quality_score:.3f} ({quality_category})")
        print(f"  Policy Decision: {policy_analysis['policy_decision']}")
        print(f"  Violations: {', '.join([v.violation_type for v in policy_analysis['violations']]) if policy_analysis['violations'] else 'None'}")
        
        # Check if prediction matches expectation
        if quality_category == review['expected_quality']:
            print(f"  ✅ CORRECT: Predicted {quality_category}, Expected {review['expected_quality']}")
        else:
            print(f"  ❌ INCORRECT: Predicted {quality_category}, Expected {review['expected_quality']}")
        
        print(f"TEXT FEATURES:")
        print(f"  Length: {features['length']} characters")
        print(f"  Words: {features['word_count']}")
        print(f"  Unique word ratio: {features['unique_word_ratio']:.3f}")
        print(f"  Capital letter ratio: {features['capital_letter_ratio']:.3f}")
        print(f"  Exclamation marks: {features['exclamation_count']}")
        
        # Key insight
        if review['rating'] >= 4 and quality_category == "LOW":
            print(f"KEY INSIGHT:")
            print(f"  ⚠️  HIGH RATING ({review['rating']}) but LOW QUALITY ({quality_category})")
            print(f"     This demonstrates why rating should NOT be used for quality assessment!")
        elif review['rating'] <= 2 and quality_category == "HIGH":
            print(f"KEY INSIGHT:")
            print(f"  ✅ LOW RATING ({review['rating']}) but HIGH QUALITY ({quality_category})")
            print(f"     This shows constructive criticism can be high quality!")
    
    # Summary statistics
    print("\n" + "="*80)
    print("TEST RESULTS SUMMARY")
    print("="*80)
    
    correct_predictions = sum(1 for r in results if r['predicted_quality'] == r['expected_quality'])
    accuracy = correct_predictions / len(results) * 100
    
    print(f"Overall Accuracy: {accuracy:.1f}% ({correct_predictions}/{len(results)} correct)")
    
    # Quality distribution
    quality_counts = {}
    for result in results:
        quality = result['predicted_quality']
        quality_counts[quality] = quality_counts.get(quality, 0) + 1
    
    print(f"\nPredicted Quality Distribution:")
    for quality, count in quality_counts.items():
        print(f"  {quality}: {count} reviews")
    
    # Rating vs Quality analysis
    print(f"\nRating vs Quality Analysis:")
    for result in results:
        rating = result['rating']
        quality = result['predicted_quality']
        expected = result['expected_quality']
        status = "✅" if quality == expected else "❌"
        print(f"  {status} Rating {rating} → Quality {quality} (Expected: {expected})")
    
    # Save results to file
    results_df = pd.DataFrame(results)
    results_df.to_csv("test_results_mixed_reviews.csv", index=False)
    print(f"\n💾 Results saved to: test_results_mixed_reviews.csv")
    
    return results

def calculate_quality_score(policy_analysis, features):
    """Calculate a simplified quality score based on policy analysis and features"""
    
    # Start with base score
    score = 0.5
    
    # Adjust based on policy violations
    if policy_analysis['policy_decision'] == 'approve':
        score += 0.3
    elif policy_analysis['policy_decision'] == 'approve_with_warning':
        score += 0.1
    elif policy_analysis['policy_decision'] == 'review':
        score -= 0.1
    elif policy_analysis['policy_decision'] == 'reject':
        score -= 0.4
    
    # Adjust based on text features
    if features['length'] >= 100:
        score += 0.1
    elif features['length'] < 50:
        score -= 0.1
    
    if features['unique_word_ratio'] >= 0.7:
        score += 0.1
    elif features['unique_word_ratio'] < 0.5:
        score -= 0.1
    
    if features['exclamation_count'] <= 2:
        score += 0.1
    elif features['exclamation_count'] > 5:
        score -= 0.2
    
    if features['capital_letter_ratio'] <= 0.1:
        score += 0.1
    elif features['capital_letter_ratio'] > 0.3:
        score -= 0.1
    
    # Ensure score is between 0 and 1
    return max(0.0, min(1.0, score))

def main():
    """Main function to run the mixed reviews test"""
    try:
        results = test_mixed_reviews()
        print(f"\n🎉 Test completed successfully!")
        print(f"Check 'test_results_mixed_reviews.csv' for detailed results.")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
