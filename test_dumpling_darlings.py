#!/usr/bin/env python3
"""
Test the enhanced system with a specific real-world example
"""

import sys
sys.path.append('src')

from models.policy_enforcement import ReviewQualityPolicyEnforcer
from preprocessing.text_preprocessor import TextPreprocessor

def test_dumpling_darlings_example():
    """Test the specific Dumpling Darlings example"""
    
    print("="*80)
    print("TESTING ENHANCED SYSTEM WITH REAL-WORLD EXAMPLE")
    print("="*80)
    
    # Initialize components
    policy = ReviewQualityPolicyEnforcer()
    preprocessor = TextPreprocessor()
    
    # The specific test case
    test_review = "Dumpling Darlings sucks, my friend went there and told me that. I suggest coming to my store at Starbucks instead."
    
    print(f"📝 TEST REVIEW:")
    print(f"\"{test_review}\"")
    print(f"\n🏪 CONTEXT: Dumpling Darlings (real restaurant in Singapore)")
    print(f"🎯 EXPECTED: Should detect rant + competitor promotion + self-promotion")
    
    # Extract features
    features = {
        'length': len(test_review),
        'word_count': len(test_review.split()),
        'exclamation_count': test_review.count('!'),
        'capital_letter_ratio': sum(1 for c in test_review if c.isupper()) / max(len(test_review), 1),
        'unique_word_ratio': len(set(test_review.split())) / max(len(test_review.split()), 1)
    }
    
    print(f"\n📊 FEATURES EXTRACTED:")
    print(f"  Length: {features['length']} characters")
    print(f"  Word Count: {features['word_count']} words")
    print(f"  Exclamation Count: {features['exclamation_count']}")
    print(f"  Capital Letter Ratio: {features['capital_letter_ratio']:.2f}")
    print(f"  Unique Word Ratio: {features['unique_word_ratio']:.2f}")
    
    # Analyze the review
    try:
        analysis = policy.analyze_review(test_review, features)
        
        print(f"\n🎯 ANALYSIS RESULTS:")
        print(f"Policy Decision: {analysis['policy_decision'].upper()}")
        print(f"Reason: {analysis['reason']}")
        
        if analysis['violations']:
            print(f"\n🚨 VIOLATIONS DETECTED:")
            for violation in analysis['violations']:
                print(f"  - {violation.violation_type.upper()} ({violation.severity}): {violation.description}")
        else:
            print(f"\n✅ NO VIOLATIONS DETECTED")
        
        # Detailed analysis
        print(f"\n📈 VIOLATION SCORES:")
        for violation_type, score in analysis['violation_scores'].items():
            print(f"  {violation_type}: {score:.2f}")
        
        print(f"\n📊 VIOLATION COUNTS:")
        for severity, count in analysis['violation_counts'].items():
            print(f"  {severity}: {count}")
        
        # Assessment
        print(f"\n🎯 ASSESSMENT:")
        if analysis['policy_decision'] in ['reject', 'review']:
            print(f"✅ SUCCESS: System correctly flagged this review as problematic")
            if 'rant' in [v.violation_type for v in analysis['violations']]:
                print(f"✅ Rant detection: Working")
            if 'advertisement' in [v.violation_type for v in analysis['violations']]:
                print(f"✅ Competitor promotion detection: Working")
            if 'irrelevant_content' in [v.violation_type for v in analysis['violations']]:
                print(f"✅ Irrelevant content detection: Working")
        else:
            print(f"❌ FAILURE: System should have flagged this review")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    test_dumpling_darlings_example()
