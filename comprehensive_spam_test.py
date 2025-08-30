#!/usr/bin/env python3
"""
Comprehensive test of spam, advertisement, and irrelevant review detection
"""

import sys
sys.path.append('src')

from models.policy_enforcement import ReviewQualityPolicyEnforcer
from preprocessing.text_preprocessor import TextPreprocessor
from feature_engineering.feature_extractor import FeatureExtractor

def test_comprehensive_spam_detection():
    """Test comprehensive spam, advertisement, and irrelevant content detection"""
    
    print("="*100)
    print("COMPREHENSIVE SPAM/ADVERTISEMENT/IRRELEVANT CONTENT DETECTION TEST")
    print("="*100)
    
    # Initialize components
    policy = ReviewQualityPolicyEnforcer()
    preprocessor = TextPreprocessor()
    
    # Comprehensive test cases covering various violation types
    test_cases = [
        # === SPAM & ADVERTISEMENTS ===
        {
            "review": "🔥🔥🔥 LIMITED TIME OFFER! 🔥🔥🔥 Buy our amazing products NOW! Call 1-800-SCAM-NOW! Earn $5000/week working from home!",
            "description": "Aggressive spam with emojis, phone numbers, and work-from-home scam"
        },
        {
            "review": "FREE TRIAL! NO CREDIT CARD REQUIRED! Visit www.scamwebsite.com for amazing deals! Click here to earn money fast!",
            "description": "Website promotion with free trial scam"
        },
        {
            "review": "💯💯💯 100% GUARANTEED RESULTS! 💯💯💯 Lose 20 pounds in 2 weeks! Special discount code: SCAM50. Order now!",
            "description": "Weight loss scam with guarantee and discount codes"
        },
        {
            "review": "INVESTMENT OPPORTUNITY! Double your money in 24 hours! Bitcoin trading signals! Join our Telegram group @scamgroup",
            "description": "Investment scam with cryptocurrency and social media"
        },
        {
            "review": "BUY ONE GET THREE FREE! LIMITED STOCK! HURRY UP! Special price only $9.99! Call 555-1234 NOW!",
            "description": "Sales pitch with urgency and phone number"
        },
        
        # === COMPETITOR PROMOTION ===
        {
            "review": "This place is okay but you should definitely try Joe's Pizza down the street instead. Much better food and service!",
            "description": "Subtle competitor promotion"
        },
        {
            "review": "Terrible experience. Go to McDonald's across the street - they have better burgers and faster service.",
            "description": "Direct competitor recommendation"
        },
        {
            "review": "The food here is mediocre. My restaurant on Main Street has much better quality and lower prices.",
            "description": "Self-promotion of own business"
        },
        {
            "review": "Not worth it. The Chinese restaurant next door has authentic food and better atmosphere.",
            "description": "Specific competitor mention"
        },
        
        # === IRRELEVANT CONTENT ===
        {
            "review": "The weather today is beautiful! Perfect for a picnic. I love sunny days like this. The sky is so blue!",
            "description": "Weather discussion completely unrelated to restaurant"
        },
        {
            "review": "Did you see the football game last night? The quarterback threw an amazing touchdown pass! Sports are great!",
            "description": "Sports discussion unrelated to restaurant"
        },
        {
            "review": "I'm voting for the Democratic candidate in the upcoming election. Politics are so important these days.",
            "description": "Political content unrelated to restaurant"
        },
        {
            "review": "Just watched the new Marvel movie. It was incredible! The special effects were amazing. Hollywood is great!",
            "description": "Entertainment discussion unrelated to restaurant"
        },
        {
            "review": "My cat is so cute! She loves to play with yarn. Pets are the best companions ever!",
            "description": "Pet discussion unrelated to restaurant"
        },
        
        # === RANTS & EXCESSIVE COMPLAINTS ===
        {
            "review": "I HATE THIS PLACE SO MUCH! WORST EXPERIENCE EVER! EVERYTHING IS TERRIBLE! NEVER COMING BACK! STAFF IS RUDE! FOOD IS DISGUSTING!",
            "description": "Excessive caps and extreme negativity"
        },
        {
            "review": "This restaurant is a complete disaster. The manager is incompetent, the chef is terrible, the waiters are lazy, the food is inedible, the prices are outrageous, and the atmosphere is depressing. I want my money back!",
            "description": "Long rant with multiple complaints"
        },
        {
            "review": "ABSOLUTELY HORRIBLE! DISGUSTING! REPULSIVE! APPALLING! OUTRAGEOUS! UNACCEPTABLE!",
            "description": "Excessive negative adjectives"
        },
        
        # === MIXED VIOLATIONS ===
        {
            "review": "This place is terrible! The weather was nice though. You should try my friend's restaurant instead - they have great deals! Call 555-9999 for reservations!",
            "description": "Rant + weather + competitor promotion + phone number"
        },
        {
            "review": "I'm so angry about the election results! Also, this restaurant is bad. Visit www.betterfood.com for better options!",
            "description": "Politics + complaint + website promotion"
        },
        {
            "review": "FREE MONEY! EARN $1000/day! Also, this place sucks. Go to my cousin's restaurant - much better!",
            "description": "Money scam + complaint + family promotion"
        },
        
        # === SUBTLE VIOLATIONS ===
        {
            "review": "The food was okay but my brother's restaurant has better quality. You should check it out sometime.",
            "description": "Subtle family business promotion"
        },
        {
            "review": "Not impressed. The Italian place around the corner is much more authentic.",
            "description": "Subtle competitor mention"
        },
        {
            "review": "The service was slow. I prefer the Mexican restaurant down the block.",
            "description": "Subtle preference for competitor"
        },
        
        # === LEGITIMATE REVIEWS (CONTROL) ===
        {
            "review": "The food was delicious! The pasta was perfectly cooked and the sauce was flavorful. Great service too!",
            "description": "Legitimate positive review"
        },
        {
            "review": "The restaurant was clean and the staff was friendly. The food took a bit long to arrive but tasted good.",
            "description": "Legitimate mixed review"
        },
        {
            "review": "I didn't enjoy the meal. The steak was overcooked and the vegetables were cold. The service was also slow.",
            "description": "Legitimate negative review"
        }
    ]
    
    # Test each case
    results = []
    for i, case in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"TEST CASE {i}: {case['description']}")
        print(f"{'='*80}")
        print(f"Review: \"{case['review']}\"")
        
        # Analyze the review
        try:
            # Extract features
            features = {
                'length': len(case['review']),
                'word_count': len(case['review'].split()),
                'exclamation_count': case['review'].count('!'),
                'capital_letter_ratio': sum(1 for c in case['review'] if c.isupper()) / max(len(case['review']), 1),
                'unique_word_ratio': len(set(case['review'].split())) / max(len(case['review'].split()), 1)
            }
            
            # Get policy analysis
            analysis = policy.analyze_review(case['review'], features)
            
            print(f"\n📊 ANALYSIS RESULTS:")
            print(f"Policy Decision: {analysis['policy_decision'].upper()}")
            print(f"Reason: {analysis['reason']}")
            
            if analysis['violations']:
                print(f"\n🚨 VIOLATIONS DETECTED:")
                for violation in analysis['violations']:
                    print(f"  - {violation.violation_type.upper()} ({violation.severity}): {violation.description}")
            else:
                print(f"\n✅ NO VIOLATIONS DETECTED")
            
            # Store results
            results.append({
                'case': i,
                'description': case['description'],
                'decision': analysis['policy_decision'],
                'violations': len(analysis['violations']),
                'expected': 'violation' if 'spam' in case['description'].lower() or 'advertisement' in case['description'].lower() or 'irrelevant' in case['description'].lower() or 'rant' in case['description'].lower() else 'legitimate'
            })
            
        except Exception as e:
            print(f"❌ ERROR: {e}")
            results.append({
                'case': i,
                'description': case['description'],
                'decision': 'ERROR',
                'violations': 0,
                'expected': 'unknown'
            })
    
    # Summary
    print(f"\n{'='*100}")
    print("SUMMARY OF RESULTS")
    print(f"{'='*100}")
    
    violation_cases = [r for r in results if r['expected'] == 'violation']
    legitimate_cases = [r for r in results if r['expected'] == 'legitimate']
    
    print(f"\n📊 VIOLATION CASES ({len(violation_cases)} total):")
    for result in violation_cases:
        status = "✅" if result['decision'] in ['reject', 'review'] else "❌"
        print(f"  {status} Case {result['case']}: {result['description']} -> {result['decision'].upper()}")
    
    print(f"\n📊 LEGITIMATE CASES ({len(legitimate_cases)} total):")
    for result in legitimate_cases:
        status = "✅" if result['decision'] in ['approve', 'approve_with_warning'] else "❌"
        print(f"  {status} Case {result['case']}: {result['description']} -> {result['decision'].upper()}")
    
    # Calculate accuracy
    correct_violations = len([r for r in violation_cases if r['decision'] in ['reject', 'review']])
    correct_legitimate = len([r for r in legitimate_cases if r['decision'] in ['approve', 'approve_with_warning']])
    
    violation_accuracy = correct_violations / len(violation_cases) if violation_cases else 0
    legitimate_accuracy = correct_legitimate / len(legitimate_cases) if legitimate_cases else 0
    overall_accuracy = (correct_violations + correct_legitimate) / len(results)
    
    print(f"\n🎯 ACCURACY METRICS:")
    print(f"  Violation Detection Accuracy: {violation_accuracy:.1%} ({correct_violations}/{len(violation_cases)})")
    print(f"  Legitimate Review Accuracy: {legitimate_accuracy:.1%} ({correct_legitimate}/{len(legitimate_cases)})")
    print(f"  Overall Accuracy: {overall_accuracy:.1%}")

if __name__ == "__main__":
    test_comprehensive_spam_detection()
