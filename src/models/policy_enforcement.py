#!/usr/bin/env python3
"""
Policy Enforcement Module for Review Quality Detection

This module enforces content policies for restaurant reviews, focusing on:
- No advertisements or promotional content
- No spam or irrelevant content  
- No excessive rants or complaints
- Quality writing standards
- Relevance to restaurant experience
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PolicyViolation:
    """Represents a policy violation"""
    violation_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    confidence: float
    affected_text: str = ""

class ReviewQualityPolicyEnforcer:
    """Enforces review quality policies and content standards"""
    
    def __init__(self):
        # Advertisement and promotional content patterns
        self.advertisement_patterns = [
            r'\b(buy|purchase|order|shop|store|website|online)\b',
            r'\b(sale|discount|offer|deal|promotion|limited|special|free|trial)\b',
            r'\b(call|phone|contact|email|message|text)\s+(now|today|immediately)\b',
            r'\b(visit|check|go\s+to)\s+(our|the)\s+(website|site|page|store)\b',
            r'\b(click|tap|follow|subscribe|like|share)\b',
            r'\b(earn|make|save|get|receive)\s+(money|cash|dollars|profit)\b',
            r'\b(work\s+from\s+home|home\s+based|remote\s+work)\b',
            r'\b(investment|opportunity|business|startup|entrepreneur)\b',
            r'\b(guarantee|warranty|refund|return|exchange)\b',
            r'\b(price|cost|fee|charge|payment|credit|loan)\b'
        ]
        
        # Spam indicators
        self.spam_patterns = [
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Phone numbers
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email addresses
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',  # URLs
            r'\b(viagra|cialis|weight\s+loss|diet\s+pills|supplements)\b',
            r'\b(casino|poker|betting|gambling|lottery)\b',
            r'\b(loan|mortgage|credit\s+card|debt|bankruptcy)\b'
        ]
        
        # Irrelevant content patterns
        self.irrelevant_patterns = [
            r'\b(politics|election|president|government|congress|senate)\b',
            r'\b(sports|football|basketball|baseball|soccer|tennis|golf)\b',
            r'\b(weather|climate|temperature|rain|snow|storm)\b',
            r'\b(news|headlines|breaking|latest|update)\b',
            r'\b(movie|film|actor|actress|director|cinema)\b',
            r'\b(music|song|album|artist|band|concert)\b',
            r'\b(technology|computer|software|hardware|gaming|video\s+game)\b',
            r'\b(car|automotive|vehicle|truck|motorcycle)\b',
            r'\b(health|medical|doctor|hospital|pharmacy|medicine)\b',
            r'\b(education|school|university|college|student|teacher)\b'
        ]
        
        # Rant and excessive complaint patterns
        self.rant_patterns = [
            r'\b(terrible|awful|horrible|worst|hate|disgusting|disgusted)\b',
            r'\b(never|again|ever|worst\s+ever|absolute\s+worst)\b',
            r'\b(complaint|angry|furious|outraged|livid|enraged)\b',
            r'\b(waste|wasted|threw\s+away|money\s+down\s+drain)\b',
            r'\b(disappointed|disappointing|let\s+down|failed)\b',
            r'\b(avoid|stay\s+away|never\s+go|boycott)\b',
            r'\b(ripoff|scam|fraud|cheat|cheated|robbed)\b',
            r'\b(disgusting|gross|nasty|filthy|dirty|unclean)\b'
        ]
        
        # Excessive punctuation and formatting
        self.excessive_patterns = [
            r'!{3,}',  # Multiple exclamation marks
            r'\?{3,}',  # Multiple question marks
            r'\.{3,}',  # Multiple periods
            r'[A-Z]{5,}',  # Excessive capitalization
            r'\b[A-Z]{3,}\b',  # All caps words
            r'[!?]{2,}',  # Mixed excessive punctuation
        ]
        
        # Minimum quality standards
        self.minimum_standards = {
            'min_length': 10,  # Minimum characters
            'min_words': 3,    # Minimum words
            'max_exclamation_ratio': 0.1,  # Max exclamation marks per character
            'max_capital_ratio': 0.3,      # Max capital letters per character
            'min_unique_word_ratio': 0.3   # Minimum unique words ratio
        }
    
    def detect_advertisements(self, text: str) -> List[PolicyViolation]:
        """Detect advertisement and promotional content"""
        violations = []
        text_lower = text.lower()
        
        for pattern in self.advertisement_patterns:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                confidence = min(1.0, len(match.group()) / 10)  # Longer matches = higher confidence
                violations.append(PolicyViolation(
                    violation_type="advertisement",
                    severity="high" if confidence > 0.7 else "medium",
                    description=f"Promotional content detected: '{match.group()}'",
                    confidence=confidence,
                    affected_text=match.group()
                ))
        
        return violations
    
    def detect_spam(self, text: str) -> List[PolicyViolation]:
        """Detect spam content"""
        violations = []
        text_lower = text.lower()
        
        for pattern in self.spam_patterns:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                confidence = 0.9 if 'phone' in pattern or 'email' in pattern or 'http' in pattern else 0.7
                violations.append(PolicyViolation(
                    violation_type="spam",
                    severity="critical" if confidence > 0.8 else "high",
                    description=f"Spam content detected: '{match.group()}'",
                    confidence=confidence,
                    affected_text=match.group()
                ))
        
        return violations
    
    def detect_irrelevant_content(self, text: str) -> List[PolicyViolation]:
        """Detect content irrelevant to restaurant experience"""
        violations = []
        text_lower = text.lower()
        
        for pattern in self.irrelevant_patterns:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                confidence = 0.8
                violations.append(PolicyViolation(
                    violation_type="irrelevant_content",
                    severity="medium",
                    description=f"Irrelevant content detected: '{match.group()}'",
                    confidence=confidence,
                    affected_text=match.group()
                ))
        
        return violations
    
    def detect_rants(self, text: str) -> List[PolicyViolation]:
        """Detect excessive rants and complaints"""
        violations = []
        text_lower = text.lower()
        
        for pattern in self.rant_patterns:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                confidence = 0.7
                violations.append(PolicyViolation(
                    violation_type="rant",
                    severity="medium",
                    description=f"Excessive complaint detected: '{match.group()}'",
                    confidence=confidence,
                    affected_text=match.group()
                ))
        
        return violations
    
    def detect_excessive_formatting(self, text: str) -> List[PolicyViolation]:
        """Detect excessive punctuation and formatting"""
        violations = []
        
        for pattern in self.excessive_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                confidence = min(1.0, len(match.group()) / 5)
                violations.append(PolicyViolation(
                    violation_type="excessive_formatting",
                    severity="low",
                    description=f"Excessive formatting detected: '{match.group()}'",
                    confidence=confidence,
                    affected_text=match.group()
                ))
        
        return violations
    
    def check_minimum_standards(self, text: str, features: Dict[str, Any]) -> List[PolicyViolation]:
        """Check if review meets minimum quality standards"""
        violations = []
        
        # Check minimum length
        if features.get('length', 0) < self.minimum_standards['min_length']:
            violations.append(PolicyViolation(
                violation_type="minimum_length",
                severity="low",
                description=f"Review too short: {features.get('length', 0)} characters",
                confidence=1.0
            ))
        
        # Check minimum words
        if features.get('word_count', 0) < self.minimum_standards['min_words']:
            violations.append(PolicyViolation(
                violation_type="minimum_words",
                severity="low",
                description=f"Review too short: {features.get('word_count', 0)} words",
                confidence=1.0
            ))
        
        # Check excessive exclamation marks
        exclamation_ratio = features.get('exclamation_count', 0) / max(features.get('length', 1), 1)
        if exclamation_ratio > self.minimum_standards['max_exclamation_ratio']:
            violations.append(PolicyViolation(
                violation_type="excessive_exclamations",
                severity="low",
                description=f"Too many exclamation marks: {exclamation_ratio:.2f} ratio",
                confidence=0.8
            ))
        
        # Check excessive capitalization
        if features.get('capital_letter_ratio', 0) > self.minimum_standards['max_capital_ratio']:
            violations.append(PolicyViolation(
                violation_type="excessive_capitalization",
                severity="low",
                description=f"Too much capitalization: {features.get('capital_letter_ratio', 0):.2f} ratio",
                confidence=0.8
            ))
        
        # Check unique word ratio
        if features.get('unique_word_ratio', 0) < self.minimum_standards['min_unique_word_ratio']:
            violations.append(PolicyViolation(
                violation_type="low_vocabulary",
                severity="medium",
                description=f"Low vocabulary diversity: {features.get('unique_word_ratio', 0):.2f} ratio",
                confidence=0.7
            ))
        
        return violations
    
    def analyze_review(self, text: str, features: Dict[str, Any]) -> Dict[str, Any]:
        """Complete policy analysis of a review"""
        
        # Detect all types of violations
        violations = []
        violations.extend(self.detect_advertisements(text))
        violations.extend(self.detect_spam(text))
        violations.extend(self.detect_irrelevant_content(text))
        violations.extend(self.detect_rants(text))
        violations.extend(self.detect_excessive_formatting(text))
        violations.extend(self.check_minimum_standards(text, features))
        
        # Calculate violation scores
        violation_scores = {
            'advertisement': 0.0,
            'spam': 0.0,
            'irrelevant_content': 0.0,
            'rant': 0.0,
            'excessive_formatting': 0.0,
            'minimum_standards': 0.0
        }
        
        for violation in violations:
            if violation.violation_type in violation_scores:
                violation_scores[violation.violation_type] = max(
                    violation_scores[violation.violation_type],
                    violation.confidence
                )
        
        # Determine overall policy compliance
        critical_violations = [v for v in violations if v.severity == 'critical']
        high_violations = [v for v in violations if v.severity == 'high']
        medium_violations = [v for v in violations if v.severity == 'medium']
        low_violations = [v for v in violations if v.severity == 'low']
        
        # Policy compliance decision
        if critical_violations:
            policy_decision = 'reject'
            reason = 'Critical policy violations detected'
        elif len(high_violations) >= 2:
            policy_decision = 'reject'
            reason = 'Multiple high-severity policy violations'
        elif len(high_violations) == 1 and len(medium_violations) >= 2:
            policy_decision = 'reject'
            reason = 'High and multiple medium-severity violations'
        elif len(high_violations) == 1:
            policy_decision = 'review'
            reason = 'High-severity policy violation detected'
        elif len(medium_violations) >= 3:
            policy_decision = 'review'
            reason = 'Multiple medium-severity violations'
        elif len(medium_violations) >= 1 or len(low_violations) >= 3:
            policy_decision = 'approve_with_warning'
            reason = 'Minor policy violations detected'
        else:
            policy_decision = 'approve'
            reason = 'No policy violations detected'
        
        return {
            'violations': violations,
            'violation_scores': violation_scores,
            'policy_decision': policy_decision,
            'reason': reason,
            'violation_counts': {
                'critical': len(critical_violations),
                'high': len(high_violations),
                'medium': len(medium_violations),
                'low': len(low_violations),
                'total': len(violations)
            }
        }
    
    def enforce_policies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply policy enforcement to entire dataset"""
        
        logger.info("Applying policy enforcement to reviews...")
        
        # Initialize policy columns
        df['policy_violations'] = None
        df['policy_decision'] = 'approve'
        df['policy_reason'] = 'No violations detected'
        df['violation_count'] = 0
        df['advertisement_score'] = 0.0
        df['spam_score'] = 0.0
        df['irrelevant_score'] = 0.0
        df['rant_score'] = 0.0
        
        # Analyze each review
        for idx, row in df.iterrows():
            text = row.get('cleaned_text', row.get('text', ''))
            
            # Extract features for this review
            features = {
                'length': row.get('text_length', len(text)),
                'word_count': row.get('word_count', len(text.split())),
                'exclamation_count': row.get('exclamation_count', text.count('!')),
                'capital_letter_ratio': row.get('capital_letter_ratio', sum(1 for c in text if c.isupper()) / max(len(text), 1)),
                'unique_word_ratio': row.get('unique_word_ratio', len(set(text.split())) / max(len(text.split()), 1))
            }
            
            # Analyze policy compliance
            analysis = self.analyze_review(text, features)
            
            # Update dataframe
            df.at[idx, 'policy_violations'] = analysis['violations']
            df.at[idx, 'policy_decision'] = analysis['policy_decision']
            df.at[idx, 'policy_reason'] = analysis['reason']
            df.at[idx, 'violation_count'] = analysis['violation_counts']['total']
            df.at[idx, 'advertisement_score'] = analysis['violation_scores']['advertisement']
            df.at[idx, 'spam_score'] = analysis['violation_scores']['spam']
            df.at[idx, 'irrelevant_score'] = analysis['violation_scores']['irrelevant_content']
            df.at[idx, 'rant_score'] = analysis['violation_scores']['rant']
        
        # Calculate policy compliance statistics
        decisions = df['policy_decision'].value_counts()
        logger.info(f"Policy enforcement completed:")
        logger.info(f"  - Approved: {decisions.get('approve', 0)} reviews")
        logger.info(f"  - Approved with warning: {decisions.get('approve_with_warning', 0)} reviews")
        logger.info(f"  - Under review: {decisions.get('review', 0)} reviews")
        logger.info(f"  - Rejected: {decisions.get('reject', 0)} reviews")
        
        return df

if __name__ == "__main__":
    # Test the policy enforcement module
    policy_module = ReviewQualityPolicyEnforcer()
    
    # Test cases
    test_cases = [
        {
            'text': "This restaurant is amazing! Best food ever!",
            'rating': 5,
            'user_history': {'user_review_count': 10, 'user_avg_rating': 4.2, 'user_rating_std': 0.8}
        },
        {
            'text': "BUY NOW!!! Special offer!!! Call 555-1234 for exclusive deals!!!",
            'rating': 1,
            'user_history': {'user_review_count': 1, 'user_avg_rating': 1.0, 'user_rating_std': 0.0}
        },
        {
            'text': "I heard this place is good. The weather was nice today and the election results are in.",
            'rating': 3,
            'user_history': {'user_review_count': 2, 'user_avg_rating': 3.0, 'user_rating_std': 1.0}
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"TEST CASE {i+1}")
        print(f"{'='*60}")
        print(f"Text: {test_case['text']}")
        print(f"Rating: {test_case['rating']}")
        
        # The new module does not use rating data, so we pass a dummy value or None
        # For demonstration, we'll pass a dummy rating or None
        # In a real scenario, you might pass the actual rating or None if not available
        # For now, we'll pass a dummy value to match the original test structure
        # The new module's analyze_review function does not use 'rating' or 'user_history'
        # as it focuses on content quality and policy violations.
        # The original test cases were designed for a rating-based policy enforcement.
        # To make the new module's test cases work, we'll pass dummy values or None.
        # Given the new module's analyze_review function, it doesn't use 'rating' or 'user_history'.
        # So, we can just pass None for both.
        
        analysis = policy_module.analyze_review(
            test_case['text'], 
            {
                'length': len(test_case['text']),
                'word_count': len(test_case['text'].split()),
                'exclamation_count': test_case['text'].count('!'),
                'capital_letter_ratio': sum(1 for c in test_case['text'] if c.isupper()) / max(len(test_case['text']), 1),
                'unique_word_ratio': len(set(test_case['text'].split())) / max(len(test_case['text'].split()), 1)
            }
        )
        
        print(f"\nOverall Policy Decision: {analysis['policy_decision']}")
        print(f"Reason: {analysis['reason']}")
        print(f"Violations: {analysis['violation_counts']['total']}")
        
        for violation in analysis['violations']:
            print(f"  - {violation.violation_type} ({violation.severity}): {violation.description} (Confidence: {violation.confidence:.2f})")
