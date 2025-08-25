#!/usr/bin/env python3
"""
Policy enforcement module for review quality detection
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Any, Tuple
import logging
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PolicyViolation:
    """Data class for policy violations"""
    violation_type: str
    severity: float  # 0.0 to 1.0
    description: str
    detected_patterns: List[str]
    confidence: float  # 0.0 to 1.0

class PolicyEnforcementModule:
    """Enforce review policies and detect violations"""
    
    def __init__(self):
        # Policy violation patterns
        self.advertisement_patterns = [
            r'\b(buy\s+now|click\s+here|visit\s+our\s+website|call\s+now|limited\s+time)\b',
            r'\b(special\s+offer|discount|free\s+trial|act\s+now|don\'t\s+miss\s+out)\b',
            r'\b(exclusive\s+deal|money\s+back\s+guarantee|no\s+risk|100%\s+free)\b',
            r'\b(earn\s+money|work\s+from\s+home|make\s+money\s+fast|investment\s+opportunity)\b',
            r'\b(promotion|deal|sale|offer|bargain|cheap|affordable|best\s+price)\b'
        ]
        
        self.irrelevant_patterns = [
            r'\b(politics|election|president|government|news|weather)\b',
            r'\b(sports|football|basketball|baseball|soccer|tennis)\b',
            r'\b(movie|music|concert|theater|cinema|album|song)\b',
            r'\b(technology|computer|software|hardware|programming|coding)\b',
            r'\b(car|automotive|vehicle|driving|traffic|parking)\b',
            r'\b(health|medical|doctor|hospital|medicine|treatment)\b'
        ]
        
        self.rant_patterns = [
            r'\b(terrible|awful|horrible|worst|hate|disgusting|never\s+again)\b',
            r'\b(complaint|angry|furious|outraged|disappointed|frustrated)\b',
            r'\b(scam|fraud|fake|false|misleading|deceptive)\b',
            r'\b(waste\s+of\s+money|ripoff|overpriced|expensive|costly)\b'
        ]
        
        self.spam_patterns = [
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Phone numbers
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email addresses
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',  # URLs
            r'\b(www\.|\.com|\.org|\.net|\.edu)\b'  # Website indicators
        ]
        
        # Excessive patterns
        self.excessive_patterns = {
            'exclamation_marks': r'!{3,}',  # 3 or more exclamation marks
            'question_marks': r'\?{3,}',    # 3 or more question marks
            'capital_letters': r'[A-Z]{5,}', # 5 or more consecutive capital letters
            'repeated_words': r'\b(\w+)(?:\s+\1){2,}\b'  # Same word repeated 3+ times
        }
    
    def check_advertisement_policy(self, text: str) -> PolicyViolation:
        """Check for advertisement/promotional content"""
        text_lower = text.lower()
        detected_patterns = []
        total_matches = 0
        
        for pattern in self.advertisement_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                detected_patterns.extend(matches)
                total_matches += len(matches)
        
        severity = min(1.0, total_matches / 3.0)  # Normalize to 0-1
        confidence = min(1.0, total_matches / 2.0)
        
        return PolicyViolation(
            violation_type="advertisement",
            severity=severity,
            description="Contains promotional or advertisement content",
            detected_patterns=detected_patterns,
            confidence=confidence
        )
    
    def check_irrelevant_content_policy(self, text: str) -> PolicyViolation:
        """Check for irrelevant content"""
        text_lower = text.lower()
        detected_patterns = []
        total_matches = 0
        
        for pattern in self.irrelevant_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                detected_patterns.extend(matches)
                total_matches += len(matches)
        
        severity = min(1.0, total_matches / 2.0)
        confidence = min(1.0, total_matches / 1.5)
        
        return PolicyViolation(
            violation_type="irrelevant_content",
            severity=severity,
            description="Contains content unrelated to the business being reviewed",
            detected_patterns=detected_patterns,
            confidence=confidence
        )
    
    def check_rant_policy(self, text: str) -> PolicyViolation:
        """Check for rants or excessive complaints"""
        text_lower = text.lower()
        detected_patterns = []
        total_matches = 0
        
        for pattern in self.rant_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                detected_patterns.extend(matches)
                total_matches += len(matches)
        
        severity = min(1.0, total_matches / 2.0)
        confidence = min(1.0, total_matches / 1.5)
        
        return PolicyViolation(
            violation_type="rant",
            severity=severity,
            description="Contains excessive complaints or rant-like content",
            detected_patterns=detected_patterns,
            confidence=confidence
        )
    
    def check_spam_policy(self, text: str) -> PolicyViolation:
        """Check for spam indicators"""
        detected_patterns = []
        total_matches = 0
        
        for pattern in self.spam_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                detected_patterns.extend(matches)
                total_matches += len(matches)
        
        severity = min(1.0, total_matches / 2.0)
        confidence = min(1.0, total_matches / 1.5)
        
        return PolicyViolation(
            violation_type="spam",
            severity=severity,
            description="Contains spam indicators (phone numbers, emails, URLs)",
            detected_patterns=detected_patterns,
            confidence=confidence
        )
    
    def check_excessive_patterns(self, text: str) -> PolicyViolation:
        """Check for excessive use of punctuation, capitalization, etc."""
        detected_patterns = []
        total_violations = 0
        
        for pattern_name, pattern in self.excessive_patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                detected_patterns.append(f"{pattern_name}: {len(matches)} instances")
                total_violations += len(matches)
        
        severity = min(1.0, total_violations / 3.0)
        confidence = min(1.0, total_violations / 2.0)
        
        return PolicyViolation(
            violation_type="excessive_patterns",
            severity=severity,
            description="Contains excessive punctuation, capitalization, or repeated words",
            detected_patterns=detected_patterns,
            confidence=confidence
        )
    
    def check_user_visit_indicators(self, text: str, rating: int, user_history: Dict[str, Any]) -> PolicyViolation:
        """Check if user likely visited the location based on content and context"""
        
        # Positive indicators of actual visit
        visit_indicators = [
            r'\b(visited|went|ate|dined|ordered|tried|tasted|experienced)\b',
            r'\b(food|dish|meal|service|staff|atmosphere|ambiance|location)\b',
            r'\b(price|cost|value|portion|size|quality|fresh|delicious)\b',
            r'\b(wait|time|reservation|table|seating|parking|accessibility)\b'
        ]
        
        # Negative indicators (suggests no visit)
        no_visit_indicators = [
            r'\b(heard|read|saw|looks|appears|seems|probably|maybe)\b',
            r'\b(never\s+been|haven\'t\s+visited|didn\'t\s+go|avoided)\b',
            r'\b(reviews\s+say|people\s+tell|friends\s+said|online\s+reviews)\b'
        ]
        
        text_lower = text.lower()
        visit_score = 0
        no_visit_score = 0
        
        for pattern in visit_indicators:
            visit_score += len(re.findall(pattern, text_lower))
        
        for pattern in no_visit_indicators:
            no_visit_score += len(re.findall(pattern, text_lower))
        
        # Consider user history
        if user_history.get('user_review_count', 0) < 2:
            no_visit_score += 1  # New users more likely to not have visited
        
        if user_history.get('user_avg_rating', 0) == 0:
            no_visit_score += 1  # No previous ratings
        
        # Calculate probability of no visit
        total_indicators = visit_score + no_visit_score
        if total_indicators == 0:
            no_visit_prob = 0.3  # Default probability
        else:
            no_visit_prob = no_visit_score / total_indicators
        
        severity = no_visit_prob
        confidence = min(1.0, total_indicators / 5.0)
        
        return PolicyViolation(
            violation_type="no_visit_indicated",
            severity=severity,
            description="Content suggests user may not have actually visited the location",
            detected_patterns=[f"visit_score: {visit_score}", f"no_visit_score: {no_visit_score}"],
            confidence=confidence
        )
    
    def check_all_policies(self, text: str, rating: int = None, user_history: Dict[str, Any] = None) -> Dict[str, PolicyViolation]:
        """Check all policies and return violations"""
        
        violations = {}
        
        # Check each policy
        violations['advertisement'] = self.check_advertisement_policy(text)
        violations['irrelevant_content'] = self.check_irrelevant_content_policy(text)
        violations['rant'] = self.check_rant_policy(text)
        violations['spam'] = self.check_spam_policy(text)
        violations['excessive_patterns'] = self.check_excessive_patterns(text)
        
        # Check visit indicators if user history is provided
        if user_history is not None:
            violations['no_visit_indicated'] = self.check_user_visit_indicators(text, rating, user_history)
        
        return violations
    
    def get_overall_violation_score(self, violations: Dict[str, PolicyViolation]) -> Dict[str, Any]:
        """Calculate overall violation score and recommendations"""
        
        if not violations:
            return {
                'overall_score': 0.0,
                'severity_level': 'none',
                'recommendation': 'approve',
                'violation_count': 0,
                'critical_violations': []
            }
        
        # Calculate weighted overall score
        weights = {
            'spam': 0.3,
            'advertisement': 0.25,
            'irrelevant_content': 0.2,
            'rant': 0.15,
            'excessive_patterns': 0.05,
            'no_visit_indicated': 0.05
        }
        
        overall_score = 0.0
        critical_violations = []
        
        for violation_type, violation in violations.items():
            weight = weights.get(violation_type, 0.1)
            weighted_score = violation.severity * weight
            overall_score += weighted_score
            
            # Mark as critical if severity > 0.7
            if violation.severity > 0.7:
                critical_violations.append(violation_type)
        
        # Determine severity level
        if overall_score >= 0.7:
            severity_level = 'high'
            recommendation = 'reject'
        elif overall_score >= 0.4:
            severity_level = 'medium'
            recommendation = 'flag_for_review'
        elif overall_score >= 0.2:
            severity_level = 'low'
            recommendation = 'approve_with_warning'
        else:
            severity_level = 'none'
            recommendation = 'approve'
        
        return {
            'overall_score': overall_score,
            'severity_level': severity_level,
            'recommendation': recommendation,
            'violation_count': len(violations),
            'critical_violations': critical_violations,
            'detailed_violations': violations
        }
    
    def enforce_policies_on_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply policy enforcement to entire dataframe"""
        
        logger.info("Applying policy enforcement to reviews...")
        
        results = []
        
        for idx, row in df.iterrows():
            # Create user history dict
            user_history = {
                'user_review_count': row.get('user_review_count', 0),
                'user_avg_rating': row.get('user_avg_rating', 0),
                'user_rating_std': row.get('user_rating_std', 0)
            }
            
            # Check all policies
            violations = self.check_all_policies(
                row['text'], 
                row.get('rating'), 
                user_history
            )
            
            # Get overall score
            overall_result = self.get_overall_violation_score(violations)
            
            # Store results
            result = {
                'overall_violation_score': overall_result['overall_score'],
                'severity_level': overall_result['severity_level'],
                'recommendation': overall_result['recommendation'],
                'violation_count': overall_result['violation_count'],
                'critical_violations': ','.join(overall_result['critical_violations']),
                'has_advertisement': violations['advertisement'].severity > 0,
                'has_irrelevant_content': violations['irrelevant_content'].severity > 0,
                'has_rant': violations['rant'].severity > 0,
                'has_spam': violations['spam'].severity > 0,
                'has_excessive_patterns': violations['excessive_patterns'].severity > 0
            }
            
            results.append(result)
        
        # Add results to dataframe
        results_df = pd.DataFrame(results, index=df.index)
        df_with_policies = pd.concat([df, results_df], axis=1)
        
        logger.info("Policy enforcement completed!")
        
        return df_with_policies

if __name__ == "__main__":
    # Test the policy enforcement module
    policy_module = PolicyEnforcementModule()
    
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
        
        violations = policy_module.check_all_policies(
            test_case['text'], 
            test_case['rating'], 
            test_case['user_history']
        )
        
        overall_result = policy_module.get_overall_violation_score(violations)
        
        print(f"\nOverall Score: {overall_result['overall_score']:.3f}")
        print(f"Severity Level: {overall_result['severity_level']}")
        print(f"Recommendation: {overall_result['recommendation']}")
        print(f"Violations: {overall_result['violation_count']}")
        
        for violation_type, violation in violations.items():
            if violation.severity > 0:
                print(f"  - {violation_type}: {violation.severity:.3f} ({violation.description})")
