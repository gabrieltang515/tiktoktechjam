#!/usr/bin/env python3
"""
Text preprocessing module for Google Maps restaurant reviews
"""

import re
import string
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextPreprocessor:
    """Text preprocessing for review quality detection"""
    
    def __init__(self, remove_stopwords: bool = True, lemmatize: bool = True):
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Common spam/advertisement indicators
        self.spam_indicators = [
            'buy now', 'click here', 'visit our website', 'call now', 'limited time',
            'special offer', 'discount', 'free trial', 'act now', 'don\'t miss out',
            'exclusive deal', 'money back guarantee', 'no risk', '100% free',
            'earn money', 'work from home', 'make money fast', 'investment opportunity'
        ]
        
        # Irrelevant content indicators
        self.irrelevant_indicators = [
            'politics', 'election', 'president', 'government', 'news', 'weather',
            'sports', 'football', 'basketball', 'baseball', 'movie', 'music',
            'technology', 'computer', 'software', 'hardware', 'car', 'automotive'
        ]
    
    def clean_text(self, text: str) -> str:
        """Clean and standardize text"""
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to string and lowercase
        text = str(text).lower().strip()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone numbers
        text = re.sub(r'[\+]?[1-9][\d]{0,15}', '', text)
        
        # Remove special characters but keep apostrophes and hyphens
        text = re.sub(r'[^\w\s\'-]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def remove_stopwords_text(self, text: str) -> str:
        """Remove stopwords from text"""
        if not self.remove_stopwords:
            return text
        
        try:
            words = word_tokenize(text)
            filtered_words = [word for word in words if word.lower() not in self.stop_words]
            return ' '.join(filtered_words)
        except:
            # Fallback to simple word splitting if NLTK tokenization fails
            words = text.split()
            filtered_words = [word for word in words if word.lower() not in self.stop_words]
            return ' '.join(filtered_words)
    
    def lemmatize_text(self, text: str) -> str:
        """Lemmatize text"""
        if not self.lemmatize:
            return text
        
        try:
            words = word_tokenize(text)
            lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words]
            return ' '.join(lemmatized_words)
        except:
            # Fallback to simple word splitting if NLTK tokenization fails
            words = text.split()
            lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words]
            return ' '.join(lemmatized_words)
    
    def detect_spam_indicators(self, text: str) -> Dict[str, Any]:
        """Detect potential spam/advertisement indicators"""
        text_lower = text.lower()
        
        spam_score = 0
        detected_indicators = []
        
        for indicator in self.spam_indicators:
            if indicator in text_lower:
                spam_score += 1
                detected_indicators.append(indicator)
        
        # Additional spam patterns
        if re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text):  # Phone numbers
            spam_score += 1
            detected_indicators.append('phone_number')
        
        if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text):  # Email
            spam_score += 1
            detected_indicators.append('email_address')
        
        if re.search(r'http[s]?://', text):  # URLs
            spam_score += 1
            detected_indicators.append('url')
        
        return {
            'spam_score': spam_score,
            'detected_indicators': detected_indicators,
            'is_spam': spam_score > 0
        }
    
    def detect_irrelevant_content(self, text: str) -> Dict[str, Any]:
        """Detect potentially irrelevant content"""
        text_lower = text.lower()
        
        irrelevant_score = 0
        detected_topics = []
        
        for topic in self.irrelevant_indicators:
            if topic in text_lower:
                irrelevant_score += 1
                detected_topics.append(topic)
        
        return {
            'irrelevant_score': irrelevant_score,
            'detected_topics': detected_topics,
            'is_irrelevant': irrelevant_score > 0
        }
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of the text"""
        try:
            blob = TextBlob(text)
            return {
                'polarity': blob.sentiment.polarity,  # -1 to 1
                'subjectivity': blob.sentiment.subjectivity  # 0 to 1
            }
        except:
            return {'polarity': 0.0, 'subjectivity': 0.0}
    
    def extract_text_features(self, text: str) -> Dict[str, Any]:
        """Extract various text features"""
        if pd.isna(text) or text == '':
            return {
                'length': 0,
                'word_count': 0,
                'avg_word_length': 0,
                'exclamation_count': 0,
                'question_count': 0,
                'capital_letter_ratio': 0,
                'digit_count': 0,
                'unique_word_ratio': 0
            }
        
        # Basic text features
        length = len(text)
        words = text.split()
        word_count = len(words)
        
        # Character-level features
        exclamation_count = text.count('!')
        question_count = text.count('?')
        capital_letters = sum(1 for c in text if c.isupper())
        capital_letter_ratio = capital_letters / length if length > 0 else 0
        digit_count = sum(1 for c in text if c.isdigit())
        
        # Word-level features
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        unique_word_ratio = len(set(words)) / word_count if word_count > 0 else 0
        
        return {
            'length': length,
            'word_count': word_count,
            'avg_word_length': avg_word_length,
            'exclamation_count': exclamation_count,
            'question_count': question_count,
            'capital_letter_ratio': capital_letter_ratio,
            'digit_count': digit_count,
            'unique_word_ratio': unique_word_ratio
        }
    
    def preprocess_text(self, text: str) -> Dict[str, Any]:
        """Complete text preprocessing pipeline"""
        
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Remove stopwords
        text_no_stopwords = self.remove_stopwords_text(cleaned_text)
        
        # Lemmatize
        lemmatized_text = self.lemmatize_text(text_no_stopwords)
        
        # Detect spam and irrelevant content
        spam_analysis = self.detect_spam_indicators(cleaned_text)
        irrelevant_analysis = self.detect_irrelevant_content(cleaned_text)
        
        # Analyze sentiment
        sentiment = self.analyze_sentiment(cleaned_text)
        
        # Extract text features
        text_features = self.extract_text_features(cleaned_text)
        
        return {
            'original_text': text,
            'cleaned_text': cleaned_text,
            'processed_text': lemmatized_text,
            'spam_analysis': spam_analysis,
            'irrelevant_analysis': irrelevant_analysis,
            'sentiment': sentiment,
            'text_features': text_features
        }
    
    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """Preprocess entire dataframe"""
        
        logger.info(f"Starting preprocessing of {len(df)} reviews...")
        
        # Create new columns for processed data
        df['cleaned_text'] = df[text_column].apply(self.clean_text)
        df['processed_text'] = df['cleaned_text'].apply(self.remove_stopwords_text).apply(self.lemmatize_text)
        
        # Extract features
        text_features = df['cleaned_text'].apply(self.extract_text_features)
        df['text_length'] = [f['length'] for f in text_features]
        df['word_count'] = [f['word_count'] for f in text_features]
        df['avg_word_length'] = [f['avg_word_length'] for f in text_features]
        df['exclamation_count'] = [f['exclamation_count'] for f in text_features]
        df['question_count'] = [f['question_count'] for f in text_features]
        df['capital_letter_ratio'] = [f['capital_letter_ratio'] for f in text_features]
        df['digit_count'] = [f['digit_count'] for f in text_features]
        df['unique_word_ratio'] = [f['unique_word_ratio'] for f in text_features]
        
        # Spam detection
        spam_analysis = df['cleaned_text'].apply(self.detect_spam_indicators)
        df['spam_score'] = [a['spam_score'] for a in spam_analysis]
        df['is_spam'] = [a['is_spam'] for a in spam_analysis]
        
        # Irrelevant content detection
        irrelevant_analysis = df['cleaned_text'].apply(self.detect_irrelevant_content)
        df['irrelevant_score'] = [a['irrelevant_score'] for a in irrelevant_analysis]
        df['is_irrelevant'] = [a['is_irrelevant'] for a in irrelevant_analysis]
        
        # Sentiment analysis
        sentiment = df['cleaned_text'].apply(self.analyze_sentiment)
        df['sentiment_polarity'] = [s['polarity'] for s in sentiment]
        df['sentiment_subjectivity'] = [s['subjectivity'] for s in sentiment]
        
        logger.info("Preprocessing completed!")
        
        return df

if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = TextPreprocessor()
    
    # Test with sample text
    sample_text = "This restaurant is AMAZING! Best food ever! Call now for special offers!!! Visit our website www.example.com"
    
    result = preprocessor.preprocess_text(sample_text)
    
    print("ORIGINAL TEXT:")
    print(sample_text)
    print("\nCLEANED TEXT:")
    print(result['cleaned_text'])
    print("\nPROCESSED TEXT:")
    print(result['processed_text'])
    print("\nSPAM ANALYSIS:")
    print(result['spam_analysis'])
    print("\nIRRELEVANT ANALYSIS:")
    print(result['irrelevant_analysis'])
    print("\nSENTIMENT:")
    print(result['sentiment'])
    print("\nTEXT FEATURES:")
    print(result['text_features'])
