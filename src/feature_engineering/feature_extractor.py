#!/usr/bin/env python3
"""
Advanced Feature Engineering Module for Review Quality Detection

This module provides comprehensive feature extraction capabilities for review
quality assessment, focusing purely on text characteristics and policy compliance
without using rating data.

Key Features:
- Textual feature extraction (TF-IDF, Count Vectors)
- Topic modeling (LDA, NMF)
- Quality indicator calculation
- Policy violation feature engineering
- Sentiment analysis features

Author: Review Quality Detection Team
Version: 2.0.0
Date: 2024
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.preprocessing import StandardScaler, LabelEncoder
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureExtractor:
    """
    Advanced feature extraction for review quality detection and content analysis.
    
    This class provides comprehensive feature engineering capabilities including:
    - Textual feature extraction using TF-IDF and count vectors
    - Topic modeling for content understanding
    - Quality indicator calculation based on text characteristics
    - Policy violation detection features
    - Sentiment analysis features
    
    The feature extractor is designed to create features that assess review
    quality independently of rating data, focusing on writing quality,
    content relevance, and policy compliance.
    
    Attributes:
        max_features: Maximum number of TF-IDF features to extract
        n_topics: Number of topics for topic modeling
        tfidf_vectorizer: TF-IDF vectorizer for text features
        count_vectorizer: Count vectorizer for word frequency
        lda_model: Latent Dirichlet Allocation model
        nmf_model: Non-negative Matrix Factorization model
        scaler: Standard scaler for feature normalization
        label_encoders: Dictionary of label encoders for categorical variables
    """
    
    def __init__(self, max_features: int = 5000, n_topics: int = 10):
        self.max_features = max_features
        self.n_topics = n_topics
        self.tfidf_vectorizer = None
        self.count_vectorizer = None
        self.lda_model = None
        self.nmf_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def extract_metadata_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract metadata-based features (excluding rating-based quality indicators)"""
        
        logger.info("Extracting metadata features...")
        
        # User-based features (for context, not quality scoring)
        df['user_review_count'] = df.groupby('author_name')['author_name'].transform('count')
        
        # Business-based features (for context, not quality scoring)
        df['business_review_count'] = df.groupby('business_name')['business_name'].transform('count')
        
        # Text quality indicators (these are the actual quality measures)
        df['text_quality_score'] = self._calculate_text_quality_score(df)
        df['readability_score'] = self._calculate_readability_score(df)
        
        return df
    
    def _encode_categorical(self, df: pd.DataFrame, column: str) -> pd.Series:
        """Encode categorical variables"""
        if column not in self.label_encoders:
            self.label_encoders[column] = LabelEncoder()
            df[f'{column}_encoded'] = self.label_encoders[column].fit_transform(df[column].fillna('unknown'))
        else:
            df[f'{column}_encoded'] = self.label_encoders[column].transform(df[column].fillna('unknown'))
        
        return df[f'{column}_encoded']
    
    def _calculate_text_quality_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate text quality score based on various indicators"""
        
        # Normalize features to 0-1 range
        max_length = df['text_length'].max()
        max_words = df['word_count'].max()
        max_exclamations = df['exclamation_count'].max()
        max_questions = df['question_count'].max()
        
        # Quality indicators (higher is better)
        length_score = df['text_length'] / max_length if max_length > 0 else 0
        word_score = df['word_count'] / max_words if max_words > 0 else 0
        unique_score = df['unique_word_ratio']
        
        # Penalty indicators (lower is better)
        exclamation_penalty = 1 - (df['exclamation_count'] / max_exclamations if max_exclamations > 0 else 0)
        question_penalty = 1 - (df['question_count'] / max_questions if max_questions > 0 else 0)
        capital_penalty = 1 - df['capital_letter_ratio']
        digit_penalty = 1 - (df['digit_count'] / df['text_length'] if df['text_length'].max() > 0 else 0)
        
        # Combined quality score
        quality_score = (
            length_score * 0.2 +
            word_score * 0.2 +
            unique_score * 0.2 +
            exclamation_penalty * 0.15 +
            question_penalty * 0.1 +
            capital_penalty * 0.1 +
            digit_penalty * 0.05
        )
        
        return quality_score
    
    def _calculate_readability_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Flesch Reading Ease score"""
        
        def flesch_reading_ease(text, word_count, sentence_count):
            if word_count == 0 or sentence_count == 0:
                return 0
            
            # Approximate syllable count (simple heuristic)
            syllables = len(re.findall(r'[aeiouy]+', text.lower()))
            
            # Flesch Reading Ease formula
            score = 206.835 - (1.015 * (word_count / sentence_count)) - (84.6 * (syllables / word_count))
            return max(0, min(100, score))  # Clamp between 0 and 100
        
        # Approximate sentence count
        sentence_counts = df['cleaned_text'].str.count(r'[.!?]+')
        sentence_counts = sentence_counts.replace(0, 1)  # Avoid division by zero
        
        readability_scores = df.apply(
            lambda row: flesch_reading_ease(
                row['cleaned_text'], 
                row['word_count'], 
                sentence_counts[row.name]
            ), axis=1
        )
        
        return readability_scores
    
    def extract_textual_features(self, df: pd.DataFrame, text_column: str = 'processed_text') -> pd.DataFrame:
        """Extract textual features using TF-IDF and topic modeling"""
        
        logger.info("Extracting textual features...")
        
        # TF-IDF features
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        tfidf_features = self.tfidf_vectorizer.fit_transform(df[text_column].fillna(''))
        tfidf_df = pd.DataFrame(
            tfidf_features.toarray(),
            columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])],
            index=df.index
        )
        
        # Count vectorizer features
        self.count_vectorizer = CountVectorizer(
            max_features=min(1000, self.max_features),
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        count_features = self.count_vectorizer.fit_transform(df[text_column].fillna(''))
        count_df = pd.DataFrame(
            count_features.toarray(),
            columns=[f'count_{i}' for i in range(count_features.shape[1])],
            index=df.index
        )
        
        # Topic modeling with LDA
        self.lda_model = LatentDirichletAllocation(
            n_components=self.n_topics,
            random_state=42,
            max_iter=10
        )
        
        lda_features = self.lda_model.fit_transform(count_features)
        lda_df = pd.DataFrame(
            lda_features,
            columns=[f'lda_topic_{i}' for i in range(self.n_topics)],
            index=df.index
        )
        
        # Topic modeling with NMF
        self.nmf_model = NMF(
            n_components=self.n_topics,
            random_state=42,
            max_iter=200
        )
        
        nmf_features = self.nmf_model.fit_transform(tfidf_features)
        nmf_df = pd.DataFrame(
            nmf_features,
            columns=[f'nmf_topic_{i}' for i in range(self.n_topics)],
            index=df.index
        )
        
        # Combine all textual features
        textual_features = pd.concat([tfidf_df, count_df, lda_df, nmf_df], axis=1)
        
        return textual_features
    
    def extract_policy_violation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features related to policy violations"""
        
        logger.info("Extracting policy violation features...")
        
        # Spam and advertisement features
        df['spam_probability'] = df['spam_score'] / df['spam_score'].max() if df['spam_score'].max() > 0 else 0
        df['advertisement_indicators'] = df['cleaned_text'].str.contains(
            r'\b(buy|sale|discount|offer|deal|promotion|limited|special|free|trial)\b', 
            case=False, 
            regex=True
        ).astype(int)
        
        # Irrelevant content features
        df['irrelevant_probability'] = df['irrelevant_score'] / df['irrelevant_score'].max() if df['irrelevant_score'].max() > 0 else 0
        df['off_topic_indicators'] = df['cleaned_text'].str.contains(
            r'\b(politics|election|sports|weather|news|movie|music|technology)\b', 
            case=False, 
            regex=True
        ).astype(int)
        
        # Rant and complaint features
        df['rant_indicators'] = df['cleaned_text'].str.contains(
            r'\b(terrible|awful|horrible|worst|hate|disgusting|never|again|complaint|angry|furious)\b', 
            case=False, 
            regex=True
        ).astype(int)
        
        # Excessive punctuation and capitalization
        df['excessive_punctuation'] = ((df['exclamation_count'] > 3) | (df['question_count'] > 3)).astype(int)
        df['excessive_capitalization'] = (df['capital_letter_ratio'] > 0.3).astype(int)
        
        # Suspicious patterns
        df['suspicious_patterns'] = (
            df['cleaned_text'].str.contains(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', regex=True) |  # Phone numbers
            df['cleaned_text'].str.contains(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', regex=True) |  # Emails
            df['cleaned_text'].str.contains(r'http[s]?://', regex=True)  # URLs
        ).astype(int)
        
        return df
    
    def create_quality_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create quality labels based purely on review text quality and policy compliance"""
        
        logger.info("Creating quality labels based on review text quality...")
        
        # Quality score based purely on review quality factors (NO RATING INFLUENCE)
        quality_score = (
            # Text quality and readability (40%)
            df['text_quality_score'] * 0.25 +
            (df['readability_score'] / 100) * 0.15 +
            # Policy compliance (45%)
            (
                (1 - df['spam_probability']) * 0.20 +
                (1 - df['irrelevant_probability']) * 0.15 +
                (1 - df['rant_indicators']) * 0.05 +
                (1 - df['suspicious_patterns']) * 0.05
            ) +
            # Writing sophistication (15%)
            (
                df['unique_word_ratio'] * 0.10 +
                (1 - df['capital_letter_ratio']) * 0.05  # Penalize excessive caps
            )
        )
        
        df['quality_score'] = quality_score
        
        # Binary quality labels - focus on policy compliance and text quality
        # High quality = good text quality AND no policy violations
        high_quality_mask = (
            (quality_score >= quality_score.quantile(0.6)) &  # Good text quality
            (df['spam_probability'] < 0.3) &  # No spam
            (df['irrelevant_probability'] < 0.2) &  # Relevant content
            (df['rant_indicators'] < 0.5)  # No excessive rants
        )
        
        df['is_high_quality'] = high_quality_mask.astype(int)
        
        # Multi-class quality labels
        df['quality_category'] = pd.cut(
            quality_score,
            bins=[0, 0.4, 0.7, 1.0],
            labels=['low', 'medium', 'high'],
            include_lowest=True
        )
        
        return df
    
    def extract_all_features(self, df: pd.DataFrame, text_column: str = 'processed_text') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Extract all features and return feature matrix and labels"""
        
        logger.info("Starting comprehensive feature extraction...")
        
        # Extract metadata features
        df = self.extract_metadata_features(df)
        
        # Extract policy violation features
        df = self.extract_policy_violation_features(df)
        
        # Create quality labels
        df = self.create_quality_labels(df)
        
        # Extract textual features
        textual_features = self.extract_textual_features(df, text_column)
        
        # Combine all features (NO RATING-BASED FEATURES)
        feature_columns = [
            # Metadata features (context only, not quality scoring)
            'user_review_count', 'business_review_count',
            
            # Text quality features
            'text_length', 'word_count', 'avg_word_length', 'unique_word_ratio',
            'exclamation_count', 'question_count', 'capital_letter_ratio', 'digit_count',
            'text_quality_score', 'readability_score',
            'sentiment_polarity', 'sentiment_subjectivity',
            
            # Policy violation features
            'spam_probability', 'advertisement_indicators',
            'irrelevant_probability', 'off_topic_indicators',
            'rant_indicators', 'excessive_punctuation', 'excessive_capitalization',
            'suspicious_patterns'
        ]
        
        # Add time-based features if available
        time_features = ['review_year', 'review_month', 'review_day_of_week', 'is_weekend']
        for feature in time_features:
            if feature in df.columns:
                feature_columns.append(feature)
        
        # Create feature matrix
        metadata_features = df[feature_columns].fillna(0)
        all_features = pd.concat([metadata_features, textual_features], axis=1)
        
        # Create labels
        labels = df[['quality_score', 'is_high_quality', 'quality_category']].copy()
        
        logger.info(f"Feature extraction completed. Shape: {all_features.shape}")
        
        return all_features, labels
    
    def get_feature_importance_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze feature importance and correlations"""
        
        # Calculate correlations with quality score
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        correlations = df[numeric_columns].corrwith(df['quality_score']).abs().sort_values(ascending=False)
        
        # Top features by correlation
        top_features = correlations.head(20).to_dict()
        
        # Feature categories (NO RATING-BASED FEATURES)
        feature_categories = {
            'text_quality': ['text_quality_score', 'readability_score', 'text_length', 'word_count'],
            'sentiment': ['sentiment_polarity', 'sentiment_subjectivity'],
            'policy_violations': ['spam_probability', 'irrelevant_probability', 'rant_indicators'],
            'user_behavior': ['user_review_count'],  # Context only
            'business_context': ['business_review_count']  # Context only
        }
        
        return {
            'top_correlated_features': top_features,
            'feature_categories': feature_categories,
            'correlation_matrix': correlations
        }

if __name__ == "__main__":
    # Test the feature extractor
    from src.preprocessing.data_loader import ReviewDataLoader
    from src.preprocessing.text_preprocessor import TextPreprocessor
    
    # Load and preprocess data
    loader = ReviewDataLoader()
    reviews_df, _ = loader.load_data()
    
    preprocessor = TextPreprocessor()
    processed_df = preprocessor.preprocess_dataframe(reviews_df)
    
    # Extract features
    extractor = FeatureExtractor(max_features=1000, n_topics=5)
    features, labels = extractor.extract_all_features(processed_df)
    
    print(f"Feature matrix shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Quality score range: {labels['quality_score'].min():.3f} - {labels['quality_score'].max():.3f}")
    print(f"High quality reviews: {labels['is_high_quality'].sum()} / {len(labels)}")
    
    # Feature importance analysis
    importance_analysis = extractor.get_feature_importance_analysis(processed_df)
    print("\nTop correlated features:")
    for feature, corr in list(importance_analysis['top_correlated_features'].items())[:10]:
        print(f"  {feature}: {corr:.3f}")
