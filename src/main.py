#!/usr/bin/env python3
"""
Review Quality Detection System - Main Pipeline

This module implements a comprehensive machine learning pipeline for automatically
detecting review quality based purely on text characteristics and policy compliance,
without using rating data.

Key Features:
- Text quality assessment (readability, vocabulary diversity, grammar)
- Policy compliance enforcement (no ads, spam, irrelevant content)
- Content moderation (detection of rants, excessive complaints)
- Multi-model ensemble approach for robust predictions

Author: Review Quality Detection Team
Version: 2.0.0
Date: 2024
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

# Import core modules
from preprocessing.data_loader import ReviewDataLoader
from preprocessing.text_preprocessor import TextPreprocessor
from feature_engineering.feature_extractor import FeatureExtractor
from models.policy_enforcement import PolicyEnforcementModule
from models.ml_models import ReviewQualityClassifier, TransformerModel
from evaluation.model_evaluator import ModelEvaluator

# Configure logging for production use
def setup_logging(log_level: str = 'INFO') -> logging.Logger:
    """
    Set up comprehensive logging configuration for the pipeline.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    
    Returns:
        Configured logger instance
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create logs directory if it doesn't exist
    Path('logs').mkdir(exist_ok=True)
    
    # Generate timestamped log filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'logs/review_quality_pipeline_{timestamp}.log'
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

class ReviewQualityDetectionPipeline:
    """
    Complete pipeline for review quality detection and content moderation.
    
    This class orchestrates the entire machine learning pipeline from data loading
    to model deployment, ensuring proper separation of rating and quality assessment.
    
    Attributes:
        config: Pipeline configuration dictionary
        data_loader: Data loading and validation component
        preprocessor: Text preprocessing and feature extraction
        feature_extractor: Advanced feature engineering
        policy_module: Content policy enforcement
        evaluator: Model evaluation and performance analysis
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the review quality detection pipeline.
        
        Args:
            config: Configuration dictionary for pipeline parameters
        """
        self.config = config or self._get_default_config()
        self.logger = setup_logging(self.config.get('log_level', 'INFO'))
        
        # Initialize core components
        self.data_loader = ReviewDataLoader()
        self.preprocessor = TextPreprocessor()
        self.feature_extractor = FeatureExtractor(
            max_features=self.config['max_features'],
            n_topics=self.config['n_topics']
        )
        self.policy_module = PolicyEnforcementModule()
        self.evaluator = ModelEvaluator()
        
        # Create necessary output directories
        self._create_directories()
        
        self.logger.info("Review Quality Detection Pipeline initialized successfully")
        
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration for the pipeline.
        
        Returns:
            Default configuration dictionary
        """
        return {
            'max_features': 2000,          # Maximum TF-IDF features
            'n_topics': 10,                # Number of topics for topic modeling
            'test_size': 0.2,              # Test set proportion
            'random_state': 42,            # Random seed for reproducibility
            'models': ['random_forest', 'xgboost', 'ensemble'],  # Models to train
            'save_models': True,           # Whether to save trained models
            'generate_visualizations': True,  # Generate performance visualizations
            'policy_enforcement': True,    # Enable policy enforcement
            'log_level': 'INFO'            # Logging level
        }
    
    def _create_directories(self) -> None:
        """Create necessary output directories for results and models."""
        directories = ['results', 'models', 'logs', 'data', 'visualizations']
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
            self.logger.debug(f"Created directory: {directory}")
    
    def run_data_loading(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Step 1: Load and validate review data.
        
        This step loads the review dataset and performs initial data validation
        to ensure data quality and completeness.
        
        Returns:
            Tuple of (reviews_df, restaurant_df) DataFrames
        """
        self.logger.info("="*60)
        self.logger.info("STEP 1: DATA LOADING AND VALIDATION")
        self.logger.info("="*60)
        
        try:
            # Load data from configured sources
            reviews_df, restaurant_df = self.data_loader.load_data()
            
            # Perform data validation and quality checks
            data_info = self.data_loader.get_data_info()
            
            self.logger.info(f"Dataset loaded successfully:")
            self.logger.info(f"  - Total reviews: {data_info['total_reviews']:,}")
            self.logger.info(f"  - Columns: {data_info['columns']}")
            self.logger.info(f"  - Missing values: {data_info['missing_values']}")
            
            # Display sample reviews for verification
            self.data_loader.display_sample_reviews(3)
            
            return reviews_df, restaurant_df
            
        except Exception as e:
            self.logger.error(f"Data loading failed: {str(e)}")
            raise
    
    def run_preprocessing(self, reviews_df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 2: Preprocess and clean review text data.
        
        This step performs comprehensive text preprocessing including:
        - Text cleaning and normalization
        - Spam and advertisement detection
        - Irrelevant content identification
        - Sentiment analysis
        - Feature extraction
        
        Args:
            reviews_df: Raw reviews DataFrame
            
        Returns:
            Preprocessed DataFrame with extracted features
        """
        self.logger.info("="*60)
        self.logger.info("STEP 2: TEXT PREPROCESSING AND FEATURE EXTRACTION")
        self.logger.info("="*60)
        
        try:
            # Apply comprehensive text preprocessing
            processed_df = self.preprocessor.preprocess_dataframe(reviews_df)
            
            # Log preprocessing statistics
            self.logger.info(f"Preprocessing completed successfully:")
            self.logger.info(f"  - Reviews processed: {len(processed_df):,}")
            self.logger.info(f"  - New features added: {len(processed_df.columns) - len(reviews_df.columns)}")
            self.logger.info(f"  - Average text length: {processed_df['text_length'].mean():.1f} characters")
            self.logger.info(f"  - Spam detected: {processed_df['is_spam'].sum():,} reviews")
            self.logger.info(f"  - Irrelevant content: {processed_df['is_irrelevant'].sum():,} reviews")
            
            return processed_df
            
        except Exception as e:
            self.logger.error(f"Preprocessing failed: {str(e)}")
            raise
    
    def run_feature_engineering(self, processed_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Step 3: Advanced feature engineering for quality detection.
        
        This step creates comprehensive features for review quality assessment:
        - Text quality metrics (readability, vocabulary diversity)
        - Policy violation indicators
        - Topic modeling features
        - Quality labels based purely on text characteristics
        
        Args:
            processed_df: Preprocessed DataFrame
            
        Returns:
            Tuple of (features, labels) DataFrames
        """
        self.logger.info("="*60)
        self.logger.info("STEP 3: ADVANCED FEATURE ENGINEERING")
        self.logger.info("="*60)
        
        try:
            # Extract comprehensive features (NO RATING DATA)
            features, labels = self.feature_extractor.extract_all_features(processed_df)
            
            # Log feature engineering results
            self.logger.info(f"Feature engineering completed:")
            self.logger.info(f"  - Feature matrix shape: {features.shape}")
            self.logger.info(f"  - Quality score range: {labels['quality_score'].min():.3f} - {labels['quality_score'].max():.3f}")
            self.logger.info(f"  - High quality reviews: {labels['is_high_quality'].sum():,} / {len(labels):,}")
            
            # Analyze feature importance
            importance_analysis = self.feature_extractor.get_feature_importance_analysis(processed_df)
            self.logger.info("Top correlated features with quality:")
            for feature, corr in list(importance_analysis['top_correlated_features'].items())[:5]:
                self.logger.info(f"  - {feature}: {corr:.3f}")
            
            return features, labels
            
        except Exception as e:
            self.logger.error(f"Feature engineering failed: {str(e)}")
            raise
    
    def run_policy_enforcement(self, processed_df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 4: Content policy enforcement and moderation.
        
        This step applies content policies to identify violations:
        - Advertisement and promotional content
        - Spam and suspicious patterns
        - Irrelevant content detection
        - Excessive complaints and rants
        
        Args:
            processed_df: Preprocessed DataFrame
            
        Returns:
            DataFrame with policy enforcement results
        """
        self.logger.info("="*60)
        self.logger.info("STEP 4: CONTENT POLICY ENFORCEMENT")
        self.logger.info("="*60)
        
        try:
            # Apply comprehensive policy enforcement
            df_with_policies = self.policy_module.enforce_policies_on_dataframe(processed_df)
            
            # Analyze policy violation statistics
            violation_stats = self._analyze_policy_violations(df_with_policies)
            
            # Log policy enforcement results
            self.logger.info(f"Policy enforcement completed:")
            self.logger.info(f"  - High severity violations: {violation_stats['high_severity']:,}")
            self.logger.info(f"  - Medium severity violations: {violation_stats['medium_severity']:,}")
            self.logger.info(f"  - Low severity violations: {violation_stats['low_severity']:,}")
            self.logger.info(f"  - No violations: {violation_stats['no_violations']:,}")
            self.logger.info(f"  - Recommended to reject: {violation_stats['recommended_reject']:,}")
            self.logger.info(f"  - Recommended to flag: {violation_stats['recommended_flag']:,}")
            self.logger.info(f"  - Recommended to approve: {violation_stats['recommended_approve']:,}")
            
            return df_with_policies
            
        except Exception as e:
            self.logger.error(f"Policy enforcement failed: {str(e)}")
            raise
    
    def _analyze_policy_violations(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Analyze policy violation statistics.
        
        Args:
            df: DataFrame with policy enforcement results
            
        Returns:
            Dictionary of violation statistics
        """
        return {
            'total_reviews': len(df),
            'high_severity': (df['severity_level'] == 'high').sum(),
            'medium_severity': (df['severity_level'] == 'medium').sum(),
            'low_severity': (df['severity_level'] == 'low').sum(),
            'no_violations': (df['severity_level'] == 'none').sum(),
            'recommended_reject': (df['recommendation'] == 'reject').sum(),
            'recommended_flag': (df['recommendation'] == 'flag_for_review').sum(),
            'recommended_approve': (df['recommendation'] == 'approve').sum()
        }
    
    def run_model_training(self, features: pd.DataFrame, labels: pd.DataFrame) -> Dict[str, Any]:
        """
        Step 5: Model training and evaluation.
        
        This step trains multiple machine learning models and evaluates their
        performance for review quality detection.
        
        Args:
            features: Feature matrix for training
            labels: Quality labels for training
            
        Returns:
            Dictionary containing trained models and evaluation results
        """
        self.logger.info("="*60)
        self.logger.info("STEP 5: MODEL TRAINING AND EVALUATION")
        self.logger.info("="*60)
        
        model_results = {}
        
        # Train each configured model
        for model_type in self.config['models']:
            self.logger.info(f"Training {model_type.upper()} model...")
            
            try:
                # Initialize and train classifier
                classifier = ReviewQualityClassifier(model_type=model_type)
                results = classifier.train_model(features, labels['is_high_quality'])
                
                # Evaluate model performance
                evaluation_results = self.evaluator.evaluate_model(
                    results['y_test'],
                    results['y_pred'],
                    results['y_pred_proba'],
                    model_type
                )
                
                # Store results
                model_results[model_type] = {
                    'classifier': classifier,
                    'training_results': results,
                    'evaluation_results': evaluation_results
                }
                
                self.logger.info(f"  - {model_type.upper()} F1 Score: {results['f1_score']:.4f}")
                self.logger.info(f"  - {model_type.upper()} Accuracy: {results['accuracy']:.4f}")
                
                # Save model if configured
                if self.config['save_models']:
                    model_path = f"models/{model_type}_model.pkl"
                    classifier.save_model(model_path, model_type)
                    self.logger.info(f"  - Model saved to: {model_path}")
                
            except Exception as e:
                self.logger.error(f"Error training {model_type} model: {str(e)}")
                continue
        
        self.logger.info(f"Model training completed. {len(model_results)} models trained successfully.")
        return model_results
    
    def run_transformer_training(self, processed_df: pd.DataFrame, labels: pd.DataFrame) -> Dict[str, Any]:
        """
        Step 6: Transformer model training (optional advanced model).
        
        This step trains a transformer-based model for enhanced text understanding.
        
        Args:
            processed_df: Preprocessed DataFrame
            labels: Quality labels
            
        Returns:
            Dictionary containing transformer model results
        """
        self.logger.info("="*60)
        self.logger.info("STEP 6: TRANSFORMER MODEL TRAINING (OPTIONAL)")
        self.logger.info("="*60)
        
        try:
            # Prepare data for transformer
            texts = processed_df['cleaned_text'].tolist()
            binary_labels = labels['is_high_quality'].tolist()
            
            # Initialize and train transformer model
            transformer = TransformerModel()
            trainer = transformer.train(texts, binary_labels, epochs=2, batch_size=8)
            
            self.logger.info("Transformer model training completed successfully")
            
            return {'transformer': transformer, 'trainer': trainer}
            
        except Exception as e:
            self.logger.error(f"Transformer training failed: {str(e)}")
            return {}
    
    def generate_reports(self, model_results: Dict[str, Any], processed_df: pd.DataFrame) -> None:
        """
        Step 7: Generate comprehensive performance reports and visualizations.
        
        This step creates detailed reports for stakeholder presentation including:
        - Model performance comparisons
        - Feature importance analysis
        - Policy enforcement statistics
        - Quality distribution analysis
        
        Args:
            model_results: Dictionary of trained models and results
            processed_df: Processed DataFrame with features
        """
        self.logger.info("="*60)
        self.logger.info("STEP 7: GENERATING COMPREHENSIVE REPORTS")
        self.logger.info("="*60)
        
        try:
            # Generate model comparison report
            if model_results:
                comparison_df = self.evaluator.compare_models(self.evaluator.evaluation_results)
                self.logger.info("\nModel Performance Comparison:")
                self.logger.info(comparison_df.to_string(index=False))
                
                # Save comparison results
                comparison_df.to_csv('results/model_comparison.csv', index=False)
                self.logger.info("Model comparison saved to: results/model_comparison.csv")
            
            # Generate comprehensive evaluation report
            report = self.evaluator.generate_evaluation_report()
            report_path = self.evaluator.save_evaluation_results()
            self.logger.info(f"Evaluation report saved to: {report_path}")
            
            # Print pipeline summary
            self._print_pipeline_summary(model_results, processed_df)
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {str(e)}")
            raise
    
    def _print_pipeline_summary(self, model_results: Dict[str, Any], processed_df: pd.DataFrame) -> None:
        """
        Print comprehensive pipeline summary for stakeholders.
        
        Args:
            model_results: Dictionary of model results
            processed_df: Processed DataFrame
        """
        self.logger.info("\n" + "="*60)
        self.logger.info("PIPELINE EXECUTION SUMMARY")
        self.logger.info("="*60)
        self.logger.info(f"Total reviews processed: {len(processed_df):,}")
        self.logger.info(f"Models trained successfully: {len(model_results)}")
        
        if model_results:
            # Find best performing model
            best_f1 = 0
            best_model = None
            for model_type, results in model_results.items():
                f1_score = results['training_results']['f1_score']
                if f1_score > best_f1:
                    best_f1 = f1_score
                    best_model = model_type
            
            self.logger.info(f"Best performing model: {best_model.upper()}")
            self.logger.info(f"Best F1 Score: {best_f1:.4f}")
        
        self.logger.info("Pipeline completed successfully!")
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """
        Execute the complete review quality detection pipeline.
        
        This method orchestrates all pipeline steps in sequence:
        1. Data loading and validation
        2. Text preprocessing and feature extraction
        3. Advanced feature engineering
        4. Content policy enforcement
        5. Model training and evaluation
        6. Transformer training (optional)
        7. Report generation
        
        Returns:
            Dictionary containing all pipeline results and artifacts
        """
        self.logger.info("Starting Review Quality Detection Pipeline")
        self.logger.info(f"Configuration: {self.config}")
        
        try:
            # Execute pipeline steps
            reviews_df, restaurant_df = self.run_data_loading()
            processed_df = self.run_preprocessing(reviews_df)
            features, labels = self.run_feature_engineering(processed_df)
            
            # Apply policy enforcement if enabled
            if self.config['policy_enforcement']:
                df_with_policies = self.run_policy_enforcement(processed_df)
            else:
                df_with_policies = processed_df
            
            # Train models
            model_results = self.run_model_training(features, labels)
            transformer_results = self.run_transformer_training(processed_df, labels)
            
            # Generate reports
            self.generate_reports(model_results, processed_df)
            
            return {
                'success': True,
                'model_results': model_results,
                'transformer_results': transformer_results,
                'processed_data': processed_df,
                'features': features,
                'labels': labels,
                'pipeline_config': self.config
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            return {'success': False, 'error': str(e)}

def main():
    """
    Main entry point for the review quality detection pipeline.
    
    This function handles command-line arguments and initiates the pipeline
    execution with appropriate configuration.
    """
    parser = argparse.ArgumentParser(
        description='Review Quality Detection Pipeline - Professional ML System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Run with default settings
  python main.py --models xgboost ensemble         # Train specific models
  python main.py --max-features 3000 --n-topics 15 # Custom feature settings
  python main.py --no-policy --no-save             # Skip policy enforcement and model saving
        """
    )
    
    # Configuration arguments
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--models', nargs='+', default=['ensemble'], 
                       choices=['random_forest', 'xgboost', 'lightgbm', 'ensemble'],
                       help='Models to train (default: ensemble)')
    parser.add_argument('--max-features', type=int, default=2000, 
                       help='Maximum number of TF-IDF features (default: 2000)')
    parser.add_argument('--n-topics', type=int, default=10, 
                       help='Number of topics for topic modeling (default: 10)')
    
    # Pipeline control arguments
    parser.add_argument('--no-policy', action='store_true', 
                       help='Skip policy enforcement step')
    parser.add_argument('--no-save', action='store_true', 
                       help='Skip saving trained models')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    
    args = parser.parse_args()
    
    # Create configuration dictionary
    config = {
        'max_features': args.max_features,
        'n_topics': args.n_topics,
        'models': args.models,
        'save_models': not args.no_save,
        'policy_enforcement': not args.no_policy,
        'generate_visualizations': True,
        'test_size': 0.2,
        'random_state': 42,
        'log_level': args.log_level
    }
    
    # Initialize and run pipeline
    pipeline = ReviewQualityDetectionPipeline(config)
    results = pipeline.run_complete_pipeline()
    
    if results['success']:
        print("\n" + "="*60)
        print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
        print("="*60)
        return 0
    else:
        print(f"\nPipeline execution failed: {results['error']}")
        return 1

if __name__ == "__main__":
    exit(main())
