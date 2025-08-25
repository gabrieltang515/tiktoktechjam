#!/usr/bin/env python3
"""
Main pipeline for ML-based review quality detection system
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent))

from preprocessing.data_loader import ReviewDataLoader
from preprocessing.text_preprocessor import TextPreprocessor
from feature_engineering.feature_extractor import FeatureExtractor
from models.policy_enforcement import PolicyEnforcementModule
from models.ml_models import ReviewQualityClassifier, TransformerModel
from evaluation.model_evaluator import ModelEvaluator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ReviewQualityDetectionPipeline:
    """Complete pipeline for review quality detection"""
    
    def __init__(self, config: dict = None):
        self.config = config or self._get_default_config()
        self.data_loader = ReviewDataLoader()
        self.preprocessor = TextPreprocessor()
        self.feature_extractor = FeatureExtractor(
            max_features=self.config['max_features'],
            n_topics=self.config['n_topics']
        )
        self.policy_module = PolicyEnforcementModule()
        self.evaluator = ModelEvaluator()
        
        # Create output directories
        self._create_directories()
        
    def _get_default_config(self) -> dict:
        """Get default configuration"""
        return {
            'max_features': 2000,
            'n_topics': 10,
            'test_size': 0.2,
            'random_state': 42,
            'models': ['random_forest', 'xgboost', 'ensemble'],
            'save_models': True,
            'generate_visualizations': True,
            'policy_enforcement': True
        }
    
    def _create_directories(self):
        """Create necessary directories"""
        directories = ['results', 'models', 'notebooks', 'data']
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
    
    def run_data_loading(self) -> tuple:
        """Step 1: Load and examine data"""
        logger.info("="*60)
        logger.info("STEP 1: DATA LOADING")
        logger.info("="*60)
        
        # Load data
        reviews_df, restaurant_df = self.data_loader.load_data()
        
        # Display data information
        info = self.data_loader.get_data_info()
        logger.info(f"Dataset loaded: {info['total_reviews']} reviews")
        logger.info(f"Columns: {info['columns']}")
        logger.info(f"Missing values: {info['missing_values']}")
        
        # Display sample reviews
        self.data_loader.display_sample_reviews(3)
        
        return reviews_df, restaurant_df
    
    def run_preprocessing(self, reviews_df: pd.DataFrame) -> pd.DataFrame:
        """Step 2: Preprocess and clean data"""
        logger.info("="*60)
        logger.info("STEP 2: DATA PREPROCESSING")
        logger.info("="*60)
        
        # Preprocess text data
        processed_df = self.preprocessor.preprocess_dataframe(reviews_df)
        
        logger.info(f"Preprocessing completed for {len(processed_df)} reviews")
        logger.info(f"New features added: {len(processed_df.columns) - len(reviews_df.columns)}")
        
        # Display preprocessing statistics
        logger.info(f"Average text length: {processed_df['text_length'].mean():.1f}")
        logger.info(f"Spam detected: {processed_df['is_spam'].sum()} reviews")
        logger.info(f"Irrelevant content detected: {processed_df['is_irrelevant'].sum()} reviews")
        
        return processed_df
    
    def run_feature_engineering(self, processed_df: pd.DataFrame) -> tuple:
        """Step 3: Feature engineering"""
        logger.info("="*60)
        logger.info("STEP 3: FEATURE ENGINEERING")
        logger.info("="*60)
        
        # Extract features
        features, labels = self.feature_extractor.extract_all_features(processed_df)
        
        logger.info(f"Feature engineering completed")
        logger.info(f"Feature matrix shape: {features.shape}")
        logger.info(f"Quality score range: {labels['quality_score'].min():.3f} - {labels['quality_score'].max():.3f}")
        logger.info(f"High quality reviews: {labels['is_high_quality'].sum()} / {len(labels)}")
        
        # Feature importance analysis
        importance_analysis = self.feature_extractor.get_feature_importance_analysis(processed_df)
        logger.info("Top correlated features:")
        for feature, corr in list(importance_analysis['top_correlated_features'].items())[:5]:
            logger.info(f"  {feature}: {corr:.3f}")
        
        return features, labels
    
    def run_policy_enforcement(self, processed_df: pd.DataFrame) -> pd.DataFrame:
        """Step 4: Policy enforcement"""
        logger.info("="*60)
        logger.info("STEP 4: POLICY ENFORCEMENT")
        logger.info("="*60)
        
        # Apply policy enforcement
        df_with_policies = self.policy_module.enforce_policies_on_dataframe(processed_df)
        
        # Analyze policy violations
        violation_stats = {
            'total_reviews': len(df_with_policies),
            'high_severity': (df_with_policies['severity_level'] == 'high').sum(),
            'medium_severity': (df_with_policies['severity_level'] == 'medium').sum(),
            'low_severity': (df_with_policies['severity_level'] == 'low').sum(),
            'no_violations': (df_with_policies['severity_level'] == 'none').sum(),
            'recommended_reject': (df_with_policies['recommendation'] == 'reject').sum(),
            'recommended_flag': (df_with_policies['recommendation'] == 'flag_for_review').sum(),
            'recommended_approve': (df_with_policies['recommendation'] == 'approve').sum()
        }
        
        logger.info("Policy enforcement completed")
        logger.info(f"High severity violations: {violation_stats['high_severity']}")
        logger.info(f"Medium severity violations: {violation_stats['medium_severity']}")
        logger.info(f"Low severity violations: {violation_stats['low_severity']}")
        logger.info(f"No violations: {violation_stats['no_violations']}")
        logger.info(f"Recommended to reject: {violation_stats['recommended_reject']}")
        logger.info(f"Recommended to flag: {violation_stats['recommended_flag']}")
        logger.info(f"Recommended to approve: {violation_stats['recommended_approve']}")
        
        return df_with_policies
    
    def run_model_training(self, features: pd.DataFrame, labels: pd.DataFrame) -> dict:
        """Step 5: Model training and evaluation"""
        logger.info("="*60)
        logger.info("STEP 5: MODEL TRAINING AND EVALUATION")
        logger.info("="*60)
        
        model_results = {}
        
        # Train multiple models
        for model_type in self.config['models']:
            logger.info(f"Training {model_type} model...")
            
            try:
                classifier = ReviewQualityClassifier(model_type=model_type)
                results = classifier.train_model(features, labels['is_high_quality'])
                
                # Evaluate model
                evaluation_results = self.evaluator.evaluate_model(
                    results['y_test'],
                    results['y_pred'],
                    results['y_pred_proba'],
                    model_type
                )
                
                model_results[model_type] = {
                    'classifier': classifier,
                    'training_results': results,
                    'evaluation_results': evaluation_results
                }
                
                logger.info(f"{model_type} - F1 Score: {results['f1_score']:.4f}")
                
                # Save model if configured
                if self.config['save_models']:
                    model_path = f"models/{model_type}_model.pkl"
                    classifier.save_model(model_path, model_type)
                
            except Exception as e:
                logger.error(f"Error training {model_type} model: {e}")
                continue
        
        return model_results
    
    def run_transformer_training(self, processed_df: pd.DataFrame, labels: pd.DataFrame) -> dict:
        """Step 6: Transformer model training (optional)"""
        logger.info("="*60)
        logger.info("STEP 6: TRANSFORMER MODEL TRAINING")
        logger.info("="*60)
        
        try:
            # Prepare data for transformer
            texts = processed_df['cleaned_text'].tolist()
            binary_labels = labels['is_high_quality'].tolist()
            
            # Initialize transformer model
            transformer = TransformerModel()
            
            # Train transformer
            trainer = transformer.train(texts, binary_labels, epochs=2, batch_size=8)
            
            logger.info("Transformer model training completed")
            
            return {'transformer': transformer, 'trainer': trainer}
            
        except Exception as e:
            logger.error(f"Error training transformer model: {e}")
            return {}
    
    def generate_reports(self, model_results: dict, processed_df: pd.DataFrame) -> None:
        """Step 7: Generate comprehensive reports"""
        logger.info("="*60)
        logger.info("STEP 7: GENERATING REPORTS")
        logger.info("="*60)
        
        # Model comparison
        if model_results:
            comparison_df = self.evaluator.compare_models(self.evaluator.evaluation_results)
            logger.info("\nModel Performance Comparison:")
            logger.info(comparison_df.to_string(index=False))
            
            # Save comparison
            comparison_df.to_csv('results/model_comparison.csv', index=False)
        
        # Generate evaluation report
        report = self.evaluator.generate_evaluation_report()
        
        # Save report
        report_path = self.evaluator.save_evaluation_results()
        logger.info(f"Evaluation report saved to: {report_path}")
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("PIPELINE SUMMARY")
        logger.info("="*60)
        logger.info(f"Total reviews processed: {len(processed_df)}")
        logger.info(f"Models trained: {len(model_results)}")
        
        if model_results:
            best_model = comparison_df.iloc[0]
            logger.info(f"Best performing model: {best_model['Model']}")
            logger.info(f"Best F1 Score: {best_model['F1_Score']:.4f}")
        
        logger.info("Pipeline completed successfully!")
    
    def run_complete_pipeline(self) -> dict:
        """Run the complete pipeline"""
        logger.info("Starting Review Quality Detection Pipeline")
        logger.info(f"Configuration: {self.config}")
        
        try:
            # Step 1: Data loading
            reviews_df, restaurant_df = self.run_data_loading()
            
            # Step 2: Preprocessing
            processed_df = self.run_preprocessing(reviews_df)
            
            # Step 3: Feature engineering
            features, labels = self.run_feature_engineering(processed_df)
            
            # Step 4: Policy enforcement
            if self.config['policy_enforcement']:
                df_with_policies = self.run_policy_enforcement(processed_df)
            else:
                df_with_policies = processed_df
            
            # Step 5: Model training
            model_results = self.run_model_training(features, labels)
            
            # Step 6: Transformer training (optional)
            transformer_results = self.run_transformer_training(processed_df, labels)
            
            # Step 7: Generate reports
            self.generate_reports(model_results, processed_df)
            
            return {
                'success': True,
                'model_results': model_results,
                'transformer_results': transformer_results,
                'processed_data': processed_df,
                'features': features,
                'labels': labels
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return {'success': False, 'error': str(e)}

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Review Quality Detection Pipeline')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--models', nargs='+', default=['ensemble'], 
                       help='Models to train (random_forest, xgboost, lightgbm, ensemble)')
    parser.add_argument('--max-features', type=int, default=2000, 
                       help='Maximum number of features')
    parser.add_argument('--n-topics', type=int, default=10, 
                       help='Number of topics for topic modeling')
    parser.add_argument('--no-policy', action='store_true', 
                       help='Skip policy enforcement')
    parser.add_argument('--no-save', action='store_true', 
                       help='Skip saving models')
    
    args = parser.parse_args()
    
    # Create configuration
    config = {
        'max_features': args.max_features,
        'n_topics': args.n_topics,
        'models': args.models,
        'save_models': not args.no_save,
        'policy_enforcement': not args.no_policy,
        'generate_visualizations': True,
        'test_size': 0.2,
        'random_state': 42
    }
    
    # Run pipeline
    pipeline = ReviewQualityDetectionPipeline(config)
    results = pipeline.run_complete_pipeline()
    
    if results['success']:
        logger.info("Pipeline completed successfully!")
        return 0
    else:
        logger.error(f"Pipeline failed: {results['error']}")
        return 1

if __name__ == "__main__":
    exit(main())
