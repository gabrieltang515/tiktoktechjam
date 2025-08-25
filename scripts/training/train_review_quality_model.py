#!/usr/bin/env python3
"""
Train Review Quality Detection Model

This script trains a model to detect review quality based purely on:
- Text quality and readability
- Policy compliance (no ads, spam, rants, irrelevant content)
- Writing sophistication
- Content relevance

NO RATING DATA IS USED - this model focuses purely on review text quality.
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from preprocessing.data_loader import ReviewDataLoader
from preprocessing.text_preprocessor import TextPreprocessor
from feature_engineering.feature_extractor import FeatureExtractor
from models.ml_models import ReviewQualityClassifier
from models.policy_enforcement import ReviewQualityPolicyEnforcer
from evaluation.model_evaluator import ModelEvaluator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/review_quality_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main training function for review quality detection model"""
    
    logger.info("="*80)
    logger.info("TRAINING REVIEW QUALITY DETECTION MODEL")
    logger.info("FOCUS: Text quality and policy compliance (NO RATING DATA)")
    logger.info("="*80)
    
    # Step 1: Load data
    logger.info("\nSTEP 1: Loading review data...")
    loader = ReviewDataLoader()
    reviews_df, restaurant_df = loader.load_data()
    
    logger.info(f"✓ Loaded {len(reviews_df)} reviews")
    logger.info(f"✓ Dataset shape: {reviews_df.shape}")
    logger.info(f"✓ Columns: {list(reviews_df.columns)}")
    
    # Verify we have the rating column but won't use it for quality scoring
    if 'rating' in reviews_df.columns:
        logger.info("✓ Rating column present (will NOT be used for quality scoring)")
        logger.info(f"  - Rating range: {reviews_df['rating'].min()} - {reviews_df['rating'].max()}")
        logger.info(f"  - Rating distribution: {reviews_df['rating'].value_counts().sort_index().to_dict()}")
    else:
        logger.warning("⚠️  No rating column found")
    
    # Step 2: Preprocess text data
    logger.info("\nSTEP 2: Preprocessing text data...")
    preprocessor = TextPreprocessor()
    processed_df = preprocessor.preprocess_dataframe(reviews_df)
    
    logger.info(f"✓ Preprocessing completed for {len(processed_df)} reviews")
    logger.info(f"✓ New features added: {len(processed_df.columns) - len(reviews_df.columns)}")
    
    # Display preprocessing statistics
    logger.info(f"✓ Average text length: {processed_df['text_length'].mean():.1f} characters")
    logger.info(f"✓ Average word count: {processed_df['word_count'].mean():.1f} words")
    logger.info(f"✓ Spam detected: {processed_df['is_spam'].sum()} reviews")
    logger.info(f"✓ Irrelevant content detected: {processed_df['is_irrelevant'].sum()} reviews")
    
    # Step 3: Apply policy enforcement
    logger.info("\nSTEP 3: Applying policy enforcement...")
    policy_enforcer = ReviewQualityPolicyEnforcer()
    df_with_policies = policy_enforcer.enforce_policies(processed_df)
    
    # Display policy enforcement results
    policy_decisions = df_with_policies['policy_decision'].value_counts()
    logger.info("✓ Policy enforcement results:")
    for decision, count in policy_decisions.items():
        logger.info(f"  - {decision}: {count} reviews ({count/len(df_with_policies)*100:.1f}%)")
    
    # Step 4: Feature engineering (NO RATING-BASED FEATURES)
    logger.info("\nSTEP 4: Extracting quality-focused features...")
    extractor = FeatureExtractor(max_features=2000, n_topics=10)
    features, labels = extractor.extract_all_features(df_with_policies)
    
    logger.info(f"✓ Feature engineering completed")
    logger.info(f"✓ Feature matrix shape: {features.shape}")
    logger.info(f"✓ Quality score range: {labels['quality_score'].min():.3f} - {labels['quality_score'].max():.3f}")
    logger.info(f"✓ High quality reviews: {labels['is_high_quality'].sum()} / {len(labels)} ({labels['is_high_quality'].sum()/len(labels)*100:.1f}%)")
    
    # Verify no rating-based features are included
    feature_columns = features.columns.tolist()
    rating_related_features = [col for col in feature_columns if 'rating' in col.lower()]
    if rating_related_features:
        logger.warning(f"⚠️  Found rating-related features: {rating_related_features}")
        logger.warning("These should be removed for pure quality detection")
    else:
        logger.info("✓ Confirmed: No rating-based features in feature matrix")
    
    # Step 5: Train multiple models
    logger.info("\nSTEP 5: Training review quality detection models...")
    
    models_to_train = ['random_forest', 'xgboost', 'ensemble']
    trained_models = {}
    evaluation_results = {}
    
    for model_type in models_to_train:
        logger.info(f"\n--- Training {model_type.upper()} ---")
        
        # Train model
        classifier = ReviewQualityClassifier(model_type=model_type, random_state=42)
        results = classifier.train_model(features, labels['is_high_quality'])
        
        trained_models[model_type] = classifier
        evaluation_results[model_type] = results
        
        logger.info(f"✓ {model_type} training completed")
        logger.info(f"  - Accuracy: {results['accuracy']:.3f}")
        logger.info(f"  - Precision: {results['precision']:.3f}")
        logger.info(f"  - Recall: {results['recall']:.3f}")
        logger.info(f"  - F1 Score: {results['f1_score']:.3f}")
    
    # Step 6: Model evaluation and comparison
    logger.info("\nSTEP 6: Evaluating and comparing models...")
    
    evaluator = ModelEvaluator()
    
    # Evaluate each model
    for model_type, results in evaluation_results.items():
        evaluator.evaluate_model(
            results['y_test'],
            results['y_pred'],
            results['y_pred_proba'],
            model_type
        )
    
    # Generate comparison
    comparison_df = evaluator.compare_models(evaluator.evaluation_results)
    logger.info("\nModel Comparison:")
    logger.info(comparison_df.to_string())
    
    # Step 7: Save models and results
    logger.info("\nSTEP 7: Saving models and results...")
    
    # Create directories
    models_dir = Path("models/review_quality")
    results_dir = Path("results/review_quality")
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save individual models
    for model_type, classifier in trained_models.items():
        model_path = models_dir / f"review_quality_{model_type}_{timestamp}.pkl"
        classifier.save_model(str(model_path), model_type)
        logger.info(f"✓ Saved {model_type} model: {model_path}")
    
    # Save ensemble model
    best_model_type = comparison_df.loc[comparison_df['f1_score'].idxmax(), 'model']
    best_model = trained_models[best_model_type]
    best_model_path = models_dir / f"review_quality_best_{timestamp}.pkl"
    best_model.save_model(str(best_model_path), best_model_type)
    logger.info(f"✓ Saved best model ({best_model_type}): {best_model_path}")
    
    # Save results
    results_path = results_dir / f"training_results_{timestamp}.csv"
    comparison_df.to_csv(results_path, index=False)
    logger.info(f"✓ Saved training results: {results_path}")
    
    # Save processed data
    data_path = results_dir / f"processed_data_{timestamp}.csv"
    df_with_policies.to_csv(data_path, index=False)
    logger.info(f"✓ Saved processed data: {data_path}")
    
    # Step 8: Generate summary report
    logger.info("\nSTEP 8: Generating summary report...")
    
    # Calculate quality distribution
    quality_distribution = labels['quality_category'].value_counts()
    
    # Calculate policy violation statistics
    policy_stats = df_with_policies['policy_decision'].value_counts()
    
    # Create summary report
    summary_report = f"""
REVIEW QUALITY DETECTION MODEL - TRAINING SUMMARY
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

DATASET OVERVIEW:
- Total reviews: {len(reviews_df)}
- Unique businesses: {reviews_df['business_name'].nunique()}
- Unique authors: {reviews_df['author_name'].nunique()}

QUALITY DISTRIBUTION:
- High quality: {quality_distribution.get('high', 0)} reviews ({quality_distribution.get('high', 0)/len(labels)*100:.1f}%)
- Medium quality: {quality_distribution.get('medium', 0)} reviews ({quality_distribution.get('medium', 0)/len(labels)*100:.1f}%)
- Low quality: {quality_distribution.get('low', 0)} reviews ({quality_distribution.get('low', 0)/len(labels)*100:.1f}%)

POLICY ENFORCEMENT RESULTS:
"""
    
    for decision, count in policy_stats.items():
        summary_report += f"- {decision.title()}: {count} reviews ({count/len(df_with_policies)*100:.1f}%)\n"
    
    summary_report += f"""
MODEL PERFORMANCE (Best Model: {best_model_type.upper()}):
- Accuracy: {comparison_df.loc[comparison_df['model'] == best_model_type, 'accuracy'].iloc[0]:.3f}
- Precision: {comparison_df.loc[comparison_df['model'] == best_model_type, 'precision'].iloc[0]:.3f}
- Recall: {comparison_df.loc[comparison_df['model'] == best_model_type, 'recall'].iloc[0]:.3f}
- F1 Score: {comparison_df.loc[comparison_df['model'] == best_model_type, 'f1_score'].iloc[0]:.3f}

IMPORTANT NOTES:
- This model focuses PURELY on review text quality and policy compliance
- NO RATING DATA is used for quality scoring
- Quality is determined by: text quality, readability, policy compliance, writing sophistication
- Policy violations include: advertisements, spam, irrelevant content, excessive rants

MODEL FILES:
- Best model: {best_model_path}
- Training results: {results_path}
- Processed data: {data_path}
"""
    
    # Save summary report
    report_path = results_dir / f"training_summary_{timestamp}.txt"
    with open(report_path, 'w') as f:
        f.write(summary_report)
    
    logger.info(f"✓ Saved summary report: {report_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(summary_report)
    
    logger.info("Review quality detection model training completed!")

if __name__ == "__main__":
    main()
