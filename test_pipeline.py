#!/usr/bin/env python3
"""
Test script for the ML-based review quality detection pipeline
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append('src')

def test_data_loading():
    """Test data loading functionality"""
    print("Testing data loading...")
    
    try:
        from src.preprocessing.data_loader import ReviewDataLoader
        
        loader = ReviewDataLoader()
        reviews_df, restaurant_df = loader.load_data()
        
        print(f"✓ Data loaded successfully")
        print(f"  - Reviews: {len(reviews_df)} rows")
        print(f"  - Restaurant data: {len(restaurant_df)} rows")
        
        return True
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return False

def test_preprocessing():
    """Test text preprocessing functionality"""
    print("\nTesting text preprocessing...")
    
    try:
        from src.preprocessing.data_loader import ReviewDataLoader
        from src.preprocessing.text_preprocessor import TextPreprocessor
        
        # Load data
        loader = ReviewDataLoader()
        reviews_df, _ = loader.load_data()
        
        # Preprocess
        preprocessor = TextPreprocessor()
        processed_df = preprocessor.preprocess_dataframe(reviews_df)
        
        print(f"✓ Text preprocessing completed")
        print(f"  - Original columns: {len(reviews_df.columns)}")
        print(f"  - Processed columns: {len(processed_df.columns)}")
        print(f"  - New features added: {len(processed_df.columns) - len(reviews_df.columns)}")
        
        return True
    except Exception as e:
        print(f"✗ Text preprocessing failed: {e}")
        return False

def test_feature_engineering():
    """Test feature engineering functionality"""
    print("\nTesting feature engineering...")
    
    try:
        from src.preprocessing.data_loader import ReviewDataLoader
        from src.preprocessing.text_preprocessor import TextPreprocessor
        from src.feature_engineering.feature_extractor import FeatureExtractor
        
        # Load and preprocess data
        loader = ReviewDataLoader()
        reviews_df, _ = loader.load_data()
        
        preprocessor = TextPreprocessor()
        processed_df = preprocessor.preprocess_dataframe(reviews_df)
        
        # Extract features
        extractor = FeatureExtractor(max_features=500, n_topics=5)
        features, labels = extractor.extract_all_features(processed_df)
        
        print(f"✓ Feature engineering completed")
        print(f"  - Feature matrix shape: {features.shape}")
        print(f"  - Labels shape: {labels.shape}")
        print(f"  - Quality score range: {labels['quality_score'].min():.3f} - {labels['quality_score'].max():.3f}")
        
        return True
    except Exception as e:
        print(f"✗ Feature engineering failed: {e}")
        return False

def test_policy_enforcement():
    """Test policy enforcement functionality"""
    print("\nTesting policy enforcement...")
    
    try:
        from src.preprocessing.data_loader import ReviewDataLoader
        from src.preprocessing.text_preprocessor import TextPreprocessor
        from src.models.policy_enforcement import PolicyEnforcementModule
        
        # Load and preprocess data
        loader = ReviewDataLoader()
        reviews_df, _ = loader.load_data()
        
        preprocessor = TextPreprocessor()
        processed_df = preprocessor.preprocess_dataframe(reviews_df)
        
        # Apply policy enforcement
        policy_module = PolicyEnforcementModule()
        df_with_policies = policy_module.enforce_policies_on_dataframe(processed_df)
        
        print(f"✓ Policy enforcement completed")
        print(f"  - High severity violations: {(df_with_policies['severity_level'] == 'high').sum()}")
        print(f"  - Medium severity violations: {(df_with_policies['severity_level'] == 'medium').sum()}")
        print(f"  - Low severity violations: {(df_with_policies['severity_level'] == 'low').sum()}")
        
        return True
    except Exception as e:
        print(f"✗ Policy enforcement failed: {e}")
        return False

def test_model_training():
    """Test model training functionality"""
    print("\nTesting model training...")
    
    try:
        from src.preprocessing.data_loader import ReviewDataLoader
        from src.preprocessing.text_preprocessor import TextPreprocessor
        from src.feature_engineering.feature_extractor import FeatureExtractor
        from src.models.ml_models import ReviewQualityClassifier
        
        # Load and preprocess data
        loader = ReviewDataLoader()
        reviews_df, _ = loader.load_data()
        
        preprocessor = TextPreprocessor()
        processed_df = preprocessor.preprocess_dataframe(reviews_df)
        
        extractor = FeatureExtractor(max_features=500, n_topics=5)
        features, labels = extractor.extract_all_features(processed_df)
        
        # Train a simple model
        classifier = ReviewQualityClassifier(model_type='random_forest')
        results = classifier.train_model(features, labels['is_high_quality'])
        
        print(f"✓ Model training completed")
        print(f"  - Model type: Random Forest")
        print(f"  - Accuracy: {results['accuracy']:.4f}")
        print(f"  - F1 Score: {results['f1_score']:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Model training failed: {e}")
        return False

def test_evaluation():
    """Test model evaluation functionality"""
    print("\nTesting model evaluation...")
    
    try:
        from src.preprocessing.data_loader import ReviewDataLoader
        from src.preprocessing.text_preprocessor import TextPreprocessor
        from src.feature_engineering.feature_extractor import FeatureExtractor
        from src.models.ml_models import ReviewQualityClassifier
        from src.evaluation.model_evaluator import ModelEvaluator
        
        # Load and preprocess data
        loader = ReviewDataLoader()
        reviews_df, _ = loader.load_data()
        
        preprocessor = TextPreprocessor()
        processed_df = preprocessor.preprocess_dataframe(reviews_df)
        
        extractor = FeatureExtractor(max_features=500, n_topics=5)
        features, labels = extractor.extract_all_features(processed_df)
        
        # Train model
        classifier = ReviewQualityClassifier(model_type='random_forest')
        results = classifier.train_model(features, labels['is_high_quality'])
        
        # Evaluate model
        evaluator = ModelEvaluator()
        evaluation_results = evaluator.evaluate_model(
            results['y_test'],
            results['y_pred'],
            results['y_pred_proba'],
            'random_forest'
        )
        
        print(f"✓ Model evaluation completed")
        print(f"  - Accuracy: {evaluation_results['accuracy']:.4f}")
        print(f"  - Precision: {evaluation_results['precision']:.4f}")
        print(f"  - Recall: {evaluation_results['recall']:.4f}")
        print(f"  - F1 Score: {evaluation_results['f1_score']:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Model evaluation failed: {e}")
        return False

def test_complete_pipeline():
    """Test the complete pipeline"""
    print("\nTesting complete pipeline...")
    
    try:
        from src.main import ReviewQualityDetectionPipeline
        
        # Create a minimal configuration
        config = {
            'max_features': 500,
            'n_topics': 5,
            'models': ['random_forest'],
            'save_models': False,
            'policy_enforcement': True,
            'generate_visualizations': False
        }
        
        # Run pipeline
        pipeline = ReviewQualityDetectionPipeline(config)
        results = pipeline.run_complete_pipeline()
        
        if results['success']:
            print(f"✓ Complete pipeline executed successfully")
            print(f"  - Models trained: {len(results['model_results'])}")
            return True
        else:
            print(f"✗ Pipeline failed: {results['error']}")
            return False
            
    except Exception as e:
        print(f"✗ Complete pipeline failed: {e}")
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("TESTING ML-BASED REVIEW QUALITY DETECTION PIPELINE")
    print("="*60)
    
    tests = [
        test_data_loading,
        test_preprocessing,
        test_feature_engineering,
        test_policy_enforcement,
        test_model_training,
        test_evaluation,
        test_complete_pipeline
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 All tests passed! The pipeline is ready to use.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
