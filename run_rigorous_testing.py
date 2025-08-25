#!/usr/bin/env python3
"""
Script to run rigorous testing and generate comprehensive performance visualizations
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from evaluation.rigorous_testing_visualizer import RigorousTestingVisualizer

def main():
    """Run rigorous testing and generate visualizations"""
    
    print("="*80)
    print("RIGOROUS MODEL TESTING AND PERFORMANCE VISUALIZATION")
    print("="*80)
    
    # Configuration for comprehensive testing
    config = {
        'max_features': 2000,
        'n_topics': 10,
        'test_size': 0.2,
        'random_state': 42,
        'cv_folds': 5,
        'models': ['random_forest', 'xgboost', 'ensemble'],
        'save_models': True,
        'generate_visualizations': True,
        'cross_validation': True,
        'threshold_analysis': True,
        'feature_importance_analysis': True
    }
    
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\nStarting rigorous testing pipeline...")
    
    # Initialize and run testing
    tester = RigorousTestingVisualizer(config)
    results = tester.run_complete_rigorous_testing()
    
    if results['success']:
        print("\n" + "="*80)
        print("✅ RIGOROUS TESTING COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        # Print summary
        print(f"\n📊 Results Summary:")
        print(f"  - Models tested: {len(config['models'])}")
        print(f"  - Cross-validation folds: {config['cv_folds']}")
        print(f"  - Visualizations created: {len(results['visualizations'])}")
        
        # Print model performance
        model_results = {k: v for k, v in results['all_results'].items() 
                        if not k.endswith('_cv') and not k.endswith('_threshold') 
                        and not k.endswith('_feature_importance')}
        
        print(f"\n🏆 Model Performance:")
        for model_name, results_data in model_results.items():
            f1_score = results_data['f1_score']
            accuracy = results_data['accuracy']
            print(f"  - {model_name}: F1={f1_score:.4f}, Accuracy={accuracy:.4f}")
        
        # Find best model
        best_model = max(model_results.items(), key=lambda x: x[1]['f1_score'])
        print(f"\n🥇 Best Model: {best_model[0]} (F1 Score: {best_model[1]['f1_score']:.4f})")
        
        print(f"\n📁 Output Files:")
        print(f"  - Plots: plots/ (HTML and PNG files)")
        print(f"  - Reports: reports/ (Detailed analysis)")
        print(f"  - Models: models/ (Trained model files)")
        print(f"  - Logs: rigorous_testing.log")
        
        print(f"\n🎯 Next Steps:")
        print(f"  1. Open plots/*.html files for interactive visualizations")
        print(f"  2. Review reports/rigorous_testing_report_*.txt for detailed analysis")
        print(f"  3. Use the best model for production deployment")
        
        return 0
    else:
        print(f"\n❌ Rigorous testing failed: {results['error']}")
        return 1

if __name__ == "__main__":
    exit(main())
