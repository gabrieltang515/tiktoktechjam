#!/usr/bin/env python3
"""
Rigorous testing and comprehensive visualization script for model performance analysis
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from preprocessing.data_loader import ReviewDataLoader
from preprocessing.text_preprocessor import TextPreprocessor
from feature_engineering.feature_extractor import FeatureExtractor
from models.ml_models import ReviewQualityClassifier
from evaluation.model_evaluator import ModelEvaluator
from evaluation.performance_visualizer import PerformanceVisualizer
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rigorous_testing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RigorousTestingVisualizer:
    """Comprehensive testing and visualization for model performance analysis"""
    
    def __init__(self, config: dict = None):
        self.config = config or self._get_default_config()
        self.data_loader = ReviewDataLoader()
        self.preprocessor = TextPreprocessor()
        self.feature_extractor = FeatureExtractor(
            max_features=self.config['max_features'],
            n_topics=self.config['n_topics']
        )
        self.evaluator = ModelEvaluator()
        self.visualizer = PerformanceVisualizer()
        
        # Create output directories
        self._create_directories()
        
    def _get_default_config(self) -> dict:
        """Get default configuration for rigorous testing"""
        return {
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
    
    def _create_directories(self):
        """Create necessary directories"""
        directories = ['results', 'models', 'plots', 'reports']
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
    
    def load_and_prepare_data(self) -> tuple:
        """Load and prepare data for rigorous testing"""
        logger.info("="*60)
        logger.info("STEP 1: LOADING AND PREPARING DATA")
        logger.info("="*60)
        
        # Load data
        reviews_df, restaurant_df = self.data_loader.load_data()
        logger.info(f"Loaded {len(reviews_df)} reviews")
        
        # Preprocess data
        processed_df = self.preprocessor.preprocess_dataframe(reviews_df)
        logger.info(f"Preprocessing completed for {len(processed_df)} reviews")
        
        # Extract features
        features, labels = self.feature_extractor.extract_all_features(processed_df)
        logger.info(f"Feature engineering completed. Shape: {features.shape}")
        
        return features, labels, processed_df
    
    def perform_cross_validation(self, features: pd.DataFrame, labels: pd.Series, 
                                model_type: str = 'ensemble') -> dict:
        """Perform cross-validation for robust performance estimation"""
        logger.info(f"Performing {self.config['cv_folds']}-fold cross-validation for {model_type}")
        
        # Initialize model with simplified parameters to avoid warnings
        if model_type == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(
                n_estimators=100, 
                max_depth=10, 
                random_state=self.config['random_state'],
                n_jobs=-1
            )
        elif model_type == 'xgboost':
            import xgboost as xgb
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.config['random_state'],
                n_jobs=-1,
                verbosity=0  # Suppress warnings
            )
        elif model_type == 'ensemble':
            # Use a simpler ensemble approach
            from sklearn.ensemble import VotingClassifier
            from sklearn.linear_model import LogisticRegression
            
            rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=self.config['random_state'])
            xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, 
                                         random_state=self.config['random_state'], verbosity=0)
            lr = LogisticRegression(random_state=self.config['random_state'], max_iter=1000)
            
            model = VotingClassifier(
                estimators=[('rf', rf), ('xgb', xgb_model), ('lr', lr)],
                voting='soft'
            )
        else:
            # Default to Random Forest
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100, random_state=self.config['random_state'])
        
        # No need to prepare data here since we're doing cross-validation on the full dataset
        
        # Perform cross-validation
        cv = StratifiedKFold(n_splits=self.config['cv_folds'], 
                           shuffle=True, random_state=self.config['random_state'])
        
        cv_scores = {
            'accuracy': cross_val_score(model, features, labels['is_high_quality'], 
                                      cv=cv, scoring='accuracy'),
            'precision': cross_val_score(model, features, labels['is_high_quality'], 
                                       cv=cv, scoring='precision_weighted'),
            'recall': cross_val_score(model, features, labels['is_high_quality'], 
                                    cv=cv, scoring='recall_weighted'),
            'f1': cross_val_score(model, features, labels['is_high_quality'], 
                                cv=cv, scoring='f1_weighted')
        }
        
        # Calculate statistics
        cv_results = {}
        for metric, scores in cv_scores.items():
            cv_results[f'{metric}_mean'] = scores.mean()
            cv_results[f'{metric}_std'] = scores.std()
            cv_results[f'{metric}_scores'] = scores.tolist()
        
        logger.info(f"Cross-validation results for {model_type}:")
        logger.info(f"  Accuracy: {cv_results['accuracy_mean']:.4f} ± {cv_results['accuracy_std']:.4f}")
        logger.info(f"  F1 Score: {cv_results['f1_mean']:.4f} ± {cv_results['f1_std']:.4f}")
        
        return cv_results
    
    def perform_rigorous_testing(self, features: pd.DataFrame, labels: pd.Series) -> dict:
        """Perform comprehensive rigorous testing of all models"""
        logger.info("="*60)
        logger.info("STEP 2: PERFORMING RIGOROUS TESTING")
        logger.info("="*60)
        
        all_results = {}
        
        for model_type in self.config['models']:
            logger.info(f"\nTesting {model_type} model...")
            
            # 1. Cross-validation
            if self.config['cross_validation']:
                cv_results = self.perform_cross_validation(features, labels, model_type)
                all_results[f'{model_type}_cv'] = cv_results
            
            # 2. Train-test split evaluation with simplified models
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels['is_high_quality'], test_size=self.config['test_size'], 
                random_state=self.config['random_state'], stratify=labels['is_high_quality']
            )
            
            # Train simplified model to avoid warnings
            if model_type == 'random_forest':
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(
                    n_estimators=100, 
                    max_depth=10, 
                    random_state=self.config['random_state'],
                    n_jobs=-1
                )
            elif model_type == 'xgboost':
                import xgboost as xgb
                model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=self.config['random_state'],
                    n_jobs=-1,
                    verbosity=0
                )
            elif model_type == 'ensemble':
                from sklearn.ensemble import VotingClassifier, RandomForestClassifier
                from sklearn.linear_model import LogisticRegression
                import xgboost as xgb
                
                rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=self.config['random_state'])
                xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, 
                                             random_state=self.config['random_state'], verbosity=0)
                lr = LogisticRegression(random_state=self.config['random_state'], max_iter=1000)
                
                model = VotingClassifier(
                    estimators=[('rf', rf), ('xgb', xgb_model), ('lr', lr)],
                    voting='soft'
                )
            else:
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(n_estimators=100, random_state=self.config['random_state'])
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # Create training results dictionary
            training_results = {
                'y_test': y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'model': model
            }
            
            # Evaluate model
            evaluation_results = self.evaluator.evaluate_model(
                training_results['y_test'],
                training_results['y_pred'],
                training_results['y_pred_proba'],
                model_type
            )
            
            all_results[model_type] = evaluation_results
            
            # 3. Threshold analysis
            if self.config['threshold_analysis']:
                threshold_results = self.perform_threshold_analysis(
                    training_results['y_test'], 
                    training_results['y_pred_proba']
                )
                all_results[f'{model_type}_threshold'] = threshold_results
            
            # 4. Feature importance analysis
            if self.config['feature_importance_analysis']:
                feature_importance = self.analyze_feature_importance(
                    model, features, labels, model_type
                )
                all_results[f'{model_type}_feature_importance'] = feature_importance
        
        return all_results
    
    def perform_threshold_analysis(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> dict:
        """Perform threshold analysis for optimal decision boundary"""
        logger.info("Performing threshold analysis...")
        
        thresholds = np.arange(0.1, 1.0, 0.05)
        results = {
            'thresholds': thresholds.tolist(),
            'precisions': [],
            'recalls': [],
            'f1_scores': [],
            'accuracies': []
        }
        
        for threshold in thresholds:
            y_pred = (y_pred_proba[:, 1] >= threshold).astype(int)
            
            # Calculate metrics
            from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
            
            results['precisions'].append(precision_score(y_true, y_pred, average='weighted'))
            results['recalls'].append(recall_score(y_true, y_pred, average='weighted'))
            results['f1_scores'].append(f1_score(y_true, y_pred, average='weighted'))
            results['accuracies'].append(accuracy_score(y_true, y_pred))
        
        # Find optimal threshold
        optimal_idx = np.argmax(results['f1_scores'])
        results['optimal_threshold'] = thresholds[optimal_idx]
        results['optimal_f1'] = results['f1_scores'][optimal_idx]
        
        logger.info(f"Optimal threshold: {results['optimal_threshold']:.3f} (F1: {results['optimal_f1']:.4f})")
        
        return results
    
    def analyze_feature_importance(self, model, features: pd.DataFrame, 
                                 labels: pd.Series, model_type: str) -> dict:
        """Analyze feature importance for the model"""
        logger.info(f"Analyzing feature importance for {model_type}")
        
        # Get feature importance directly from the trained model
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
        elif hasattr(model, 'estimators_'):
            # For ensemble models, try to get importance from base models
            try:
                importance = np.mean([est.feature_importances_ for est in model.estimators_], axis=0)
            except:
                importance = None
        else:
            importance = None
        
        if importance is not None:
            feature_importance_df = pd.DataFrame({
                'feature': features.columns,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            return {
                'feature_importance': feature_importance_df,
                'top_features': feature_importance_df.head(20).to_dict('records')
            }
        
        return {'feature_importance': None, 'top_features': []}
    
    def create_comprehensive_visualizations(self, all_results: dict, 
                                          features: pd.DataFrame, 
                                          labels: pd.Series) -> dict:
        """Create comprehensive visualizations for all results"""
        logger.info("="*60)
        logger.info("STEP 3: CREATING COMPREHENSIVE VISUALIZATIONS")
        logger.info("="*60)
        
        # Extract model evaluation results
        model_results = {k: v for k, v in all_results.items() 
                        if not k.endswith('_cv') and not k.endswith('_threshold') 
                        and not k.endswith('_feature_importance')}
        
        # Create performance dashboard
        dashboard = self.visualizer.create_comprehensive_performance_dashboard(
            model_results, save_plots=True
        )
        
        # Create additional specialized visualizations
        additional_plots = {}
        
        # 1. Cross-validation results visualization
        if self.config['cross_validation']:
            additional_plots['cross_validation'] = self.create_cv_visualization(all_results)
        
        # 2. Threshold analysis visualization
        if self.config['threshold_analysis']:
            additional_plots['threshold_analysis'] = self.create_threshold_visualization(all_results)
        
        # 3. Feature importance visualization
        if self.config['feature_importance_analysis']:
            additional_plots['feature_importance'] = self.create_feature_importance_visualization(all_results)
        
        # 4. Model comparison summary
        additional_plots['model_summary'] = self.create_model_summary_visualization(all_results)
        
        # Save additional plots
        for plot_name, fig in additional_plots.items():
            if fig is not None:
                html_path = self.visualizer.plots_dir / f"{plot_name}.html"
                fig.write_html(str(html_path))
                logger.info(f"Saved {plot_name} visualization")
        
        all_visualizations = {**dashboard, **additional_plots}
        return all_visualizations
    
    def create_cv_visualization(self, all_results: dict) -> go.Figure:
        """Create cross-validation results visualization"""
        
        cv_results = {k: v for k, v in all_results.items() if k.endswith('_cv')}
        
        if not cv_results:
            return None
        
        fig = go.Figure()
        
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        colors = px.colors.qualitative.Set3[:len(cv_results)]
        
        for i, (model_name, results) in enumerate(cv_results.items()):
            model_clean = model_name.replace('_cv', '')
            
            for metric in metrics:
                mean_score = results[f'{metric}_mean']
                std_score = results[f'{metric}_std']
                
                fig.add_trace(go.Bar(
                    name=f'{model_clean} {metric}',
                    x=[f'{model_clean} {metric}'],
                    y=[mean_score],
                    error_y=dict(type='data', array=[std_score], visible=True),
                    marker_color=colors[i],
                    showlegend=True
                ))
        
        fig.update_layout(
            title="Cross-Validation Results (Mean ± Std)",
            yaxis_title="Score",
            template="plotly_white",
            height=600,
            barmode='group'
        )
        
        return fig
    
    def create_threshold_visualization(self, all_results: dict) -> go.Figure:
        """Create threshold analysis visualization"""
        
        threshold_results = {k: v for k, v in all_results.items() if k.endswith('_threshold')}
        
        if not threshold_results:
            return None
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=["Precision vs Threshold", "Recall vs Threshold",
                           "F1 Score vs Threshold", "Accuracy vs Threshold"],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        colors = px.colors.qualitative.Set3[:len(threshold_results)]
        
        for i, (model_name, results) in enumerate(threshold_results.items()):
            model_clean = model_name.replace('_threshold', '')
            thresholds = results['thresholds']
            
            # Precision
            fig.add_trace(
                go.Scatter(x=thresholds, y=results['precisions'], 
                          mode='lines+markers', name=f'{model_clean} Precision',
                          line=dict(color=colors[i])),
                row=1, col=1
            )
            
            # Recall
            fig.add_trace(
                go.Scatter(x=thresholds, y=results['recalls'], 
                          mode='lines+markers', name=f'{model_clean} Recall',
                          line=dict(color=colors[i], dash='dash')),
                row=1, col=2
            )
            
            # F1 Score
            fig.add_trace(
                go.Scatter(x=thresholds, y=results['f1_scores'], 
                          mode='lines+markers', name=f'{model_clean} F1',
                          line=dict(color=colors[i], dash='dot')),
                row=2, col=1
            )
            
            # Accuracy
            fig.add_trace(
                go.Scatter(x=thresholds, y=results['accuracies'], 
                          mode='lines+markers', name=f'{model_clean} Accuracy',
                          line=dict(color=colors[i], dash='longdash')),
                row=2, col=2
            )
        
        fig.update_layout(
            title="Threshold Analysis Across Models",
            template="plotly_white",
            height=800
        )
        
        return fig
    
    def create_feature_importance_visualization(self, all_results: dict) -> go.Figure:
        """Create feature importance visualization"""
        
        feature_results = {k: v for k, v in all_results.items() if k.endswith('_feature_importance')}
        
        if not feature_results:
            return None
        
        # Create subplots for each model
        n_models = len(feature_results)
        fig = make_subplots(
            rows=1, cols=n_models,
            subplot_titles=[name.replace('_feature_importance', '') for name in feature_results.keys()],
            specs=[[{"type": "bar"} for _ in range(n_models)]]
        )
        
        colors = px.colors.qualitative.Set3[:n_models]
        
        for i, (model_name, results) in enumerate(feature_results.items()):
            if results['feature_importance'] is not None:
                top_features = results['feature_importance'].head(10)
                
                fig.add_trace(
                    go.Bar(
                        x=top_features['importance'],
                        y=top_features['feature'],
                        orientation='h',
                        marker_color=colors[i],
                        name=model_name.replace('_feature_importance', ''),
                        showlegend=False
                    ),
                    row=1, col=i+1
                )
        
        fig.update_layout(
            title="Top 10 Feature Importance by Model",
            template="plotly_white",
            height=600
        )
        
        return fig
    
    def create_model_summary_visualization(self, all_results: dict) -> go.Figure:
        """Create comprehensive model summary visualization"""
        
        model_results = {k: v for k, v in all_results.items() 
                        if not k.endswith('_cv') and not k.endswith('_threshold') 
                        and not k.endswith('_feature_importance')}
        
        # Create radar chart for comprehensive comparison
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc']
        
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set3[:len(model_results)]
        
        for i, (model_name, results) in enumerate(model_results.items()):
            values = [results.get(metric, 0) for metric in metrics]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=[metric.replace('_', ' ').title() for metric in metrics],
                fill='toself',
                name=model_name,
                line_color=colors[i]
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Model Performance Radar Chart",
            template="plotly_white",
            height=600
        )
        
        return fig
    
    def generate_comprehensive_report(self, all_results: dict, 
                                    features: pd.DataFrame, 
                                    labels: pd.Series) -> str:
        """Generate comprehensive testing report"""
        logger.info("="*60)
        logger.info("STEP 4: GENERATING COMPREHENSIVE REPORT")
        logger.info("="*60)
        
        # Get model evaluation results
        model_results = {k: v for k, v in all_results.items() 
                        if not k.endswith('_cv') and not k.endswith('_threshold') 
                        and not k.endswith('_feature_importance')}
        
        # Generate summary report
        summary_report = self.visualizer.create_summary_report(model_results)
        
        # Add cross-validation results
        cv_results = {k: v for k, v in all_results.items() if k.endswith('_cv')}
        if cv_results:
            summary_report += "\n\nCROSS-VALIDATION RESULTS:\n" + "="*50 + "\n"
            for model_name, results in cv_results.items():
                model_clean = model_name.replace('_cv', '')
                summary_report += f"\n{model_clean}:\n"
                summary_report += f"  Accuracy: {results['accuracy_mean']:.4f} ± {results['accuracy_std']:.4f}\n"
                summary_report += f"  F1 Score: {results['f1_mean']:.4f} ± {results['f1_std']:.4f}\n"
        
        # Add threshold analysis results
        threshold_results = {k: v for k, v in all_results.items() if k.endswith('_threshold')}
        if threshold_results:
            summary_report += "\n\nTHRESHOLD ANALYSIS:\n" + "="*50 + "\n"
            for model_name, results in threshold_results.items():
                model_clean = model_name.replace('_threshold', '')
                summary_report += f"\n{model_clean}:\n"
                summary_report += f"  Optimal Threshold: {results['optimal_threshold']:.3f}\n"
                summary_report += f"  Optimal F1 Score: {results['optimal_f1']:.4f}\n"
        
        # Save report
        report_path = f"reports/rigorous_testing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_path, 'w') as f:
            f.write(summary_report)
        
        logger.info(f"Comprehensive report saved to: {report_path}")
        
        return summary_report
    
    def run_complete_rigorous_testing(self) -> dict:
        """Run complete rigorous testing pipeline"""
        logger.info("Starting Rigorous Testing and Visualization Pipeline")
        logger.info(f"Configuration: {self.config}")
        
        try:
            # Step 1: Load and prepare data
            features, labels, processed_df = self.load_and_prepare_data()
            
            # Step 2: Perform rigorous testing
            all_results = self.perform_rigorous_testing(features, labels)
            
            # Step 3: Create comprehensive visualizations
            all_visualizations = self.create_comprehensive_visualizations(all_results, features, labels)
            
            # Step 4: Generate comprehensive report
            report = self.generate_comprehensive_report(all_results, features, labels)
            
            logger.info("="*60)
            logger.info("RIGOROUS TESTING PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("="*60)
            logger.info(f"Total visualizations created: {len(all_visualizations)}")
            logger.info(f"Plots saved to: {self.visualizer.plots_dir}")
            logger.info(f"Report saved to: reports/")
            
            return {
                'success': True,
                'all_results': all_results,
                'visualizations': all_visualizations,
                'report': report,
                'features': features,
                'labels': labels,
                'processed_data': processed_df
            }
            
        except Exception as e:
            logger.error(f"Rigorous testing pipeline failed: {e}")
            return {'success': False, 'error': str(e)}

def main():
    """Main function for running rigorous testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Rigorous Testing and Visualization Pipeline')
    parser.add_argument('--models', nargs='+', default=['ensemble', 'xgboost', 'random_forest'], 
                       help='Models to test')
    parser.add_argument('--cv-folds', type=int, default=5, 
                       help='Number of cross-validation folds')
    parser.add_argument('--max-features', type=int, default=2000, 
                       help='Maximum number of features')
    parser.add_argument('--no-cv', action='store_true', 
                       help='Skip cross-validation')
    parser.add_argument('--no-threshold', action='store_true', 
                       help='Skip threshold analysis')
    parser.add_argument('--no-feature-importance', action='store_true', 
                       help='Skip feature importance analysis')
    
    args = parser.parse_args()
    
    # Create configuration
    config = {
        'max_features': args.max_features,
        'n_topics': 10,
        'test_size': 0.2,
        'random_state': 42,
        'cv_folds': args.cv_folds,
        'models': args.models,
        'save_models': True,
        'generate_visualizations': True,
        'cross_validation': not args.no_cv,
        'threshold_analysis': not args.no_threshold,
        'feature_importance_analysis': not args.no_feature_importance
    }
    
    # Run rigorous testing
    tester = RigorousTestingVisualizer(config)
    results = tester.run_complete_rigorous_testing()
    
    if results['success']:
        logger.info("Rigorous testing completed successfully!")
        return 0
    else:
        logger.error(f"Rigorous testing failed: {results['error']}")
        return 1

if __name__ == "__main__":
    exit(main())
