#!/usr/bin/env python3
"""
Model evaluation and reporting module
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple
import logging
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    precision_recall_fscore_support, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation and reporting"""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        self.evaluation_results = {}
        
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      y_pred_proba: np.ndarray = None, model_name: str = "model") -> Dict[str, Any]:
        """Evaluate model performance"""
        
        logger.info(f"Evaluating {model_name}...")
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        
        # Detailed classification report
        class_report = classification_report(y_true, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # ROC and PR curves if probabilities are available
        roc_auc = None
        pr_auc = None
        if y_pred_proba is not None:
            try:
                roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
                pr_auc = average_precision_score(y_true, y_pred_proba[:, 1])
            except:
                logger.warning("Could not calculate ROC/PR AUC scores")
        
        # Store results
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': support,
            'classification_report': class_report,
            'confusion_matrix': cm,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        self.evaluation_results[model_name] = results
        
        logger.info(f"Evaluation completed for {model_name}")
        logger.info(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        return results
    
    def compare_models(self, model_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """Compare multiple models"""
        
        comparison_data = []
        
        for model_name, results in model_results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1_Score': results['f1_score'],
                'ROC_AUC': results.get('roc_auc', np.nan),
                'PR_AUC': results.get('pr_auc', np.nan)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('F1_Score', ascending=False)
        
        return comparison_df
    
    def create_confusion_matrix_plot(self, cm: np.ndarray, model_name: str, 
                                   class_names: List[str] = None) -> go.Figure:
        """Create confusion matrix visualization"""
        
        if class_names is None:
            class_names = ['Low Quality', 'High Quality']
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=cm_percent,
            x=class_names,
            y=class_names,
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 16},
            colorscale='Blues',
            showscale=True,
            colorbar=dict(title="Percentage")
        ))
        
        fig.update_layout(
            title=f"Confusion Matrix - {model_name}",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            width=600,
            height=500
        )
        
        return fig
    
    def create_roc_curve_plot(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                             model_name: str) -> go.Figure:
        """Create ROC curve visualization"""
        
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
        roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
        
        fig = go.Figure()
        
        # ROC curve
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {roc_auc:.3f})',
            line=dict(color='blue', width=2)
        ))
        
        # Diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title=f"ROC Curve - {model_name}",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            width=600,
            height=500,
            showlegend=True
        )
        
        return fig
    
    def create_precision_recall_curve_plot(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                                         model_name: str) -> go.Figure:
        """Create Precision-Recall curve visualization"""
        
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba[:, 1])
        pr_auc = average_precision_score(y_true, y_pred_proba[:, 1])
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=recall, y=precision,
            mode='lines',
            name=f'PR Curve (AUC = {pr_auc:.3f})',
            line=dict(color='green', width=2)
        ))
        
        fig.update_layout(
            title=f"Precision-Recall Curve - {model_name}",
            xaxis_title="Recall",
            yaxis_title="Precision",
            width=600,
            height=500,
            showlegend=True
        )
        
        return fig
    
    def create_feature_importance_plot(self, feature_importance: pd.DataFrame, 
                                     model_name: str, top_n: int = 20) -> go.Figure:
        """Create feature importance visualization"""
        
        # Get top N features
        top_features = feature_importance.head(top_n)
        
        fig = go.Figure(data=go.Bar(
            x=top_features['importance'],
            y=top_features['feature'],
            orientation='h',
            marker_color='lightblue'
        ))
        
        fig.update_layout(
            title=f"Top {top_n} Feature Importance - {model_name}",
            xaxis_title="Importance",
            yaxis_title="Features",
            width=800,
            height=max(400, top_n * 15)
        )
        
        return fig
    
    def create_performance_summary_plot(self, comparison_df: pd.DataFrame) -> go.Figure:
        """Create performance comparison visualization"""
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score']
        
        fig = go.Figure()
        
        for metric in metrics:
            fig.add_trace(go.Bar(
                name=metric,
                x=comparison_df['Model'],
                y=comparison_df[metric],
                text=comparison_df[metric].round(3),
                textposition='auto'
            ))
        
        fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Models",
            yaxis_title="Score",
            barmode='group',
            width=800,
            height=500
        )
        
        return fig
    
    def generate_evaluation_report(self, model_name: str = None) -> str:
        """Generate comprehensive evaluation report"""
        
        if model_name is None:
            # Generate report for all models
            report_parts = []
            
            for name in self.evaluation_results.keys():
                report_parts.append(self._generate_single_model_report(name))
            
            return "\n\n" + "="*80 + "\n\n".join(report_parts)
        else:
            return self._generate_single_model_report(model_name)
    
    def _generate_single_model_report(self, model_name: str) -> str:
        """Generate report for a single model"""
        
        if model_name not in self.evaluation_results:
            raise ValueError(f"Model {model_name} not found in evaluation results")
        
        results = self.evaluation_results[model_name]
        
        report = f"""
{'='*80}
MODEL EVALUATION REPORT: {model_name.upper()}
{'='*80}

PERFORMANCE METRICS:
- Accuracy:  {results['accuracy']:.4f}
- Precision: {results['precision']:.4f}
- Recall:    {results['recall']:.4f}
- F1 Score:  {results['f1_score']:.4f}

"""
        
        if results['roc_auc'] is not None:
            report += f"- ROC AUC:   {results['roc_auc']:.4f}\n"
        if results['pr_auc'] is not None:
            report += f"- PR AUC:    {results['pr_auc']:.4f}\n"
        
        report += f"""
DETAILED CLASSIFICATION REPORT:
{'-'*40}
"""
        
        # Convert classification report to string
        class_report = results['classification_report']
        for class_name, metrics in class_report.items():
            if isinstance(metrics, dict):
                report += f"\n{class_name}:\n"
                for metric, value in metrics.items():
                    if isinstance(value, float):
                        report += f"  {metric}: {value:.4f}\n"
                    else:
                        report += f"  {metric}: {value}\n"
        
        report += f"""
CONFUSION MATRIX:
{'-'*40}
{results['confusion_matrix']}

RECOMMENDATIONS:
{'-'*40}
"""
        
        # Add recommendations based on performance
        f1_score = results['f1_score']
        if f1_score >= 0.9:
            report += "- Excellent performance! Model is ready for production use.\n"
        elif f1_score >= 0.8:
            report += "- Good performance. Consider fine-tuning for better results.\n"
        elif f1_score >= 0.7:
            report += "- Acceptable performance. May need feature engineering or model selection.\n"
        else:
            report += "- Poor performance. Requires significant improvements in data quality or model selection.\n"
        
        # Add specific recommendations
        precision = results['precision']
        recall = results['recall']
        
        if precision < recall:
            report += "- Model has high recall but low precision. Consider adjusting threshold to reduce false positives.\n"
        elif recall < precision:
            report += "- Model has high precision but low recall. Consider adjusting threshold to catch more positive cases.\n"
        
        return report
    
    def save_evaluation_results(self, filename: str = None):
        """Save evaluation results to file"""
        
        if filename is None:
            filename = f"evaluation_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        filepath = f"{self.results_dir}/{filename}"
        
        # Create results directory if it doesn't exist
        import os
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Generate and save report
        report = self.generate_evaluation_report()
        
        with open(filepath, 'w') as f:
            f.write(report)
        
        logger.info(f"Evaluation report saved to {filepath}")
        
        return filepath
    
    def create_comprehensive_visualization(self, model_name: str = None) -> go.Figure:
        """Create comprehensive visualization dashboard"""
        
        if model_name is None:
            # Create comparison dashboard
            comparison_df = self.compare_models(self.evaluation_results)
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Model Performance Comparison', 'Feature Importance', 
                              'Confusion Matrix', 'ROC Curve'),
                specs=[[{"type": "bar"}, {"type": "bar"}],
                       [{"type": "heatmap"}, {"type": "scatter"}]]
            )
            
            # Performance comparison
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score']
            for i, metric in enumerate(metrics):
                fig.add_trace(
                    go.Bar(name=metric, x=comparison_df['Model'], y=comparison_df[metric]),
                    row=1, col=1
                )
            
            # Add other visualizations as needed
            fig.update_layout(height=800, title_text="Model Evaluation Dashboard")
            
        else:
            # Create single model dashboard
            results = self.evaluation_results[model_name]
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Confusion Matrix', 'ROC Curve', 
                              'Precision-Recall Curve', 'Performance Metrics'),
                specs=[[{"type": "heatmap"}, {"type": "scatter"}],
                       [{"type": "scatter"}, {"type": "bar"}]]
            )
            
            # Confusion matrix
            cm_fig = self.create_confusion_matrix_plot(results['confusion_matrix'], model_name)
            fig.add_trace(cm_fig.data[0], row=1, col=1)
            
            # ROC curve
            if results['y_pred_proba'] is not None:
                roc_fig = self.create_roc_curve_plot(results['y_true'], results['y_pred_proba'], model_name)
                fig.add_trace(roc_fig.data[0], row=1, col=2)
                fig.add_trace(roc_fig.data[1], row=1, col=2)
            
            fig.update_layout(height=800, title_text=f"Evaluation Dashboard - {model_name}")
        
        return fig

if __name__ == "__main__":
    # Test the evaluator
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
    
    # Train multiple models
    models = ['random_forest', 'xgboost', 'ensemble']
    evaluator = ModelEvaluator()
    
    for model_type in models:
        classifier = ReviewQualityClassifier(model_type=model_type)
        results = classifier.train_model(features, labels['is_high_quality'])
        
        # Evaluate model
        evaluator.evaluate_model(
            results['y_test'],
            results['y_pred'],
            results['y_pred_proba'],
            model_type
        )
    
    # Generate comparison
    comparison_df = evaluator.compare_models(evaluator.evaluation_results)
    print("\nModel Comparison:")
    print(comparison_df)
    
    # Generate report
    report = evaluator.generate_evaluation_report()
    print(report)
    
    # Save results
    evaluator.save_evaluation_results()
