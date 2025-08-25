#!/usr/bin/env python3
"""
Comprehensive performance visualization module for model evaluation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple, Optional
import logging
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    precision_recall_fscore_support, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score, f1_score
)
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceVisualizer:
    """Comprehensive performance visualization for model evaluation"""
    
    def __init__(self, results_dir: str = "results", plots_dir: str = "plots"):
        self.results_dir = Path(results_dir)
        self.plots_dir = Path(plots_dir)
        self.plots_dir.mkdir(exist_ok=True)
        
        # Set style for matplotlib
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def create_comprehensive_performance_dashboard(self, model_results: Dict[str, Dict[str, Any]], 
                                                 save_plots: bool = True) -> Dict[str, go.Figure]:
        """Create comprehensive performance dashboard with all visualizations"""
        
        logger.info("Creating comprehensive performance dashboard...")
        
        dashboard = {}
        
        # 1. Model Comparison Overview
        dashboard['model_comparison'] = self.create_model_comparison_plot(model_results)
        
        # 2. ROC Curves
        dashboard['roc_curves'] = self.create_roc_curves_plot(model_results)
        
        # 3. Precision-Recall Curves
        dashboard['pr_curves'] = self.create_precision_recall_curves_plot(model_results)
        
        # 4. Confusion Matrices
        dashboard['confusion_matrices'] = self.create_confusion_matrices_plot(model_results)
        
        # 5. Performance Metrics Heatmap
        dashboard['metrics_heatmap'] = self.create_metrics_heatmap(model_results)
        
        # 6. F1 Score Comparison
        dashboard['f1_comparison'] = self.create_f1_score_comparison(model_results)
        
        # 7. Error Analysis
        dashboard['error_analysis'] = self.create_error_analysis_plot(model_results)
        
        # 8. Threshold Analysis
        dashboard['threshold_analysis'] = self.create_threshold_analysis_plot(model_results)
        
        # 9. Performance by Class
        dashboard['class_performance'] = self.create_class_performance_plot(model_results)
        
        # 10. Model Confidence Distribution
        dashboard['confidence_distribution'] = self.create_confidence_distribution_plot(model_results)
        
        # Save plots if requested
        if save_plots:
            self.save_all_plots(dashboard)
        
        return dashboard
    
    def create_model_comparison_plot(self, model_results: Dict[str, Dict[str, Any]]) -> go.Figure:
        """Create comprehensive model comparison visualization"""
        
        # Extract metrics
        models = list(model_results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[metric.replace('_', ' ').title() for metric in metrics],
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        
        colors = px.colors.qualitative.Set3[:len(models)]
        
        for i, metric in enumerate(metrics):
            row = (i // 3) + 1
            col = (i % 3) + 1
            
            values = [model_results[model].get(metric, 0) for model in models]
            
            fig.add_trace(
                go.Bar(
                    x=models,
                    y=values,
                    name=metric.replace('_', ' ').title(),
                    marker_color=colors,
                    text=[f'{v:.3f}' for v in values],
                    textposition='auto',
                    showlegend=False
                ),
                row=row, col=col
            )
            
            # Add horizontal line for baseline (0.5 for most metrics)
            baseline = 0.5 if metric in ['accuracy', 'precision', 'recall', 'f1_score'] else 0.7
            fig.add_hline(y=baseline, line_dash="dash", line_color="red", 
                         annotation_text=f"Baseline ({baseline})", 
                         row=row, col=col)
        
        fig.update_layout(
            title="Comprehensive Model Performance Comparison",
            height=800,
            showlegend=False,
            template="plotly_white"
        )
        
        return fig
    
    def create_roc_curves_plot(self, model_results: Dict[str, Dict[str, Any]]) -> go.Figure:
        """Create ROC curves comparison plot"""
        
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set3[:len(model_results)]
        
        for i, (model_name, results) in enumerate(model_results.items()):
            if results.get('y_pred_proba') is not None:
                y_true = results['y_true']
                y_pred_proba = results['y_pred_proba'][:, 1]
                
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
                auc = roc_auc_score(y_true, y_pred_proba)
                
                fig.add_trace(
                    go.Scatter(
                        x=fpr, y=tpr,
                        mode='lines',
                        name=f'{model_name} (AUC = {auc:.3f})',
                        line=dict(color=colors[i], width=3),
                        fill='tonexty' if i == 0 else None
                    )
                )
        
        # Add diagonal line
        fig.add_trace(
            go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random Classifier',
                line=dict(color='red', dash='dash', width=2),
                showlegend=True
            )
        )
        
        fig.update_layout(
            title="ROC Curves Comparison",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            template="plotly_white",
            height=600
        )
        
        return fig
    
    def create_precision_recall_curves_plot(self, model_results: Dict[str, Dict[str, Any]]) -> go.Figure:
        """Create Precision-Recall curves comparison plot"""
        
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set3[:len(model_results)]
        
        for i, (model_name, results) in enumerate(model_results.items()):
            if results.get('y_pred_proba') is not None:
                y_true = results['y_true']
                y_pred_proba = results['y_pred_proba'][:, 1]
                
                precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
                auc = average_precision_score(y_true, y_pred_proba)
                
                fig.add_trace(
                    go.Scatter(
                        x=recall, y=precision,
                        mode='lines',
                        name=f'{model_name} (AP = {auc:.3f})',
                        line=dict(color=colors[i], width=3)
                    )
                )
        
        # Add baseline (proportion of positive class)
        if model_results:
            first_model = list(model_results.values())[0]
            baseline = np.mean(first_model['y_true'])
            fig.add_hline(y=baseline, line_dash="dash", line_color="red",
                         annotation_text=f"Baseline ({baseline:.3f})")
        
        fig.update_layout(
            title="Precision-Recall Curves Comparison",
            xaxis_title="Recall",
            yaxis_title="Precision",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            template="plotly_white",
            height=600
        )
        
        return fig
    
    def create_confusion_matrices_plot(self, model_results: Dict[str, Dict[str, Any]]) -> go.Figure:
        """Create confusion matrices comparison plot"""
        
        n_models = len(model_results)
        fig = make_subplots(
            rows=1, cols=n_models,
            subplot_titles=list(model_results.keys()),
            specs=[[{"type": "heatmap"} for _ in range(n_models)]]
        )
        
        for i, (model_name, results) in enumerate(model_results.items()):
            cm = results['confusion_matrix']
            
            # Normalize confusion matrix
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            fig.add_trace(
                go.Heatmap(
                    z=cm_normalized,
                    x=['Predicted 0', 'Predicted 1'],
                    y=['Actual 0', 'Actual 1'],
                    colorscale='Blues',
                    text=cm,
                    texttemplate="%{text}",
                    textfont={"size": 16},
                    showscale=False
                ),
                row=1, col=i+1
            )
        
        fig.update_layout(
            title="Confusion Matrices Comparison (Normalized)",
            height=400,
            template="plotly_white"
        )
        
        return fig
    
    def create_metrics_heatmap(self, model_results: Dict[str, Dict[str, Any]]) -> go.Figure:
        """Create metrics heatmap for easy comparison"""
        
        # Prepare data
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc']
        models = list(model_results.keys())
        
        data = []
        for model in models:
            row = []
            for metric in metrics:
                value = model_results[model].get(metric, 0)
                row.append(value)
            data.append(row)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=data,
            x=[metric.replace('_', ' ').title() for metric in metrics],
            y=models,
            colorscale='RdYlGn',
            text=[[f'{val:.3f}' for val in row] for row in data],
            texttemplate="%{text}",
            textfont={"size": 12},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Model Performance Metrics Heatmap",
            xaxis_title="Metrics",
            yaxis_title="Models",
            template="plotly_white",
            height=400
        )
        
        return fig
    
    def create_f1_score_comparison(self, model_results: Dict[str, Dict[str, Any]]) -> go.Figure:
        """Create detailed F1 score comparison"""
        
        models = list(model_results.keys())
        f1_scores = [model_results[model]['f1_score'] for model in models]
        
        # Create bar plot with error bars (if available)
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=models,
            y=f1_scores,
            text=[f'{score:.3f}' for score in f1_scores],
            textposition='auto',
            marker_color='lightblue',
            name='F1 Score'
        ))
        
        # Add horizontal line for good performance threshold
        fig.add_hline(y=0.8, line_dash="dash", line_color="orange",
                     annotation_text="Good Performance (0.8)")
        fig.add_hline(y=0.9, line_dash="dash", line_color="green",
                     annotation_text="Excellent Performance (0.9)")
        
        fig.update_layout(
            title="F1 Score Comparison Across Models",
            xaxis_title="Models",
            yaxis_title="F1 Score",
            template="plotly_white",
            height=500
        )
        
        return fig
    
    def create_error_analysis_plot(self, model_results: Dict[str, Dict[str, Any]]) -> go.Figure:
        """Create error analysis visualization"""
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["False Positives vs False Negatives", "Error Rate by Model"],
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        # False Positives vs False Negatives
        models = list(model_results.keys())
        false_positives = []
        false_negatives = []
        
        for model in models:
            cm = model_results[model]['confusion_matrix']
            false_positives.append(cm[0, 1])  # Predicted 1, Actual 0
            false_negatives.append(cm[1, 0])  # Predicted 0, Actual 1
        
        fig.add_trace(
            go.Bar(name='False Positives', x=models, y=false_positives, marker_color='red'),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(name='False Negatives', x=models, y=false_negatives, marker_color='orange'),
            row=1, col=1
        )
        
        # Error Rate
        total_errors = [fp + fn for fp, fn in zip(false_positives, false_negatives)]
        total_samples = [len(model_results[model]['y_true']) for model in models]
        error_rates = [err/total for err, total in zip(total_errors, total_samples)]
        
        fig.add_trace(
            go.Bar(x=models, y=error_rates, marker_color='purple', name='Error Rate'),
            row=1, col=2
        )
        
        fig.update_layout(
            title="Error Analysis",
            template="plotly_white",
            height=500,
            barmode='group'
        )
        
        return fig
    
    def create_threshold_analysis_plot(self, model_results: Dict[str, Dict[str, Any]]) -> go.Figure:
        """Create threshold analysis plot"""
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Precision vs Threshold", "Recall vs Threshold"],
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        thresholds = np.arange(0.1, 1.0, 0.05)
        colors = px.colors.qualitative.Set3[:len(model_results)]
        
        for i, (model_name, results) in enumerate(model_results.items()):
            if results.get('y_pred_proba') is not None:
                y_true = results['y_true']
                y_pred_proba = results['y_pred_proba'][:, 1]
                
                precisions = []
                recalls = []
                
                for threshold in thresholds:
                    y_pred_threshold = (y_pred_proba >= threshold).astype(int)
                    precision = precision_recall_fscore_support(y_true, y_pred_threshold, average='weighted')[0]
                    recall = precision_recall_fscore_support(y_true, y_pred_threshold, average='weighted')[1]
                    precisions.append(precision)
                    recalls.append(recall)
                
                fig.add_trace(
                    go.Scatter(x=thresholds, y=precisions, mode='lines+markers',
                              name=f'{model_name} Precision', line=dict(color=colors[i])),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=thresholds, y=recalls, mode='lines+markers',
                              name=f'{model_name} Recall', line=dict(color=colors[i], dash='dash')),
                    row=1, col=2
                )
        
        fig.update_layout(
            title="Threshold Analysis",
            template="plotly_white",
            height=500
        )
        
        return fig
    
    def create_class_performance_plot(self, model_results: Dict[str, Dict[str, Any]]) -> go.Figure:
        """Create class-wise performance comparison"""
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Class 0 Performance", "Class 1 Performance"],
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        models = list(model_results.keys())
        metrics = ['precision', 'recall', 'f1-score']
        
        # Class 0 performance
        for i, metric in enumerate(metrics):
            values = []
            for model in models:
                class_report = model_results[model]['classification_report']
                if '0' in class_report and isinstance(class_report['0'], dict):
                    values.append(class_report['0'].get(metric, 0))
                else:
                    values.append(0)
            
            fig.add_trace(
                go.Bar(name=f'Class 0 {metric}', x=models, y=values,
                       marker_color=px.colors.qualitative.Set3[i]),
                row=1, col=1
            )
        
        # Class 1 performance
        for i, metric in enumerate(metrics):
            values = []
            for model in models:
                class_report = model_results[model]['classification_report']
                if '1' in class_report and isinstance(class_report['1'], dict):
                    values.append(class_report['1'].get(metric, 0))
                else:
                    values.append(0)
            
            fig.add_trace(
                go.Bar(name=f'Class 1 {metric}', x=models, y=values,
                       marker_color=px.colors.qualitative.Set3[i+3]),
                row=1, col=2
            )
        
        fig.update_layout(
            title="Class-wise Performance Comparison",
            template="plotly_white",
            height=500,
            barmode='group'
        )
        
        return fig
    
    def create_confidence_distribution_plot(self, model_results: Dict[str, Dict[str, Any]]) -> go.Figure:
        """Create confidence distribution plot"""
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Prediction Confidence Distribution", "Confidence vs Accuracy"],
            specs=[[{"type": "histogram"}, {"type": "scatter"}]]
        )
        
        colors = px.colors.qualitative.Set3[:len(model_results)]
        
        for i, (model_name, results) in enumerate(model_results.items()):
            if results.get('y_pred_proba') is not None:
                y_pred_proba = results['y_pred_proba'][:, 1]
                
                # Confidence distribution
                fig.add_trace(
                    go.Histogram(x=y_pred_proba, nbinsx=20, name=model_name,
                                marker_color=colors[i], opacity=0.7),
                    row=1, col=1
                )
                
                # Confidence vs accuracy (binned)
                bins = np.linspace(0, 1, 11)
                bin_centers = (bins[:-1] + bins[1:]) / 2
                accuracies = []
                
                for j in range(len(bins)-1):
                    mask = (y_pred_proba >= bins[j]) & (y_pred_proba < bins[j+1])
                    if mask.sum() > 0:
                        y_true_bin = results['y_true'][mask]
                        y_pred_bin = (y_pred_proba[mask] >= 0.5).astype(int)
                        accuracy = accuracy_score(y_true_bin, y_pred_bin)
                        accuracies.append(accuracy)
                    else:
                        accuracies.append(np.nan)
                
                fig.add_trace(
                    go.Scatter(x=bin_centers, y=accuracies, mode='lines+markers',
                              name=f'{model_name} Accuracy', line=dict(color=colors[i])),
                    row=1, col=2
                )
        
        fig.update_layout(
            title="Prediction Confidence Analysis",
            template="plotly_white",
            height=500
        )
        
        return fig
    
    def save_all_plots(self, dashboard: Dict[str, go.Figure]) -> None:
        """Save all plots to files"""
        
        logger.info(f"Saving plots to {self.plots_dir}...")
        
        for plot_name, fig in dashboard.items():
            # Save as HTML (interactive)
            html_path = self.plots_dir / f"{plot_name}.html"
            fig.write_html(str(html_path))
            
            logger.info(f"Saved {plot_name} to {html_path}")
    
    def create_summary_report(self, model_results: Dict[str, Dict[str, Any]]) -> str:
        """Create a summary report of all visualizations"""
        
        report = f"""
{'='*80}
COMPREHENSIVE MODEL PERFORMANCE ANALYSIS REPORT
{'='*80}

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Models Analyzed: {len(model_results)}

{'='*80}
EXECUTIVE SUMMARY
{'='*80}
"""
        
        # Find best model
        best_model = max(model_results.items(), key=lambda x: x[1]['f1_score'])
        report += f"Best Performing Model: {best_model[0]} (F1 Score: {best_model[1]['f1_score']:.4f})\n\n"
        
        # Performance summary
        report += "Performance Summary:\n"
        for model_name, results in model_results.items():
            report += f"- {model_name}: F1={results['f1_score']:.4f}, "
            report += f"Accuracy={results['accuracy']:.4f}, "
            report += f"ROC AUC={results.get('roc_auc', 'N/A')}\n"
        
        report += f"""
{'='*80}
DETAILED ANALYSIS
{'='*80}

1. MODEL COMPARISON PLOT: Shows comprehensive comparison across all metrics
2. ROC CURVES: Displays discrimination ability of each model
3. PRECISION-RECALL CURVES: Shows precision-recall trade-offs
4. CONFUSION MATRICES: Visualizes classification errors
5. METRICS HEATMAP: Easy-to-read performance comparison
6. F1 SCORE COMPARISON: Detailed F1 score analysis
7. ERROR ANALYSIS: False positive vs false negative breakdown
8. THRESHOLD ANALYSIS: Optimal threshold selection
9. CLASS PERFORMANCE: Per-class performance comparison
10. CONFIDENCE DISTRIBUTION: Prediction confidence analysis

{'='*80}
RECOMMENDATIONS
{'='*80}
"""
        
        # Add recommendations based on analysis
        f1_scores = [results['f1_score'] for results in model_results.values()]
        avg_f1 = np.mean(f1_scores)
        
        if avg_f1 >= 0.9:
            report += "- Excellent overall performance! Models are ready for production.\n"
        elif avg_f1 >= 0.8:
            report += "- Good performance. Consider fine-tuning for marginal improvements.\n"
        elif avg_f1 >= 0.7:
            report += "- Acceptable performance. May benefit from feature engineering.\n"
        else:
            report += "- Performance needs improvement. Consider data quality and model selection.\n"
        
        # Check for class imbalance issues
        for model_name, results in model_results.items():
            cm = results['confusion_matrix']
            if cm[0, 1] > cm[1, 0]:  # More false positives than false negatives
                report += f"- {model_name}: Consider adjusting threshold to reduce false positives.\n"
            elif cm[1, 0] > cm[0, 1]:  # More false negatives than false positives
                report += f"- {model_name}: Consider adjusting threshold to catch more positive cases.\n"
        
        report += f"""
{'='*80}
PLOTS GENERATED
{'='*80}
All plots have been saved to: {self.plots_dir}

Files generated:
- HTML files (interactive): Available in plots directory

{'='*80}
"""
        
        return report
