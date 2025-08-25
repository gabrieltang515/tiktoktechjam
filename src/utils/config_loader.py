#!/usr/bin/env python3
"""
Configuration loader utility
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class ConfigLoader:
    """Load and manage configuration settings"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.config = {}
        self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as file:
                    self.config = yaml.safe_load(file)
                logger.info(f"Configuration loaded from {self.config_path}")
            else:
                logger.warning(f"Configuration file not found: {self.config_path}")
                self.config = self._get_default_config()
                
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            self.config = self._get_default_config()
        
        return self.config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'data': {
                'max_features': 2000,
                'n_topics': 10,
                'test_size': 0.2,
                'random_state': 42
            },
            'models': {
                'classical': ['random_forest', 'xgboost', 'ensemble'],
                'transformer': {
                    'model_name': 'distilbert-base-uncased',
                    'max_length': 512,
                    'epochs': 3,
                    'batch_size': 16
                }
            },
            'features': {
                'text': {
                    'tfidf_max_features': 2000,
                    'count_max_features': 1000,
                    'ngram_range': [1, 2],
                    'min_df': 2,
                    'max_df': 0.95
                },
                'topics': {
                    'n_topics': 10,
                    'max_iter': 200
                },
                'sentiment': {
                    'use_textblob': True,
                    'use_vader': False
                }
            },
            'policy': {
                'thresholds': {
                    'spam': 0.3,
                    'advertisement': 0.25,
                    'irrelevant_content': 0.2,
                    'rant': 0.15,
                    'excessive_patterns': 0.05
                },
                'severity': {
                    'high': 0.7,
                    'medium': 0.4,
                    'low': 0.2
                }
            },
            'evaluation': {
                'metrics': ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc'],
                'cv_folds': 5,
                'scoring': 'f1_weighted'
            },
            'output': {
                'save_models': True,
                'save_reports': True,
                'generate_visualizations': True,
                'results_dir': 'results',
                'models_dir': 'models',
                'logs_dir': 'logs'
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file': 'pipeline.log'
            },
            'performance': {
                'n_jobs': -1,
                'memory_efficient': False,
                'early_stopping': True
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports nested keys with dot notation)"""
        
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value by key (supports nested keys with dot notation)"""
        
        keys = key.split('.')
        config = self.config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
    
    def save_config(self, filepath: str = None) -> None:
        """Save configuration to file"""
        
        if filepath is None:
            filepath = self.config_path
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        try:
            with open(filepath, 'w') as file:
                yaml.dump(self.config, file, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def update_from_dict(self, updates: Dict[str, Any]) -> None:
        """Update configuration from dictionary"""
        
        def update_nested(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = update_nested(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        self.config = update_nested(self.config, updates)
        logger.info("Configuration updated from dictionary")
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model-specific configuration"""
        return {
            'classical_models': self.get('models.classical', []),
            'transformer_config': self.get('models.transformer', {}),
            'max_features': self.get('data.max_features', 2000),
            'n_topics': self.get('data.n_topics', 10),
            'test_size': self.get('data.test_size', 0.2),
            'random_state': self.get('data.random_state', 42)
        }
    
    def get_feature_config(self) -> Dict[str, Any]:
        """Get feature engineering configuration"""
        return {
            'text_features': self.get('features.text', {}),
            'topic_features': self.get('features.topics', {}),
            'sentiment_features': self.get('features.sentiment', {})
        }
    
    def get_policy_config(self) -> Dict[str, Any]:
        """Get policy enforcement configuration"""
        return {
            'thresholds': self.get('policy.thresholds', {}),
            'severity_levels': self.get('policy.severity', {})
        }
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration"""
        return {
            'metrics': self.get('evaluation.metrics', []),
            'cv_folds': self.get('evaluation.cv_folds', 5),
            'scoring': self.get('evaluation.scoring', 'f1_weighted')
        }
    
    def get_output_config(self) -> Dict[str, Any]:
        """Get output configuration"""
        return {
            'save_models': self.get('output.save_models', True),
            'save_reports': self.get('output.save_reports', True),
            'generate_visualizations': self.get('output.generate_visualizations', True),
            'results_dir': self.get('output.results_dir', 'results'),
            'models_dir': self.get('output.models_dir', 'models'),
            'logs_dir': self.get('output.logs_dir', 'logs')
        }

if __name__ == "__main__":
    # Test the config loader
    config_loader = ConfigLoader()
    
    print("Default configuration:")
    print(config_loader.config)
    
    print("\nModel configuration:")
    print(config_loader.get_model_config())
    
    print("\nFeature configuration:")
    print(config_loader.get_feature_config())
