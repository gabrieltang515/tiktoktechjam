#!/usr/bin/env python3
"""
ML Models for review quality detection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReviewQualityClassifier:
    """Machine learning classifier for review quality detection"""
    
    def __init__(self, model_type: str = 'ensemble', random_state: int = 42):
        self.model_type = model_type
        self.random_state = random_state
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_importance = {}
        self.best_model = None
        self.best_score = 0.0
        
    def prepare_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Tuple:
        """Prepare data for training"""
        
        # Handle missing values
        X = X.fillna(0)
        
        # Encode labels if needed
        if y.dtype == 'object':
            y = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X_train.columns
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Train Random Forest classifier"""
        
        logger.info("Training Random Forest...")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Grid search
        rf = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_rf = grid_search.best_estimator_
        
        # Feature importance
        feature_importance = dict(zip(feature_names, best_rf.feature_importances_))
        
        return {
            'model': best_rf,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'feature_importance': feature_importance
        }
    
    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Train XGBoost classifier"""
        
        logger.info("Training XGBoost...")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        # Grid search
        xgb_model = xgb.XGBClassifier(random_state=self.random_state, n_jobs=-1)
        grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_xgb = grid_search.best_estimator_
        
        # Feature importance
        feature_importance = dict(zip(feature_names, best_xgb.feature_importances_))
        
        return {
            'model': best_xgb,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'feature_importance': feature_importance
        }
    
    def train_lightgbm(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Train LightGBM classifier"""
        
        logger.info("Training LightGBM...")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'num_leaves': [31, 62, 127]
        }
        
        # Grid search
        lgb_model = lgb.LGBMClassifier(random_state=self.random_state, n_jobs=-1)
        grid_search = GridSearchCV(lgb_model, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_lgb = grid_search.best_estimator_
        
        # Feature importance
        feature_importance = dict(zip(feature_names, best_lgb.feature_importances_))
        
        return {
            'model': best_lgb,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'feature_importance': feature_importance
        }
    
    def train_logistic_regression(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Train Logistic Regression classifier"""
        
        logger.info("Training Logistic Regression...")
        
        # Define parameter grid
        param_grid = {
            'C': [0.1, 1.0, 10.0, 100.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }
        
        # Grid search
        lr = LogisticRegression(random_state=self.random_state, max_iter=1000)
        grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_lr = grid_search.best_estimator_
        
        # Feature importance (coefficients)
        feature_importance = dict(zip(feature_names, np.abs(best_lr.coef_[0])))
        
        return {
            'model': best_lr,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'feature_importance': feature_importance
        }
    
    def train_svm(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Train SVM classifier"""
        
        logger.info("Training SVM...")
        
        # Define parameter grid
        param_grid = {
            'C': [0.1, 1.0, 10.0],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto']
        }
        
        # Grid search
        svm = SVC(random_state=self.random_state, probability=True)
        grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_svm = grid_search.best_estimator_
        
        return {
            'model': best_svm,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'feature_importance': {}  # SVM doesn't provide feature importance
        }
    
    def train_ensemble(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Train ensemble of multiple models"""
        
        logger.info("Training Ensemble Model...")
        
        # Train individual models
        rf_result = self.train_random_forest(X_train, y_train, feature_names)
        xgb_result = self.train_xgboost(X_train, y_train, feature_names)
        lgb_result = self.train_lightgbm(X_train, y_train, feature_names)
        lr_result = self.train_logistic_regression(X_train, y_train, feature_names)
        
        # Create ensemble
        ensemble = VotingClassifier(
            estimators=[
                ('rf', rf_result['model']),
                ('xgb', xgb_result['model']),
                ('lgb', lgb_result['model']),
                ('lr', lr_result['model'])
            ],
            voting='soft'
        )
        
        # Train ensemble
        ensemble.fit(X_train, y_train)
        
        # Cross-validation score
        cv_scores = cross_val_score(ensemble, X_train, y_train, cv=5, scoring='f1_weighted')
        
        # Combine feature importance
        combined_importance = {}
        for feature in feature_names:
            importance_sum = 0
            count = 0
            for result in [rf_result, xgb_result, lgb_result, lr_result]:
                if feature in result['feature_importance']:
                    importance_sum += result['feature_importance'][feature]
                    count += 1
            if count > 0:
                combined_importance[feature] = importance_sum / count
        
        return {
            'model': ensemble,
            'best_params': 'ensemble',
            'best_score': cv_scores.mean(),
            'feature_importance': combined_importance,
            'individual_models': {
                'random_forest': rf_result,
                'xgboost': xgb_result,
                'lightgbm': lgb_result,
                'logistic_regression': lr_result
            }
        }
    
    def train_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train the specified model type"""
        
        logger.info(f"Training {self.model_type} model...")
        
        # Prepare data
        X_train, X_test, y_train, y_test, feature_names = self.prepare_data(X, y)
        
        # Train model based on type
        if self.model_type == 'random_forest':
            result = self.train_random_forest(X_train, y_train, feature_names)
        elif self.model_type == 'xgboost':
            result = self.train_xgboost(X_train, y_train, feature_names)
        elif self.model_type == 'lightgbm':
            result = self.train_lightgbm(X_train, y_train, feature_names)
        elif self.model_type == 'logistic_regression':
            result = self.train_logistic_regression(X_train, y_train, feature_names)
        elif self.model_type == 'svm':
            result = self.train_svm(X_train, y_train, feature_names)
        elif self.model_type == 'ensemble':
            result = self.train_ensemble(X_train, y_train, feature_names)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Evaluate on test set
        y_pred = result['model'].predict(X_test)
        y_pred_proba = result['model'].predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        # Store results
        self.models[self.model_type] = result
        self.feature_importance = result['feature_importance']
        
        # Update best model
        if f1 > self.best_score:
            self.best_model = result['model']
            self.best_score = f1
        
        evaluation_results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'y_test': y_test,
            'X_test': X_test,
            'feature_names': feature_names
        }
        
        logger.info(f"Model training completed. F1 Score: {f1:.4f}")
        
        return evaluation_results
    
    def predict(self, X: pd.DataFrame, model_name: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions using trained model"""
        
        if model_name is None:
            model_name = self.model_type
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
        
        # Prepare data
        X_scaled = self.scaler.transform(X.fillna(0))
        
        # Make predictions
        model = self.models[model_name]['model']
        y_pred = model.predict(X_scaled)
        y_pred_proba = model.predict_proba(X_scaled)
        
        return y_pred, y_pred_proba
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get top feature importance"""
        
        if not self.feature_importance:
            raise ValueError("No feature importance available. Train a model first.")
        
        # Sort features by importance
        sorted_features = sorted(
            self.feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Create DataFrame
        feature_df = pd.DataFrame(
            sorted_features[:top_n], 
            columns=['feature', 'importance']
        )
        
        return feature_df
    
    def save_model(self, filepath: str, model_name: str = None):
        """Save trained model"""
        
        if model_name is None:
            model_name = self.model_type
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        # Save model and preprocessing components
        model_data = {
            'model': self.models[model_name]['model'],
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_importance': self.feature_importance,
            'model_type': model_name
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        
        model_data = joblib.load(filepath)
        
        self.models[model_data['model_type']] = {
            'model': model_data['model'],
            'feature_importance': model_data['feature_importance']
        }
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.model_type = model_data['model_type']
        
        logger.info(f"Model loaded from {filepath}")

class TransformerModel:
    """Transformer-based model for review quality detection"""
    
    def __init__(self, model_name: str = 'distilbert-base-uncased', max_length: int = 512):
        self.model_name = model_name
        self.max_length = max_length
        self.model = None
        self.tokenizer = None
        
    def setup_model(self):
        """Setup transformer model and tokenizer"""
        
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
            import torch
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, 
                num_labels=2  # Binary classification
            )
            
            logger.info(f"Transformer model {self.model_name} loaded successfully")
            
        except ImportError:
            logger.error("Transformers library not available. Please install: pip install transformers torch")
            raise
    
    def prepare_transformer_data(self, texts: List[str], labels: List[int]) -> Any:
        """Prepare data for transformer model"""
        
        from torch.utils.data import Dataset
        
        class ReviewDataset(Dataset):
            def __init__(self, texts, labels, tokenizer, max_length):
                self.texts = texts
                self.labels = labels
                self.tokenizer = tokenizer
                self.max_length = max_length
            
            def __len__(self):
                return len(self.texts)
            
            def __getitem__(self, idx):
                text = str(self.texts[idx])
                label = self.labels[idx]
                
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                return {
                    'input_ids': encoding['input_ids'].flatten(),
                    'attention_mask': encoding['attention_mask'].flatten(),
                    'labels': torch.tensor(label, dtype=torch.long)
                }
        
        return ReviewDataset(texts, labels, self.tokenizer, self.max_length)
    
    def train(self, texts: List[str], labels: List[int], epochs: int = 3, batch_size: int = 16):
        """Train transformer model"""
        
        if self.model is None:
            self.setup_model()
        
        from transformers import TrainingArguments, Trainer
        import torch
        
        # Prepare datasets
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        train_dataset = self.prepare_transformer_data(train_texts, train_labels)
        val_dataset = self.prepare_transformer_data(val_texts, val_labels)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir="./transformer_model",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
        
        # Train
        trainer.train()
        
        logger.info("Transformer model training completed")
        
        return trainer

if __name__ == "__main__":
    # Test the ML models
    from src.preprocessing.data_loader import ReviewDataLoader
    from src.preprocessing.text_preprocessor import TextPreprocessor
    from src.feature_engineering.feature_extractor import FeatureExtractor
    
    # Load and preprocess data
    loader = ReviewDataLoader()
    reviews_df, _ = loader.load_data()
    
    preprocessor = TextPreprocessor()
    processed_df = preprocessor.preprocess_dataframe(reviews_df)
    
    extractor = FeatureExtractor(max_features=500, n_topics=5)
    features, labels = extractor.extract_all_features(processed_df)
    
    # Train model
    classifier = ReviewQualityClassifier(model_type='ensemble')
    results = classifier.train_model(features, labels['is_high_quality'])
    
    print(f"\nModel Performance:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    
    print(f"\nClassification Report:")
    print(results['classification_report'])
    
    # Feature importance
    feature_importance = classifier.get_feature_importance(top_n=10)
    print(f"\nTop 10 Feature Importance:")
    print(feature_importance)
