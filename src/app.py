"""
Main application for multivariate anomaly detection.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, Any, Tuple
import logging

from .data_utils import TimeSeriesGenerator, DataPreprocessor
from .models import AnomalyDetector
from .visualization import AnomalyVisualizer
from .config import Config, setup_logging

logger = logging.getLogger(__name__)


class AnomalyDetectionApp:
    """Main application class for anomaly detection."""
    
    def __init__(self, config_path: str = None):
        """Initialize the application.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = Config(config_path)
        setup_logging(
            self.config.get('logging.level', 'INFO'),
            self.config.get('logging.file')
        )
        
        self.data_generator = TimeSeriesGenerator(
            random_state=self.config.get('data.random_state', 42)
        )
        self.preprocessor = DataPreprocessor()
        self.visualizer = AnomalyVisualizer()
        
        self.data = None
        self.labels = None
        self.scaled_data = None
        self.model = None
        self.results = {}
    
    def generate_data(self, data_type: str = 'multivariate') -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic data.
        
        Args:
            data_type: Type of data to generate ('multivariate' or 'time_series')
            
        Returns:
            Tuple of (data, labels)
        """
        logger.info(f"Generating {data_type} data...")
        
        if data_type == 'multivariate':
            self.data, self.labels = self.data_generator.generate_multivariate_normal(
                n_samples=self.config.get('data.n_samples', 1000),
                n_features=self.config.get('data.n_features', 3),
                anomaly_ratio=self.config.get('data.anomaly_ratio', 0.05)
            )
        elif data_type == 'time_series':
            df, self.labels = self.data_generator.generate_time_series_with_trends(
                n_samples=self.config.get('data.n_samples', 1000),
                n_features=self.config.get('data.n_features', 3),
                anomaly_ratio=self.config.get('data.anomaly_ratio', 0.05)
            )
            self.data = df.values
            self.time_index = df.index
        else:
            raise ValueError("data_type must be 'multivariate' or 'time_series'")
        
        logger.info(f"Generated {len(self.data)} samples with {np.sum(self.labels)} anomalies")
        return self.data, self.labels
    
    def preprocess_data(self) -> np.ndarray:
        """Preprocess the data.
        
        Returns:
            Scaled data
        """
        logger.info("Preprocessing data...")
        self.scaled_data = self.preprocessor.fit_transform(self.data)
        return self.scaled_data
    
    def train_model(self, model_type: str = None) -> Dict[str, Any]:
        """Train anomaly detection model.
        
        Args:
            model_type: Type of model to train
            
        Returns:
            Training results
        """
        if model_type is None:
            model_type = self.config.get('model.type', 'autoencoder')
        
        logger.info(f"Training {model_type} model...")
        
        # Initialize model
        if model_type == 'autoencoder':
            self.model = AnomalyDetector(
                model_type='autoencoder',
                input_dim=self.data.shape[1],
                encoding_dim=self.config.get('model.encoding_dim', 3),
                hidden_dims=self.config.get('model.hidden_dims', [8])
            )
        elif model_type == 'isolation_forest':
            self.model = AnomalyDetector(
                model_type='isolation_forest',
                contamination=self.config.get('data.anomaly_ratio', 0.05),
                random_state=self.config.get('data.random_state', 42)
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train model
        training_results = self.model.train(
            self.scaled_data,
            epochs=self.config.get('model.epochs', 100),
            lr=self.config.get('model.learning_rate', 0.001)
        )
        
        logger.info(f"Model training completed. Final loss: {training_results.get('final_loss', 'N/A')}")
        return training_results
    
    def evaluate_model(self) -> Dict[str, float]:
        """Evaluate model performance.
        
        Returns:
            Evaluation metrics
        """
        logger.info("Evaluating model...")
        
        metrics = self.model.evaluate(
            self.scaled_data,
            self.labels,
            threshold_percentile=self.config.get('model.threshold_percentile', 95)
        )
        
        self.results[self.model.model_type] = metrics
        logger.info(f"Model evaluation completed. Accuracy: {metrics['accuracy']:.3f}")
        
        return metrics
    
    def predict_anomalies(self) -> Tuple[np.ndarray, np.ndarray]:
        """Predict anomalies in the data.
        
        Returns:
            Tuple of (anomaly_scores, predictions)
        """
        logger.info("Predicting anomalies...")
        
        anomaly_scores, predictions = self.model.predict(
            self.scaled_data,
            threshold_percentile=self.config.get('model.threshold_percentile', 95)
        )
        
        logger.info(f"Predicted {np.sum(predictions)} anomalies out of {len(predictions)} samples")
        return anomaly_scores, predictions
    
    def create_visualizations(self, anomaly_scores: np.ndarray, predictions: np.ndarray) -> Dict[str, Any]:
        """Create visualization plots.
        
        Args:
            anomaly_scores: Anomaly scores
            predictions: Predicted labels
            
        Returns:
            Dictionary of figures
        """
        logger.info("Creating visualizations...")
        
        figures = {}
        
        # Reconstruction error plot
        threshold = np.percentile(anomaly_scores, self.config.get('model.threshold_percentile', 95))
        figures['reconstruction_errors'] = self.visualizer.plot_reconstruction_errors(
            anomaly_scores, threshold, self.labels
        )
        
        # Multivariate data plot
        figures['multivariate_data'] = self.visualizer.plot_multivariate_data(
            self.data, self.labels
        )
        
        # Model comparison plot (if multiple models)
        if len(self.results) > 1:
            figures['model_comparison'] = self.visualizer.plot_model_comparison(self.results)
        
        return figures
    
    def run_full_pipeline(self, data_type: str = 'multivariate', model_type: str = None) -> Dict[str, Any]:
        """Run the complete anomaly detection pipeline.
        
        Args:
            data_type: Type of data to generate
            model_type: Type of model to use
            
        Returns:
            Complete results dictionary
        """
        logger.info("Starting full anomaly detection pipeline...")
        
        # Generate data
        self.generate_data(data_type)
        
        # Preprocess data
        self.preprocess_data()
        
        # Train model
        training_results = self.train_model(model_type)
        
        # Evaluate model
        evaluation_metrics = self.evaluate_model()
        
        # Predict anomalies
        anomaly_scores, predictions = self.predict_anomalies()
        
        # Create visualizations
        figures = self.create_visualizations(anomaly_scores, predictions)
        
        # Compile results
        results = {
            'data_info': {
                'n_samples': len(self.data),
                'n_features': self.data.shape[1],
                'n_anomalies': np.sum(self.labels),
                'anomaly_ratio': np.mean(self.labels)
            },
            'model_info': {
                'type': self.model.model_type,
                'training_results': training_results
            },
            'evaluation_metrics': evaluation_metrics,
            'predictions': {
                'anomaly_scores': anomaly_scores,
                'predicted_labels': predictions,
                'threshold': np.percentile(anomaly_scores, self.config.get('model.threshold_percentile', 95))
            },
            'figures': figures
        }
        
        logger.info("Pipeline completed successfully!")
        return results
