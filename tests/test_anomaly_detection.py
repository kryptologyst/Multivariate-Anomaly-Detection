"""
Unit tests for the multivariate anomaly detection package.
"""

import pytest
import numpy as np
import pandas as pd
import torch
from unittest.mock import patch, MagicMock

from src.data_utils import TimeSeriesGenerator, DataPreprocessor
from src.models import Autoencoder, LSTMAutoencoder, AnomalyDetector
from src.config import Config
from src.app import AnomalyDetectionApp


class TestTimeSeriesGenerator:
    """Test cases for TimeSeriesGenerator."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.generator = TimeSeriesGenerator(random_state=42)
    
    def test_generate_multivariate_normal(self):
        """Test multivariate normal data generation."""
        data, labels = self.generator.generate_multivariate_normal(
            n_samples=100, n_features=3, anomaly_ratio=0.1
        )
        
        assert data.shape == (100, 3)
        assert labels.shape == (100,)
        assert np.sum(labels) == 10  # 10% of 100
        assert set(labels) == {0, 1}
    
    def test_generate_time_series_with_trends(self):
        """Test time series data generation."""
        df, labels = self.generator.generate_time_series_with_trends(
            n_samples=100, n_features=2, anomaly_ratio=0.05
        )
        
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (100, 2)
        assert labels.shape == (100,)
        assert np.sum(labels) == 5  # 5% of 100
        assert isinstance(df.index, pd.DatetimeIndex)


class TestDataPreprocessor:
    """Test cases for DataPreprocessor."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.preprocessor = DataPreprocessor(scaler_type='standard')
        self.data = np.random.normal(0, 1, (100, 3))
    
    def test_fit_transform(self):
        """Test fit and transform."""
        scaled_data = self.preprocessor.fit_transform(self.data)
        
        assert scaled_data.shape == self.data.shape
        assert np.allclose(np.mean(scaled_data, axis=0), 0, atol=1e-10)
        assert np.allclose(np.std(scaled_data, axis=0), 1, atol=1e-10)
    
    def test_transform_inverse_transform(self):
        """Test transform and inverse transform."""
        self.preprocessor.fit_transform(self.data)
        scaled_data = self.preprocessor.transform(self.data)
        original_data = self.preprocessor.inverse_transform(scaled_data)
        
        assert np.allclose(original_data, self.data, atol=1e-10)
    
    def test_train_test_split_with_labels(self):
        """Test train test split."""
        labels = np.random.randint(0, 2, 100)
        X_train, X_test, y_train, y_test = self.preprocessor.train_test_split_with_labels(
            self.data, labels, test_size=0.2, random_state=42
        )
        
        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20


class TestAutoencoder:
    """Test cases for Autoencoder."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.model = Autoencoder(input_dim=3, encoding_dim=2, hidden_dims=[4])
    
    def test_forward_pass(self):
        """Test forward pass."""
        x = torch.randn(10, 3)
        output = self.model(x)
        
        assert output.shape == x.shape
    
    def test_encode(self):
        """Test encoding."""
        x = torch.randn(10, 3)
        encoded = self.model.encode(x)
        
        assert encoded.shape == (10, 2)


class TestLSTMAutoencoder:
    """Test cases for LSTMAutoencoder."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.model = LSTMAutoencoder(input_dim=3, hidden_dim=16, num_layers=2)
    
    def test_forward_pass(self):
        """Test forward pass."""
        x = torch.randn(10, 20, 3)  # batch_size, seq_len, input_dim
        output = self.model(x)
        
        assert output.shape == x.shape


class TestAnomalyDetector:
    """Test cases for AnomalyDetector."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.data = np.random.normal(0, 1, (100, 3))
        self.labels = np.random.randint(0, 2, 100)
    
    def test_autoencoder_detector(self):
        """Test autoencoder anomaly detector."""
        detector = AnomalyDetector(
            model_type='autoencoder',
            input_dim=3,
            encoding_dim=2,
            hidden_dims=[4]
        )
        
        # Train
        training_results = detector.train(self.data, epochs=5, lr=0.01)
        assert 'losses' in training_results
        
        # Predict
        scores, predictions = detector.predict(self.data)
        assert len(scores) == len(self.data)
        assert len(predictions) == len(self.data)
        assert set(predictions) == {0, 1}
    
    def test_isolation_forest_detector(self):
        """Test isolation forest anomaly detector."""
        detector = AnomalyDetector(
            model_type='isolation_forest',
            contamination=0.1,
            random_state=42
        )
        
        # Train
        training_results = detector.train(self.data)
        assert training_results['status'] == 'trained'
        
        # Predict
        scores, predictions = detector.predict(self.data)
        assert len(scores) == len(self.data)
        assert len(predictions) == len(self.data)
        assert set(predictions) == {0, 1}
    
    def test_evaluate(self):
        """Test model evaluation."""
        detector = AnomalyDetector(
            model_type='isolation_forest',
            contamination=0.1,
            random_state=42
        )
        
        detector.train(self.data)
        metrics = detector.evaluate(self.data, self.labels)
        
        assert 'accuracy' in metrics
        assert 'auc_score' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics


class TestConfig:
    """Test cases for Config."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = Config()
    
    def test_default_config(self):
        """Test default configuration."""
        assert self.config.get('data.n_samples') == 1000
        assert self.config.get('model.type') == 'autoencoder'
    
    def test_set_get(self):
        """Test set and get methods."""
        self.config.set('test.value', 42)
        assert self.config.get('test.value') == 42
    
    def test_get_with_default(self):
        """Test get with default value."""
        value = self.config.get('nonexistent.key', 'default')
        assert value == 'default'


class TestAnomalyDetectionApp:
    """Test cases for AnomalyDetectionApp."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.app = AnomalyDetectionApp()
    
    def test_generate_data_multivariate(self):
        """Test multivariate data generation."""
        data, labels = self.app.generate_data('multivariate')
        
        assert data.shape[0] == labels.shape[0]
        assert data.shape[1] == self.app.config.get('data.n_features', 3)
        assert set(labels) == {0, 1}
    
    def test_generate_data_time_series(self):
        """Test time series data generation."""
        data, labels = self.app.generate_data('time_series')
        
        assert data.shape[0] == labels.shape[0]
        assert data.shape[1] == self.app.config.get('data.n_features', 3)
        assert set(labels) == {0, 1}
        assert hasattr(self.app, 'time_index')
    
    def test_preprocess_data(self):
        """Test data preprocessing."""
        self.app.generate_data('multivariate')
        scaled_data = self.app.preprocess_data()
        
        assert scaled_data.shape == self.app.data.shape
        assert np.allclose(np.mean(scaled_data, axis=0), 0, atol=1e-10)
    
    @patch('src.app.AnomalyDetector')
    def test_train_model(self, mock_detector_class):
        """Test model training."""
        mock_detector = MagicMock()
        mock_detector_class.return_value = mock_detector
        mock_detector.train.return_value = {'losses': [1.0, 0.5, 0.3]}
        
        self.app.generate_data('multivariate')
        self.app.preprocess_data()
        training_results = self.app.train_model('autoencoder')
        
        assert 'losses' in training_results
        mock_detector.train.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
