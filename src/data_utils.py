"""
Data generation and preprocessing utilities for time series anomaly detection.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


class TimeSeriesGenerator:
    """Generate realistic synthetic time series data with various patterns."""
    
    def __init__(self, random_state: int = 42):
        """Initialize the generator with a random seed.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
    
    def generate_multivariate_normal(
        self, 
        n_samples: int = 1000, 
        n_features: int = 3,
        anomaly_ratio: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate multivariate normal data with injected anomalies.
        
        Args:
            n_samples: Number of samples to generate
            n_features: Number of features/dimensions
            anomaly_ratio: Proportion of data points that are anomalies
            
        Returns:
            Tuple of (data, labels) where labels are 0 for normal, 1 for anomaly
        """
        n_anomalies = int(n_samples * anomaly_ratio)
        n_normal = n_samples - n_anomalies
        
        # Generate normal multivariate data
        normal_data = np.random.normal(0, 1, (n_normal, n_features))
        
        # Generate anomalies (random spikes)
        anomalies = np.random.uniform(-6, 6, (n_anomalies, n_features))
        
        # Combine and shuffle
        data = np.vstack([normal_data, anomalies])
        labels = np.array([0] * n_normal + [1] * n_anomalies)
        
        idx = np.random.permutation(len(data))
        data, labels = data[idx], labels[idx]
        
        logger.info(f"Generated {n_samples} samples with {n_anomalies} anomalies")
        return data, labels
    
    def generate_time_series_with_trends(
        self,
        n_samples: int = 1000,
        n_features: int = 3,
        trend_strength: float = 0.1,
        seasonal_period: int = 24,
        noise_level: float = 0.1,
        anomaly_ratio: float = 0.05
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Generate time series data with trends, seasonality, and anomalies.
        
        Args:
            n_samples: Number of time points
            n_features: Number of features
            trend_strength: Strength of linear trend
            seasonal_period: Period of seasonal component
            noise_level: Level of random noise
            anomaly_ratio: Proportion of anomalies
            
        Returns:
            Tuple of (dataframe with timestamps, labels)
        """
        time_index = pd.date_range(start='2023-01-01', periods=n_samples, freq='H')
        n_anomalies = int(n_samples * anomaly_ratio)
        
        data = np.zeros((n_samples, n_features))
        
        for i in range(n_features):
            # Linear trend
            trend = np.linspace(0, trend_strength * n_samples, n_samples)
            
            # Seasonal component
            seasonal = 0.5 * np.sin(2 * np.pi * np.arange(n_samples) / seasonal_period)
            
            # Random noise
            noise = np.random.normal(0, noise_level, n_samples)
            
            # Combine components
            data[:, i] = trend + seasonal + noise
        
        # Inject anomalies
        anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
        labels = np.zeros(n_samples)
        
        for idx in anomaly_indices:
            labels[idx] = 1
            # Add spike anomaly
            data[idx] += np.random.normal(0, 3, n_features)
        
        # Create DataFrame
        df = pd.DataFrame(data, index=time_index, columns=[f'feature_{i}' for i in range(n_features)])
        
        logger.info(f"Generated time series with {n_anomalies} anomalies")
        return df, labels


class DataPreprocessor:
    """Preprocessing utilities for anomaly detection."""
    
    def __init__(self, scaler_type: str = 'standard'):
        """Initialize preprocessor.
        
        Args:
            scaler_type: Type of scaler ('standard' or 'minmax')
        """
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("scaler_type must be 'standard' or 'minmax'")
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit scaler and transform data.
        
        Args:
            data: Input data to scale
            
        Returns:
            Scaled data
        """
        return self.scaler.fit_transform(data)
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data using fitted scaler.
        
        Args:
            data: Input data to scale
            
        Returns:
            Scaled data
        """
        return self.scaler.transform(data)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform scaled data.
        
        Args:
            data: Scaled data
            
        Returns:
            Original scale data
        """
        return self.scaler.inverse_transform(data)
    
    def train_test_split_with_labels(
        self, 
        data: np.ndarray, 
        labels: np.ndarray, 
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data and labels into train/test sets.
        
        Args:
            data: Input data
            labels: Corresponding labels
            test_size: Proportion of data for testing
            random_state: Random seed
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        return train_test_split(data, labels, test_size=test_size, random_state=random_state)
