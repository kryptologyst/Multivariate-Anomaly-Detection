"""
Advanced anomaly detection models including autoencoders, isolation forests, and deep learning approaches.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Dict, Any, Optional, List
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, roc_auc_score
import logging

logger = logging.getLogger(__name__)


class Autoencoder(nn.Module):
    """Autoencoder neural network for anomaly detection."""
    
    def __init__(self, input_dim: int, encoding_dim: int = 3, hidden_dims: List[int] = [8]):
        """Initialize autoencoder.
        
        Args:
            input_dim: Input dimension
            encoding_dim: Encoding dimension (bottleneck)
            hidden_dims: List of hidden layer dimensions
        """
        super(Autoencoder, self).__init__()
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        encoder_layers.append(nn.Linear(prev_dim, encoding_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        prev_dim = encoding_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through autoencoder.
        
        Args:
            x: Input tensor
            
        Returns:
            Reconstructed tensor
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation.
        
        Args:
            x: Input tensor
            
        Returns:
            Encoded tensor
        """
        return self.encoder(x)


class LSTMAutoencoder(nn.Module):
    """LSTM-based autoencoder for time series anomaly detection."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        """Initialize LSTM autoencoder.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            num_layers: Number of LSTM layers
        """
        super(LSTMAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Encoder
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Decoder
        self.decoder = nn.LSTM(hidden_dim, input_dim, num_layers, batch_first=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LSTM autoencoder.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Reconstructed tensor
        """
        # Encode
        encoded, (hidden, cell) = self.encoder(x)
        
        # Decode
        decoded, _ = self.decoder(encoded, (hidden, cell))
        
        return decoded


class AnomalyDetector:
    """Main anomaly detection class that wraps different models."""
    
    def __init__(self, model_type: str = 'autoencoder', **kwargs):
        """Initialize anomaly detector.
        
        Args:
            model_type: Type of model ('autoencoder', 'lstm_autoencoder', 'isolation_forest')
            **kwargs: Additional model parameters
        """
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.threshold = None
        self.kwargs = kwargs
        
        if model_type == 'autoencoder':
            self.model = Autoencoder(**kwargs)
        elif model_type == 'lstm_autoencoder':
            self.model = LSTMAutoencoder(**kwargs)
        elif model_type == 'isolation_forest':
            self.model = IsolationForest(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train(self, data: np.ndarray, epochs: int = 100, lr: float = 0.001) -> Dict[str, Any]:
        """Train the anomaly detection model.
        
        Args:
            data: Training data
            epochs: Number of training epochs (for neural networks)
            lr: Learning rate (for neural networks)
            
        Returns:
            Training metrics
        """
        if self.model_type in ['autoencoder', 'lstm_autoencoder']:
            return self._train_neural_network(data, epochs, lr)
        elif self.model_type == 'isolation_forest':
            return self._train_isolation_forest(data)
        else:
            raise ValueError(f"Training not implemented for {self.model_type}")
    
    def _train_neural_network(self, data: np.ndarray, epochs: int, lr: float) -> Dict[str, Any]:
        """Train neural network models."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        data_tensor = torch.FloatTensor(data).to(device)
        losses = []
        
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self.model(data_tensor)
            loss = criterion(output, data_tensor)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}, Loss: {loss.item():.6f}")
        
        return {'losses': losses, 'final_loss': losses[-1]}
    
    def _train_isolation_forest(self, data: np.ndarray) -> Dict[str, Any]:
        """Train isolation forest model."""
        self.model.fit(data)
        return {'status': 'trained'}
    
    def predict(self, data: np.ndarray, threshold_percentile: float = 95) -> Tuple[np.ndarray, np.ndarray]:
        """Predict anomalies in the data.
        
        Args:
            data: Input data
            threshold_percentile: Percentile for anomaly threshold
            
        Returns:
            Tuple of (anomaly_scores, predictions)
        """
        if self.model_type in ['autoencoder', 'lstm_autoencoder']:
            return self._predict_neural_network(data, threshold_percentile)
        elif self.model_type == 'isolation_forest':
            return self._predict_isolation_forest(data)
        else:
            raise ValueError(f"Prediction not implemented for {self.model_type}")
    
    def _predict_neural_network(self, data: np.ndarray, threshold_percentile: float) -> Tuple[np.ndarray, np.ndarray]:
        """Predict using neural network models."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.model.eval()
        
        data_tensor = torch.FloatTensor(data).to(device)
        
        with torch.no_grad():
            reconstructed = self.model(data_tensor)
            reconstruction_errors = torch.mean((reconstructed - data_tensor) ** 2, dim=1).cpu().numpy()
        
        threshold = np.percentile(reconstruction_errors, threshold_percentile)
        predictions = (reconstruction_errors > threshold).astype(int)
        
        self.threshold = threshold
        
        return reconstruction_errors, predictions
    
    def _predict_isolation_forest(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict using isolation forest."""
        anomaly_scores = self.model.decision_function(data)
        predictions = self.model.predict(data)
        
        # Convert -1/1 to 0/1
        predictions = (predictions == -1).astype(int)
        
        return anomaly_scores, predictions
    
    def evaluate(self, data: np.ndarray, labels: np.ndarray, threshold_percentile: float = 95) -> Dict[str, float]:
        """Evaluate model performance.
        
        Args:
            data: Test data
            labels: True labels (0: normal, 1: anomaly)
            threshold_percentile: Percentile for anomaly threshold
            
        Returns:
            Evaluation metrics
        """
        anomaly_scores, predictions = self.predict(data, threshold_percentile)
        
        # Calculate metrics
        accuracy = (predictions == labels).mean()
        auc_score = roc_auc_score(labels, anomaly_scores)
        
        # Classification report
        report = classification_report(labels, predictions, output_dict=True)
        
        metrics = {
            'accuracy': accuracy,
            'auc_score': auc_score,
            'precision': report['1']['precision'],
            'recall': report['1']['recall'],
            'f1_score': report['1']['f1-score']
        }
        
        logger.info(f"Model evaluation - Accuracy: {accuracy:.3f}, AUC: {auc_score:.3f}")
        
        return metrics
