"""
Configuration management and logging utilities.
"""

import yaml
import logging
import os
from typing import Dict, Any, Optional
from pathlib import Path


def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None) -> None:
    """Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    # Create logs directory if it doesn't exist
    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )


class Config:
    """Configuration management class."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_default_config()
        
        if config_path and os.path.exists(config_path):
            self._load_config(config_path)
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration."""
        return {
            'data': {
                'n_samples': 1000,
                'n_features': 3,
                'anomaly_ratio': 0.05,
                'test_size': 0.2,
                'random_state': 42
            },
            'model': {
                'type': 'autoencoder',
                'encoding_dim': 3,
                'hidden_dims': [8],
                'epochs': 100,
                'learning_rate': 0.001,
                'threshold_percentile': 95
            },
            'training': {
                'batch_size': 32,
                'validation_split': 0.1,
                'early_stopping_patience': 10
            },
            'visualization': {
                'figure_size': [12, 8],
                'dpi': 100,
                'style': 'seaborn-v0_8'
            },
            'logging': {
                'level': 'INFO',
                'file': 'logs/anomaly_detection.log'
            }
        }
    
    def _load_config(self, config_path: str) -> None:
        """Load configuration from file.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            file_config = yaml.safe_load(f)
            self._update_config(self.config, file_config)
    
    def _update_config(self, base_config: Dict[str, Any], new_config: Dict[str, Any]) -> None:
        """Recursively update configuration.
        
        Args:
            base_config: Base configuration dictionary
            new_config: New configuration dictionary
        """
        for key, value in new_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self._update_config(base_config[key], value)
            else:
                base_config[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, config_path: str) -> None:
        """Save configuration to file.
        
        Args:
            config_path: Path to save configuration
        """
        config_dir = Path(config_path).parent
        config_dir.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
