"""
Visualization utilities for anomaly detection results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class AnomalyVisualizer:
    """Visualization utilities for anomaly detection."""
    
    def __init__(self, style: str = 'seaborn-v0_8', figure_size: Tuple[int, int] = (12, 8)):
        """Initialize visualizer.
        
        Args:
            style: Matplotlib style
            figure_size: Default figure size
        """
        plt.style.use(style)
        self.figure_size = figure_size
        sns.set_palette("husl")
    
    def plot_reconstruction_errors(
        self, 
        errors: np.ndarray, 
        threshold: float, 
        labels: Optional[np.ndarray] = None,
        title: str = "Reconstruction Error Analysis"
    ) -> plt.Figure:
        """Plot reconstruction errors with threshold.
        
        Args:
            errors: Reconstruction errors
            threshold: Anomaly threshold
            labels: True labels (optional)
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figure_size)
        
        # Plot 1: Reconstruction errors over time
        ax1.plot(errors, alpha=0.7, label='Reconstruction Error')
        ax1.axhline(threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.3f})')
        
        if labels is not None:
            anomaly_indices = np.where(labels == 1)[0]
            ax1.scatter(anomaly_indices, errors[anomaly_indices], 
                       color='red', s=50, alpha=0.8, label='True Anomalies', zorder=5)
        
        ax1.set_title(title)
        ax1.set_xlabel('Sample Index')
        ax1.set_ylabel('Reconstruction Error')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Error distribution
        ax2.hist(errors, bins=50, alpha=0.7, label='Error Distribution')
        ax2.axvline(threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.3f})')
        ax2.set_xlabel('Reconstruction Error')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Error Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_multivariate_data(
        self, 
        data: np.ndarray, 
        labels: np.ndarray,
        feature_names: Optional[List[str]] = None,
        title: str = "Multivariate Data Visualization"
    ) -> plt.Figure:
        """Plot multivariate data with anomaly highlighting.
        
        Args:
            data: Input data
            labels: Anomaly labels
            feature_names: Feature names
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        n_features = data.shape[1]
        if feature_names is None:
            feature_names = [f'Feature {i+1}' for i in range(n_features)]
        
        fig, axes = plt.subplots(n_features, 1, figsize=(self.figure_size[0], 3*n_features))
        if n_features == 1:
            axes = [axes]
        
        for i in range(n_features):
            normal_indices = labels == 0
            anomaly_indices = labels == 1
            
            axes[i].scatter(np.where(normal_indices)[0], data[normal_indices, i], 
                           alpha=0.6, s=20, label='Normal', color='blue')
            axes[i].scatter(np.where(anomaly_indices)[0], data[anomaly_indices, i], 
                           alpha=0.8, s=50, label='Anomaly', color='red')
            
            axes[i].set_title(f'{feature_names[i]}')
            axes[i].set_xlabel('Sample Index')
            axes[i].set_ylabel('Value')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        return fig
    
    def plot_time_series_with_anomalies(
        self, 
        df: pd.DataFrame, 
        labels: np.ndarray,
        title: str = "Time Series with Anomalies"
    ) -> plt.Figure:
        """Plot time series data with anomaly highlighting.
        
        Args:
            df: Time series DataFrame with datetime index
            labels: Anomaly labels
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        n_features = len(df.columns)
        fig, axes = plt.subplots(n_features, 1, figsize=(self.figure_size[0], 3*n_features))
        if n_features == 1:
            axes = [axes]
        
        for i, col in enumerate(df.columns):
            normal_indices = labels == 0
            anomaly_indices = labels == 1
            
            axes[i].plot(df.index[normal_indices], df[col].iloc[normal_indices], 
                        alpha=0.7, label='Normal', color='blue')
            axes[i].scatter(df.index[anomaly_indices], df[col].iloc[anomaly_indices], 
                           alpha=0.8, s=50, label='Anomaly', color='red', zorder=5)
            
            axes[i].set_title(f'{col}')
            axes[i].set_xlabel('Time')
            axes[i].set_ylabel('Value')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        return fig
    
    def plot_interactive_time_series(
        self, 
        df: pd.DataFrame, 
        labels: np.ndarray,
        title: str = "Interactive Time Series with Anomalies"
    ) -> go.Figure:
        """Create interactive time series plot with Plotly.
        
        Args:
            df: Time series DataFrame with datetime index
            labels: Anomaly labels
            title: Plot title
            
        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=len(df.columns), 
            cols=1,
            subplot_titles=df.columns.tolist(),
            vertical_spacing=0.05
        )
        
        colors = px.colors.qualitative.Set1
        
        for i, col in enumerate(df.columns):
            normal_indices = labels == 0
            anomaly_indices = labels == 1
            
            # Normal data
            fig.add_trace(
                go.Scatter(
                    x=df.index[normal_indices],
                    y=df[col].iloc[normal_indices],
                    mode='lines',
                    name=f'{col} (Normal)',
                    line=dict(color=colors[i % len(colors)], width=1),
                    opacity=0.7
                ),
                row=i+1, col=1
            )
            
            # Anomaly data
            fig.add_trace(
                go.Scatter(
                    x=df.index[anomaly_indices],
                    y=df[col].iloc[anomaly_indices],
                    mode='markers',
                    name=f'{col} (Anomaly)',
                    marker=dict(color='red', size=8),
                    opacity=0.8
                ),
                row=i+1, col=1
            )
        
        fig.update_layout(
            title=title,
            height=300 * len(df.columns),
            showlegend=True
        )
        
        return fig
    
    def plot_model_comparison(
        self, 
        results: Dict[str, Dict[str, float]],
        title: str = "Model Performance Comparison"
    ) -> plt.Figure:
        """Plot model performance comparison.
        
        Args:
            results: Dictionary of model results
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_score']
        models = list(results.keys())
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        x = np.arange(len(metrics))
        width = 0.8 / len(models)
        
        for i, model in enumerate(models):
            values = [results[model].get(metric, 0) for metric in metrics]
            ax.bar(x + i * width, values, width, label=model, alpha=0.8)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title(title)
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(metrics, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def save_plots(self, figures: List[plt.Figure], filenames: List[str], output_dir: str = 'plots') -> None:
        """Save multiple figures to files.
        
        Args:
            figures: List of matplotlib figures
            filenames: List of filenames
            output_dir: Output directory
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for fig, filename in zip(figures, filenames):
            filepath = os.path.join(output_dir, filename)
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot: {filepath}")
        
        plt.close('all')
