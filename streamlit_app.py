"""
Streamlit web interface for multivariate anomaly detection.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.app import AnomalyDetectionApp
from src.config import Config
from src.visualization import AnomalyVisualizer

# Page configuration
st.set_page_config(
    page_title="Multivariate Anomaly Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .anomaly-highlight {
        background-color: #ffebee;
        padding: 0.5rem;
        border-radius: 0.25rem;
        border-left: 4px solid #f44336;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🔍 Multivariate Anomaly Detection</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    This application demonstrates state-of-the-art multivariate anomaly detection using various machine learning approaches.
    Generate synthetic data, train different models, and visualize anomaly detection results.
    """)
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Data generation parameters
    st.sidebar.subheader("Data Generation")
    data_type = st.sidebar.selectbox(
        "Data Type",
        ["multivariate", "time_series"],
        help="Choose between multivariate normal data or time series with trends"
    )
    
    n_samples = st.sidebar.slider(
        "Number of Samples",
        min_value=100,
        max_value=5000,
        value=1000,
        step=100
    )
    
    n_features = st.sidebar.slider(
        "Number of Features",
        min_value=2,
        max_value=10,
        value=3,
        step=1
    )
    
    anomaly_ratio = st.sidebar.slider(
        "Anomaly Ratio",
        min_value=0.01,
        max_value=0.2,
        value=0.05,
        step=0.01,
        format="%.2f"
    )
    
    # Model parameters
    st.sidebar.subheader("Model Configuration")
    model_type = st.sidebar.selectbox(
        "Model Type",
        ["autoencoder", "isolation_forest"],
        help="Choose the anomaly detection model"
    )
    
    if model_type == "autoencoder":
        encoding_dim = st.sidebar.slider(
            "Encoding Dimension",
            min_value=2,
            max_value=min(n_features, 10),
            value=min(3, n_features),
            step=1
        )
        
        epochs = st.sidebar.slider(
            "Training Epochs",
            min_value=10,
            max_value=500,
            value=100,
            step=10
        )
        
        learning_rate = st.sidebar.slider(
            "Learning Rate",
            min_value=0.0001,
            max_value=0.01,
            value=0.001,
            step=0.0001,
            format="%.4f"
        )
    
    threshold_percentile = st.sidebar.slider(
        "Anomaly Threshold Percentile",
        min_value=80,
        max_value=99,
        value=95,
        step=1
    )
    
    # Random seed
    random_state = st.sidebar.number_input(
        "Random Seed",
        min_value=0,
        max_value=10000,
        value=42,
        step=1
    )
    
    # Run button
    if st.sidebar.button("🚀 Run Anomaly Detection", type="primary"):
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Initialize app
            status_text.text("Initializing application...")
            progress_bar.progress(10)
            
            # Create configuration
            config = Config()
            config.set('data.n_samples', n_samples)
            config.set('data.n_features', n_features)
            config.set('data.anomaly_ratio', anomaly_ratio)
            config.set('data.random_state', random_state)
            config.set('model.type', model_type)
            config.set('model.threshold_percentile', threshold_percentile)
            
            if model_type == "autoencoder":
                config.set('model.encoding_dim', encoding_dim)
                config.set('model.epochs', epochs)
                config.set('model.learning_rate', learning_rate)
            
            # Initialize app
            app = AnomalyDetectionApp()
            app.config = config
            
            # Generate data
            status_text.text("Generating synthetic data...")
            progress_bar.progress(20)
            app.generate_data(data_type)
            
            # Preprocess data
            status_text.text("Preprocessing data...")
            progress_bar.progress(30)
            app.preprocess_data()
            
            # Train model
            status_text.text(f"Training {model_type} model...")
            progress_bar.progress(50)
            training_results = app.train_model(model_type)
            
            # Evaluate model
            status_text.text("Evaluating model performance...")
            progress_bar.progress(70)
            evaluation_metrics = app.evaluate_model()
            
            # Predict anomalies
            status_text.text("Predicting anomalies...")
            progress_bar.progress(85)
            anomaly_scores, predictions = app.predict_anomalies()
            
            # Create visualizations
            status_text.text("Creating visualizations...")
            progress_bar.progress(95)
            
            # Results
            status_text.text("Analysis complete!")
            progress_bar.progress(100)
            
            # Display results
            display_results(app, evaluation_metrics, anomaly_scores, predictions, data_type)
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)
    
    # Display sample data info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Sample Data Info")
    st.sidebar.info(f"""
    **Data Type:** {data_type.title()}
    **Samples:** {n_samples:,}
    **Features:** {n_features}
    **Expected Anomalies:** {int(n_samples * anomaly_ratio):,}
    **Model:** {model_type.title()}
    """)


def display_results(app, evaluation_metrics, anomaly_scores, predictions, data_type):
    """Display analysis results."""
    
    # Metrics
    st.header("📊 Performance Metrics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Accuracy",
            f"{evaluation_metrics['accuracy']:.3f}",
            help="Overall classification accuracy"
        )
    
    with col2:
        st.metric(
            "Precision",
            f"{evaluation_metrics['precision']:.3f}",
            help="Precision for anomaly detection"
        )
    
    with col3:
        st.metric(
            "Recall",
            f"{evaluation_metrics['recall']:.3f}",
            help="Recall for anomaly detection"
        )
    
    with col4:
        st.metric(
            "F1-Score",
            f"{evaluation_metrics['f1_score']:.3f}",
            help="F1-score for anomaly detection"
        )
    
    with col5:
        st.metric(
            "AUC Score",
            f"{evaluation_metrics['auc_score']:.3f}",
            help="Area under ROC curve"
        )
    
    # Data summary
    st.header("📈 Data Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Samples", f"{len(app.data):,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("True Anomalies", f"{np.sum(app.labels):,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Predicted Anomalies", f"{np.sum(predictions):,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Visualizations
    st.header("📊 Visualizations")
    
    # Reconstruction errors
    st.subheader("Reconstruction Error Analysis")
    threshold = np.percentile(anomaly_scores, 95)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Reconstruction errors over time
    ax1.plot(anomaly_scores, alpha=0.7, label='Reconstruction Error')
    ax1.axhline(threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.3f})')
    
    anomaly_indices = np.where(app.labels == 1)[0]
    ax1.scatter(anomaly_indices, anomaly_scores[anomaly_indices], 
               color='red', s=50, alpha=0.8, label='True Anomalies', zorder=5)
    
    ax1.set_title("Reconstruction Error Analysis")
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Reconstruction Error')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Error distribution
    ax2.hist(anomaly_scores, bins=50, alpha=0.7, label='Error Distribution')
    ax2.axvline(threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.3f})')
    ax2.set_xlabel('Reconstruction Error')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Error Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Multivariate data visualization
    st.subheader("Multivariate Data Visualization")
    
    n_features = app.data.shape[1]
    fig, axes = plt.subplots(n_features, 1, figsize=(12, 3*n_features))
    if n_features == 1:
        axes = [axes]
    
    for i in range(n_features):
        normal_indices = app.labels == 0
        anomaly_indices = app.labels == 1
        
        axes[i].scatter(np.where(normal_indices)[0], app.data[normal_indices, i], 
                       alpha=0.6, s=20, label='Normal', color='blue')
        axes[i].scatter(np.where(anomaly_indices)[0], app.data[anomaly_indices, i], 
                       alpha=0.8, s=50, label='Anomaly', color='red')
        
        axes[i].set_title(f'Feature {i+1}')
        axes[i].set_xlabel('Sample Index')
        axes[i].set_ylabel('Value')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Interactive time series plot (if time series data)
    if data_type == 'time_series' and hasattr(app, 'time_index'):
        st.subheader("Interactive Time Series Visualization")
        
        df = pd.DataFrame(app.data, index=app.time_index, 
                         columns=[f'Feature {i+1}' for i in range(n_features)])
        
        fig = make_subplots(
            rows=n_features, 
            cols=1,
            subplot_titles=df.columns.tolist(),
            vertical_spacing=0.05
        )
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, col in enumerate(df.columns):
            normal_indices = app.labels == 0
            anomaly_indices = app.labels == 1
            
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
            title="Interactive Time Series with Anomalies",
            height=300 * n_features,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Download results
    st.header("💾 Download Results")
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'sample_index': range(len(app.data)),
        'anomaly_score': anomaly_scores,
        'predicted_anomaly': predictions,
        'true_anomaly': app.labels
    })
    
    # Add feature columns
    for i in range(n_features):
        results_df[f'feature_{i+1}'] = app.data[:, i]
    
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name="anomaly_detection_results.csv",
        mime="text/csv"
    )


if __name__ == "__main__":
    main()
