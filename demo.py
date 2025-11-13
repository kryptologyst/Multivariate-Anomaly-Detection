#!/usr/bin/env python3
"""
Demonstration script for the modernized multivariate anomaly detection project.
This script showcases the key features and capabilities of the toolkit.
"""

import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.app import AnomalyDetectionApp
from src.config import Config
from src.visualization import AnomalyVisualizer


def demonstrate_multivariate_detection():
    """Demonstrate multivariate anomaly detection."""
    print("=" * 60)
    print("MULTIVARIATE ANOMALY DETECTION DEMONSTRATION")
    print("=" * 60)
    
    # Initialize application
    app = AnomalyDetectionApp()
    
    # Generate multivariate data
    print("\n1. Generating multivariate data...")
    data, labels = app.generate_data('multivariate')
    print(f"   Generated {len(data)} samples with {np.sum(labels)} anomalies")
    
    # Preprocess data
    print("\n2. Preprocessing data...")
    scaled_data = app.preprocess_data()
    print(f"   Data scaled and normalized")
    
    # Train autoencoder
    print("\n3. Training autoencoder model...")
    training_results = app.train_model('autoencoder')
    print(f"   Training completed. Final loss: {training_results['final_loss']:.6f}")
    
    # Evaluate model
    print("\n4. Evaluating model performance...")
    metrics = app.evaluate_model()
    print(f"   Accuracy: {metrics['accuracy']:.3f}")
    print(f"   Precision: {metrics['precision']:.3f}")
    print(f"   Recall: {metrics['recall']:.3f}")
    print(f"   F1-Score: {metrics['f1_score']:.3f}")
    print(f"   AUC Score: {metrics['auc_score']:.3f}")
    
    # Predict anomalies
    print("\n5. Predicting anomalies...")
    anomaly_scores, predictions = app.predict_anomalies()
    print(f"   Predicted {np.sum(predictions)} anomalies")
    
    return app, metrics, anomaly_scores, predictions


def demonstrate_time_series_detection():
    """Demonstrate time series anomaly detection."""
    print("\n" + "=" * 60)
    print("TIME SERIES ANOMALY DETECTION DEMONSTRATION")
    print("=" * 60)
    
    # Initialize application
    app = AnomalyDetectionApp()
    
    # Generate time series data
    print("\n1. Generating time series data...")
    df, labels = app.generate_data('time_series')
    print(f"   Generated {len(df)} time points with {np.sum(labels)} anomalies")
    print(f"   Time range: {df.index[0]} to {df.index[-1]}")
    
    # Preprocess data
    print("\n2. Preprocessing data...")
    scaled_data = app.preprocess_data()
    
    # Train autoencoder
    print("\n3. Training autoencoder model...")
    training_results = app.train_model('autoencoder')
    print(f"   Training completed. Final loss: {training_results['final_loss']:.6f}")
    
    # Evaluate model
    print("\n4. Evaluating model performance...")
    metrics = app.evaluate_model()
    print(f"   Accuracy: {metrics['accuracy']:.3f}")
    print(f"   AUC Score: {metrics['auc_score']:.3f}")
    
    return app, metrics


def demonstrate_model_comparison():
    """Demonstrate comparison between different models."""
    print("\n" + "=" * 60)
    print("MODEL COMPARISON DEMONSTRATION")
    print("=" * 60)
    
    # Generate data
    app = AnomalyDetectionApp()
    data, labels = app.generate_data('multivariate')
    scaled_data = app.preprocess_data()
    
    results = {}
    
    # Test Autoencoder
    print("\n1. Testing Autoencoder...")
    app.train_model('autoencoder')
    autoencoder_metrics = app.evaluate_model()
    results['Autoencoder'] = autoencoder_metrics
    print(f"   Accuracy: {autoencoder_metrics['accuracy']:.3f}")
    
    # Test Isolation Forest
    print("\n2. Testing Isolation Forest...")
    app.train_model('isolation_forest')
    if_metrics = app.evaluate_model()
    results['Isolation Forest'] = if_metrics
    print(f"   Accuracy: {if_metrics['accuracy']:.3f}")
    
    # Compare results
    print("\n3. Model Comparison:")
    print("   " + "-" * 40)
    for model_name, metrics in results.items():
        print(f"   {model_name:20} | Accuracy: {metrics['accuracy']:.3f} | AUC: {metrics['auc_score']:.3f}")
    
    return results


def create_visualizations(app, metrics, anomaly_scores, predictions):
    """Create and save visualizations."""
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    visualizer = AnomalyVisualizer()
    
    # Reconstruction error plot
    print("\n1. Creating reconstruction error plot...")
    threshold = np.percentile(anomaly_scores, 95)
    fig1 = visualizer.plot_reconstruction_errors(
        anomaly_scores, threshold, app.labels,
        title="Reconstruction Error Analysis"
    )
    fig1.savefig(plots_dir / 'reconstruction_errors.png', dpi=300, bbox_inches='tight')
    print(f"   Saved: {plots_dir / 'reconstruction_errors.png'}")
    
    # Multivariate data plot
    print("\n2. Creating multivariate data plot...")
    fig2 = visualizer.plot_multivariate_data(
        app.data, app.labels,
        feature_names=['Temperature', 'Pressure', 'Vibration'],
        title="Multivariate Data with Anomalies"
    )
    fig2.savefig(plots_dir / 'multivariate_data.png', dpi=300, bbox_inches='tight')
    print(f"   Saved: {plots_dir / 'multivariate_data.png'}")
    
    plt.close('all')
    print(f"\n   All plots saved to: {plots_dir}")


def save_results(app, metrics, anomaly_scores, predictions):
    """Save results to files."""
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'sample_index': range(len(app.data)),
        'anomaly_score': anomaly_scores,
        'predicted_anomaly': predictions,
        'true_anomaly': app.labels
    })
    
    # Add feature columns
    for i in range(app.data.shape[1]):
        results_df[f'feature_{i+1}'] = app.data[:, i]
    
    results_file = output_dir / 'anomaly_detection_results.csv'
    results_df.to_csv(results_file, index=False)
    print(f"Results saved to: {results_file}")
    
    # Save metrics
    metrics_file = output_dir / 'performance_metrics.txt'
    with open(metrics_file, 'w') as f:
        f.write("ANOMALY DETECTION PERFORMANCE METRICS\n")
        f.write("=" * 40 + "\n\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.3f}\n")
    
    print(f"Metrics saved to: {metrics_file}")


def main():
    """Main demonstration function."""
    print("MULTIVARIATE ANOMALY DETECTION TOOLKIT")
    print("Modernized and Enhanced Version")
    print("=" * 60)
    
    try:
        # Demonstrate multivariate detection
        app, metrics, anomaly_scores, predictions = demonstrate_multivariate_detection()
        
        # Demonstrate time series detection
        ts_app, ts_metrics = demonstrate_time_series_detection()
        
        # Demonstrate model comparison
        comparison_results = demonstrate_model_comparison()
        
        # Create visualizations
        create_visualizations(app, metrics, anomaly_scores, predictions)
        
        # Save results
        save_results(app, metrics, anomaly_scores, predictions)
        
        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nKey Features Demonstrated:")
        print("✓ Multivariate anomaly detection")
        print("✓ Time series analysis with trends and seasonality")
        print("✓ Multiple model types (Autoencoder, Isolation Forest)")
        print("✓ Comprehensive evaluation metrics")
        print("✓ Visualization capabilities")
        print("✓ Results export functionality")
        
        print("\nNext Steps:")
        print("• Run 'streamlit run streamlit_app.py' for web interface")
        print("• Run 'python cli.py --help' for command-line options")
        print("• Check 'notebooks/tutorial.ipynb' for detailed examples")
        print("• Review 'output/' directory for generated results")
        
    except Exception as e:
        print(f"\nError during demonstration: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
