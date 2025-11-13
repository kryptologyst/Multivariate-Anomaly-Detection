#!/usr/bin/env python3
"""
Command-line interface for multivariate anomaly detection.
"""

import argparse
import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.app import AnomalyDetectionApp
from src.config import Config
from src.visualization import AnomalyVisualizer


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Multivariate Anomaly Detection CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python cli.py

  # Run with custom configuration
  python cli.py --config config/config.yaml

  # Generate time series data
  python cli.py --data-type time_series --model-type autoencoder

  # Run with custom parameters
  python cli.py --samples 2000 --features 5 --anomaly-ratio 0.1
        """
    )
    
    # Data parameters
    parser.add_argument(
        '--data-type',
        choices=['multivariate', 'time_series'],
        default='multivariate',
        help='Type of data to generate (default: multivariate)'
    )
    
    parser.add_argument(
        '--samples',
        type=int,
        default=1000,
        help='Number of samples to generate (default: 1000)'
    )
    
    parser.add_argument(
        '--features',
        type=int,
        default=3,
        help='Number of features (default: 3)'
    )
    
    parser.add_argument(
        '--anomaly-ratio',
        type=float,
        default=0.05,
        help='Proportion of anomalies (default: 0.05)'
    )
    
    # Model parameters
    parser.add_argument(
        '--model-type',
        choices=['autoencoder', 'isolation_forest'],
        default='autoencoder',
        help='Type of anomaly detection model (default: autoencoder)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate (default: 0.001)'
    )
    
    parser.add_argument(
        '--threshold-percentile',
        type=int,
        default=95,
        help='Anomaly threshold percentile (default: 95)'
    )
    
    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Output directory for results (default: output)'
    )
    
    parser.add_argument(
        '--save-plots',
        action='store_true',
        help='Save visualization plots'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = 'DEBUG' if args.verbose else 'INFO'
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Initialize application
        print("Initializing anomaly detection application...")
        app = AnomalyDetectionApp(args.config)
        
        # Update configuration with command line arguments
        app.config.set('data.n_samples', args.samples)
        app.config.set('data.n_features', args.features)
        app.config.set('data.anomaly_ratio', args.anomaly_ratio)
        app.config.set('model.type', args.model_type)
        app.config.set('model.epochs', args.epochs)
        app.config.set('model.learning_rate', args.learning_rate)
        app.config.set('model.threshold_percentile', args.threshold_percentile)
        app.config.set('logging.level', log_level)
        
        # Run full pipeline
        print(f"Running anomaly detection pipeline...")
        print(f"Data type: {args.data_type}")
        print(f"Model type: {args.model_type}")
        print(f"Samples: {args.samples:,}")
        print(f"Features: {args.features}")
        print(f"Anomaly ratio: {args.anomaly_ratio:.1%}")
        
        results = app.run_full_pipeline(
            data_type=args.data_type,
            model_type=args.model_type
        )
        
        # Display results
        print("\n" + "="*50)
        print("RESULTS SUMMARY")
        print("="*50)
        
        metrics = results['evaluation_metrics']
        print(f"Accuracy:     {metrics['accuracy']:.3f}")
        print(f"Precision:    {metrics['precision']:.3f}")
        print(f"Recall:       {metrics['recall']:.3f}")
        print(f"F1-Score:     {metrics['f1_score']:.3f}")
        print(f"AUC Score:    {metrics['auc_score']:.3f}")
        
        data_info = results['data_info']
        print(f"\nData Summary:")
        print(f"Total samples: {data_info['n_samples']:,}")
        print(f"True anomalies: {data_info['n_anomalies']:,}")
        print(f"Predicted anomalies: {np.sum(results['predictions']['predicted_labels']):,}")
        
        # Save results
        if args.save_plots:
            print(f"\nSaving plots to {output_dir}...")
            visualizer = AnomalyVisualizer()
            
            plots_dir = output_dir / 'plots'
            plots_dir.mkdir(exist_ok=True)
            
            # Save reconstruction error plot
            threshold = results['predictions']['threshold']
            fig1 = visualizer.plot_reconstruction_errors(
                results['predictions']['anomaly_scores'],
                threshold,
                app.labels
            )
            fig1.savefig(plots_dir / 'reconstruction_errors.png', dpi=300, bbox_inches='tight')
            
            # Save multivariate data plot
            fig2 = visualizer.plot_multivariate_data(app.data, app.labels)
            fig2.savefig(plots_dir / 'multivariate_data.png', dpi=300, bbox_inches='tight')
            
            print(f"Plots saved to {plots_dir}")
        
        # Save results to CSV
        results_file = output_dir / 'results.csv'
        import pandas as pd
        
        results_df = pd.DataFrame({
            'sample_index': range(len(app.data)),
            'anomaly_score': results['predictions']['anomaly_scores'],
            'predicted_anomaly': results['predictions']['predicted_labels'],
            'true_anomaly': app.labels
        })
        
        # Add feature columns
        for i in range(args.features):
            results_df[f'feature_{i+1}'] = app.data[:, i]
        
        results_df.to_csv(results_file, index=False)
        print(f"Results saved to {results_file}")
        
        print(f"\nAnalysis completed successfully!")
        print(f"Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
