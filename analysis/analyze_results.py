"""
Analyze experiment results and generate summary statistics.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from utils import (
    load_all_results, load_best_models, load_best_model_eval,
    compute_cv_statistics, get_experiment_configs, format_metric
)


def analyze_experiment(model_name: str, timestamp: str) -> dict:
    """
    Analyze a single experiment and return comprehensive statistics.
    
    Args:
        model_name: Name of the model (mlp, tabnet, tabtransformer)
        timestamp: Experiment timestamp
        
    Returns:
        Dictionary with analysis results
    """
    # Load data
    all_results = load_all_results(model_name, timestamp)
    best_models = load_best_models(model_name, timestamp)
    best_eval = load_best_model_eval(model_name, timestamp)
    
    # Metrics to analyze
    metric_cols = ['mae', 'mse', 'rmse', 'r2', 'mape']
    
    # Aggregate CV statistics for each configuration
    cv_stats = []
    for gs_id in all_results['grid_search_id'].unique():
        config_data = all_results[all_results['grid_search_id'] == gs_id]
        stats = {'grid_search_id': gs_id}
        stats.update(compute_cv_statistics(config_data, metric_cols))
        
        # Add hyperparameters
        for col in config_data.columns:
            if col not in ['fold', 'grid_search_id'] + metric_cols + ['final_train_loss', 'final_val_loss', 'best_val_loss', 'best_epoch', 'actual_epochs', 'num_epochs', 'stopped_early', 'model_path']:
                stats[col] = config_data[col].iloc[0]
        
        cv_stats.append(stats)
    
    cv_stats_df = pd.DataFrame(cv_stats)
    
    # Sort by mean validation MAE
    if 'mae_mean' in cv_stats_df.columns:
        cv_stats_df = cv_stats_df.sort_values('mae_mean')
    
    analysis = {
        'model_name': model_name,
        'timestamp': timestamp,
        'num_configurations': len(all_results['grid_search_id'].unique()),
        'num_folds': len(all_results['fold'].unique()),
        'cv_statistics': cv_stats_df,
        'best_models': best_models,
        'best_eval': best_eval,
        'all_results': all_results
    }
    
    return analysis


def compare_architectures(analyses: dict) -> pd.DataFrame:
    """
    Compare different architectures.
    
    Args:
        analyses: Dictionary with model_name as keys and analysis results as values
        
    Returns:
        DataFrame with comparison results
    """
    comparison = []
    
    for model_name, analysis in analyses.items():
        best_eval = analysis['best_eval']
        cv_stats = analysis['cv_statistics']
        
        # Get best configuration statistics
        best_config = cv_stats.iloc[0]
        
        comparison.append({
            'Architecture': analysis.get('display_name', model_name),
            'MAE (Test)': best_eval['mae'].iloc[0],
            'RMSE (Test)': best_eval['rmse'].iloc[0],
            'R² (Test)': best_eval['r2'].iloc[0],
            'MAPE (Test)': best_eval['mape'].iloc[0],
            'MAE (CV Mean)': best_config.get('mae_mean', np.nan),
            'MAE (CV Std)': best_config.get('mae_std', np.nan),
            'Num Configurations': analysis['num_configurations']
        })
    
    return pd.DataFrame(comparison).sort_values('MAE (Test)')


def print_summary(model_name: str, analysis: dict):
    """Print summary of analysis results."""
    print(f"\n{'='*80}")
    print(f"ANALYSIS SUMMARY: {model_name.upper()}")
    print(f"{'='*80}")
    print(f"Experiment timestamp: {analysis['timestamp']}")
    print(f"Number of configurations: {analysis['num_configurations']}")
    print(f"Number of folds: {analysis['num_folds']}")
    
    print(f"\n--- Best Model (Test Set) ---")
    best_eval = analysis['best_eval']
    for col in best_eval.columns:
        print(f"{col}: {best_eval[col].iloc[0]:.4f}")
    
    print(f"\n--- Top 5 Configurations (CV Mean MAE) ---")
    cv_stats = analysis['cv_statistics']
    top5 = cv_stats.head(5)
    
    for idx, row in top5.iterrows():
        print(f"\nRank {idx+1}:")
        print(f"  Grid Search ID: {row['grid_search_id']}")
        if 'mae_mean' in row:
            print(f"  MAE (CV): {row['mae_mean']:.4f} ± {row['mae_std']:.4f}")
        if 'rmse_mean' in row:
            print(f"  RMSE (CV): {row['rmse_mean']:.4f} ± {row['rmse_std']:.4f}")
        if 'r2_mean' in row:
            print(f"  R² (CV): {row['r2_mean']:.4f} ± {row['r2_std']:.4f}")


def main():
    """Main analysis function."""
    configs = get_experiment_configs()
    
    # Analyze each experiment
    analyses = {}
    for key, config in configs.items():
        print(f"\nAnalyzing {config['display_name']}...")
        analysis = analyze_experiment(config['name'], config['timestamp'])
        analysis['display_name'] = config['display_name']
        analyses[key] = analysis
        print_summary(config['name'], analysis)
    
    # Compare architectures
    print(f"\n{'='*80}")
    print("ARCHITECTURE COMPARISON")
    print(f"{'='*80}")
    comparison = compare_architectures(analyses)
    print(comparison.to_string(index=False))
    
    # Save analyses for later use
    import pickle
    output_path = Path(__file__).parent / 'analyses.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(analyses, f)
    print(f"\nAnalyses saved to {output_path}")


if __name__ == '__main__':
    main()
