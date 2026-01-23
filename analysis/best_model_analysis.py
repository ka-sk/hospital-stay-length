"""
Detailed analysis of best models from each architecture.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from utils import (
    load_all_results, load_best_models, load_best_model_eval,
    load_predictions, get_experiment_configs, format_metric
)


def analyze_residuals(predictions_df: pd.DataFrame) -> dict:
    """
    Analyze residuals for normality and patterns.
    
    Args:
        predictions_df: DataFrame with y_true, y_pred, error columns
        
    Returns:
        Dictionary with residual analysis results
    """
    residuals = predictions_df['error'].values
    
    # Normality tests
    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    
    # Anderson-Darling test
    anderson_result = stats.anderson(residuals, dist='norm')
    
    # Skewness and kurtosis
    skewness = stats.skew(residuals)
    kurtosis = stats.kurtosis(residuals)
    
    # Heteroscedasticity check (Breusch-Pagan-like)
    # Correlation between squared residuals and predictions
    squared_residuals = residuals ** 2
    predictions = predictions_df['y_pred'].values
    heterosced_corr = np.corrcoef(predictions, squared_residuals)[0, 1]
    
    analysis = {
        'mean_residual': np.mean(residuals),
        'std_residual': np.std(residuals),
        'min_residual': np.min(residuals),
        'max_residual': np.max(residuals),
        'median_residual': np.median(residuals),
        'shapiro_statistic': shapiro_stat,
        'shapiro_pvalue': shapiro_p,
        'anderson_statistic': anderson_result.statistic,
        'anderson_critical_values': anderson_result.critical_values,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'heteroscedasticity_correlation': heterosced_corr
    }
    
    return analysis


def identify_worst_predictions(predictions_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Identify predictions with largest errors.
    
    Args:
        predictions_df: DataFrame with y_true, y_pred, error columns
        n: Number of worst predictions to return
        
    Returns:
        DataFrame with worst predictions
    """
    # Sort by absolute error
    predictions_df['abs_error'] = predictions_df['error'].abs()
    worst = predictions_df.nlargest(n, 'abs_error').copy()
    worst['relative_error'] = (worst['error'] / worst['y_true']) * 100
    
    return worst[['y_true', 'y_pred', 'error', 'abs_error', 'relative_error']]


def analyze_best_model(model_name: str, timestamp: str, gs_id: int) -> dict:
    """
    Comprehensive analysis of a single best model.
    
    Args:
        model_name: Name of the model
        timestamp: Experiment timestamp
        gs_id: Grid search ID of the best model
        
    Returns:
        Dictionary with comprehensive analysis
    """
    # Load data
    all_results = load_all_results(model_name, timestamp)
    best_eval = load_best_model_eval(model_name, timestamp)
    predictions = load_predictions(model_name, timestamp)
    
    # Get configuration data for all folds
    config_data = all_results[all_results['grid_search_id'] == gs_id]
    
    # CV metrics statistics
    metric_cols = ['mae', 'mse', 'rmse', 'r2', 'mape', 'final_train_loss', 
                   'final_val_loss', 'best_val_loss', 'actual_epochs']
    
    cv_metrics = {}
    for col in metric_cols:
        if col in config_data.columns:
            cv_metrics[f'{col}_mean'] = config_data[col].mean()
            cv_metrics[f'{col}_std'] = config_data[col].std()
            cv_metrics[f'{col}_min'] = config_data[col].min()
            cv_metrics[f'{col}_max'] = config_data[col].max()
    
    # Hyperparameters
    hyperparams = {}
    first_row = config_data.iloc[0]
    for col in config_data.columns:
        if col not in metric_cols + ['fold', 'grid_search_id', 'model_path']:
            hyperparams[col] = first_row[col]
    
    # Test set metrics
    test_metrics = best_eval.iloc[0].to_dict()
    
    # Residual analysis
    residual_analysis = analyze_residuals(predictions)
    
    # Worst predictions
    worst_predictions = identify_worst_predictions(predictions, n=20)
    
    # Prediction statistics
    pred_stats = {
        'num_predictions': len(predictions),
        'mean_true': predictions['y_true'].mean(),
        'std_true': predictions['y_true'].std(),
        'min_true': predictions['y_true'].min(),
        'max_true': predictions['y_true'].max(),
        'mean_pred': predictions['y_pred'].mean(),
        'std_pred': predictions['y_pred'].std(),
        'min_pred': predictions['y_pred'].min(),
        'max_pred': predictions['y_pred'].max()
    }
    
    analysis = {
        'model_name': model_name,
        'timestamp': timestamp,
        'grid_search_id': gs_id,
        'hyperparameters': hyperparams,
        'cv_metrics': cv_metrics,
        'test_metrics': test_metrics,
        'residual_analysis': residual_analysis,
        'worst_predictions': worst_predictions,
        'prediction_statistics': pred_stats,
        'num_folds': len(config_data)
    }
    
    return analysis


def print_detailed_analysis(analysis: dict):
    """Print detailed analysis report."""
    print(f"\n{'='*80}")
    print(f"DETAILED ANALYSIS: {analysis['model_name'].upper()}")
    print(f"Grid Search ID: {analysis['grid_search_id']}")
    print(f"{'='*80}")
    
    print(f"\n--- Hyperparameters ---")
    for key, value in analysis['hyperparameters'].items():
        print(f"  {key}: {value}")
    
    print(f"\n--- Cross-Validation Metrics (5 folds) ---")
    cv = analysis['cv_metrics']
    print(f"  MAE: {cv.get('mae_mean', 0):.4f} ± {cv.get('mae_std', 0):.4f}")
    print(f"  RMSE: {cv.get('rmse_mean', 0):.4f} ± {cv.get('rmse_std', 0):.4f}")
    print(f"  R²: {cv.get('r2_mean', 0):.4f} ± {cv.get('r2_std', 0):.4f}")
    print(f"  MAPE: {cv.get('mape_mean', 0):.4f} ± {cv.get('mape_std', 0):.4f}")
    print(f"  Avg Epochs: {cv.get('actual_epochs_mean', 0):.1f} ± {cv.get('actual_epochs_std', 0):.1f}")
    
    print(f"\n--- Test Set Metrics ---")
    test = analysis['test_metrics']
    print(f"  MAE: {test['mae']:.4f}")
    print(f"  RMSE: {test['rmse']:.4f}")
    print(f"  R²: {test['r2']:.4f}")
    print(f"  MAPE: {test['mape']:.4f}")
    print(f"  MedAE: {test['medae']:.4f}")
    print(f"  Max Error: {test['max_error']:.4f}")
    
    print(f"\n--- Residual Analysis ---")
    res = analysis['residual_analysis']
    print(f"  Mean: {res['mean_residual']:.4f}")
    print(f"  Std: {res['std_residual']:.4f}")
    print(f"  Median: {res['median_residual']:.4f}")
    print(f"  Range: [{res['min_residual']:.4f}, {res['max_residual']:.4f}]")
    print(f"  Shapiro-Wilk p-value: {res['shapiro_pvalue']:.4f} {'(normal)' if res['shapiro_pvalue'] > 0.05 else '(not normal)'}")
    print(f"  Skewness: {res['skewness']:.4f}")
    print(f"  Kurtosis: {res['kurtosis']:.4f}")
    print(f"  Heteroscedasticity correlation: {res['heteroscedasticity_correlation']:.4f}")
    
    print(f"\n--- Prediction Statistics ---")
    pred = analysis['prediction_statistics']
    print(f"  Number of predictions: {pred['num_predictions']}")
    print(f"  True values: {pred['mean_true']:.4f} ± {pred['std_true']:.4f} [{pred['min_true']:.1f}, {pred['max_true']:.1f}]")
    print(f"  Predicted values: {pred['mean_pred']:.4f} ± {pred['std_pred']:.4f} [{pred['min_pred']:.1f}, {pred['max_pred']:.1f}]")
    
    print(f"\n--- Top 10 Worst Predictions ---")
    worst = analysis['worst_predictions'].head(10)
    print(worst.to_string(index=False))


def main():
    """Main function for detailed analysis."""
    configs = get_experiment_configs()
    
    detailed_analyses = {}
    
    for key, config in configs.items():
        print(f"\nAnalyzing {config['display_name']} (Best Model)...")
        analysis = analyze_best_model(
            config['name'],
            config['timestamp'],
            config['best_gs_id']
        )
        analysis['display_name'] = config['display_name']
        detailed_analyses[key] = analysis
        print_detailed_analysis(analysis)
    
    # Save analyses
    import pickle
    output_path = Path(__file__).parent / 'detailed_analyses.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(detailed_analyses, f)
    print(f"\n\nDetailed analyses saved to {output_path}")


if __name__ == '__main__':
    main()
