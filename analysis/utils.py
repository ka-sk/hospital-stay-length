"""
Utility functions for analysis scripts.
"""
import pandas as pd
import numpy as np
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
RESULTS_DIR = EXPERIMENTS_DIR / "results"


def load_all_results(model_name: str, experiment_timestamp: str) -> pd.DataFrame:
    """Load all results CSV for a given experiment."""
    csv_path = RESULTS_DIR / f"{model_name}_{experiment_timestamp}_all_results.csv"
    return pd.read_csv(csv_path)


def load_best_models(model_name: str, experiment_timestamp: str) -> pd.DataFrame:
    """Load best models CSV for a given experiment."""
    csv_path = RESULTS_DIR / f"{model_name}_{experiment_timestamp}_best_models.csv"
    return pd.read_csv(csv_path)


def load_best_model_eval(model_name: str, experiment_timestamp: str) -> pd.DataFrame:
    """Load best model evaluation CSV for a given experiment."""
    csv_path = RESULTS_DIR / f"{model_name}_{experiment_timestamp}" / "best_model_eval.csv"
    return pd.read_csv(csv_path)


def load_predictions(model_name: str, experiment_timestamp: str) -> pd.DataFrame:
    """Load predictions CSV for a given experiment."""
    csv_path = RESULTS_DIR / f"{model_name}_{experiment_timestamp}" / "best_model_eval.predictions.csv"
    return pd.read_csv(csv_path)


def format_metric(value: float, decimals: int = 4) -> str:
    """Format metric value to specified decimal places."""
    return f"{value:.{decimals}f}"


def escape_latex(text: str) -> str:
    """Escape special LaTeX characters."""
    replacements = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\^{}',
        '\\': r'\textbackslash{}',
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text


def compute_cv_statistics(df: pd.DataFrame, metric_cols: list) -> dict:
    """
    Compute cross-validation statistics (mean and std) for specified metrics.
    
    Args:
        df: DataFrame with fold results for a single configuration
        metric_cols: List of metric column names
        
    Returns:
        Dictionary with mean and std for each metric
    """
    stats = {}
    for col in metric_cols:
        if col in df.columns:
            stats[f'{col}_mean'] = df[col].mean()
            stats[f'{col}_std'] = df[col].std()
    return stats


def get_experiment_configs():
    """Return experiment configurations."""
    return {
        'mlp': {
            'name': 'mlp',
            'timestamp': '20251216_221732',
            'display_name': 'SimpleMLP',
            'best_gs_id': 13
        },
        'tabnet': {
            'name': 'tabnet',
            'timestamp': '20251217_103631',
            'display_name': 'TabNet',
            'best_gs_id': 22
        },
        'tabtransformer': {
            'name': 'tabtransformer',
            'timestamp': '20251217_101214',
            'display_name': 'TabTransformer',
            'best_gs_id': 11
        }
    }
