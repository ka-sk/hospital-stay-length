"""
Evaluation module for trained models.
Provides comprehensive regression metrics and model evaluation utilities.
"""
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    r2_score,
    median_absolute_error,
    max_error as sklearn_max_error
)
from typing import Union
import matplotlib.pyplot as plt


# ============== PATHS ==============
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
RESULTS_DIR = EXPERIMENTS_DIR / "results"
MODELS_DIR = EXPERIMENTS_DIR / "models"
BEST_DIR = EXPERIMENTS_DIR / "best"


# ============== REGRESSION METRICS ==============
def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:

    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # Basic metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    max_err = sklearn_max_error(y_true, y_pred)
    
    # MAPE - handle zero values
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan
    
    # Explained Variance
    explained_var = 1 - np.var(y_true - y_pred) / np.var(y_true)
    
    # Additional statistics
    errors = y_true - y_pred
    mean_error = np.mean(errors)  # Bias
    std_error = np.std(errors)
    
    return {
        # Primary metrics
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        
        # Robust metrics
        "medae": medae,
        "max_error": max_err,
        
        # Percentage metrics
        "mape": mape,
        
        # Additional
        "explained_variance": explained_var,
        "mean_error": mean_error,  # Bias (ujemny = niedoszacowanie)
        "std_error": std_error,
    }


def evaluate_model(model: torch.nn.Module, dataloader: DataLoader, device: str = None) -> dict:
    
    if device is None:
        from utils import get_device
        device = get_device()
    
    model = model.to(device)
    model.eval()
    
    all_y_true = []
    all_y_pred = []
    
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            all_y_true.append(y.cpu().numpy())
            all_y_pred.append(y_pred.cpu().numpy())
    
    y_true = np.concatenate(all_y_true).flatten()
    y_pred = np.concatenate(all_y_pred).flatten()
    
    metrics = compute_regression_metrics(y_true, y_pred)
    metrics["y_true"] = y_true
    metrics["y_pred"] = y_pred
    
    return metrics


def load_and_evaluate_model(
    model_class: type,
    model_path: Union[str, Path],
    dataloader: DataLoader,
    model_kwargs: dict = None,
    device: str = None
) -> dict:

    if device is None:
        from utils import get_device
        device = get_device()
    
    model_kwargs = model_kwargs or {}
    model = model_class(**model_kwargs)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    
    metrics = evaluate_model(model, dataloader, device)
    metrics["model_path"] = str(model_path)
    metrics["model_class"] = model_class.__name__
    
    return metrics


def evaluate_best_model(experiment_name: str, dataloader: DataLoader) -> dict:

    import models  # Import here to avoid circular imports
    
    # Load metadata
    meta_path = BEST_DIR / f"{experiment_name}_best_metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")
    
    metadata = pd.read_csv(meta_path).iloc[0]
    model_name = metadata["model_name"]
    model_path = BEST_DIR / f"{experiment_name}_best_{model_name}_loss{metadata['best_val_loss']:.4f}.pt"
    
    # Get model class and hyperparameters
    if model_name == "SimpleMLP":
        model_class = models.SimpleMLP
        model_kwargs = {
            "in_features": 22,  # Default
            "hidden_features": int(metadata.get("hidden_features", 64)),
            "activation_layer": str(metadata.get("activation", "relu")).lower(),
            "dropout": float(metadata.get("dropout", 0.2))
        }
    elif model_name == "SimpleTabTransformer":
        model_class = models.SimpleTabTransformer
        model_kwargs = {
            "in_features": 22,
            "d_model": int(metadata.get("d_model", 64)),
            "n_heads": int(metadata.get("n_heads", 4)),
            "num_layers": int(metadata.get("num_layers", 2))
        }
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return load_and_evaluate_model(model_class, model_path, dataloader, model_kwargs)


# ============== REPORTING ==============
def print_metrics(metrics: dict, title: str = "Evaluation Results"):

    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}")
    
    # Primary metrics
    print(f"\n  Primary Metrics:")
    print(f"  {'─' * 40}")
    print(f"  MAE  (Mean Absolute Error):     {metrics['mae']:.4f} days")
    print(f"  RMSE (Root Mean Squared Error): {metrics['rmse']:.4f} days")
    print(f"  R²   (Coefficient of Determ.):  {metrics['r2']:.4f}")
    
    # Robust metrics (resistant to outliers)
    print(f"\n  Robust Metrics (outlier resistant):")
    print(f"  {'─' * 40}")
    print(f"  MedAE (Median Absolute Error):  {metrics['medae']:.4f} days")
    print(f"  Max Error (worst case):         {metrics['max_error']:.4f} days")
    
    # Percentage metrics
    if not np.isnan(metrics['mape']):
        print(f"\n  Percentage Metrics:")
        print(f"  {'─' * 40}")
        print(f"  MAPE (Mean Abs. % Error):       {metrics['mape']:.2f}%")
    
    # Bias analysis
    print(f"\n  Bias Analysis:")
    print(f"  {'─' * 40}")
    print(f"  Mean Error (bias):              {metrics['mean_error']:.4f} days")
    if metrics['mean_error'] > 0:
        print(f"    → Model tends to UNDERESTIMATE")
    elif metrics['mean_error'] < 0:
        print(f"    → Model tends to OVERESTIMATE")
    print(f"  Std Error:                      {metrics['std_error']:.4f} days")
    
    print(f"\n{'═' * 60}\n")


def metrics_to_dataframe(metrics: dict) -> pd.DataFrame:
    """Convert metrics dict to DataFrame (excluding arrays)."""
    return pd.DataFrame({
        k: [v] for k, v in metrics.items() 
        if not isinstance(v, np.ndarray)
    })


def save_evaluation_results(
    metrics: dict, 
    save_path: Union[str, Path],
    include_predictions: bool = False
):

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Metrics CSV
    df = metrics_to_dataframe(metrics)
    df.to_csv(save_path, index=False)
    print(f"Metrics saved: {save_path}")
    
    # Predictions CSV (optional)
    if include_predictions and "y_true" in metrics and "y_pred" in metrics:
        pred_path = save_path.with_suffix(".predictions.csv")
        pred_df = pd.DataFrame({
            "y_true": metrics["y_true"],
            "y_pred": metrics["y_pred"],
            "error": metrics["y_true"] - metrics["y_pred"]
        })
        pred_df.to_csv(pred_path, index=False)
        print(f"Predictions saved: {pred_path}")


# ============== VISUALIZATION ==============
def plot_predictions(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    title: str = "Predictions vs Actual",
    save_path: Union[str, Path] = None
):

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter plot
    ax1 = axes[0]
    ax1.scatter(y_true, y_pred, alpha=0.5, s=10)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')
    
    ax1.set_xlabel('Actual (days)')
    ax1.set_ylabel('Predicted (days)')
    ax1.set_title(f'{title}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Error distribution
    ax2 = axes[1]
    errors = y_true - y_pred
    ax2.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    ax2.axvline(x=0, color='r', linestyle='--', label='Zero error')
    ax2.axvline(x=np.mean(errors), color='g', linestyle='-', label=f'Mean: {np.mean(errors):.2f}')
    ax2.set_xlabel('Error (days)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Error Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {save_path}")
    
    plt.show()
    return fig


def plot_error_by_value(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    save_path: Union[str, Path] = None
):

    errors = np.abs(y_true - y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(y_true, errors, alpha=0.5, s=10)
    ax.set_xlabel('Actual Length of Stay (days)')
    ax.set_ylabel('Absolute Error (days)')
    ax.set_title('Error vs Actual Value')
    ax.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(y_true, errors, 1)
    p = np.poly1d(z)
    x_line = np.linspace(y_true.min(), y_true.max(), 100)
    ax.plot(x_line, p(x_line), 'r--', label=f'Trend (slope: {z[0]:.3f})')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {save_path}")
    
    plt.show()
    return fig


if __name__ == "__main__":
    import data_loader as data
    import models
    from torch.utils.data import random_split
    
    # Load data
    print("Loading data...")
    dataset = data.data_filtration(data.load_data())
    
    # Split data
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    _, test_dataset = random_split(dataset, [train_size, test_size], 
                                   generator=torch.Generator().manual_seed(42))
    
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Example: Evaluate a simple model (untrained - just for testing)
    print("\nTesting evaluation with untrained model...")
    model = models.SimpleMLP(in_features=22, hidden_features=64)
    
    metrics = evaluate_model(model, test_loader)
    print_metrics(metrics, title="Test Evaluation (untrained model)")
    
    # Plot if we have predictions
    if "y_true" in metrics and "y_pred" in metrics:
        plot_predictions(metrics["y_true"], metrics["y_pred"], 
                        title="Untrained Model Predictions")
