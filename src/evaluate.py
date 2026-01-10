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

# Optional matplotlib and scipy imports for plotting
try:
    import matplotlib.pyplot as plt
    from scipy import stats
    from scipy.signal import savgol_filter
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
    stats = None
    savgol_filter = None


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


def evaluate_model(model, dataloader: DataLoader = None, device: str = None, X_data: np.ndarray = None, y_data: np.ndarray = None) -> dict:
    """
    Evaluate model on validation data.

    Args:
        model: PyTorch model or TabNetRegressor
        dataloader: DataLoader for PyTorch models (optional if X_data/y_data provided)
        device: Device for PyTorch models
        X_data: Numpy array for TabNet models
        y_data: Numpy array for TabNet models (ground truth)

    Returns:
        dict with metrics and predictions
    """
    from pytorch_tabnet.tab_model import TabNetRegressor

    # TabNet uses different API
    if isinstance(model, TabNetRegressor):
        if X_data is None or y_data is None:
            raise ValueError("TabNet requires X_data and y_data")

        y_pred = model.predict(X_data).flatten()
        y_true = y_data.flatten()

        metrics = compute_regression_metrics(y_true, y_pred)
        metrics["y_true"] = y_true
        metrics["y_pred"] = y_pred
        return metrics

    # PyTorch models
    if device is None:
        from utils import get_device
        device = get_device()

    if dataloader is None:
        raise ValueError("PyTorch models require dataloader")

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


def evaluate_best_model(
    experiment_name: str,
    X_test,
    y_test,
    save_results: bool = False,
    show_plots: bool = False
) -> dict:
    """
    Evaluate the best model from an experiment.

    Args:
        experiment_name: Name of the experiment
        X_test: Test features (tensor or numpy array)
        y_test: Test labels (tensor or numpy array)
        save_results: Save metrics to CSV
        show_plots: Generate and show evaluatn metrics
    """
    import models  # Import here to avoid circular imports
    from pytorch_tabnet.tab_model import TabNetRegressor

    # Load metadata
    meta_path = BEST_DIR / f"{experiment_name}_best_metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")

    metadata = pd.read_csv(meta_path).iloc[0]
    model_name = metadata["model_name"]

    # Convert tensors to numpy for consistency
    if isinstance(X_test, torch.Tensor):
        X_test_np = X_test.cpu().numpy()
        y_test_np = y_test.cpu().numpy().flatten()
    else:
        X_test_np = X_test
        y_test_np = y_test.flatten()

    # Handle TabNet differently (uses .zip format, not .pt)
    if model_name == "TabNetRegressor":
        # Find TabNet model directory
        model_dir_name = f"{experiment_name}_best_{model_name}_loss{metadata['best_val_loss']:.4f}"
        model_path = BEST_DIR / model_dir_name

        # Load TabNet model
        model = TabNetRegressor()
        model.load_model(str(model_path) + ".zip")

        # Evaluate with numpy arrays
        metrics = evaluate_model(
            model=model,
            X_data=X_test_np,
            y_data=y_test_np
        )

    else:
        # PyTorch models
        model_path = BEST_DIR / f"{experiment_name}_best_{model_name}_loss{metadata['best_val_loss']:.4f}.pt"

        # Get model class and hyperparameters
        if model_name == "SimpleMLP":
            model_class = models.SimpleMLP
            model_kwargs = {
                "in_features": int(metadata.get("in_features", 22)),
                "hidden_features": int(metadata.get("hidden_features", 64)),
                "activation_layer": str(metadata.get("activation", "relu")).lower(),
                "dropout": float(metadata.get("dropout", 0.2))
            }
        elif model_name == "SimpleTabTransformer":
            model_class = models.SimpleTabTransformer
            model_kwargs = {
                "in_features": int(metadata.get("in_features", 22)),
                "d_model": int(metadata.get("d_model", 64)),
                "n_heads": int(metadata.get("n_heads", 4)),
                "num_layers": int(metadata.get("num_layers", 2))
            }
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Load PyTorch model
        from utils import get_device
        device = get_device()

        model = model_class(**model_kwargs)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

        # Create dataloader
        test_dataset = TensorDataset(
            torch.tensor(X_test_np, dtype=torch.float32),
            torch.tensor(y_test_np, dtype=torch.float32)
        )
        test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        # Evaluate
        metrics = evaluate_model(model=model, dataloader=test_dataloader, device=device)

    # Print metrics
    print_metrics(metrics, title=f"Best Model Evaluation: {experiment_name}")

    # Save results
    if save_results:
        results_path = RESULTS_DIR / experiment_name / "best_model_eval.csv"
        save_evaluation_results(metrics, results_path, include_predictions=True)

    # Generate plots
    if show_plots:
        y_true = metrics["y_true"]
        y_pred = metrics["y_pred"]

        # Create plots directory
        plots_dir = RESULTS_DIR / experiment_name / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        plot_predictions(y_true, y_pred, title=f"{experiment_name} - Predictions vs Actual",
                        save_path=plots_dir / "predictions.png")
        plot_residuals(y_true, y_pred,
                      save_path=plots_dir / "residuals.png")
        plot_qq(y_true, y_pred,
               save_path=plots_dir / "qq_plot.png")
        plot_comprehensive_evaluation(y_true, y_pred, title=f"{experiment_name} - Comprehensive Evaluation",
                                     save_path=plots_dir / "comprehensive.png")

    return metrics


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
        # Save as PNG
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {save_path}")
        # Save as EPS
        eps_path = Path(save_path).with_suffix('.eps')
        plt.savefig(eps_path, format='eps', bbox_inches='tight')
        print(f"Plot saved: {eps_path}")

    plt.close(fig)
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
        # Save as PNG
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {save_path}")
        # Save as EPS
        eps_path = Path(save_path).with_suffix('.eps')
        plt.savefig(eps_path, format='eps', bbox_inches='tight')
        print(f"Plot saved: {eps_path}")

    plt.close(fig)
    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Union[str, Path] = None
):
    """
    Plot residuals vs fitted values to check for homoscedasticity.
    Ideal pattern: random scatter around zero line.
    """
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not installed, skipping plot")
        return None

    residuals = y_true - y_pred

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(y_pred, residuals, alpha=0.5, s=10)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero residual')

    # Add smoothed trend line
    from scipy.signal import savgol_filter
    try:
        # Sort by predicted values for smooth line
        sorted_idx = np.argsort(y_pred)
        y_pred_sorted = y_pred[sorted_idx]
        residuals_sorted = residuals[sorted_idx]

        # Apply smoothing if enough points
        if len(y_pred) > 50:
            window = min(51, len(y_pred) // 10 * 2 + 1)  # Odd number
            residuals_smooth = savgol_filter(residuals_sorted, window, 3)
            ax.plot(y_pred_sorted, residuals_smooth, 'g-', linewidth=2, alpha=0.7, label='Trend')
    except:
        pass  # Skip if smoothing fails

    ax.set_xlabel('Fitted Values (Predicted Length of Stay)')
    ax.set_ylabel('Residuals (Actual - Predicted)')
    ax.set_title('Residual Plot')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        # Save as PNG
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {save_path}")
        # Save as EPS
        eps_path = Path(save_path).with_suffix('.eps')
        plt.savefig(eps_path, format='eps', bbox_inches='tight')
        print(f"Plot saved: {eps_path}")

    plt.close(fig)
    return fig


def plot_qq(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Union[str, Path] = None
):
    """
    Q-Q plot to check if residuals follow normal distribution.
    Points should fall along the diagonal line for normal distribution.
    """
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not installed, skipping plot")
        return None

    from scipy import stats

    residuals = y_true - y_pred

    fig, ax = plt.subplots(figsize=(8, 8))

    # Q-Q plot
    stats.probplot(residuals, dist="norm", plot=ax)

    ax.set_title('Q-Q Plot (Normal Distribution Check)')
    ax.grid(True, alpha=0.3)

    # Add text with normality test
    _, p_value = stats.shapiro(residuals[:5000] if len(residuals) > 5000 else residuals)
    textstr = f'Shapiro-Wilk p-value: {p_value:.4f}\n'
    if p_value < 0.05:
        textstr += 'Result: Residuals NOT normally distributed'
    else:
        textstr += 'Result: Residuals appear normally distributed'

    ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        # Save as PNG
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {save_path}")
        # Save as EPS
        eps_path = Path(save_path).with_suffix('.eps')
        plt.savefig(eps_path, format='eps', bbox_inches='tight')
        print(f"Plot saved: {eps_path}")

    plt.close(fig)
    return fig


def plot_comprehensive_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Model Evaluation",
    save_path: Union[str, Path] = None
):
    """
    Create comprehensive evaluation plot with 4 subplots:
    1. Predictions vs Actual
    2. Residuals vs Fitted
    3. Error Distribution
    4. Q-Q Plot
    """
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not installed, skipping plot")
        return None

    from scipy import stats

    residuals = y_true - y_pred

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # 1. Predictions vs Actual
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y_true, y_pred, alpha=0.5, s=10)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')
    ax1.set_xlabel('Actual (days)')
    ax1.set_ylabel('Predicted (days)')
    ax1.set_title('Predictions vs Actual')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Residuals vs Fitted
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(y_pred, residuals, alpha=0.5, s=10)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('Fitted Values')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residual Plot')
    ax2.grid(True, alpha=0.3)

    # 3. Error Distribution
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(residuals, bins=50, edgecolor='black', alpha=0.7, density=True)
    ax3.axvline(x=0, color='r', linestyle='--', label='Zero error')
    ax3.axvline(x=np.mean(residuals), color='g', linestyle='-', label=f'Mean: {np.mean(residuals):.2f}')

    # Overlay normal distribution
    mu, sigma = residuals.mean(), residuals.std()
    x = np.linspace(residuals.min(), residuals.max(), 100)
    ax3.plot(x, stats.norm.pdf(x, mu, sigma), 'k--', linewidth=2, label='Normal fit')

    ax3.set_xlabel('Residuals (days)')
    ax3.set_ylabel('Density')
    ax3.set_title('Residual Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Q-Q Plot
    ax4 = fig.add_subplot(gs[1, 1])
    stats.probplot(residuals, dist="norm", plot=ax4)
    ax4.set_title('Q-Q Plot')
    ax4.grid(True, alpha=0.3)

    # Overall title
    fig.suptitle(title, fontsize=16, fontweight='bold')

    if save_path:
        # Save as PNG
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {save_path}")
        # Save as EPS
        eps_path = Path(save_path).with_suffix('.eps')
        plt.savefig(eps_path, format='eps', bbox_inches='tight')
        print(f"Plot saved: {eps_path}")

    plt.close(fig)
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

    # Plot if we have predictions and matplotlib is available
    if HAS_MATPLOTLIB and "y_true" in metrics and "y_pred" in metrics:
        print("\nGenerating plots...")

        # Individual plots
        print("1. Predictions vs Actual + Error Distribution")
        plot_predictions(metrics["y_true"], metrics["y_pred"],
                        title="Untrained Model Predictions")

        print("2. Residuals Plot")
        plot_residuals(metrics["y_true"], metrics["y_pred"])

        print("3. Q-Q Plot")
        plot_qq(metrics["y_true"], metrics["y_pred"])

        print("4. Comprehensive Evaluation Plot")
        plot_comprehensive_evaluation(metrics["y_true"], metrics["y_pred"],
                                     title="Comprehensive Model Evaluation")
    elif not HAS_MATPLOTLIB:
        print("\nMatplotlib not installed. Skipping plots.")
        print("Install with: pip install matplotlib scipy")
