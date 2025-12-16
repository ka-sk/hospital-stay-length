"""
Training module with grid search and cross validation.
Saves models, results to CSV, and tracks best models.
Includes early stopping and live preview of training progress.
"""
import torch
import data_loader as data
import models
import copy
import shutil
import pandas as pd
from datetime import datetime
from pytorch_tabnet.tab_model import TabNetRegressor
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from itertools import product
from collections.abc import Iterable
from sklearn.model_selection import KFold
from torch.utils.data import Subset
from collections.abc import Callable
from pathlib import Path

# Import evaluation functions from evaluate module
from evaluate import compute_regression_metrics, evaluate_model


# ============== PATHS ==============
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
RESULTS_DIR = EXPERIMENTS_DIR / "results"
MODELS_DIR = EXPERIMENTS_DIR / "models"
BEST_DIR = EXPERIMENTS_DIR / "best"


# ============== COLORS FOR LIVE PREVIEW ==============
class Colors:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"

    @staticmethod
    def success(text: str) -> str:
        return f"{Colors.GREEN}{text}{Colors.RESET}"

    @staticmethod
    def warning(text: str) -> str:
        return f"{Colors.YELLOW}{text}{Colors.RESET}"

    @staticmethod
    def error(text: str) -> str:
        return f"{Colors.RED}{text}{Colors.RESET}"

    @staticmethod
    def info(text: str) -> str:
        return f"{Colors.CYAN}{text}{Colors.RESET}"

    @staticmethod
    def bold(text: str) -> str:
        return f"{Colors.BOLD}{text}{Colors.RESET}"


# ============== EARLY STOPPING ==============
class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0, restore_best: bool = True):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            restore_best: Whether to restore best model weights
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.counter = 0
        self.best_loss = None
        self.best_epoch = 0
        self.best_model_state = None
        self.should_stop = False

    def __call__(self, val_loss: float, model: torch.nn.Module, epoch: int) -> bool:
        """
        Check if training should stop.
        Returns True if should stop.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.best_model_state = copy.deepcopy(model.state_dict())
            return False

        if val_loss < self.best_loss - self.min_delta:
            # Improvement
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0
            return False
        else:
            # No improvement
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
            return False

    def restore(self, model: torch.nn.Module) -> torch.nn.Module:
        """Restore best model weights."""
        if self.restore_best and self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
        return model


# ============== LIVE PREVIEW ==============
class LivePreview:
    """Live preview of training progress with trend indicators."""

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.fold_history = []
        self.current_best_loss = float('inf')
        self.trend_window = 3  # Compare with last N folds

    def update(self, fold_idx: int, grid_search_id: int, model_name: str,
               train_loss: float, val_loss: float, metrics: dict,
               stopped_early: bool = False, best_epoch: int = None, total_epochs: int = None):
        """Update and display live preview after each fold."""

        # Track history
        self.fold_history.append({
            "fold": fold_idx,
            "grid_search_id": grid_search_id,
            "val_loss": val_loss
        })

        # Determine trend
        trend = self._get_trend(val_loss)

        # Update best
        is_new_best = val_loss < self.current_best_loss
        if is_new_best:
            self.current_best_loss = val_loss

        # Display
        self._display(
            fold_idx=fold_idx,
            grid_search_id=grid_search_id,
            model_name=model_name,
            train_loss=train_loss,
            val_loss=val_loss,
            metrics=metrics,
            trend=trend,
            is_new_best=is_new_best,
            stopped_early=stopped_early,
            best_epoch=best_epoch,
            total_epochs=total_epochs
        )

    def _get_trend(self, current_loss: float) -> str:
        """Get trend indicator based on recent history."""
        if len(self.fold_history) < 2:
            return "→"  # Neutral

        recent = [h["val_loss"] for h in self.fold_history[-self.trend_window:]]
        avg_recent = np.mean(recent[:-1]) if len(recent) > 1 else recent[0]

        if current_loss < avg_recent * 0.95:
            return "↓↓"  # Strong improvement
        elif current_loss < avg_recent:
            return "↓"   # Improvement
        elif current_loss > avg_recent * 1.05:
            return "↑↑"  # Strong degradation
        elif current_loss > avg_recent:
            return "↑"   # Degradation
        else:
            return "→"   # Stable

    def _display(self, fold_idx: int, grid_search_id: int, model_name: str,
                 train_loss: float, val_loss: float, metrics: dict, trend: str,
                 is_new_best: bool, stopped_early: bool, best_epoch: int, total_epochs: int):
        """Display formatted live preview."""

        # Trend coloring
        if "↓" in trend:
            trend_colored = Colors.success(trend)
        elif "↑" in trend:
            trend_colored = Colors.error(trend)
        else:
            trend_colored = Colors.warning(trend)

        # Status
        status_parts = []
        if is_new_best:
            status_parts.append(Colors.success(" NEW BEST"))
        if stopped_early:
            status_parts.append(Colors.warning(f"Early stop @ epoch {best_epoch}/{total_epochs}"))
        status = " | ".join(status_parts) if status_parts else ""

        # Main output
        print(f"\n    {'─' * 50}")
        print(f"    {Colors.bold(f'Fold {fold_idx}')} | {model_name} | GS#{grid_search_id}")
        print(f"    {'─' * 50}")
        print(f"    Train Loss: {train_loss:.4f}  |  Val Loss: {val_loss:.4f}  {trend_colored}")
        print(f"    MAE: {metrics['mae']:.4f}  |  RMSE: {metrics['rmse']:.4f}  |  R²: {metrics['r2']:.4f}")
        if status:
            print(f"    {status}")
        print(f"    {'─' * 50}")

    def summary(self):
        """Print final summary."""
        print(f"\n{'═' * 60}")
        print(Colors.bold(f"  EXPERIMENT SUMMARY: {self.experiment_name}"))
        print(f"{'═' * 60}")
        print(f"  Total folds completed: {len(self.fold_history)}")
        print(f"  Best validation loss: {Colors.success(f'{self.current_best_loss:.4f}')}")
        print(f"{'═' * 60}\n")


def ensure_dirs():
    """Create necessary directories."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    BEST_DIR.mkdir(parents=True, exist_ok=True)


def get_model_hyperparams(model: torch.nn.Module) -> dict:
    """Extract hyperparameters from model for logging."""
    model_name = model.__class__.__name__
    params = {"model_name": model_name}

    if hasattr(model, 'linear1'):  # SimpleMLP
        params["hidden_features"] = model.linear1.out_features
        params["dropout"] = model.dropout.p
        params["activation"] = model.act_layer.__class__.__name__
    elif hasattr(model, 'transformer'):  # SimpleTabTransformer
        params["d_model"] = model.embedding.out_features
        params["n_heads"] = model.transformer.layers[0].self_attn.num_heads
        params["num_layers"] = len(model.transformer.layers)

    return params


def train_tabnet(X_train: np.ndarray,
                 y_train: np.ndarray,
                 X_val: np.ndarray,
                 y_val: np.ndarray,
                 model: TabNetRegressor,
                 num_epochs: int = 100,
                 early_stopping_patience: int = 10) -> tuple:
    """
    Train TabNet model using its native .fit() method.
    Returns: model, train_loss_array, val_loss_array, actual_epochs, stopped_early
    """
    # TabNet uses its own training method
    model.fit(
        X_train=X_train,
        y_train=y_train.reshape(-1, 1),
        eval_set=[(X_val, y_val.reshape(-1, 1))],
        eval_metric=['mae'],
        max_epochs=num_epochs,
        patience=early_stopping_patience,
        batch_size=64,
        virtual_batch_size=32,
        num_workers=0,
        drop_last=False
    )

    # Extract training history
    history = model.history
    train_loss = np.array([h['loss'] for h in history])
    val_loss = np.array([h['val_0_mae'] for h in history])  # Using MAE as validation metric

    actual_epochs = len(train_loss)
    stopped_early = actual_epochs < num_epochs

    return model, train_loss, val_loss, actual_epochs, stopped_early


def train(train_dataloader: DataLoader,
          test_dataloader: DataLoader,
          model: torch.nn.Module,
          loss_funct: Callable,
          optim: torch.optim.Optimizer,
          num_epochs: int = 100,
          early_stopping: EarlyStopping = None):
    """
    Train model for specified epochs with optional early stopping.
    Returns: model, train_loss_array, test_loss_array, actual_epochs, stopped_early
    """
    train_loss_list = []
    test_loss_list = []
    stopped_early = False
    actual_epochs = num_epochs

    for epoch in range(num_epochs):
        model, train_loss = train_step(train_dataloader, model, loss_funct, optim)
        test_loss = test_step(test_dataloader, model, loss_funct)

        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)

        # Early stopping check
        if early_stopping is not None:
            if early_stopping(test_loss, model, epoch):
                stopped_early = True
                actual_epochs = epoch + 1
                model = early_stopping.restore(model)
                break

    return model, np.array(train_loss_list), np.array(test_loss_list), actual_epochs, stopped_early


def train_step(dataloader: DataLoader, model: torch.nn.Module, loss_funct: Callable, optim: torch.optim.Optimizer):
    """Single training epoch."""
    model.train()
    loss_list = np.zeros(len(dataloader))

    for idx, (X_batch, y_batch) in enumerate(dataloader):
        optim.zero_grad()
        y_pred = model(X_batch)
        loss_train = loss_funct(y_pred, y_batch)
        loss_train.backward()
        loss_list[idx] = loss_train.item()
        optim.step()

    return model, loss_list.mean()


def test_step(dataloader: DataLoader, model: torch.nn.Module, loss_funct: Callable):
    """Evaluation step."""
    model.eval()
    loss_list = np.zeros(len(dataloader))

    with torch.inference_mode():
        for idx, (X, y) in enumerate(dataloader):
            y_pred = model(X)
            loss_list[idx] = loss_funct(y_pred, y).cpu().item()

    return np.mean(loss_list)


def save_model(model: torch.nn.Module, save_path: Path):
    """Save model state dict."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved: {save_path}")


def cross_val_tabnet(X_data: np.ndarray,
                     y_data: np.ndarray,
                     model: TabNetRegressor,
                     k_fold: int = 5,
                     num_epochs: int = 100,
                     model_save_dir: Path = None,
                     grid_search_id: int = 0,
                     early_stopping_patience: int = 10,
                     live_preview: LivePreview = None) -> list[dict]:
    """
    K-Fold cross validation for TabNet models.
    Returns list of dicts with results for each fold.
    """
    cv = KFold(n_splits=k_fold, shuffle=True, random_state=42)
    fold_results = []

    model_name = model.__class__.__name__
    hyperparams = {
        "model_name": model_name,
        "n_d": model.n_d,
        "n_a": model.n_a,
        "n_steps": model.n_steps
    }

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_data)):
        # TabNet needs fresh instance for each fold
        fold_model = TabNetRegressor(
            n_d=model.n_d,
            n_a=model.n_a,
            n_steps=model.n_steps,
            seed=42
        )

        # Split data
        X_train, X_val = X_data[train_idx], X_data[val_idx]
        y_train, y_val = y_data[train_idx], y_data[val_idx]

        # Train TabNet
        fold_model, train_loss_arr, val_loss_arr, actual_epochs, stopped_early = train_tabnet(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            model=fold_model,
            num_epochs=num_epochs,
            early_stopping_patience=early_stopping_patience
        )

        # Final losses
        final_train_loss = train_loss_arr[-1]
        final_val_loss = val_loss_arr[-1]
        best_val_loss = np.min(val_loss_arr)
        best_epoch = int(np.argmin(val_loss_arr)) + 1

        # Compute metrics
        y_pred = fold_model.predict(X_val)
        metrics = compute_regression_metrics(
            y_true=y_val,
            y_pred=y_pred.flatten()
        )

        # Save model (TabNet uses .zip format)
        if model_save_dir is not None:
            model_filename = f"gs{grid_search_id:04d}_fold{fold_idx + 1}"
            model_path = model_save_dir / model_filename
            model_save_dir.mkdir(parents=True, exist_ok=True)
            fold_model.save_model(str(model_path))
        else:
            model_path = None

        # Live preview
        if live_preview is not None:
            live_preview.update(
                fold_idx=fold_idx + 1,
                grid_search_id=grid_search_id,
                model_name=model_name,
                train_loss=final_train_loss,
                val_loss=final_val_loss,
                metrics=metrics,
                stopped_early=stopped_early,
                best_epoch=best_epoch,
                total_epochs=num_epochs
            )

        # Record results
        result = {
            "grid_search_id": grid_search_id,
            "fold": fold_idx + 1,
            "final_train_loss": final_train_loss,
            "final_val_loss": final_val_loss,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "actual_epochs": actual_epochs,
            "num_epochs": num_epochs,
            "stopped_early": stopped_early,
            "optimizer": "Adam",  # TabNet uses Adam internally
            "learning_rate": 0.02,  # TabNet default
            "loss_function": "MAE",  # TabNet uses MAE for validation
            "model_path": str(model_path) if model_path else None,
            # Metrics
            "mae": metrics["mae"],
            "mse": metrics["mse"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
            "mape": metrics["mape"],
            **hyperparams
        }
        fold_results.append(result)

    return fold_results


def cross_val(dataset: TensorDataset,
              model: torch.nn.Module,
              loss_funct: Callable,
              optim_class,
              lr: float = 0.001,
              k_fold: int = 5,
              num_epochs: int = 100,
              model_save_dir: Path = None,
              grid_search_id: int = 0,
              early_stopping_patience: int = 10,
              live_preview: LivePreview = None) -> list[dict]:
    """
    K-Fold cross validation with early stopping and live preview.
    Returns list of dicts with results for each fold.
    """
    cv = KFold(n_splits=k_fold, shuffle=True, random_state=42)
    fold_results = []

    hyperparams = get_model_hyperparams(model)
    model_name = model.__class__.__name__

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(dataset)):
        # Deep copy model for each fold
        fold_model = copy.deepcopy(model)
        optimizer = optim_class(fold_model.parameters(), lr=lr)

        # Early stopping for this fold
        early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            min_delta=0.0001,
            restore_best=True
        )

        # Subsets
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, test_idx)

        # Dataloaders
        train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
        test_loader = DataLoader(val_subset, batch_size=64, shuffle=False)

        # Train with early stopping
        fold_model, train_loss_arr, test_loss_arr, actual_epochs, stopped_early = train(
            train_dataloader=train_loader,
            test_dataloader=test_loader,
            model=fold_model,
            loss_funct=loss_funct,
            optim=optimizer,
            num_epochs=num_epochs,
            early_stopping=early_stopping
        )

        # Final losses
        final_train_loss = train_loss_arr[-1]
        final_val_loss = test_loss_arr[-1]
        best_val_loss = np.min(test_loss_arr)
        best_epoch = int(np.argmin(test_loss_arr)) + 1

        # Compute full metrics on validation set
        metrics = evaluate_model(fold_model, test_loader)

        # Save model
        if model_save_dir is not None:
            model_filename = f"gs{grid_search_id:04d}_fold{fold_idx + 1}.pt"
            model_path = model_save_dir / model_filename
            save_model(fold_model, model_path)
        else:
            model_path = None

        # Live preview
        if live_preview is not None:
            live_preview.update(
                fold_idx=fold_idx + 1,
                grid_search_id=grid_search_id,
                model_name=model_name,
                train_loss=final_train_loss,
                val_loss=final_val_loss,
                metrics=metrics,
                stopped_early=stopped_early,
                best_epoch=early_stopping.best_epoch + 1 if stopped_early else best_epoch,
                total_epochs=num_epochs
            )

        # Record results
        result = {
            "grid_search_id": grid_search_id,
            "fold": fold_idx + 1,
            "final_train_loss": final_train_loss,
            "final_val_loss": final_val_loss,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "actual_epochs": actual_epochs,
            "num_epochs": num_epochs,
            "stopped_early": stopped_early,
            "optimizer": optim_class.__name__,
            "learning_rate": lr,
            "loss_function": loss_funct.__class__.__name__,
            "model_path": str(model_path) if model_path else None,
            # Metrics
            "mae": metrics["mae"],
            "mse": metrics["mse"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
            "mape": metrics["mape"],
            **hyperparams
        }
        fold_results.append(result)

    return fold_results


def grid_search_legacy(dataset: TensorDataset,
                model_list: Iterable | torch.nn.Module,
                optim_class_list: Iterable = None,
                loss_funct_list: Iterable | Callable = None,
                lr_list: list[float] = None,
                k_fold: int = 5,
                num_epochs: int = 100,
                early_stopping_patience: int = 10,
                experiment_name: str = None):
    """
    Grid search over models, optimizers, loss functions.
    Saves all results to CSV and tracks best models.
    Includes early stopping and live preview.
    """
    ensure_dirs()

    # Default values
    if optim_class_list is None:
        optim_class_list = [torch.optim.Adam]
    if loss_funct_list is None:
        loss_funct_list = [torch.nn.L1Loss()]
    if lr_list is None:
        lr_list = [0.001]

    # Ensure lists
    if not isinstance(model_list, Iterable):
        model_list = [model_list]
    if not isinstance(optim_class_list, Iterable):
        optim_class_list = [optim_class_list]
    if not isinstance(loss_funct_list, Iterable):
        loss_funct_list = [loss_funct_list]
    if not isinstance(lr_list, Iterable):
        lr_list = [lr_list]

    # Convert to lists (in case of generators)
    model_list = list(model_list)
    optim_class_list = list(optim_class_list)
    loss_funct_list = list(loss_funct_list)
    lr_list = list(lr_list)

    # Experiment timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if experiment_name is None:
        experiment_name = f"experiment_{timestamp}"

    # Model save directory for this experiment
    experiment_model_dir = MODELS_DIR / experiment_name

    # Initialize live preview
    live_preview = LivePreview(experiment_name)

    # Collect all results
    all_results = []
    grid_search_id = 0
    total_combinations = len(model_list) * len(optim_class_list) * len(loss_funct_list) * len(lr_list)

    print(f"\n{Colors.bold('═' * 60)}")
    print(Colors.bold(f"  GRID SEARCH: {experiment_name}"))
    print(f"{Colors.bold('═' * 60)}")
    print(f"  Total combinations: {total_combinations}")
    print(f"  Models: {len(model_list)}, Optimizers: {len(optim_class_list)}")
    print(f"  Loss Functions: {len(loss_funct_list)}, Learning Rates: {len(lr_list)}")
    print(f"  K-Folds: {k_fold}, Max Epochs: {num_epochs}")
    print(f"  Early Stopping Patience: {early_stopping_patience}")
    print(f"{Colors.bold('═' * 60)}\n")

    for model in model_list:
        # TabNet requires special handling (different API)
        if isinstance(model, TabNetRegressor):
            grid_search_id += 1
            model_name = model.__class__.__name__

            print(f"\n{Colors.info(f'[{grid_search_id}/{total_combinations}]')} {Colors.bold(model_name)} | TabNet Native Training")

            # Extract numpy arrays from dataset
            X_data = dataset.tensors[0].cpu().numpy()
            y_data = dataset.tensors[1].cpu().numpy()

            # Run TabNet-specific cross validation
            fold_results = cross_val_tabnet(
                X_data=X_data,
                y_data=y_data,
                model=model,
                k_fold=k_fold,
                num_epochs=num_epochs,
                model_save_dir=experiment_model_dir,
                grid_search_id=grid_search_id,
                early_stopping_patience=early_stopping_patience,
                live_preview=live_preview
            )
            all_results.extend(fold_results)
            continue

        for optim_class, loss_funct, lr in product(optim_class_list, loss_funct_list, lr_list):
            grid_search_id += 1
            model_name = model.__class__.__name__

            print(f"\n{Colors.info(f'[{grid_search_id}/{total_combinations}]')} {Colors.bold(model_name)} | {optim_class.__name__} | {loss_funct.__class__.__name__} | lr={lr}")

            # Run cross validation with early stopping and live preview
            fold_results = cross_val(
                dataset=dataset,
                model=model,
                loss_funct=loss_funct,
                optim_class=optim_class,
                lr=lr,
                k_fold=k_fold,
                num_epochs=num_epochs,
                model_save_dir=experiment_model_dir,
                grid_search_id=grid_search_id,
                early_stopping_patience=early_stopping_patience,
                live_preview=live_preview
            )

            all_results.extend(fold_results)

    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)

    # Save all results
    all_results_path = RESULTS_DIR / f"{experiment_name}_all_results.csv"
    results_df.to_csv(all_results_path, index=False)
    print(f"\n{Colors.success(f'All results saved: {all_results_path}')}")

    # Compute best models (average across folds)
    if len(results_df) > 0:
        best_models_df = compute_best_models(results_df, experiment_name)

        # Copy best model to best folder
        copy_best_model(best_models_df, experiment_name)

    # Final summary
    live_preview.summary()

    return results_df


def compute_best_models(results_df: pd.DataFrame, experiment_name: str) -> pd.DataFrame:
    """
    Compute average metrics per grid_search_id and find best models.
    """
    # Add hyperparameter columns that exist
    hyperparam_cols = ["hidden_features", "dropout", "activation", "d_model", "n_heads", "num_layers"]
    metric_cols = ["mae", "mse", "rmse", "r2", "mape"]

    # Build aggregation dict
    agg_dict = {
        "final_train_loss": "mean",
        "final_val_loss": "mean",
        "best_val_loss": "mean",
        "model_name": "first",
        "optimizer": "first",
        "learning_rate": "first",
        "loss_function": "first",
        "num_epochs": "first",
        "actual_epochs": "mean",
        "stopped_early": "sum",  # Count how many folds stopped early
    }

    # Add metric columns
    for col in metric_cols:
        if col in results_df.columns:
            agg_dict[col] = "mean"

    # Add hyperparameter columns
    for col in hyperparam_cols:
        if col in results_df.columns:
            agg_dict[col] = "first"

    agg_df = results_df.groupby("grid_search_id").agg(agg_dict).reset_index()

    # Sort by best_val_loss
    agg_df = agg_df.sort_values("best_val_loss").reset_index(drop=True)
    agg_df["rank"] = agg_df.index + 1

    # Get best fold model path for each grid_search_id
    best_fold_paths = results_df.loc[
        results_df.groupby("grid_search_id")["best_val_loss"].idxmin()
    ][["grid_search_id", "model_path", "fold"]]

    agg_df = agg_df.merge(best_fold_paths, on="grid_search_id", how="left")

    # Save best models CSV
    best_models_path = RESULTS_DIR / f"{experiment_name}_best_models.csv"
    agg_df.to_csv(best_models_path, index=False)
    print(f"{Colors.success(f'Best models saved: {best_models_path}')}")

    # Print top 5
    print(f"\n{Colors.bold('Top 5 Models:')}")
    print("-" * 60)
    for i, row in agg_df.head(5).iterrows():
        r2_str = f"R²: {row['r2']:.4f}" if 'r2' in row else ""
        mae_str = f"MAE: {row['mae']:.4f}" if 'mae' in row else ""
        print(f"  #{row['rank']}: {row['model_name']} | Val Loss: {row['best_val_loss']:.4f} | {mae_str} | {r2_str}")

    return agg_df


def copy_best_model(best_models_df: pd.DataFrame, experiment_name: str):
    """Copy the best model to the 'best' folder."""
    if len(best_models_df) == 0:
        return

    best_row = best_models_df.iloc[0]
    best_model_path = best_row.get("model_path")

    if best_model_path and Path(best_model_path).exists():
        # Create descriptive filename
        model_name = best_row["model_name"]
        val_loss = best_row["best_val_loss"]

        dest_filename = f"{experiment_name}_best_{model_name}_loss{val_loss:.4f}.pt"
        dest_path = BEST_DIR / dest_filename

        shutil.copy2(best_model_path, dest_path)
        print(f"\n{Colors.success(f'Best model copied to: {dest_path}')}")

        # Also save a metadata file
        meta_path = BEST_DIR / f"{experiment_name}_best_metadata.csv"
        best_row.to_frame().T.to_csv(meta_path, index=False)
        print(f"{Colors.info(f'Best model metadata: {meta_path}')}")


# ============== NEW API FOR CONFIG-BASED GRID SEARCH ==============

def grid_search(config,
                X_train,
                y_train,
                X_test=None,
                y_test=None,
                k_folds: int = 5,
                early_stopping_patience: int = 10) -> str:
    """
    Grid search using config file (new API for main.py).

    Args:
        config: OmegaConf configuration with model hyperparameters
        X_train: Training features (tensor or numpy)
        y_train: Training labels (tensor or numpy)
        X_test: Optional test features
        y_test: Optional test labels
        k_folds: Number of folds for cross-validation
        early_stopping_patience: Patience for early stopping

    Returns:
        experiment_name: Name of the experiment
    """
    import utils
    device = utils.get_device()

    # Convert to tensors if needed
    if not isinstance(X_train, torch.Tensor):
        X_train = torch.tensor(X_train, dtype=torch.float32)
    if not isinstance(y_train, torch.Tensor):
        y_train = torch.tensor(y_train, dtype=torch.float32)

    X_train = X_train.to(device)
    y_train = y_train.to(device)

    # Create dataset
    dataset = TensorDataset(X_train, y_train)

    # Generate model instances from config
    model_list = models.load_model_instances(config)

    # Generate experiment name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{config.model.name}_{timestamp}"

    # Run grid search with default optimizers and loss functions
    grid_search_legacy(
        dataset=dataset,
        model_list=model_list,
        optim_class_list=[torch.optim.Adam],
        loss_funct_list=[torch.nn.L1Loss()],
        lr_list=[0.001],
        k_fold=k_folds,
        num_epochs=100,
        early_stopping_patience=early_stopping_patience,
        experiment_name=experiment_name
    )

    return experiment_name


if __name__ == '__main__':
    # Load data
    dataset = data.data_filtration(data.load_data())

    # Load all models from grid search configs
    model_list = models.get_all_models()

    print(f"Total models to train: {len(model_list)}")

    # Run grid search with cross validation
    # Includes: early stopping, live preview, regression metrics
    results = grid_search_legacy(
        dataset=dataset,
        model_list=model_list,
        optim_class_list=[torch.optim.Adam],
        loss_funct_list=[torch.nn.L1Loss(), torch.nn.MSELoss()],
        lr_list=[0.001, 0.0001],
        k_fold=5,
        num_epochs=100,
        early_stopping_patience=10,  # Stop if no improvement for 10 epochs
        experiment_name="hospital_stay_gridsearch"
    )
