"""
Training module with grid search and cross validation.
Saves models, results to CSV, and tracks best models.
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


# ============== PATHS ==============
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
RESULTS_DIR = EXPERIMENTS_DIR / "results"
MODELS_DIR = EXPERIMENTS_DIR / "models"
BEST_DIR = EXPERIMENTS_DIR / "best"


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


def train(train_dataloader: DataLoader,
          test_dataloader: DataLoader,
          model: torch.nn.Module, 
          loss_funct: Callable, 
          optim: torch.optim.Optimizer,
          num_epochs: int = 100):
    """
    Train model for specified epochs.
    Returns: model, train_loss_array, test_loss_array
    """
    test_loss_array = np.zeros(num_epochs)
    train_loss_array = np.zeros(num_epochs)

    for epoch in range(num_epochs):
        model, train_loss_array[epoch] = train_step(train_dataloader, model, loss_funct, optim)
        test_loss_array[epoch] = test_step(test_dataloader, model, loss_funct)

    return model, train_loss_array, test_loss_array


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


def cross_val(dataset: TensorDataset, 
              model: torch.nn.Module, 
              loss_funct: Callable, 
              optim_class,
              lr: float = 0.001,
              k_fold: int = 5,
              num_epochs: int = 100,
              model_save_dir: Path = None,
              grid_search_id: int = 0) -> list[dict]:
    """
    K-Fold cross validation.
    Returns list of dicts with results for each fold.
    """
    cv = KFold(n_splits=k_fold, shuffle=True, random_state=42)
    fold_results = []
    
    hyperparams = get_model_hyperparams(model)
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(dataset)):
        print(f"  Fold {fold_idx + 1}/{k_fold}")
        
        # Deep copy model for each fold
        fold_model = copy.deepcopy(model)
        optimizer = optim_class(fold_model.parameters(), lr=lr)
        
        # Subsets
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, test_idx)

        # Dataloaders
        train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
        test_loader = DataLoader(val_subset, batch_size=64, shuffle=False)

        # Train
        fold_model, train_loss_arr, test_loss_arr = train(
            train_dataloader=train_loader,
            test_dataloader=test_loader,
            model=fold_model,
            loss_funct=loss_funct,
            optim=optimizer,
            num_epochs=num_epochs,
        )
        
        # Final losses
        final_train_loss = train_loss_arr[-1]
        final_val_loss = test_loss_arr[-1]
        best_val_loss = np.min(test_loss_arr)
        best_epoch = int(np.argmin(test_loss_arr)) + 1
        
        # Save model
        if model_save_dir is not None:
            model_filename = f"gs{grid_search_id:04d}_fold{fold_idx + 1}.pt"
            model_path = model_save_dir / model_filename
            save_model(fold_model, model_path)
        else:
            model_path = None
        
        # Record results
        result = {
            "grid_search_id": grid_search_id,
            "fold": fold_idx + 1,
            "final_train_loss": final_train_loss,
            "final_val_loss": final_val_loss,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "num_epochs": num_epochs,
            "optimizer": optim_class.__name__,
            "learning_rate": lr,
            "loss_function": loss_funct.__class__.__name__,
            "model_path": str(model_path) if model_path else None,
            **hyperparams
        }
        fold_results.append(result)
        
        print(f"    Train Loss: {final_train_loss:.4f}, Val Loss: {final_val_loss:.4f}, Best Val: {best_val_loss:.4f}")
    
    return fold_results


def grid_search(dataset: TensorDataset,
                model_list: Iterable | torch.nn.Module, 
                optim_class_list: Iterable = None,
                loss_funct_list: Iterable | Callable = None,
                lr_list: list[float] = None,
                k_fold: int = 5,
                num_epochs: int = 100,
                experiment_name: str = None):
    """
    Grid search over models, optimizers, loss functions.
    Saves all results to CSV and tracks best models.
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
    
    # Collect all results
    all_results = []
    grid_search_id = 0
    total_combinations = len(model_list) * len(optim_class_list) * len(loss_funct_list) * len(lr_list)
    
    print(f"Starting Grid Search: {total_combinations} combinations")
    print(f"Models: {len(model_list)}, Optimizers: {len(optim_class_list)}, Loss Functions: {len(loss_funct_list)}, LRs: {len(lr_list)}")
    print("=" * 60)
    
    for model in model_list:
        # Skip TabNet for now (requires different training)
        if isinstance(model, TabNetRegressor):
            print(f"Skipping TabNet (requires special handling)")
            continue
            
        for optim_class, loss_funct, lr in product(optim_class_list, loss_funct_list, lr_list):
            grid_search_id += 1
            model_name = model.__class__.__name__
            
            print(f"\n[{grid_search_id}/{total_combinations}] {model_name} | {optim_class.__name__} | {loss_funct.__class__.__name__} | lr={lr}")
            
            # Run cross validation
            fold_results = cross_val(
                dataset=dataset,
                model=model,
                loss_funct=loss_funct,
                optim_class=optim_class,
                lr=lr,
                k_fold=k_fold,
                num_epochs=num_epochs,
                model_save_dir=experiment_model_dir,
                grid_search_id=grid_search_id
            )
            
            all_results.extend(fold_results)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save all results
    all_results_path = RESULTS_DIR / f"{experiment_name}_all_results.csv"
    results_df.to_csv(all_results_path, index=False)
    print(f"\nAll results saved: {all_results_path}")
    
    # Compute best models (average across folds)
    if len(results_df) > 0:
        best_models_df = compute_best_models(results_df, experiment_name)
        
        # Copy best model to best folder
        copy_best_model(best_models_df, experiment_name)
    
    print("\n" + "=" * 60)
    print("Grid Search Complete!")
    
    return results_df


def compute_best_models(results_df: pd.DataFrame, experiment_name: str) -> pd.DataFrame:
    """
    Compute average metrics per grid_search_id and find best models.
    """
    # Group by grid_search_id and compute mean metrics
    groupby_cols = ["grid_search_id", "model_name", "optimizer", "learning_rate", "loss_function"]
    
    # Add hyperparameter columns that exist
    hyperparam_cols = ["hidden_features", "dropout", "activation", "d_model", "n_heads", "num_layers"]
    groupby_cols.extend([c for c in hyperparam_cols if c in results_df.columns])
    
    agg_df = results_df.groupby("grid_search_id").agg({
        "final_train_loss": "mean",
        "final_val_loss": "mean",
        "best_val_loss": "mean",
        "model_name": "first",
        "optimizer": "first",
        "learning_rate": "first",
        "loss_function": "first",
        "num_epochs": "first",
        **{c: "first" for c in hyperparam_cols if c in results_df.columns}
    }).reset_index()
    
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
    print(f"Best models saved: {best_models_path}")
    
    # Print top 5
    print("\nTop 5 Models:")
    print("-" * 40)
    for i, row in agg_df.head(5).iterrows():
        print(f"  #{row['rank']}: {row['model_name']} | Val Loss: {row['best_val_loss']:.4f}")
    
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
        print(f"\nBest model copied to: {dest_path}")
        
        # Also save a metadata file
        meta_path = BEST_DIR / f"{experiment_name}_best_metadata.csv"
        best_row.to_frame().T.to_csv(meta_path, index=False)
        print(f"Best model metadata: {meta_path}")


if __name__ == '__main__':
    # Load data
    dataset = data.data_filtration(data.load_data())
    
    # Load all models from grid search configs
    model_list = models.get_all_models()
    
    # Filter out TabNet for now
    model_list = [m for m in model_list if not isinstance(m, TabNetRegressor)]
    
    print(f"Total models to train: {len(model_list)}")
    
    # Run grid search with cross validation
    results = grid_search(
        dataset=dataset,
        model_list=model_list,
        optim_class_list=[torch.optim.Adam],
        loss_funct_list=[torch.nn.L1Loss(), torch.nn.MSELoss()],
        lr_list=[0.001, 0.0001],
        k_fold=5,
        num_epochs=50,
        experiment_name="hospital_stay_gridsearch"
    )

