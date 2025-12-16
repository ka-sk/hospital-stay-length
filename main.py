"""
Main experiment script for hospital stay length prediction.

This script runs the complete experimental pipeline:
1. Data loading and preprocessing
2. Grid search with cross-validation for multiple models
3. Model training with early stopping
4. Results aggregation and best model selection
5. Evaluation of best models

Models tested:
- SimpleMLP: Multi-layer perceptron with configurable architecture
- SimpleTabTransformer: Transformer-based tabular model

Run with:
    python main.py
"""

import sys
from pathlib import Path
from omegaconf import OmegaConf

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_loader import load_data
from train import grid_search
from evaluate import evaluate_best_model


def main():
    """Run complete experimental pipeline."""
    
    print("=" * 80)
    print("HOSPITAL STAY LENGTH PREDICTION - COMPLETE EXPERIMENT")
    print("=" * 80)
    
    # ============== STEP 1: Data Loading ==============
    print("\n[1/4] Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = load_data(return_split=True, test_size=0.2, random_state=42)
    print(f"✓ Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"✓ Features: {X_train.shape[1]}")
    
    # ============== STEP 2: Model Configurations ==============
    print("\n[2/4] Loading model configurations...")
    
    configs_dir = PROJECT_ROOT / "experiments" / "configs" / "grid-search"
    model_configs = {
        'mlp': configs_dir / "mlp.yaml",
        'tabtransformer': configs_dir / "tabtransformer.yaml",
        'tabnet': configs_dir / "tabnet.yaml",
    }
    
    # Filter existing configs
    available_configs = {
        name: path for name, path in model_configs.items() 
        if path.exists()
    }
    
    if not available_configs:
        print("✗ No model configurations found!")
        print(f"Expected configs in: {configs_dir}")
        return
    
    print(f"✓ Found {len(available_configs)} model configuration(s):")
    for name in available_configs:
        print(f"  - {name}")
    
    # ============== STEP 3: Grid Search Training ==============
    print("\n[3/4] Running grid search with cross-validation...")
    print("-" * 80)
    
    all_experiment_names = []
    
    for model_name, config_path in available_configs.items():
        print(f"\n{'=' * 80}")
        print(f"MODEL: {model_name.upper()}")
        print(f"{'=' * 80}")
        
        # Load configuration
        config = OmegaConf.load(config_path)
        
        # Run grid search
        experiment_name = grid_search(
            config=config,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            k_folds=5,
            early_stopping_patience=10
        )
        
        all_experiment_names.append(experiment_name)
        print(f"\n✓ Completed grid search for {model_name}")
        print(f"  Experiment: {experiment_name}")
    
    # ============== STEP 4: Evaluation of Best Models ==============
    print("\n" + "=" * 80)
    print("[4/4] Evaluating best models from each experiment...")
    print("=" * 80)
    
    for experiment_name in all_experiment_names:
        print(f"\n{'─' * 80}")
        print(f"Evaluating experiment: {experiment_name}")
        print(f"{'─' * 80}")
        
        try:
            evaluate_best_model(
                experiment_name=experiment_name,
                X_test=X_test,
                y_test=y_test,
                save_results=True,
                show_plots=True
            )
        except Exception as e:
            print(f"✗ Error evaluating {experiment_name}: {e}")
            continue
    
    # ============== SUMMARY ==============
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE!")
    print("=" * 80)
    print(f"✓ Trained and evaluated {len(all_experiment_names)} model(s)")
    print(f"\nResults saved in:")
    print(f"  - Models: experiments/models/")
    print(f"  - Results: experiments/results/")
    print(f"  - Best models: experiments/best/")
    print(f"\nTo compare results, check:")
    print(f"  - experiments/results/<experiment>/all_results.csv")
    print(f"  - experiments/results/<experiment>/best_models.csv")
    print("=" * 80)


if __name__ == "__main__":
    main()
