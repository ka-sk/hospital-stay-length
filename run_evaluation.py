"""
Script to evaluate already-trained models with plotting enabled.
"""

import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_loader import load_data
from evaluate import evaluate_best_model


def main():
    """Evaluate all trained models with plots."""

    print("=" * 80)
    print("EVALUATING TRAINED MODELS WITH PLOTS")
    print("=" * 80)

    # Load test data
    print("\n[1/2] Loading test data...")
    X_train, X_test, y_train, y_test = load_data(return_split=True, test_size=0.2, random_state=42)
    print(f"✓ Test samples: {len(X_test)}")
    print(f"✓ Features: {X_test.shape[1]}")

    # Define experiments to evaluate
    experiments = [
        "mlp_20251216_212136",
        "mlp_20251216_221732",
        "tabtransformer_20251216_213147",
        "tabtransformer_20251217_101214",
    ]

    print(f"\n[2/2] Evaluating {len(experiments)} trained models...")
    print("=" * 80)

    for experiment_name in experiments:
        print(f"\n{'─' * 80}")
        print(f"Evaluating: {experiment_name}")
        print(f"{'─' * 80}")

        try:
            evaluate_best_model(
                experiment_name=experiment_name,
                X_test=X_test,
                y_test=y_test,
                save_results=True,
                show_plots=True  # Enable plotting
            )
            print(f"✓ Successfully evaluated {experiment_name}")
        except Exception as e:
            print(f"✗ Error evaluating {experiment_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE!")
    print("=" * 80)
    print("✓ All plots should be displayed")
    print(f"✓ Results saved in: experiments/results/<experiment>/best_model_eval.csv")
    print("=" * 80)


if __name__ == "__main__":
    main()
