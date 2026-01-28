"""
Generate comparison plots using only matplotlib.
"""
import pickle
import sys
sys.path.insert(0, '/usr/lib/python3/dist-packages')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Globalne ustawienia czcionek dla prezentacji
FONT_SIZE_TITLE = 24
FONT_SIZE_AXIS_LABEL = 20
FONT_SIZE_TICK = 16
FONT_SIZE_LEGEND = 16
FONT_SIZE_ANNOTATION = 16

def set_presentation_style():
    """Ustaw styl wykresów odpowiedni dla prezentacji."""
    plt.rcParams.update({
        'font.size': FONT_SIZE_TICK,
        'axes.titlesize': FONT_SIZE_TITLE,
        'axes.labelsize': FONT_SIZE_AXIS_LABEL,
        'xtick.labelsize': FONT_SIZE_TICK,
        'ytick.labelsize': FONT_SIZE_TICK,
        'legend.fontsize': FONT_SIZE_LEGEND,
        'figure.titlesize': FONT_SIZE_TITLE,
    })


def plot_architecture_comparison(detailed_analyses, output_dir):
    """Plot comparison of architectures - combined 2x2 plot."""
    set_presentation_style()

    architectures = []
    mae_vals = []
    rmse_vals = []
    r2_vals = []
    mape_vals = []

    for key in ['mlp', 'tabnet', 'tabtransformer']:
        analysis = detailed_analyses[key]
        test_metrics = analysis['test_metrics']
        architectures.append(analysis.get('display_name', key))
        mae_vals.append(test_metrics['mae'])
        rmse_vals.append(test_metrics['rmse'])
        r2_vals.append(test_metrics['r2'])
        mape_vals.append(test_metrics['mape'])

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Porównanie Architektur - Metryki na Zbiorze Testowym', fontsize=FONT_SIZE_TITLE, fontweight='bold')

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # MAE
    axes[0, 0].bar(architectures, mae_vals, color=colors)
    axes[0, 0].set_ylabel('MAE [dni]')
    axes[0, 0].set_title('Średni Błąd Bezwzględny')
    axes[0, 0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(mae_vals):
        axes[0, 0].text(i, v, f'{v:.4f}', ha='center', va='bottom', fontsize=FONT_SIZE_ANNOTATION)

    # RMSE
    axes[0, 1].bar(architectures, rmse_vals, color=colors)
    axes[0, 1].set_ylabel('RMSE [dni]')
    axes[0, 1].set_title('Pierwiastek Błędu Średniokwadratowego')
    axes[0, 1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(rmse_vals):
        axes[0, 1].text(i, v, f'{v:.4f}', ha='center', va='bottom', fontsize=FONT_SIZE_ANNOTATION)

    # R²
    axes[1, 0].bar(architectures, r2_vals, color=colors)
    axes[1, 0].set_ylabel('R²')
    axes[1, 0].set_title('Współczynnik Determinacji')
    axes[1, 0].grid(axis='y', alpha=0.3)
    axes[1, 0].set_ylim([0, 1])
    for i, v in enumerate(r2_vals):
        axes[1, 0].text(i, v, f'{v:.4f}', ha='center', va='bottom', fontsize=FONT_SIZE_ANNOTATION)

    # MAPE
    axes[1, 1].bar(architectures, mape_vals, color=colors)
    axes[1, 1].set_ylabel('MAPE [%]')
    axes[1, 1].set_title('Średni Błąd Procentowy')
    axes[1, 1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(mape_vals):
        axes[1, 1].text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=FONT_SIZE_ANNOTATION)

    plt.tight_layout()

    output_file = output_dir / 'architecture_comparison.pdf'
    plt.savefig(output_file, bbox_inches='tight', format='pdf')
    print(f"Generated: {output_file}")
    plt.close()


def plot_architecture_comparison_separate(detailed_analyses, output_dir):
    """Plot comparison of architectures - separate plots for each metric."""
    set_presentation_style()

    architectures = []
    mae_vals = []
    rmse_vals = []
    r2_vals = []
    mape_vals = []

    for key in ['mlp', 'tabnet', 'tabtransformer']:
        analysis = detailed_analyses[key]
        test_metrics = analysis['test_metrics']
        architectures.append(analysis.get('display_name', key))
        mae_vals.append(test_metrics['mae'])
        rmse_vals.append(test_metrics['rmse'])
        r2_vals.append(test_metrics['r2'])
        mape_vals.append(test_metrics['mape'])

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # Osobne wykresy dla każdej metryki
    metrics_data = [
        ('mae', 'Średni Błąd Bezwzględny (MAE)', mae_vals, 'MAE [dni]', None),
        ('rmse', 'Pierwiastek Błędu Średniokwadratowego (RMSE)', rmse_vals, 'RMSE [dni]', None),
        ('r2', 'Współczynnik Determinacji (R²)', r2_vals, 'R²', [0, 1]),
        ('mape', 'Średni Błąd Procentowy (MAPE)', mape_vals, 'MAPE [%]', None),
    ]

    for metric_name, title, values, ylabel, ylim in metrics_data:
        fig, ax = plt.subplots(figsize=(10, 7))

        bars = ax.bar(architectures, values, color=colors)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        if ylim:
            ax.set_ylim(ylim)

        # Dodaj wartości na słupkach
        for i, v in enumerate(values):
            fmt = f'{v:.4f}' if metric_name != 'mape' else f'{v:.2f}'
            ax.text(i, v + 0.01 * max(values), fmt, ha='center', va='bottom',
                    fontsize=FONT_SIZE_ANNOTATION, fontweight='bold')

        plt.tight_layout()

        output_file = output_dir / f'comparison_{metric_name}.pdf'
        plt.savefig(output_file, bbox_inches='tight', format='pdf')
        print(f"Generated: {output_file}")
        plt.close()


def plot_cv_stability(detailed_analyses, output_dir):
    """Plot CV stability comparison."""
    set_presentation_style()

    fig, ax = plt.subplots(figsize=(10, 7))

    models = []
    mae_means = []
    mae_stds = []

    for key in ['mlp', 'tabnet', 'tabtransformer']:
        analysis = detailed_analyses[key]
        cv_metrics = analysis['cv_metrics']
        models.append(analysis.get('display_name', key))
        mae_means.append(cv_metrics.get('mae_mean', 0))
        mae_stds.append(cv_metrics.get('mae_std', 0))

    x_pos = np.arange(len(models))
    bars = ax.bar(x_pos, mae_means, yerr=mae_stds, capsize=10, alpha=0.7,
                   color=['#1f77b4', '#ff7f0e', '#2ca02c'])

    ax.set_xticks(x_pos)
    ax.set_xticklabels(models)
    ax.set_ylabel('MAE (walidacja krzyżowa) [dni]')
    ax.set_title('Stabilność Modeli w Walidacji Krzyżowej', fontweight='bold')
    ax.set_ylim([0, 1.6])
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, (mean, std) in enumerate(zip(mae_means, mae_stds)):
        ax.text(i, mean + std + 0.01, f'{mean:.4f}\n±{std:.4f}',
                ha='center', va='bottom', fontsize=FONT_SIZE_ANNOTATION, fontweight='bold')

    plt.tight_layout()

    output_file = output_dir / 'cv_stability.pdf'
    plt.savefig(output_file, bbox_inches='tight', format='pdf')
    print(f"Generated: {output_file}")
    plt.close()


def load_predictions_for_model(model_key: str, detailed_analyses: dict) -> pd.DataFrame:
    """Load predictions CSV for a given model."""
    analysis = detailed_analyses[model_key]
    model_name = analysis['model_name']
    timestamp = analysis['timestamp']

    # Sprawdź obie możliwe lokalizacje
    base_dir = Path(__file__).parent.parent
    possible_paths = [
        base_dir / 'experiments' / 'results' / f"{model_name}_{timestamp}" / "best_model_eval.predictions.csv",
        base_dir / 'results' / f"{model_name}_{timestamp}" / "best_model_eval.predictions.csv",
    ]

    for csv_path in possible_paths:
        if csv_path.exists():
            return pd.read_csv(csv_path)

    print(f"Warning: Predictions file not found in any location for {model_name}")
    return None


def plot_predictions_single(y_true, y_pred, model_name: str, output_dir: Path):
    """Plot predictions vs actual - single large plot for presentation."""
    set_presentation_style()

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(y_true, y_pred, alpha=0.5, s=30, c='#1f77b4')

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Idealna predykcja')

    ax.set_xlabel('Wartości rzeczywiste [dni]')
    ax.set_ylabel('Wartości przewidywane [dni]')
    ax.set_title(f'{model_name} - Predykcje vs Wartości Rzeczywiste', fontweight='bold')
    ax.legend(fontsize=FONT_SIZE_LEGEND)
    ax.grid(True, alpha=0.3)

    # Oblicz metryki ręcznie
    mae = np.mean(np.abs(y_true - y_pred))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    ax.text(0.05, 0.95, f'MAE = {mae:.3f}\nR² = {r2:.3f}',
            transform=ax.transAxes, fontsize=FONT_SIZE_ANNOTATION,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    # Generate filename from model name
    filename = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
    output_file = output_dir / f'{filename}_predictions.pdf'
    plt.savefig(output_file, bbox_inches='tight', format='pdf')
    print(f"Generated: {output_file}")
    plt.close()


def plot_residuals_single(y_true, y_pred, model_name: str, output_dir: Path):
    """Plot residuals - single large plot for presentation."""
    set_presentation_style()

    residuals = y_true - y_pred

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(y_pred, residuals, alpha=0.5, s=30, c='#1f77b4')
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero reszt')

    # Add smoothed trend line
    try:
        from scipy.signal import savgol_filter
        sorted_idx = np.argsort(y_pred)
        y_pred_sorted = y_pred[sorted_idx]
        residuals_sorted = residuals[sorted_idx]

        if len(y_pred) > 50:
            window = min(51, len(y_pred) // 10 * 2 + 1)
            residuals_smooth = savgol_filter(residuals_sorted, window, 3)
            ax.plot(y_pred_sorted, residuals_smooth, 'g-', linewidth=2, alpha=0.7, label='Trend')
    except:
        pass

    ax.set_xlabel('Wartości przewidywane [dni]')
    ax.set_ylabel('Reszty (Rzeczywiste - Przewidywane) [dni]')
    ax.set_title(f'{model_name} - Wykres Reszt', fontweight='bold')
    ax.legend(fontsize=FONT_SIZE_LEGEND)
    ax.grid(True, alpha=0.3)

    # Add statistics annotation
    mean_res = np.mean(residuals)
    std_res = np.std(residuals)
    ax.text(0.05, 0.95, f'Średnia = {mean_res:.3f}\nStd = {std_res:.3f}',
            transform=ax.transAxes, fontsize=FONT_SIZE_ANNOTATION,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    # Generate filename from model name
    filename = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
    output_file = output_dir / f'{filename}_residuals.pdf'
    plt.savefig(output_file, bbox_inches='tight', format='pdf')
    print(f"Generated: {output_file}")
    plt.close()


def plot_model_predictions_and_residuals(detailed_analyses, output_dir):
    """Generate prediction and residual plots for all models."""

    model_display_names = {
        'mlp': 'SimpleMLP',
        'tabnet': 'TabNet',
        'tabtransformer': 'TabTransformer'
    }

    for model_key, display_name in model_display_names.items():
        print(f"Generating plots for {display_name}...")

        predictions_df = load_predictions_for_model(model_key, detailed_analyses)

        if predictions_df is not None:
            y_true = predictions_df['y_true'].values
            y_pred = predictions_df['y_pred'].values

            plot_predictions_single(y_true, y_pred, display_name, output_dir)
            plot_residuals_single(y_true, y_pred, display_name, output_dir)
        else:
            print(f"  Skipping {display_name} - no predictions data")


def main():
    """Generate all plots."""

    output_dir = Path(__file__).parent.parent / 'report' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load detailed analyses
    analyses_path = Path(__file__).parent / 'detailed_analyses.pkl'

    if not analyses_path.exists():
        print("Error: detailed_analyses.pkl not found.")
        return

    with open(analyses_path, 'rb') as f:
        detailed_analyses = pickle.load(f)

    print("Generating architecture comparison plot (combined)...")
    plot_architecture_comparison(detailed_analyses, output_dir)

    print("\nGenerating architecture comparison plots (separate)...")
    plot_architecture_comparison_separate(detailed_analyses, output_dir)

    print("\nGenerating CV stability plot...")
    plot_cv_stability(detailed_analyses, output_dir)

    print("\nGenerating prediction and residual plots...")
    plot_model_predictions_and_residuals(detailed_analyses, output_dir)

    print(f"\nAll plots generated in {output_dir}")


if __name__ == '__main__':
    main()
