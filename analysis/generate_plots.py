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
from pathlib import Path

def plot_architecture_comparison(detailed_analyses, output_dir):
    """Plot comparison of architectures."""
    
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
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle('Porównanie Architektur - Metryki na Zbiorze Testowym', fontsize=14, fontweight='bold')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # MAE
    axes[0, 0].bar(architectures, mae_vals, color=colors)
    axes[0, 0].set_ylabel('MAE')
    axes[0, 0].set_title('Mean Absolute Error')
    axes[0, 0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(mae_vals):
        axes[0, 0].text(i, v, f'{v:.4f}', ha='center', va='bottom')
    
    # RMSE
    axes[0, 1].bar(architectures, rmse_vals, color=colors)
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].set_title('Root Mean Square Error')
    axes[0, 1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(rmse_vals):
        axes[0, 1].text(i, v, f'{v:.4f}', ha='center', va='bottom')
    
    # R²
    axes[1, 0].bar(architectures, r2_vals, color=colors)
    axes[1, 0].set_ylabel('R²')
    axes[1, 0].set_title('Współczynnik Determinacji')
    axes[1, 0].grid(axis='y', alpha=0.3)
    axes[1, 0].set_ylim([0, 1])
    for i, v in enumerate(r2_vals):
        axes[1, 0].text(i, v, f'{v:.4f}', ha='center', va='bottom')
    
    # MAPE
    axes[1, 1].bar(architectures, mape_vals, color=colors)
    axes[1, 1].set_ylabel('MAPE (%)')
    axes[1, 1].set_title('Mean Absolute Percentage Error')
    axes[1, 1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(mape_vals):
        axes[1, 1].text(i, v, f'{v:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    output_file = output_dir / 'architecture_comparison.pdf'
    plt.savefig(output_file, bbox_inches='tight', format='pdf')
    print(f"Generated: {output_file}")
    plt.close()


def plot_cv_stability(detailed_analyses, output_dir):
    """Plot CV stability comparison."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
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
    ax.set_ylabel('MAE (Cross-Validation)')
    ax.set_title('Stabilność Modeli w Walidacji Krzyżowej', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (mean, std) in enumerate(zip(mae_means, mae_stds)):
        ax.text(i, mean + std + 0.01, f'{mean:.4f}\n±{std:.4f}', 
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    output_file = output_dir / 'cv_stability.pdf'
    plt.savefig(output_file, bbox_inches='tight', format='pdf')
    print(f"Generated: {output_file}")
    plt.close()


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
    
    print("Generating architecture comparison plot...")
    plot_architecture_comparison(detailed_analyses, output_dir)
    
    print("Generating CV stability plot...")
    plot_cv_stability(detailed_analyses, output_dir)
    
    print(f"\nAll plots generated in {output_dir}")


if __name__ == '__main__':
    main()
