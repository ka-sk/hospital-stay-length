"""
Generate LaTeX tables from analysis results.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from utils import format_metric, escape_latex, get_experiment_configs


def generate_hyperparameter_space_table(model_name: str) -> str:
    """Generate LaTeX table for hyperparameter search space."""
    
    hyperparams = {
        'mlp': [
            ('hidden\\_channels', '[64, 128, 256]', 'Liczba neuronów w warstwie ukrytej'),
            ('activation\\_layer', "['relu', 'tanh']", 'Funkcja aktywacji'),
            ('dropout', '[0.2, 0.3, 0.4]', 'Współczynnik dropout'),
        ],
        'tabnet': [
            ('n\\_d', '[8, 32, 64]', 'Wymiar feature transformers'),
            ('n\\_a', '[8, 32, 64]', 'Wymiar attention layers'),
            ('n\\_steps', '[3, 5, 7]', 'Liczba kroków decyzyjnych'),
        ],
        'tabtransformer': [
            ('d\\_model', '[64, 128, 256]', 'Wymiar modelu'),
            ('n\\_heads', '[2, 4, 8]', 'Liczba głów attention'),
            ('num\\_layers', '[1, 2, 3]', 'Liczba warstw transformera'),
        ]
    }
    
    params = hyperparams[model_name]
    
    latex = r"""\begin{table}[h]
\centering
\caption{Przestrzeń hiperparametrów dla """ + model_name.upper() + r"""}
\label{tab:hyperparams_""" + model_name + r"""}
\begin{tabular}{lll}
\toprule
Hiperparametr & Wartości & Opis \\
\midrule
"""
    
    for param, values, desc in params:
        latex += f"{param} & {values} & {desc} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    return latex


def generate_cv_results_table(analysis: dict, top_n: int = 10) -> str:
    """Generate LaTeX table for cross-validation results."""
    
    cv_stats = analysis['cv_statistics'].head(top_n)
    model_name = analysis['model_name']
    
    latex = r"""\begin{table}[h]
\centering
\caption{Top """ + str(top_n) + r""" konfiguracji dla """ + analysis.get('display_name', model_name) + r""" (walidacja krzyżowa)}
\label{tab:cv_results_""" + model_name + r"""}
\small
\begin{tabular}{cccccc}
\toprule
Rank & Grid ID & MAE & RMSE & R² & MAPE \\
\midrule
"""
    
    for idx, (_, row) in enumerate(cv_stats.iterrows(), 1):
        gs_id = int(row['grid_search_id'])
        mae = f"{row.get('mae_mean', 0):.4f}"
        rmse = f"{row.get('rmse_mean', 0):.4f}"
        r2 = f"{row.get('r2_mean', 0):.4f}"
        mape = f"{row.get('mape_mean', 0):.2f}"
        
        latex += f"{idx} & {gs_id} & {mae} & {rmse} & {r2} & {mape} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    return latex


def generate_best_config_table(analysis: dict) -> str:
    """Generate LaTeX table for best configuration details."""
    
    model_name = analysis['model_name']
    hyperparams = analysis['hyperparameters']
    
    latex = r"""\begin{table}[h]
\centering
\caption{Najlepsza konfiguracja dla """ + analysis.get('display_name', model_name) + r"""}
\label{tab:best_config_""" + model_name + r"""}
\begin{tabular}{ll}
\toprule
Parametr & Wartość \\
\midrule
"""
    
    # Format hyperparameters
    for key, value in hyperparams.items():
        if key not in ['model_path', 'fold']:
            key_formatted = key.replace('_', '\\_')
            latex += f"{key_formatted} & {value} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    return latex


def generate_metrics_comparison_table(detailed_analyses: dict) -> str:
    """Generate LaTeX table comparing best models from each architecture."""
    
    latex = r"""\begin{table}[h]
\centering
\caption{Porównanie najlepszych modeli z każdej architektury}
\label{tab:architecture_comparison}
\begin{tabular}{lcccccc}
\toprule
Architektura & MAE & RMSE & R² & MAPE & MedAE & Max Error \\
\midrule
"""
    
    # Collect results
    results = []
    for key in ['mlp', 'tabnet', 'tabtransformer']:
        analysis = detailed_analyses[key]
        test_metrics = analysis['test_metrics']
        results.append({
            'name': analysis.get('display_name', key),
            'mae': test_metrics['mae'],
            'rmse': test_metrics['rmse'],
            'r2': test_metrics['r2'],
            'mape': test_metrics['mape'],
            'medae': test_metrics['medae'],
            'max_error': test_metrics['max_error']
        })
    
    # Sort by MAE
    results.sort(key=lambda x: x['mae'])
    
    for i, res in enumerate(results):
        name = res['name']
        mae = f"{res['mae']:.4f}"
        rmse = f"{res['rmse']:.4f}"
        r2 = f"{res['r2']:.4f}"
        mape = f"{res['mape']:.2f}"
        medae = f"{res['medae']:.4f}"
        max_err = f"{res['max_error']:.2f}"
        
        # Bold the best model
        if i == 0:
            latex += f"\\textbf{{{name}}} & \\textbf{{{mae}}} & \\textbf{{{rmse}}} & \\textbf{{{r2}}} & \\textbf{{{mape}}} & \\textbf{{{medae}}} & \\textbf{{{max_err}}} \\\\\n"
        else:
            latex += f"{name} & {mae} & {rmse} & {r2} & {mape} & {medae} & {max_err} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    return latex


def generate_cv_metrics_table(analysis: dict) -> str:
    """Generate LaTeX table for CV metrics with standard deviations."""
    
    model_name = analysis['model_name']
    cv_metrics = analysis['cv_metrics']
    
    latex = r"""\begin{table}[h]
\centering
\caption{Metryki walidacji krzyżowej dla najlepszego modelu """ + analysis.get('display_name', model_name) + r"""}
\label{tab:cv_metrics_""" + model_name + r"""}
\begin{tabular}{lcc}
\toprule
Metryka & Średnia & Odch. std. \\
\midrule
"""
    
    metrics = [
        ('MAE', 'mae'),
        ('RMSE', 'rmse'),
        ('R²', 'r2'),
        ('MAPE', 'mape'),
    ]
    
    for display_name, key in metrics:
        mean_val = cv_metrics.get(f'{key}_mean', 0)
        std_val = cv_metrics.get(f'{key}_std', 0)
        latex += f"{display_name} & {mean_val:.4f} & {std_val:.4f} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    return latex


def main():
    """Generate all LaTeX tables."""
    
    # Create output directory
    output_dir = Path(__file__).parent.parent / 'report' / 'tables'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load detailed analyses
    analyses_path = Path(__file__).parent / 'detailed_analyses.pkl'
    
    if not analyses_path.exists():
        print("Error: detailed_analyses.pkl not found. Run best_model_analysis.py first.")
        return
    
    with open(analyses_path, 'rb') as f:
        detailed_analyses = pickle.load(f)
    
    # Generate hyperparameter space tables
    for model_name in ['mlp', 'tabnet', 'tabtransformer']:
        table = generate_hyperparameter_space_table(model_name)
        output_file = output_dir / f'hyperparams_{model_name}.tex'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(table)
        print(f"Generated: {output_file}")
    
    # Generate CV results tables
    for key, analysis in detailed_analyses.items():
        table = generate_cv_results_table(analysis, top_n=10)
        output_file = output_dir / f'cv_results_{key}.tex'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(table)
        print(f"Generated: {output_file}")
    
    # Generate best configuration tables
    for key, analysis in detailed_analyses.items():
        table = generate_best_config_table(analysis)
        output_file = output_dir / f'best_config_{key}.tex'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(table)
        print(f"Generated: {output_file}")
    
    # Generate CV metrics tables
    for key, analysis in detailed_analyses.items():
        table = generate_cv_metrics_table(analysis)
        output_file = output_dir / f'cv_metrics_{key}.tex'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(table)
        print(f"Generated: {output_file}")
    
    # Generate architecture comparison table
    table = generate_metrics_comparison_table(detailed_analyses)
    output_file = output_dir / 'architecture_comparison.tex'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(table)
    print(f"Generated: {output_file}")
    
    print(f"\nAll tables generated in {output_dir}")


if __name__ == '__main__':
    main()
