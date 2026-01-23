# Hospital Stay Length Prediction - LaTeX Report

This directory contains the comprehensive LaTeX report for the Hospital Stay Length Prediction project.

## Report Structure

The report is organized into the following sections:

1. **Introduction** (`sections/introduction.tex`)
   - Problem formulation
   - Project goals
   - Task type (regression)
   - Overview of methods used

2. **Data** (`sections/data.tex`)
   - Dataset description (Healthcare Risk Factors Dataset)
   - Preprocessing steps
   - Train/test split
   - Data statistics

3. **Methodology** (`sections/methodology.tex`)
   - Grid Search
   - 5-fold Cross-Validation
   - Early Stopping
   - Evaluation metrics (MAE, RMSE, R², MAPE, MedAE)
   - Optimization (Adam optimizer, L1Loss)

4. **Architectures** (`sections/architectures.tex`)
   - SimpleMLP
   - TabNet
   - TabTransformer
   - Comparison of architectures

5. **Training Process** (`sections/training.tex`)
   - Grid search with cross-validation procedure
   - Early stopping implementation
   - Checkpoint saving
   - Resume functionality
   - Results storage

6. **Experiment Results** (`sections/results.tex`)
   - Results for SimpleMLP (8 configurations)
   - Results for TabNet (27 configurations)
   - Results for TabTransformer (27 configurations)
   - Architecture comparison

7. **Detailed Analysis** (`sections/analysis.tex`)
   - In-depth analysis of best MLP model
   - In-depth analysis of best TabNet model
   - In-depth analysis of best TabTransformer model
   - Comparison of top 3 models

8. **Final Analysis** (`sections/final_analysis.tex`)
   - Why TabNet won
   - Strengths and limitations
   - Practical significance
   - Deployment recommendations

9. **Conclusions** (`sections/conclusions.tex`)
   - Summary of results
   - Applicability of solution
   - Study limitations
   - Future research directions

## Compiling the Report

### Prerequisites

You need a LaTeX distribution installed with the following packages:
- babel (polish)
- inputenc
- fontenc
- graphicx
- booktabs
- amsmath
- amssymb
- hyperref
- caption
- subcaption
- float
- geometry
- fancyhdr
- setspace

On Ubuntu/Debian:
```bash
sudo apt-get install texlive-full
```

On Fedora:
```bash
sudo dnf install texlive-scheme-full
```

### Compilation

To compile the PDF report:

```bash
cd report
pdflatex main.tex
pdflatex main.tex  # Run twice for references
```

Or use the provided compile script:

```bash
./compile.sh
```

The generated PDF will be `main.pdf`.

## Tables

All tables are stored in the `tables/` directory and include:
- Hyperparameter space definitions
- Best configuration details
- Cross-validation metrics
- Top configurations for each architecture
- Architecture comparison

## Figures

The report references figures from `../experiments/results/` directory:
- Prediction vs. actual plots
- Residual plots
- Q-Q plots
- Comprehensive evaluation plots

## Analysis Scripts

Python scripts for generating analysis data and tables are located in `../analysis/`:
- `analyze_results.py` - Analyze all experiment results
- `best_model_analysis.py` - Detailed analysis of best models
- `generate_tables.py` - Generate LaTeX tables
- `generate_plots.py` - Generate comparison plots
- `utils.py` - Utility functions

To run the analysis scripts:

```bash
cd ../analysis
python analyze_results.py
python best_model_analysis.py
python generate_tables.py
python generate_plots.py
```

## Key Results

The best model is **TabNet (Grid Search ID 22)** with:
- MAE: 1.2640 days
- RMSE: 1.5971 days
- R²: 0.6739
- MAPE: 34.60%

## Language

The report is written in Polish as per the project requirements.

## Author Information

Update the author information in `main.tex` before final compilation:
```latex
\author{
    [Imię i Nazwisko]\\
    [Numer indeksu]\\
    [Wydział]\\
    [Uniwersytet]
}
```

## Notes

- The report includes all figures generated during experiments
- All tables are automatically generated from experiment results
- The bibliography contains references to the dataset and key papers
- The report follows academic standards for technical documentation
