# Hospital Stay Length Prediction

A machine learning project to predict hospital stay length based on healthcare risk factors.

## Dataset

This project uses the [Healthcare Risk Factors Dataset](https://www.kaggle.com/datasets/abdallaahmed77/healthcare-risk-factors-dataset) from Kaggle.

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installing Requirements

1. Clone this repository:
   ```bash
   git clone https://github.com/ka-sk/hospital-stay-length.git
   cd hospital-stay-length
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
hospital-stay-length/
├── dataset/
│   └── download_data.py        # Script to download the dataset
├── src/
│   └── parameters/
│       └── hyperparameters.py  # Model hyperparameters
├── requirements.txt
└── README.md
```

## Notes

Final project should have following steps:
1. Data aquisition and loading
2. Loading prepared models with initial hiperparameters
3. Choosing model with the best performance (quality metrics are to be chosen)
4. Fune-tuning of a chosen model using grid-search (hyperparameters defined in file)
5. Model evaluation, calculating metrics