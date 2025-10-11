import kagglehub
import pandas as pd
from pathlib import Path


def load_data():
    
    # Download data
    path = Path(kagglehub.dataset_download("abdallaahmed77/healthcare-risk-factors-dataset"))
    
    # Load data
    dataframe = pd.read_csv(path / 'dirty_v3_path.csv')
    
    return dataframe


def data_filtration(dataframe: pd.DataFrame):
    # filter out unnecessary colums 
    dataframe = dataframe.drop(['random_notes', 'noise_col'], axis=1)

    # nb of rows before filtration
    len1 = dataframe.shape[0]

    # filtrations of lacking data
    dataframe = dataframe.dropna()
    # nb of rows after filtration
    len2 = dataframe.shape[0]

    print(f"Rows with lacking data: {len1-len2}")

    return dataframe


if __name__ == "__main__":
    data_filtration(load_data())