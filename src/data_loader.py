import kagglehub
import pandas as pd
from pathlib import Path
import torch
from torch import cuda
from torch.utils.data import TensorDataset


def load_data()-> pd.DataFrame:

    # Download data
    path = Path(kagglehub.dataset_download("abdallaahmed77/healthcare-risk-factors-dataset"))
    
    # Load data
    dataframe = pd.read_csv(path / 'dirty_v3_path.csv')
    
    return dataframe


def data_filtration(dataframe: pd.DataFrame) -> pd.DataFrame:
    device = 'cuda' if cuda.is_available() else 'cpu' 

    # filter out unnecessary colums 
    dataframe = dataframe.drop(['random_notes', 'noise_col'], axis=1)

    # nb of rows before filtration
    len1 = dataframe.shape[0]

    # filtrations of lacking data
    dataframe = dataframe.dropna()
    # nb of rows after filtration
    len2 = dataframe.shape[0]

    print(f"Rows with lacking data: {len1-len2}")

    # "Male" "Female" into 1 and 0
    mapping_dict = {'Male': 1, 'Female': 0}
    dataframe['Gender'] = dataframe['Gender'].map(mapping_dict)

    # Get all unique conditions except 'Healthy'
    conditions = dataframe['Medical Condition'].unique()
    conditions = [cond for cond in conditions if cond != 'Healthy']

    # Create a column for each condition
    for cond in conditions:
        dataframe[cond] = (dataframe['Medical Condition'] == cond).astype(int)

    # Optionally, drop the original column
    dataframe = dataframe.drop('Medical Condition', axis=1)

    # Should there be data normalisation? 
    y = torch.from_numpy(dataframe['LengthOfStay'].values).to(dtype=torch.float32, device=device)
    y = y.unsqueeze(dim=1)
    X = torch.from_numpy(dataframe.drop('LengthOfStay', axis=1).values).to(dtype=torch.float32, device=device)

    return TensorDataset(X, y)


if __name__ == "__main__":
    print(data_filtration(load_data()))