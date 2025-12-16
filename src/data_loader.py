import kagglehub
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from utils import get_device


def load_data(return_split: bool = False, test_size: float = 0.2, random_state: int = 42):
    """
    Load and preprocess hospital stay data.
    
    Args:
        return_split: If True, returns (X_train, X_test, y_train, y_test)
                     If False, returns TensorDataset (legacy behavior)
        test_size: Proportion of data for test set
        random_state: Random seed for reproducibility
    
    Returns:
        If return_split=True: tuple of (X_train, X_test, y_train, y_test) as tensors
        If return_split=False: TensorDataset (X, y)
    """
    # Download data
    path = Path(kagglehub.dataset_download("abdallaahmed77/healthcare-risk-factors-dataset"))
    
    # Load data
    dataframe = pd.read_csv(path / 'dirty_v3_path.csv')
    
    # Preprocess
    dataframe = data_filtration(dataframe, return_tensors=False)
    
    if return_split:
        # Split into train/test
        y = dataframe['LengthOfStay'].values
        X = dataframe.drop('LengthOfStay', axis=1).values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Convert to tensors
        device = get_device()
        X_train = torch.from_numpy(X_train).to(dtype=torch.float32, device=device)
        X_test = torch.from_numpy(X_test).to(dtype=torch.float32, device=device)
        y_train = torch.from_numpy(y_train).to(dtype=torch.float32, device=device).unsqueeze(1)
        y_test = torch.from_numpy(y_test).to(dtype=torch.float32, device=device).unsqueeze(1)
        
        return X_train, X_test, y_train, y_test
    else:
        # Legacy: return TensorDataset
        device = get_device()
        y = torch.from_numpy(dataframe['LengthOfStay'].values).to(dtype=torch.float32, device=device)
        y = y.unsqueeze(dim=1)
        X = torch.from_numpy(dataframe.drop('LengthOfStay', axis=1).values).to(dtype=torch.float32, device=device)
        return TensorDataset(X, y)


def data_filtration(dataframe: pd.DataFrame, return_tensors: bool = True):
    device = get_device() 

    # filter out unnecessary colums 
    dataframe = dataframe.drop(['random_notes', 'noise_col'], axis=1)

    # nb of rows before filtration
    len1 = dataframe.shape[0]

    # filtrations of lacking data
    dataframe = dataframe.dropna()
    # nb of rows after filtration
    len2 = dataframe.shape[0]

    print(f"Rows with missing data: {len1-len2}")

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

def data_filtration(dataframe: pd.DataFrame, return_tensors: bool = True):
    """
    Clean and preprocess dataframe.
    
    Args:
        dataframe: Raw dataframe
        return_tensors: If True, returns TensorDataset. If False, returns DataFrame.
    
    Returns:
        TensorDataset or DataFrame depending on return_tensors flag
    """
    device = get_device() 

    # filter out unnecessary colums 
    dataframe = dataframe.drop(['random_notes', 'noise_col'], axis=1)

    # nb of rows before filtration
    len1 = dataframe.shape[0]

    # filtrations of lacking data
    dataframe = dataframe.dropna()
    # nb of rows after filtration
    len2 = dataframe.shape[0]

    print(f"Rows with missing data: {len1-len2}")

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

    if return_tensors:
        # Should there be data normalisation? 
        y = torch.from_numpy(dataframe['LengthOfStay'].values).to(dtype=torch.float32, device=device)
        y = y.unsqueeze(dim=1)
        X = torch.from_numpy(dataframe.drop('LengthOfStay', axis=1).values).to(dtype=torch.float32, device=device)
        return TensorDataset(X, y)
    else:
        return dataframe


if __name__ == "__main__":
    # Test legacy behavior
    dataset = load_data(return_split=False)
    print(f"Dataset size: {len(dataset)}")
    
    # Test new split behavior
    X_train, X_test, y_train, y_test = load_data(return_split=True)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")