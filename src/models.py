# lstm i gru nie są dobrym wyborem
# mlp
# tab transformer
# tab net

#dokładne dane w plikach yml w folderze experiments/configs
#tutaj tylko wczytywanie modeli i zwracanie instancji z torch

from omegaconf import OmegaConf
from itertools import product
import torch.nn as nn
from pathlib import Path


def load_model_instances(path: str):
    config = OmegaConf.load(path)
    models = []

    if config.model.name == "tabtransformer":

        # Simple tab transformer
        class SimpleTabTransformer(nn.Module):
            def __init__(self, num_features, d_model=64, n_heads=4, num_layers=2):
                super().__init__()
                self.embedding = nn.Linear(num_features, d_model)
                encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads)
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                self.fc_out = nn.Linear(d_model, 1)

            def forward(self, x):
                # Add sequence dimension (required by transformer)
                x = self.embedding(x).unsqueeze(1)
                x = self.transformer(x)
                x = x.mean(dim=1)  # pool across sequence dimension
                return self.fc_out(x)


        # Extract grid search parameters
        hidden_sizes = config.model.hidden_size
        num_layers = config.model.num_layers
        dropouts = config.model.dropout

        # Generate all combinations
        for hs, nl, do in product(hidden_sizes, num_layers, dropouts):
            model = SimpleTabTransformer()
            models.append(model)

    elif config.model.name == "tabnet":
        hidden_sizes = config.model.hidden_size
        num_layers = config.model.num_layers 
        dropouts = config.model.dropout

        for hs, nl, do in product(hidden_sizes, num_layers, dropouts):
            model = nn.LSTM(
                input_size=1,  # Adjust as needed
                hidden_size=hs,
                num_layers=nl,
                dropout=do if nl > 1 else 0,
                batch_first=True
            )
            models.append(model)

    elif config.model.name == "mlp":
        hidden_channels = config.model.hidden_channels
        activation_layers = config.model.activation_layer
        dropouts = config.model.dropout

        activation_map = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh
        }

        for hc, act, do in product(hidden_channels, activation_layers, dropouts):
            layers = [
                nn.Linear(1, hc),  # Adjust input size as needed
                activation_map[act](),
                nn.Dropout(do),
                nn.Linear(hc, 1)   # Adjust output size as needed
            ]
            model = nn.Sequential(*layers)
            models.append(model)

    else:
        raise ValueError(f"Unknown model name: {config.model.name} in {path}")
    return models


def get_all_models(path=''):
    if path == '':
        path = Path('experiments/configs/grid-search/')
        all_models_list = []
    for filepath in path.iterdir():
        all_models_list += load_model_instances(filepath)
    return all_models_list


if __name__ == "__main__":
    models = get_all_models()
    [print(f"{num+1}: {model}") for num, model in enumerate(models)]