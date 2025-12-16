from omegaconf import OmegaConf
from itertools import product
import torch.nn as nn
from pathlib import Path
from pytorch_tabnet.tab_model import TabNetRegressor
import torch
from utils import get_device


# Simple tab transformer
class SimpleTabTransformer(nn.Module):
    def __init__(self, in_features, d_model=64, n_heads=4, num_layers=2):
        super().__init__()
        self.in_features = in_features
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers

        self.embedding = nn.Linear(in_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, x):
        # Add sequence dimension (required by transformer)
        x = self.embedding(x).unsqueeze(1)
        x = self.transformer(x)
        x = x.mean(dim=1)  # pool across sequence dimension
        return self.fc_out(x)

    def get_hyperparams(self) -> dict:
        return {
            "in_features": self.in_features,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "num_layers": self.num_layers
        }


# Simple MLP
class SimpleMLP(nn.Module):
    def __init__(self, in_features=22, hidden_features=8, activation_layer='relu', dropout=0.2):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.activation_name = activation_layer
        self.dropout_rate = dropout

        activation_map = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh
        }
        self.linear1 = nn.Linear(in_features=in_features, out_features=hidden_features)
        self.act_layer = activation_map[activation_layer]()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_features, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act_layer(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

    def get_hyperparams(self) -> dict:
        return {
            "in_features": self.in_features,
            "hidden_features": self.hidden_features,
            "activation": self.activation_name,
            "dropout": self.dropout_rate
        }


def load_model_instances(config_or_path):
    # Accept either a path (str/Path) or an already-loaded OmegaConf config
    if isinstance(config_or_path, (str, Path)):
        config = OmegaConf.load(config_or_path)
    else:
        config = config_or_path
    models = []
    device = get_device()

    in_features = config.model.in_features

    if config.model.name == "tabtransformer":

        # Extract grid search parameters
        d_model = config.model.d_model
        num_layers = config.model.num_layers
        n_heads = config.model.n_heads

        # Generate all combinations
        for dm, nh, nl in product(d_model, n_heads, num_layers):
            model = SimpleTabTransformer(in_features=in_features, d_model=dm, n_heads=nh, num_layers=nl)
            models.append(model.to(device=device))

    elif config.model.name == "tabnet":
        n_d = config.model.n_d
        n_a = config.model.n_a
        n_steps = config.model.n_steps

        for nd, na, ns in product(n_d, n_a, n_steps):
            model = TabNetRegressor(n_d=nd, n_a=na, n_steps=ns)
            models.append(model)

    elif config.model.name == "mlp":
        hidden_channels = config.model.hidden_channels
        activation_layers = config.model.activation_layer
        dropouts = config.model.dropout

        for hc, act, do in product(hidden_channels, activation_layers, dropouts):
            model = SimpleMLP(hidden_features=hc, activation_layer=act, dropout=do)
            models.append(model.to(device=device))

    else:
        raise ValueError(f"Unknown model name: {config.model.name}")
    return models


def get_all_models(path=''):
    if path == '':
        path = Path('experiments/configs/grid-search/')

    all_models_list = []
    for filepath in path.iterdir():
        all_models_list += load_model_instances(filepath)
    return all_models_list


if __name__ == "__main__":
    # Test loading all models
    all_models = get_all_models()
    print(f"Total models loaded: {len(all_models)}")
    for i, model in enumerate(all_models[:5]):
        print(f"  {i+1}. {model.__class__.__name__}")
        if hasattr(model, 'get_hyperparams'):
            print(f"      Hyperparams: {model.get_hyperparams()}")
