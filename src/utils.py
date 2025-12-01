"""
Utility functions for hospital stay prediction project.
"""
import torch
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"


def get_device() -> str:
    """Get available device (cuda or cpu)."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def model_filepath(model: torch.nn.Module, grid_search_id: int, fold: int, base_dir: Path = None) -> Path:
    """
    Generate filepath for saving a model.
    
    Structure: models/{model_name}/gs{id:04d}_fold{fold}.pt
    """
    if base_dir is None:
        base_dir = EXPERIMENTS_DIR / "models"
    
    model_name = model.__class__.__name__
    filename = f"gs{grid_search_id:04d}_fold{fold}.pt"
    
    return base_dir / model_name / filename


def load_model(model_class: type, path: Path, **init_kwargs) -> torch.nn.Module:
    """Load model from saved state dict."""
    model = model_class(**init_kwargs)
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model


if __name__ == '__main__':
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Experiments dir: {EXPERIMENTS_DIR}")
    print(f"Device: {get_device()}")
