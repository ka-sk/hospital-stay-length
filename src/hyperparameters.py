# #dokładne dane w plikach yml w folderze experiments/configs
#tutaj tylko wczytywanie funkcji strat, optymalizacji, learning rate itp
from omegaconf import OmegaConf
from pathlib import Path
from torch import cuda
import torch

def load_loss_function(path=''):
    device = 'cuda' if cuda.is_available() else 'gpu' 

    if path == '':
            path = Path('experiments/configs/loss-functions')

    all_loss_list = []

    for filepath in path.iterdir():

        config = OmegaConf.load(filepath)

        if config.loss.name == 'mse':
            from torch.nn import MSELoss
            loss = MSELoss(reduction=config.loss.reduction)
        elif config.loss.name == 'mae':
            from torch.nn import L1Loss
            loss = L1Loss(reduction=config.loss.reduction)
        else:
            raise ValueError(f"Unknown loss function name: {config.loss.name} in {path}")
        all_loss_list.append(loss.to(device=device))
    return all_loss_list


def load_optim(path='', model: torch.nn.Module=None):
    device = 'cuda' if cuda.is_available() else 'gpu' 

    if path == '':
            path = Path('experiments/configs/optimizers')
    
    all_funct_list = []

    for filepath in path.iterdir():

        config = OmegaConf.load(filepath)

        if config.optimizer.name == 'adam':
            from torch.optim import Adam

            if model is None:
                optim = (Adam, {'lr':config.optimizer.lr, 
                            'weight_decay':config.optimizer.weight_decay})
            else:
                optim = Adam(model.parameters(),
                            lr=config.optimizer.lr,
                            weight_decay=config.optimizer.weight_decay)
            
        elif config.optimizer.name == 'sgd':
            from torch.optim import SGD

            if model is None:
                optim = (SGD, {'lr':config.optimizer.lr,
                            'momentum':config.optimizer.momentum,
                            'weight_decay':config.optimizer.weight_decay})
            else:
                optim = SGD(model.parameters(), 
                            lr=config.optimizer.lr,
                            momentum=config.optimizer.momentum,
                            weight_decay=config.optimizer.weight_decay)
        else:
            raise ValueError(f"Unknown optim function name: {config.optimizer.name} in {path}")
        
        all_funct_list.append(optim)
        
    return all_funct_list


if __name__ == '__main__':
    import models
    model_list = models.get_all_models()

    print(load_loss_function())
    print(load_optim(model=model_list[28]))
    print(load_optim())