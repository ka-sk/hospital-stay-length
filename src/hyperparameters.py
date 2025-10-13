# #dokładne dane w plikach yml w folderze experiments/configs
#tutaj tylko wczytywanie funkcji strat, optymalizacji, learning rate itp
from omegaconf import OmegaConf
from itertools import product
from pathlib import Path
from typing import Literal

def load_loss_function(path):
    config = OmegaConf.load(path)
    if config.loss.name == 'mse':
        from torch.nn import MSELoss
        loss = MSELoss(reduction=config.loss.reduction)
    elif config.loss.name == 'mae':
        from torch.nn import L1Loss
        loss = L1Loss(reduction=config.loss.reduction)
    else:
        raise ValueError(f"Unknown loss function name: {config.loss.name} in {path}")
    return loss


def load_optim(path):
    config = OmegaConf.load(path)
    if config.optimizer.name == 'adam':
        from torch.optim import Adam
        optim = (Adam, {'lr':config.optimizer.lr, 
                        'weight_decay':config.optimizer.weight_decay})
        
    elif config.optimizer.name == 'sgd':
        from torch.optim import SGD
        optim = (SGD, {'lr':config.optimizer.lr,
                        'momentum':config.optimizer.momentum,
                        'weight_decay':config.optimizer.weight_decay})
    else:
        raise ValueError(f"Unknown optim function name: {config.optimizer.name} in {path}")
    return optim


def get_all_functions(path='', funct: Literal['loss', 'optim'] = 'loss'):
    if funct == 'loss':
        getter = load_loss_function

        if path == '':
            path = Path('experiments/configs/loss-functions')

    elif funct == 'optim':
        getter = load_optim

        if path == '':
            path = Path('experiments/configs/optimizers')

    all_funct_list = []
    for filepath in path.iterdir():
        all_funct_list.append(getter(filepath))
    return all_funct_list


if __name__ == '__main__':
    print(get_all_functions())
    print(get_all_functions(funct='optim'))