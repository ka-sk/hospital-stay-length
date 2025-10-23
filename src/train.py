# signle step function

# whole training with hyperparameters, model and data
# check if file already exists (if so, skip the training)

#grid-search function 
import torch
import data_loader as data
import models
from pytorch_tabnet.tab_model import TabNetRegressor
from torch.utils.data import DataLoader, random_split
import numpy as np
from itertools import product
from collections.abc import Iterable

def train(train_dataloader: DataLoader,
         test_dataloader: DataLoader,
         model: torch.nn.Module, 
         loss_funct: callable, 
         optim: torch.optim.Optimizer,
         num_epochs: int=100):
    
    test_loss_array = np.zeros([1, num_epochs])
    train_loss_array = np.zeros([1, num_epochs])

    for epoch in range(num_epochs):

        model, train_loss_array[epoch] = train_step(train_dataloader, model, loss_funct, optim)

        test_loss_array[epoch] = test_step(test_dataloader, model, loss_funct)

    return model, train_loss_array, test_loss_array


def train_step(dataloader: DataLoader, model: torch.nn.Module, loss_funct: callable, optim: torch.optim.Optimizer):
    # gets only train data
    #split into batches
    # train

    loss_list = np.zeros(len(dataloader))

    for idx, (X_batch, y_batch) in enumerate(dataloader):

        model.train()

        optim.zero_grad()

        y_pred = model(X_batch)

        loss_train = loss_funct(y_pred, y_batch)

        loss_train.backward()

        loss_list[idx] = loss_train.item()

        optim.step()

    return model, loss_list.mean()


def test_step(dataloader:DataLoader, model: torch.nn.Module, loss_funct: callable):

    model.eval()

    loss_list = np.zeros(len(dataloader))
    with torch.inference_mode():
        for idx, (X, y) in enumerate(dataloader):
            y_pred = model(X)

            loss_list[idx] = loss_funct(y_pred, y).cpu()
    return np.mean(loss_list)


def grid_search(train_dataloader: DataLoader,
                test_dataloader: DataLoader,
                model_list: Iterable | torch.nn.Module, 
                optim_list: Iterable | torch.optim.Optimizer, 
                loss_funct_list: Iterable | callable,
                num_epochs: int=100):
    
    if not isinstance(model_list, Iterable):
        model_list = [model_list]
    
    if not isinstance(optim_list, Iterable):
        model_list = [optim_list]

    if not isinstance(loss_funct_list, Iterable):
        model_list = [loss_funct_list]

    for model, optim, loss_f in product(model_list, optim_list, loss_funct_list):
        # cross val        
        pass


if __name__ == '__main__':

    dataset = data.data_filtration(data.load_data())

    train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])

    train_dataloader = DataLoader(train_dataset, batch_size=10)
    test_dataloader = DataLoader(test_dataset)

    model_list = models.get_all_models()

    loss_funct = torch.nn.L1Loss()

    work_count = 0
    skipped = 0

    for model in model_list:
        if type(model)!=TabNetRegressor:
            train(train_dataloader, test_dataloader, model, loss_funct, torch.optim.Adam(model.parameters()), num_epochs=1)
            work_count += 1
            print(f'Done: {work_count}/{len(model_list)}')
        else:
            skipped += 1
    print(f"Done: {work_count}, skipped: {skipped}")

