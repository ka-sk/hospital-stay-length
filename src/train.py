# signle step function

# whole training with hyperparameters, model and data
# check if file already exists (if so, skip the training)

#grid-search function 
import torch
import data_loader as data
import models
import pytorch_tabnet
from torch.utils.data import DataLoader
import numpy as np

def step(X_train: torch.Tensor, 
         X_test: torch.Tensor, 
         y_train: torch.Tensor, 
         y_test: torch.Tensor, 
         model: torch.nn.Module, 
         loss_funct: callable, 
         optim: torch.optim.Optimizer):

    model.train()

    optim.zero_grad()

    y_pred = model(X_train)

    loss_train = loss_funct(y_pred, y_train)

    loss_train.backward()

    optim.step()

    model.eval()

    with torch.inference_mode():
        y_test_pred = model(X_test)

        loss_test = loss_funct(y_test_pred, y_test)

    return model, loss_train.item(), loss_test.item()


def train_step(dataloader: DataLoader, loss_funct: callable, optim: torch.optim.Optimizer):
    # gets only train data
    #split into batches
    # train
    loss_list = np.zeros()

    for X_batch, y_batch in dataloader:

        model.train()

        optim.zero_grad()

        y_pred = model(X_batch)

        loss_train = loss_funct(y_pred, y_batch)

        loss_train.backward()

        loss_list.append(loss_train.item())

        optim.step()

    return loss_list.mean()
    pass


def test_step():
    
    pass


if __name__ == '__main__':
    df = data.load_data()
    X, y = data.data_filtration(df)

    dataloader = DataLoader((X, y), batch_size=10)

    model_list = models.get_all_models()

    loss_funct = torch.nn.L1Loss()

    work_count = 0
    skipped = 0

    for model in model_list:
        if type(model)!=pytorch_tabnet.tab_model.TabNetRegressor:
            step(X, y, model, loss_funct, torch.optim.Adam(model.parameters()))
            work_count += 1
            print(f'Done: {work_count}/{len(model_list)}')
        else:
            skipped += 1
    print(f"Done: {work_count}, skipped: {skipped}")


'''
from torch.utils.data import DataLoader, TensorDataset
import torch

# data to define a simple dataset
inputs = torch.arange(1, 51).float().reshape(-1, 1)  # a 1D tensor dataset (input)
targets = inputs ** 2  # square of the input values (simulating a regression task)

# create a TensorDataset and DataLoader
dataset = TensorDataset(inputs, targets)
dataloader = DataLoader(dataset, batch_size=5, shuffle=True)

# iterate through DataLoader
for batch in dataloader:
    print(batch)

# sample output:
# [tensor([[46.],
#         [42.],
#         [25.],
#         [10.],
#         [34.]]), tensor([[2116.],
#         [1764.],
#         [ 625.],
#         [ 100.],
#         [1156.]])]
# ...
# [tensor([[21.],
#         [ 9.],
#         [ 2.],
#         [38.],
#         [44.]]), tensor([[ 441.],
#         [  81.],
#         [   4.],
#         [1444.],
#         [1936.]])]
# '''
