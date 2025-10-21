# signle step function

# whole training with hyperparameters, model and data
# check if file already exists (if so, skip the training)

#grid-search function 
import torch
import data_loader as data
import models
import pytorch_tabnet

def step(X_train: torch.Tensor, y_train: torch.Tensor, model: torch.nn.Module, loss_funct: torch.nn, optim: torch.optim.Optimizer):
    if model == pytorch_tabnet.tab_model.TabNetRegressor:
        print('here')
        #model.fit()
    else:
        model.train()

        optim.zero_grad()

        y_pred = model(X_train)
        loss = loss_funct(y_pred, y_train)

        loss.backward()

        optim.step()

        model.eval()

        with torch.inference_mode():
            pass





if __name__ == '__main__':
    df = data.load_data()
    X, y = data.data_filtration(df)
    model_list = models.get_all_models()
    model = model_list[10]

    [print(type(i)) for i in model_list]

    loss_funct = torch.nn.L1Loss()
    optim = torch.optim.Adam(model.parameters())
    [step(X, y, model, loss_funct, optim) for model in model_list]
