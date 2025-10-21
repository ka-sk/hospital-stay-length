# signle step function

# whole training with hyperparameters, model and data
# check if file already exists (if so, skip the training)

#grid-search function 
import torch
import data_loader as data
import models
import pytorch_tabnet

def step(X_train: torch.Tensor, y_train: torch.Tensor, model: torch.nn.Module, loss_funct: torch.nn, optim: torch.optim.Optimizer):

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