# signle step function

# whole training with hyperparameters, model and data
# check if file already exists (if so, skip the training)

#grid-search function 
import torch

def step(X_train: torch.Tensor, y_train: torch.Tensor, model: torch.nn.Module, loss_funct: torch.nn, optim: torch.optim.Optimizer):
    # , X_test: torch.Tensor, y_test: torch.Tensor
    model.train()

    optim.zero_grad()

    y_pred = model(X_train)
    loss = loss_funct(y_pred, y_train)
    
    loss.backward()

    optim.step()

    model.eval()

    with model.mode.inference_mode():
        pass



if __name__ == '__main__':
    import data_loader as data
    df = data.load_data()
    df = data.data_filtration(df)
    y = torch.from_numpy(df['LengthOfStay'].values).to(dtype=torch.float32)
    y = y.unsqueeze(dim=1)

    X = torch.from_numpy(df.drop('LengthOfStay', axis=1).values).to(dtype=torch.float32)
    model = torch.nn.GRU(X.shape[1], 1)
    loss_funct = torch.nn.L1Loss()
    optim = torch.optim.Adam(model.parameters())
    step(X, y, model, loss_funct, optim)
