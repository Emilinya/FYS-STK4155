import copy
import torch
from torch import nn
from torch.utils.data import DataLoader

import numpy as np


def trial_function(x, t, model: nn.Sequential):
    network_out = model.forward(torch.cat((x, t), 1))
    return torch.sin(torch.pi * x) + x * (1 - x) * t * network_out


def loss_function(x, t, model: nn.Sequential):
    x.requires_grad = True
    t.requires_grad = True

    u_t = trial_function(x, t, model)

    # calculate derivatives automatically
    du_t_dt, = torch.autograd.grad(
        u_t, t, grad_outputs=torch.ones_like(t), retain_graph=True
    )
    du_t_dx, = torch.autograd.grad(
        u_t, x, grad_outputs=torch.ones_like(x), create_graph=True
    )
    d2u_t_dx2, = torch.autograd.grad(
        du_t_dx, x, grad_outputs=torch.ones_like(x), create_graph=True
    )

    return torch.mean((du_t_dt - d2u_t_dx2) ** 2)


def train_model(
    dataloader: DataLoader, epochs,
    model: nn.Sequential, optimizer: torch.optim.Optimizer
):
    min_loss = float("infinity")
    optimal_model = copy.deepcopy(model)

    for epoch in range(epochs):
        for X in dataloader:
            x, t = X[:, [0]], X[:, [1]]

            loss = loss_function(x, t, model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if loss.item() < min_loss:
            min_loss = loss.item()
            optimal_model = copy.deepcopy(model)

        p = int(epoch*100 / (epochs-1))
        if p != int((epoch+1)*100 / (epochs-1)):
            print(f"\r{p:>3d} % [loss={loss.item():.12f}]     ", end="")

        # if loss increased a lot, we have probably reached a minimum
        if loss.item() > 100*min_loss:
            break

    model = copy.deepcopy(optimal_model)
    loss = loss_function(x, t, model)

    print(f"\n  min loss: {loss.item():.12f}")


def u_analytic(x, t):
    return np.exp(-np.pi**2 * t) * np.sin(np.pi * x)


def main():
    # create training and test data
    Nx = 20
    Nt = 20
    T = np.log(100) / (np.pi**2)
   
    x_ray_test = torch.linspace(0, 1, Nx)
    t_ray_test = torch.linspace(0, T, Nt)

    x_ray_train = (x_ray_test + 0.5/Nx)[:-1]
    t_ray_train = (t_ray_test + 0.5*T/Nt)[:-1]

    x_grid_train, t_grid_train = torch.meshgrid(
        x_ray_train, t_ray_train, indexing="xy"
    )
    x_flat_train = x_grid_train.reshape(-1, 1)
    t_flat_train = t_grid_train.reshape(-1, 1)

    x_grid_test, t_grid_test = torch.meshgrid(
        x_ray_test, t_ray_test, indexing="xy"
    )
    x_flat_test = x_grid_test.reshape(-1, 1)
    t_flat_test = t_grid_test.reshape(-1, 1)

    # define hyperparameters
    num_hidden_neurons = 10
    learning_rate = 0.01
    batch_size = 10
    epochs = 500

    # create model
    train_dataloader = DataLoader(
        torch.cat((x_flat_train, t_flat_train), 1),
        batch_size=batch_size, shuffle=False
    )
    model = nn.Sequential(
        nn.Linear(2, num_hidden_neurons),
        nn.Sigmoid(),
        nn.Linear(num_hidden_neurons, 1)
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # train model
    train_model(train_dataloader, epochs, model, optimizer)

    # test model
    with torch.no_grad():
        u_ffnn_grid = trial_function(x_flat_test, t_flat_test, model).reshape(Nx, Nt)
        u_ana_grid = u_analytic(x_grid_test, t_grid_test)

        np.savez(
            f"data/pytorch_solver/hn={num_hidden_neurons}_tp=SIGMOID.npz",
            x_ray=x_grid_test[0, :].numpy(), t_ray=t_grid_test[:, 0].numpy(),
            u_pytorch_grid=u_ffnn_grid.numpy(), u_ana_grid=u_ana_grid.numpy()
        )


if __name__ == "__main__":
    main()
