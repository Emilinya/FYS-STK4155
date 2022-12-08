from pytorch_solver import trial_function, train_model, u_analytic

import torch
from torch import nn
from torch.utils.data import DataLoader

import numpy as np

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
    num_hidden_neurons = 11
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
        nn.Conv1d(num_hidden_neurons, 1, kernel_size=3),
        nn.Linear(num_hidden_neurons, 1),
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
            u_ffnn_grid=u_ffnn_grid.numpy(), u_ana_grid=u_ana_grid.numpy()
        )


if __name__ == "__main__":
    main()

