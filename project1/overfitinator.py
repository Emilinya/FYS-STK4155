from resampling import Resampler
from linfitUtils import SplitData, LinFit
from franke_func import initialize_franke
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np


def get_MSE(split_data: SplitData):
    linfit = LinFit(split_data)

    beta = linfit.get_beta()
    test_MSE, _ = linfit.test_fit(beta)
    train_MSE, _ = linfit.test_train_fit(beta)
    return test_MSE, train_MSE


def MSE_bootstrap(poly_degree, x_grid, y_grid, z_grid, n_resamples):
    split_data = SplitData.from_2d_polynomial(
        poly_degree, x_grid, y_grid, z_grid, normalize=True
    )

    test_MSE, train_MSE = get_MSE(split_data)

    resampler = Resampler(split_data, n_resamples)
    while resampler.bootstrap():
        teMSE, trMSE = get_MSE(resampler.resampled_data)
        test_MSE += teMSE
        train_MSE += trMSE

    return test_MSE / (n_resamples + 1), train_MSE / (n_resamples + 1)


if __name__ == "__main__":
    np.random.seed(1337)

    n = 15
    max_degree = 16
    x_grid, y_grid, z_grid = initialize_franke(n)
    degree_list = list(range(max_degree+1))
    MSE_test_ray = np.zeros(len(degree_list))
    MSE_train_ray = np.zeros_like(MSE_test_ray)

    for i, poly_degree in enumerate(tqdm(degree_list)):
        MSE_test_ray[i], MSE_train_ray[i] = MSE_bootstrap(
            poly_degree, x_grid, y_grid, z_grid, n*n - 1
        )

    plt.figure(figsize=(7, 5), constrained_layout=True)
    plt.plot(degree_list, MSE_test_ray, ".-", label="test MSE")
    plt.plot(degree_list, MSE_train_ray, ".-", label="train MSE")
    plt.legend()
    plt.ylim(-0.05, 1.9)
    plt.xlabel("polynomial degree []")
    plt.ylabel("MSE []")
    plt.title(f"Overfitting example, ${n}^2$ datapoints")
    plt.savefig("imgs/franke/overfitting_example.png", dpi=200)
    plt.clf()
