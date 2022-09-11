from linfitUtils import get_design_matrix, get_beta, test_fit
from franke_func import frankeFunction
import matplotlib.pyplot as plt
import numpy as np
import time

def plot_MSE_R2_fig(x_grid, y_grid, z_grid, max_degree, filename, scaling=False):
    degree_list = list(range(max_degree+1))
    MSE_train_list = []
    R2_train_list = []
    MSE_test_list = []
    R2_test_list = []

    if scaling:
        # first subtract mean
        x_grid = x_grid - np.average(x_grid)
        y_grid = y_grid - np.average(y_grid)
        z_grid = z_grid - np.average(z_grid)

        # then scale such that largest absolute value is 1
        x_grid = x_grid / np.max(np.abs(x_grid))
        y_grid = y_grid / np.max(np.abs(y_grid))
        z_grid = z_grid / np.max(np.abs(z_grid))

    start = time.time()
    for i, poly_degree in enumerate(degree_list):
        print(f"\r  {i}/{max_degree}", end="")
        X_train, X_test, y_train, y_test = get_design_matrix(poly_degree, x_grid, y_grid, z_grid)
        beta = get_beta(X_train, y_train)
        
        MSE_train, R2_train = test_fit(y_train, X_train, beta)
        MSE_train_list.append(MSE_train)
        R2_train_list.append(R2_train)

        MSE_test, R2_test = test_fit(y_test, X_test, beta)
        MSE_test_list.append(MSE_test)
        R2_test_list.append(R2_test)
    print(f" - {time.time() - start:.3f} s")

    plt.plot(degree_list, MSE_train_list, label="training MSE")
    plt.plot(degree_list, R2_train_list, label="training R2")
    plt.plot(degree_list, MSE_test_list, label="testing MSE")
    plt.plot(degree_list, R2_test_list, label="testing R2")
    plt.xlabel("polynomial degree []")
    plt.ylabel("y []")
    plt.title(f"MSE and R2 of polynomial fits with different degree, {'scaled data' if scaling else 'no scaling'}, n={n}")
    plt.legend()
    plt.savefig(filename, dpi=200)
    plt.clf()

    return degree_list, R2_test_list

for n in [5, 20, 100, 400]:
    x_ray = np.linspace(0, 1, n)
    x_grid, y_grid = np.meshgrid(x_ray, x_ray)
    z_grid = frankeFunction(x_grid, y_grid)

    print(f"n = {n}:")
    degree_list, R2_test_noscale = plot_MSE_R2_fig(x_grid, y_grid, z_grid, 6, f"imgs/MSE_R2_comp/MSE_R2_n={n}_noscale.png")
    _, R2_test_scale = plot_MSE_R2_fig(x_grid, y_grid, z_grid, 6, f"imgs/MSE_R2_comp/MSE_R2_n={n}_scale.png", scaling=True)

    plt.plot(degree_list, np.array(R2_test_noscale) - np.array(R2_test_scale), label="R2 no scale - R2 scale")
    plt.legend()
    plt.savefig(f"imgs/MSE_R2_comp/MSE_R2_n={n}_diff.png", dpi=200)
    plt.clf()
