from linfitUtils import SplitData, LinFit
from franke_func import initialize_franke
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time

def plot_MSE_R2(
    degree_list, MSE_train_list, MSE_test_list, 
    R2_train_list, R2_test_list, title, filename
):
    plt.figure(figsize=(6, 4), constrained_layout=True)
    plt.plot(degree_list, MSE_train_list, ".--", label="train MSE")
    plt.plot(degree_list, R2_train_list, ".--", label="train R2")
    plt.plot(degree_list, MSE_test_list, ".--", label="test MSE")
    plt.plot(degree_list, R2_test_list, ".--", label="test R2")
    plt.xlabel("polynomial degree []")
    plt.ylabel("y []")
    if "no_noise" in title:
        plt.ylim(-0.1, 1.1)
    else:
        plt.ylim(-0.5, 1.5)
    plt.title(title)
    plt.legend()
    plt.savefig(filename, dpi=200)
    plt.clf()

def plot_beta(beta_grid, title, filename):
    plt.figure(figsize=(14, 3), constrained_layout=True)
    sns.heatmap(beta_grid, cbar=False, fmt=".3g", annot=True, linewidths=0.25, linecolor="k")
    plt.ylabel("polynomial degree []")
    plt.xlabel("$\\beta_i$ []")
    plt.title(title)
    plt.savefig(".".join(filename.split(".")[:-1]) + "_beta.png", dpi=200)
    plt.clf()

def linfit_franke(n, max_degree, add_noise, normalize):
    x_grid, y_grid, z_grid = initialize_franke(n)
    if add_noise:
        z_grid += np.random.normal(0, 1, z_grid.shape)

    degree_list = list(range(max_degree+1))
    MSE_train_list = []
    R2_train_list = []
    MSE_test_list = []
    R2_test_list = []

    beta_grid = np.zeros((max_degree+1, int((max_degree+1)*(max_degree+2)/2)))
    beta_grid.fill(np.nan)

    noise_str = "noise" if add_noise else "no_noise"
    normalize_str = 'normalized' if normalize else 'not_normalized'
    filename = f"imgs/franke/MSE_R2_comp/n={n}_{noise_str}_{normalize_str}.png"

    start = time.time()
    for i, poly_degree in enumerate(degree_list):
        print(f"\r{noise_str} {normalize_str} {i}/{max_degree}", end="")

        split_data = SplitData.from_2d_polynomial(poly_degree, x_grid, y_grid, z_grid, normalize)
        linfit = LinFit(split_data)
    
        beta = linfit.get_beta()
        beta_grid[i, :len(beta)] = beta
        
        MSE_train, R2_train = linfit.test_train_fit(beta)
        MSE_train_list.append(MSE_train)
        R2_train_list.append(R2_train)

        MSE_test, R2_test = linfit.test_fit(beta)
        MSE_test_list.append(MSE_test)
        R2_test_list.append(R2_test)
    print(f" - {time.time() - start:.3f} s")

    MSE_R2_title = f"{normalize_str},{noise_str},n={n}"    
    plot_MSE_R2(
        degree_list, MSE_train_list, MSE_test_list, 
        R2_train_list, R2_test_list, MSE_R2_title, filename
    )

    beta_title = f"optimal $\\beta$ of fits, {normalize_str}, {noise_str}, n={n}"
    plot_beta(beta_grid, beta_title, filename)

n = 15
max_degree = 5
np.random.seed(1)

for add_noise in [False, True]:
    R2_list = []
    for normalize in [False, True]:
        linfit_franke(n, max_degree, add_noise, normalize)
