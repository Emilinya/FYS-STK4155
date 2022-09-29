from distutils.log import warn
from linfitUtils import LinFit, SplitData
from resampling import Resampler
from franke_func import initialize_franke
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np


def er_bi_va(poly_degree, x_grid, y_grid, z_grid, n_resamples, beta_func, *beta_args):
    split_data = SplitData.from_2d_polynomial(
        poly_degree, x_grid, y_grid, z_grid, True
    )
    linfit = LinFit(split_data)

    Y_pred = np.zeros((split_data.test_shape()[0], n_resamples+1))
    Y_test = np.zeros_like(Y_pred)

    Y_pred[:, 0] = linfit.get_fit(beta_func(linfit, *beta_args))
    Y_test[:, 0] = split_data.test_data.y

    i = 0
    resampler = Resampler(split_data, n_resamples)
    while resampler.bootstrap():
        linfit.split_data = resampler.resampled_data
        Y_pred[:, 1+i] = linfit.get_fit(beta_func(linfit, *beta_args))
        Y_test[:, 1+i] = linfit.split_data.test_data.y
        i += 1

    error = np.mean(np.mean((Y_test - Y_pred)**2, axis=1, keepdims=True))
    bias = np.mean((Y_test - np.mean(Y_pred, axis=1, keepdims=True))**2)
    variance = np.mean(np.var(Y_pred, axis=1, keepdims=True))

    return error, bias, variance


def er_bi_va_bootstrap_OLS(poly_degree, x_grid, y_grid, z_grid, n_resamples):
    return er_bi_va(poly_degree, x_grid, y_grid, z_grid, n_resamples, LinFit.get_beta)


def er_bi_va_bootstrap_Ridge(poly_degree, x_grid, y_grid, z_grid, n_resamples, lda):
    return er_bi_va(poly_degree, x_grid, y_grid, z_grid, n_resamples, LinFit.get_beta_ridge, lda)


def er_bi_va_bootstrap_Lasso(poly_degree, x_grid, y_grid, z_grid, n_resamples, lda):
    return er_bi_va(poly_degree, x_grid, y_grid, z_grid, n_resamples, LinFit.get_beta_lasso, lda)


def plot_erbiva_lda(d, x_grid, y_grid, z_grid, lda_ray, regtype):
    n = len(x_grid.flat)

    error_OLS, bias_OLS, variance_OLS = er_bi_va_bootstrap_OLS(
        d, x_grid, y_grid, z_grid, n-1)

    error_ray = np.zeros(len(lda_ray))
    bias_ray = np.zeros_like(error_ray)
    variance_ray = np.zeros_like(error_ray)

    if regtype == "ridge":
        erbiva_func = er_bi_va_bootstrap_Ridge
    elif regtype == "lasso":
        erbiva_func = er_bi_va_bootstrap_Lasso
    else:
        warn(f"Unknown regtype: {regtype}")
        return

    for i, lda in enumerate(tqdm(lda_ray)):
        error_ray[i], bias_ray[i], variance_ray[i] = erbiva_func(
            d, x_grid, y_grid, z_grid, n-1, lda
        )
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    minmax = [np.min(lda_ray), np.max(lda_ray)]
    plt.figure(figsize=(8, 5), constrained_layout=True)
    plt.plot(lda_ray, error_ray, ".-", markersize=4, linewidth=1, label="MSE")
    plt.plot(
        lda_ray, bias_ray, ".-", markersize=4,
        linewidth=1, label="Bias$^2$"
    )
    plt.plot(
        lda_ray, variance_ray, ".-", markersize=4,
        linewidth=1, label="variance"
    )
    plt.plot(
        minmax, [error_OLS, error_OLS], "--",
             color=colors[0], label=f"OLS MSE", linewidth=1)
    plt.plot(minmax, [bias_OLS, bias_OLS], "--",
             color=colors[1], label=f"OLS Bias$^2$", linewidth=1)
    plt.plot(minmax, [variance_OLS, variance_OLS], "--",
             color=colors[2], label=f"OLS variance", linewidth=1)
    plt.legend()
    plt.xscale('log')
    plt.xlabel(r"$\lambda$ []")
    plt.ylabel("[]")
    plt.title(
        f"bias-variance as a function of $\\lambda$, "
        + f"{regtype.capitalize()} regression, "
        + f"n={n}, d={d}"
    )
    plt.savefig(f"imgs/franke/{regtype}_ldaplot_d={d}.png", dpi=200)


if __name__ == "__main__":
    np.random.seed(1337)

    n = 15
    max_degree = 16
    x_grid, y_grid, z_grid = initialize_franke(n)
    degree_list = list(range(max_degree+1))
    error_ray = np.zeros(len(degree_list))
    bias_ray = np.zeros_like(error_ray)
    variance_ray = np.zeros_like(error_ray)

    for i, poly_degree in enumerate(tqdm(degree_list)):
        error_ray[i], bias_ray[i], variance_ray[i] = er_bi_va_bootstrap_OLS(
            poly_degree, x_grid, y_grid, z_grid, n*n-1
        )

    plt.figure(figsize=(7, 5), constrained_layout=True)
    plt.plot(degree_list, error_ray, ".-", label="MSE")
    plt.plot(degree_list, bias_ray, ".-", label="Bias$^2$")
    plt.plot(degree_list, variance_ray, ".-", label="Variance")
    plt.legend()
    plt.ylim(-0.05, 1.9)
    plt.xlabel("polynomial degree []")
    plt.ylabel("[]")
    plt.title(f"bias-variance tradeoff, OLS, ${n}^2$ datapoints")
    plt.savefig("imgs/franke/bias-variance_OLS.png", dpi=200)
    plt.clf()
