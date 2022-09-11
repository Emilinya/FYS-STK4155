from linfitUtils import get_design_matrix, get_beta, get_beta_ridge, test_fit
from franke_func import initialize_franke
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import time

n = 100
d = 3


x_grid, y_grid, z_grid = initialize_franke(n)
X_train, X_test, y_train, y_test = get_design_matrix(d, x_grid, y_grid, z_grid)

beta = get_beta(X_train, y_train)
MSE, _ = test_fit(y_test, X_test, beta)

lda_ray = np.logspace(-2, 2, 100)
MSE_ray = np.zeros_like(lda_ray)

for i, lda in enumerate(tqdm(lda_ray)):
    beta_ridge = get_beta_ridge(X_train, y_train, lda)
    MSE_ridge, _ = test_fit(y_test, X_test, beta_ridge)
    MSE_ray[i] = MSE_ridge

plt.plot(lda_ray, MSE_ray, ".-", markersize=4, linewidth=1)
plt.plot([np.min(lda_ray), np.max(lda_ray)], [MSE, MSE], "--k", label=f"OLS", linewidth=1)
plt.legend()
plt.xscale('log')
plt.xlabel(r"$\lambda$ []")
plt.ylabel("MSE []")
plt.title(f"MSE as a function of $\\lambda$ in Ridge regression, n={n}, d={d}")
plt.savefig("imgs/ridge_comp.png", dpi=200)
