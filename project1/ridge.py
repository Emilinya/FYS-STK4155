from bias_variance import plot_erbiva_lda, er_bi_va_bootstrap_Ridge
from sklearn.model_selection import GridSearchCV
from franke_func import initialize_franke
from sklearn.linear_model import Ridge
from linfitUtils import SplitData
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

np.random.seed(1337)

n = 15
max_degree = 16
lda_ray = np.logspace(-8, 4, 300)
degree_list = list(range(max_degree+1))
error_ray = np.zeros(len(degree_list))
bias_ray = np.zeros_like(error_ray)
variance_ray = np.zeros_like(error_ray)
optimal_lda_ray = np.zeros_like(error_ray)

x_grid, y_grid, z_grid = initialize_franke(n)

for i, poly_degree in enumerate(tqdm(degree_list)):
    split_data = SplitData.from_2d_polynomial(poly_degree, x_grid, y_grid, z_grid, True)
    gridsearch = GridSearchCV(estimator=Ridge(), param_grid=dict(alpha=lda_ray))
    gridsearch.fit(*split_data.train_data)
    optimal_lda_ray[i] = gridsearch.best_params_['alpha']

    error_ray[i], bias_ray[i], variance_ray[i] = er_bi_va_bootstrap_Ridge(
        poly_degree, x_grid, y_grid, z_grid, n*n-1, optimal_lda_ray[i]
    )
print(optimal_lda_ray)

plt.figure(figsize=(7, 5), constrained_layout=True)
plt.plot(degree_list, error_ray, ".-", label="MSE")
plt.plot(degree_list, bias_ray, ".-", label="Bias$^2$")
plt.plot(degree_list, variance_ray, ".-", label="Variance")
plt.legend()
plt.ylim(-0.05, 1.9)
plt.xlabel("polynomial degree []")
plt.ylabel("[]")
plt.title(f"bias-variance tradeoff, ridge with optimal $\\lambda$, ${n}^2$ datapoints")
plt.savefig("imgs/franke/bias-variance_ridge.png", dpi=200)
plt.clf()

plot_erbiva_lda(7, x_grid, y_grid, z_grid, lda_ray, regtype="ridge")
plot_erbiva_lda(13, x_grid, y_grid, z_grid, lda_ray, regtype="ridge")
