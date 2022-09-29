from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge, Lasso
from linfitUtils import LinFit, SplitData
from mpl_toolkits.mplot3d import Axes3D
from resampling import Resampler
import matplotlib.pyplot as plt
from terrain import get_terrain
from tqdm import tqdm
import numpy as np
import warnings

np.random.seed(1337)

x_grid, y_grid, z_grid = get_terrain(20)
n = len(x_grid.flat)

lda_ray = np.logspace(-8, 4, 200)
degree_list = list(range(20))
OLS_error_list = []
ridge_error_list = []
lasso_error_list = []

bootstrap_itrs = 50
for poly_degree in tqdm(degree_list):
    split_data = SplitData.from_2d_polynomial(poly_degree, x_grid, y_grid, z_grid, True)
    linfit = LinFit(split_data)
    resampler = Resampler(split_data, bootstrap_itrs)

    ridge_gridsearch = GridSearchCV(estimator=Ridge(), param_grid=dict(alpha=lda_ray))
    ridge_gridsearch.fit(*split_data.train_data)
    optimal_ridge_lda = ridge_gridsearch.best_params_['alpha']

    lasso_gridsearch = GridSearchCV(estimator=Lasso(), param_grid=dict(alpha=lda_ray))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lasso_gridsearch.fit(*split_data.train_data)
    optimal_lasso_lda = lasso_gridsearch.best_params_['alpha']

    OLS_error = linfit.test_fit()[0]
    ridge_error = linfit.test_fit(linfit.get_beta_ridge(optimal_ridge_lda))[0]
    lasso_error = linfit.test_fit(linfit.get_beta_lasso(optimal_lasso_lda))[0]
    while resampler.bootstrap():
        linfit.split_data = resampler.resampled_data
        OLS_error += linfit.test_fit()[0]
        ridge_error += linfit.test_fit(linfit.get_beta_ridge(optimal_ridge_lda))[0]
        lasso_error += linfit.test_fit(linfit.get_beta_lasso(optimal_lasso_lda))[0]
    OLS_error /= (bootstrap_itrs+1)
    ridge_error /= (bootstrap_itrs+1)
    lasso_error /= (bootstrap_itrs+1)

    OLS_error_list.append(OLS_error)
    ridge_error_list.append(ridge_error)
    lasso_error_list.append(lasso_error)

min_list = [min(ols, ridge, lasso) for ols, ridge, lasso in zip(OLS_error_list, ridge_error_list, lasso_error_list)]
best_degree = np.argmin(min_list)

plt.figure(figsize=(7, 5), constrained_layout=True)
plt.plot(degree_list, OLS_error_list, ".-", label="OLS MSE")
plt.plot(degree_list, ridge_error_list, ".-", label="Ridge MSE")
plt.plot(degree_list, lasso_error_list, ".-", label="Lasso MSE")
plt.plot([best_degree, best_degree], [0, min_list[best_degree]], "k--", label="optimal degree")
plt.xlabel(degree_list)
plt.legend()
plt.ylim(0, OLS_error_list[0]*1.1)
plt.xlabel("polynomial degree []")
plt.ylabel("[]")
plt.title(f"MSE for terrain fit with increasing polynomial degree, using ${n}$ datapoints")
plt.savefig("imgs/terrain/terrain_MSE.png", dpi=200)
plt.clf()