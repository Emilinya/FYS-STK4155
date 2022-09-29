from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge, Lasso
from linfitUtils import SplitData, LinFit
from franke_func import initialize_franke
from resampling import Resampler
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import warnings

np.random.seed(1337)

n = 15
max_degree = 16
lda_ray = np.logspace(-8, 4, 300)
degree_list = list(range(max_degree+1))

OLS_nosample_error_ray = np.zeros(len(degree_list))
OLS_bootstrap_error_ray = np.zeros_like(OLS_nosample_error_ray)
OLS_cross_validation_error_ray = np.zeros_like(OLS_nosample_error_ray)

ridge_nosample_error_ray = np.zeros_like(OLS_nosample_error_ray)
ridge_bootstrap_error_ray = np.zeros_like(OLS_nosample_error_ray)
ridge_cross_validation_error_ray = np.zeros_like(OLS_nosample_error_ray)

lasso_nosample_error_ray = np.zeros_like(OLS_nosample_error_ray)
lasso_bootstrap_error_ray = np.zeros_like(OLS_nosample_error_ray)
lasso_cross_validation_error_ray = np.zeros_like(OLS_nosample_error_ray)

x_grid, y_grid, z_grid = initialize_franke(n)

for i, poly_degree in enumerate(tqdm(degree_list)):
    split_data = SplitData.from_2d_polynomial(poly_degree, x_grid, y_grid, z_grid, True)

    ridge_gridsearch = GridSearchCV(estimator=Ridge(), param_grid=dict(alpha=lda_ray))
    ridge_gridsearch.fit(*split_data.train_data)
    optimal_ridge_lda = ridge_gridsearch.best_params_['alpha']

    lasso_gridsearch = GridSearchCV(estimator=Lasso(), param_grid=dict(alpha=lda_ray))
    with warnings.catch_warnings():
        # if the model can't find a good fit it produces an annoying warning
        warnings.simplefilter("ignore")
        lasso_gridsearch.fit(*split_data.train_data)
    optimal_lasso_lda = lasso_gridsearch.best_params_['alpha']


    linfit = LinFit(split_data)

    OLS_nosample_error_ray[i] = linfit.test_fit(linfit.get_beta())[0]
    ridge_nosample_error_ray[i] = linfit.test_fit(linfit.get_beta_ridge(optimal_ridge_lda))[0]
    lasso_nosample_error_ray[i] = linfit.test_fit(linfit.get_beta_lasso(optimal_lasso_lda))[0]


    bootstrap_resampler = Resampler(split_data, n*n - 1)
    OLS_bootstrap_error = 0
    ridge_bootstrap_error = 0
    lasso_bootstrap_error = 0
    while bootstrap_resampler.bootstrap():
        linfit.split_data = bootstrap_resampler.resampled_data
        OLS_bootstrap_error += linfit.test_fit(linfit.get_beta())[0]
        ridge_bootstrap_error += linfit.test_fit(linfit.get_beta_ridge(optimal_ridge_lda))[0]
        lasso_bootstrap_error += linfit.test_fit(linfit.get_beta_lasso(optimal_lasso_lda))[0]
    OLS_bootstrap_error_ray[i] = OLS_bootstrap_error / (n*n - 1)
    ridge_bootstrap_error_ray[i] = ridge_bootstrap_error / (n*n - 1)
    lasso_bootstrap_error_ray[i] = lasso_bootstrap_error / (n*n - 1)


    cross_validation_resampler = Resampler(split_data, 5)
    OLS_cross_validation_error = 0
    ridge_cross_validation_error = 0
    lasso_cross_validation_error = 0
    while cross_validation_resampler.cross_validation():
        linfit.split_data = cross_validation_resampler.resampled_data
        OLS_cross_validation_error += linfit.test_fit(linfit.get_beta())[0]
        ridge_cross_validation_error += linfit.test_fit(linfit.get_beta_ridge(optimal_ridge_lda))[0]
        lasso_cross_validation_error += linfit.test_fit(linfit.get_beta_lasso(optimal_lasso_lda))[0]
    OLS_cross_validation_error_ray[i] = OLS_cross_validation_error / 5
    ridge_cross_validation_error_ray[i] = ridge_cross_validation_error / 5
    lasso_cross_validation_error_ray[i] = lasso_cross_validation_error / 5

plt.figure(figsize=(10, 6), constrained_layout=True)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

plt.plot(degree_list, OLS_nosample_error_ray, ".--", color=colors[0], label="OLS no resampling")
plt.plot(degree_list, OLS_bootstrap_error_ray, "*--", color=colors[0], label="OLS bootstrap resampling")
plt.plot(degree_list, OLS_cross_validation_error_ray, "+--", color=colors[0], label="OLS cross-validation resampling")
plt.plot(degree_list, ridge_nosample_error_ray, ".--", color=colors[1], label="Ridge no resampling")
plt.plot(degree_list, ridge_bootstrap_error_ray, "*--", color=colors[1], label="Ridge bootstrap resampling")
plt.plot(degree_list, ridge_cross_validation_error_ray, "+--", color=colors[1], label="Ridge cross-validation resampling")
plt.plot(degree_list, lasso_nosample_error_ray, ".--", color=colors[2], label="Lasso no resampling")
plt.plot(degree_list, lasso_bootstrap_error_ray, "*--", color=colors[2], label="Lasso bootstrap resampling")
plt.plot(degree_list, lasso_cross_validation_error_ray, "+--", color=colors[2], label="Lasso cross-validation resampling")
plt.ylim(0, 1.1)
plt.legend()
plt.xlabel("polynomial degree []")
plt.ylabel("[]")
plt.title(f"MSE for various resampling- and regression methods, using optimal $\\lambda$, ${n}^2$ datapoints")
plt.savefig("imgs/franke/resampling_regression_comp.png", dpi=200)
plt.clf()