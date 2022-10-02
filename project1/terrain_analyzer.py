from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge, Lasso
from linfitUtils import LinFit, SplitData
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

minimum_errors = []

bootstrap_itrs = 50
for poly_degree in tqdm(degree_list):
    split_data = SplitData.from_2d_polynomial(poly_degree, x_grid, y_grid, z_grid, True)
    linfit = LinFit(split_data)
    resampler = Resampler(split_data, bootstrap_itrs)

    # find optimal Ridge lambda
    ridge_gridsearch = GridSearchCV(estimator=Ridge(), param_grid=dict(alpha=lda_ray))
    ridge_gridsearch.fit(*split_data.train_data)
    optimal_ridge_lda = ridge_gridsearch.best_params_['alpha']

    # find optimal Lasso lambda, supressing warnings
    lasso_gridsearch = GridSearchCV(estimator=Lasso(), param_grid=dict(alpha=lda_ray))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lasso_gridsearch.fit(*split_data.train_data)
    optimal_lasso_lda = lasso_gridsearch.best_params_['alpha']

    # calculate MSE for each regression type
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

    # find which regression type is best
    if OLS_error < ridge_error and OLS_error < lasso_error:
        # OLS is smallest
        minimum_errors.append({
            'MSE': OLS_error,
            'type': 'OLS',
            'lambda': 0
        })
    elif OLS_error < ridge_error:
        # Lasso is smallest
        minimum_errors.append({
            'MSE': lasso_error,
            'type': 'Lasso',
            'lambda': optimal_lasso_lda
        })
    else:
        # Ridge is smallest
        minimum_errors.append({
            'MSE': ridge_error,
            'type': 'Ridge',
            'lambda': optimal_ridge_lda
        })
    minimum_errors[-1]['degree'] = poly_degree

min_err = minimum_errors[np.argmin([err['MSE'] for err in minimum_errors])]

plt.figure(figsize=(7, 5), constrained_layout=True)
plt.plot([0, max(degree_list)], [min_err['MSE'], min_err['MSE']], "k--", linewidth=1, label="Min MSE")
plt.plot(degree_list, OLS_error_list, ".-", label="OLS MSE")
plt.plot(degree_list, ridge_error_list, ".-", label="Ridge MSE")
plt.plot(degree_list, lasso_error_list, ".-", label="Lasso MSE")

def label(minmum):
    plt.plot([minmum['degree'], minmum['degree']], [0, minmum['MSE']], "k--", linewidth=1)
    if minmum['type'] != "OLS":
        plt.text(minmum['degree']+0.1, minmum['MSE']*(0.5 - 0.1), f"$\\lambda_{{{minmum['type'][0]}}}$={minmum['lambda']:.3g}")
label(minimum_errors[6])
label(minimum_errors[13])
label(minimum_errors[16])

plt.xticks(degree_list)
plt.legend()
plt.ylim(0, OLS_error_list[0]*1.1)
plt.xlabel("polynomial degree []")
plt.ylabel("[]")
plt.title(f"MSE for terrain fit with increasing polynomial degree, using ${n}$ datapoints")
plt.savefig("imgs/terrain/terrain_MSE.png", dpi=200)
plt.clf()
