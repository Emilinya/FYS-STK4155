import numpy as np
from polyUtils import power_itr
from sklearn.model_selection import train_test_split

def get_design_matrix(poly_degree, x_grid, y_grid, z_grid):
    X = np.array(
        [[
            x**xd * y**yd for xd, yd in power_itr(poly_degree)
        ] for x, y in zip(x_grid.flat, y_grid.flat)]
    )
    X_train, X_test, y_train, y_test = train_test_split(X, z_grid.flat, test_size=0.2)

    return X_train, X_test, y_train, y_test

def get_beta(X, y):
    return np.linalg.solve(X.T @ X, X.T @ y)

def get_fit(X, beta):
    return X @ beta

def MSE_R2(y_dat, y_fit):
    n = len(y_dat)
    average = np.sum(y_dat) / n
    squared_error = np.sum((y_dat - y_fit)**2)
    return squared_error / n, 1 - squared_error / np.sum((y_dat - average)**2)

def test_fit(z, X, beta):
    return MSE_R2(z, get_fit(X, beta))