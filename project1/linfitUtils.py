import numpy as np
from tqdm import trange
from polyUtils import power_itr
from sklearn.model_selection import train_test_split

def get_design_matrix(poly_degree, x_grid, y_grid, z_grid):
    x_flat, y_flat = x_grid.flatten(), y_grid.flatten()
    n = len(x_flat)
    m = int((poly_degree + 2) * (poly_degree + 1) / 2)

    X = np.zeros((n, m))
    for i, (xd, yd) in enumerate(power_itr(poly_degree)):
        X[:, i] = x_flat**xd * y_flat**yd
        
    X_train, X_test, y_train, y_test = train_test_split(X, z_grid.flat, test_size=0.2)

    return X_train, X_test, y_train, y_test

def get_beta(X, y):
    return np.linalg.solve(X.T @ X, X.T @ y)

def get_beta_ridge(X, y, lda):
    return np.linalg.solve(X.T @ X + lda * np.identity(X.shape[1]), X.T @ y)

def get_fit(X, beta):
    return X @ beta

def MSE_R2(y_dat, y_fit):
    n = len(y_dat)
    average = np.sum(y_dat) / n
    squared_error = np.sum((y_dat - y_fit)**2)
    MSE = squared_error / n
    R2 = 1 - squared_error / np.sum((y_dat - average)**2)
    return MSE, R2

def test_fit(z, X, beta):
    return MSE_R2(z, get_fit(X, beta))