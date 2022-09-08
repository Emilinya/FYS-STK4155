import numpy as np
from sklearn.model_selection import train_test_split

def get_design_matrix(poly_degree, x_ray, y_ray):
    X = np.array(
        [[
            x**d for d in range(poly_degree+1)
        ] for x in x_ray]
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y_ray, test_size=0.2)

    return X_train, X_test, y_train, y_test

def get_beta(X, y):
    return np.linalg.pinv(X.T @ X) @ X.T @ y

def get_fit(X, beta):
    return X @ beta

def MSE_R2(y_dat, y_fit):
    n = len(y_dat.flat)
    average = np.sum(y_dat) / n
    squared_error = np.sum((y_dat - y_fit)**2)
    return squared_error / n, 1 - squared_error / np.sum((y_dat - average)**2)

def test_fit(z, X, beta):
    return MSE_R2(z, get_fit(X, beta))