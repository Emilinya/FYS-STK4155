import numpy as np
from polyUtils import power_itr
from sklearn.model_selection import train_test_split

def complicated_train_test_split(X, z_grid, test_size):
    # If X is shape (100, 100, 4), train_test_split gives (20, 100, 4), not (20, 20, 4),
    # this function returns the correct (20, 20, 4) matrix

    X0, z0 = X[0], z_grid[0]
    _, _, z0_train, z0_test = train_test_split(X0, z0, test_size=test_size)
    trn, ten = len(z0_train), len(z0_test) 

    X_train, X_test = np.zeros((trn, trn, X.shape[2])), np.zeros((ten, ten, X.shape[2]))
    z_train, z_test = np.zeros((trn, trn)), np.zeros((ten, ten))

    trn_buckets, ten_buckets = trn, ten
    trn_i, ten_i = 0, 0
    for i in range(trn + ten):
        sub_X_train, sub_X_test, sub_z_train, sub_z_test = train_test_split(X[i], z_grid[i], test_size=0.2)
        selector = np.random.randint(0, trn_buckets + ten_buckets)
        if selector < trn_buckets:
            X_train[trn_i] = sub_X_train
            z_train[trn_i] = sub_z_train
            trn_i += 1
            trn_buckets -= 1
        else:
            X_test[ten_i] = sub_X_test
            z_test[ten_i] = sub_z_test
            ten_i += 1
            ten_buckets -= 1

    return X_train, X_test, z_train, z_test

def get_design_matrix(poly_degree, x_ray, y_ray, z_grid):
    X = np.array(
        [[[
            x**xd * y**yd for (xd, yd) in power_itr(poly_degree)
        ] for x in x_ray 
        ] for y in y_ray]
    )
    complicated_train_test_split(X, z_grid, test_size=0.2)
    exit()
    X_train, X_test, z_train, z_test = train_test_split(X, z_grid, test_size=0.2)

    return X_train, X_test, z_train, z_test

def get_beta(X, z):
    return np.linalg.pinv(np.tensordot(X.T, X)) @ np.tensordot(X.T, z)

def get_fit(X, beta):
    return np.tensordot(X, beta, axes=1)

def MSE_R2(z_dat, z_fit):
    n = len(z_dat.flat)
    average = np.sum(z_dat) / n
    squared_error = np.sum((z_dat - z_fit)**2)
    return squared_error / n, 1 - squared_error / np.sum((z_dat - average)**2)

def test_fit(z, X, beta):
    return MSE_R2(z, get_fit(X, beta))