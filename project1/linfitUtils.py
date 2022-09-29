import warnings
import numpy as np
from polyUtils import power_itr
from sklearn import linear_model
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

def normalize_ray(ray: np.ndarray):
    # normalize array in the same way as sklearn.preprocessing.StandardScaler
    std = np.std(ray)
    if (std != 0):
        return (ray - np.mean(ray)) / std
    else:
        return ray - np.mean(ray)

@dataclass
class Data:
    # simple struct containing X and y data
    X: np.ndarray 
    y: np.ndarray

    def normalize(self):
        self.y = normalize_ray(self.y)
        for i in range(self.X.shape[1]):
            self.X[:, i] = normalize_ray(self.X[:, i])

    def shape(self):
        return self.X.shape

    def copy(self):
        return Data(self.X.copy(), self.y.copy())

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i == 0:
            self.i += 1
            return self.X
        elif self.i == 1:
            self.i += 1
            return self.y
        else:
            raise StopIteration

@dataclass
class SplitData:
    # simple class containing test and training data, with som usefull constructors
    test_data: Data
    train_data: Data

    def train_shape(self):
        return self.train_data.shape()

    def test_shape(self):
        return self.test_data.shape()

    def copy(self):
        return SplitData(self.test_data.copy(), self.train_data.copy())

    @classmethod
    def from_data(cls, data, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(data.X, data.y, test_size=test_size)
        return cls(Data(X_test, y_test), Data(X_train, y_train))

    @classmethod
    def from_1d_polynomial(cls, poly_degree, x_ray, y_ray, normalize=False):
        n = len(x_ray)
        m = poly_degree+1
        X = np.zeros((n, m))
    
        for d in range(m):
            X[:, d] = x_ray**d
            
        data = Data(X, y_ray)
        if normalize:
            data.normalize()
            

        return cls.from_data(data)

    @classmethod
    def from_2d_polynomial(cls, poly_degree, x_grid, y_grid, z_grid, normalize=False):
        x_flat, y_flat = x_grid.flatten(), y_grid.flatten()
        n = len(x_flat)
        m = int((poly_degree + 2) * (poly_degree + 1) / 2)

        X = np.zeros((n, m))
        for i, (xd, yd) in enumerate(power_itr(poly_degree)):
            X[:, i] = x_flat**xd * y_flat**yd
        y = z_grid.flatten()
        
        data = Data(X, y)
        if normalize:
            data.normalize()

        return cls.from_data(data)

class LinFit:
    # a class for linear regression
    def __init__(self, split_data: SplitData):
        self.split_data = split_data

    def train_shape(self):
        return self.split_data.train_shape()

    def test_shape(self):
        return self.split_data.test_shape()

    def get_beta(self):
        X, y = self.split_data.train_data
        return np.linalg.pinv(X.T @ X) @ X.T @ y

    def get_beta_ridge(self, lda):
        X, y = self.split_data.train_data
        return np.linalg.pinv(X.T @ X + lda * np.identity(X.shape[1])) @ X.T @ y

    def get_beta_lasso(self, lda):
        X, y = self.split_data.train_data
        # We Lasso regression from sklearn
        model = linear_model.Lasso(alpha=lda, fit_intercept=False)
        with warnings.catch_warnings():
            # if the model can't find a good fit it produces an annoying warning
            warnings.simplefilter("ignore")
            model.fit(X, y)
        return model.coef_

    def get_fit(self, beta):
        if beta is None:
            beta = self.get_beta()
        X = self.split_data.test_data.X
        return X @ beta

    def test_fit(self, beta=None):
        y_fit = self.get_fit(beta)
        y_test = self.split_data.test_data.y

        n = len(y_test)
        average = np.sum(y_test) / n
        squared_error = np.sum((y_test - y_fit)**2)
        MSE = squared_error / n
        R2 = 1 - squared_error / np.sum((y_test - average)**2)
        return MSE, R2

    def get_train_fit(self, beta):
        if beta is None:
            beta = self.get_beta()
        X = self.split_data.train_data.X
        return X @ beta

    def test_train_fit(self, beta=None):
        y_fit = self.get_train_fit(beta)
        y_train = self.split_data.train_data.y

        n = len(y_train)
        average = np.average(y_train)
        squared_error = np.sum((y_train - y_fit)**2)
        MSE = squared_error / n
        R2 = 1 - squared_error / np.sum((y_train - average)**2)
        return MSE, R2

