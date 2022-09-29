import numpy as np
from linfitUtils import *

from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

def print_compare(a, b):
    try:
        a[0]
        a = a.flatten()
        b = b.flatten()
        abss = [f"{abs(ai - bi):.3g}" for ai, bi in zip(a, b)]
        rels = ["0 %" if ai==0 and bi==0 else "∞ %" if bi==0 else f"{abs(ai - bi)/abs(bi)*100:.3g} %" for ai, bi in zip(a, b)]
        print(f"  abs:({' '.join(abss)}), rel:({', '.join(rels)})")
        return
    except:
        print(f"  abs:{abs(a - b):.3g}, rel:{abs(a - b)/abs(b)*100:.3g} %")

def f(x):
    term1 = 0.5*np.exp(-(9*x-7)**2/4.0)
    term2 = -0.2*np.exp(-(9*x-4)**2)
    return term1 + term2

x_ray = np.linspace(0, 1, 1000000)
y_ray = f(x_ray)
poly_degree = 4

split_data = SplitData.from_1d_polynomial(poly_degree, x_ray, y_ray, normalize=False)


# no scaling


# my program

my_model = LinFit(split_data)
my_beta = my_model.get_beta()
my_MSE, my_R2 = my_model.test_fit(my_beta)

# sklearn

sklearn_model = linear_model.LinearRegression(fit_intercept=False)
sklearn_model.fit(*split_data.train_data)
sklearn_beta = sklearn_model.coef_
sklearn_fit = sklearn_model.predict(split_data.test_data.X)
sklearn_MSE = mean_squared_error(split_data.test_data.y, sklearn_fit)
sklearn_R2 = r2_score(split_data.test_data.y, sklearn_fit)

# comp

print("no scaling differences:")
print(" parameters:")
print_compare(my_beta, sklearn_beta)
print(" MSE:")
print_compare(my_MSE, sklearn_MSE)
print(" R2:")
print_compare(my_R2, sklearn_R2)


# scaling


# sklearn

Xscaler = StandardScaler()
Xscaler.fit(split_data.train_data.X)
X_train_scaled = Xscaler.transform(split_data.train_data.X)
X_test_scaled = Xscaler.transform(split_data.test_data.X)

Yscaler = StandardScaler()
Yscaler.fit(split_data.train_data.y.reshape(-1, 1))
y_train_scaled = Yscaler.transform(split_data.train_data.y.reshape(-1, 1))
y_test_scaled = Yscaler.transform(split_data.test_data.y.reshape(-1, 1))

model = linear_model.LinearRegression(fit_intercept=False)
model.fit(X_train_scaled, y_train_scaled)
sklearn_beta = model.coef_
sklearn_fit = model.predict(X_test_scaled)
sklearn_MSE = mean_squared_error(y_test_scaled, sklearn_fit)
sklearn_R2 = r2_score(y_test_scaled, sklearn_fit)

# my program

my_model.split_data.train_data.normalize()
my_model.split_data.test_data.normalize()
my_beta = my_model.get_beta()
my_MSE, my_R2 = my_model.test_fit(my_beta)

# comp

print("\nscaling differences:")
print(" parameters:")
print_compare(my_beta, sklearn_beta)
print(" MSE:")
print_compare(my_MSE, sklearn_MSE)
print(" R2:")
print_compare(my_R2, sklearn_R2)
