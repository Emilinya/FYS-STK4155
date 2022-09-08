from linfitUtils import get_design_matrix, get_beta, test_fit
import matplotlib.pyplot as plt
import numpy as np

def simplefunc(x):
    term1 = 0.75*np.exp(-0.25*(9*x-2)**2)
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0)
    return term1 + term2

def plot_MSE_R2_fig(x_ray, y_ray, max_degree, filename, scaling=False):
    degree_list = list(range(max_degree+1))
    MSE_train_list = []
    R2_train_list = []
    MSE_test_list = []
    R2_test_list = []

    if scaling:
        # first subtract mean
        x_ray = x_ray - np.average(x_ray)
        y_ray = y_ray - np.average(y_ray)

        # then scale such that max = 1
        x_ray = x_ray / np.max(x_ray)
        y_ray = y_ray / np.max(y_ray)

    for poly_degree in degree_list:
        X_train, X_test, y_train, y_test = get_design_matrix(poly_degree, x_ray, y_ray)
        beta = get_beta(X_train, y_train)
        
        MSE_train, R2_train = test_fit(y_train, X_train, beta)
        MSE_train_list.append(MSE_train)
        R2_train_list.append(R2_train)

        MSE_test, R2_test = test_fit(y_test, X_test, beta)
        MSE_test_list.append(MSE_test)
        R2_test_list.append(R2_test)

    plt.plot(degree_list, MSE_train_list, label="training MSE")
    plt.plot(degree_list, R2_train_list, label="training R2")
    plt.plot(degree_list, MSE_test_list, label="testing MSE")
    plt.plot(degree_list, R2_test_list, label="testing R2")
    plt.xlabel("polynomial degree []")
    plt.ylabel("y []")
    plt.legend()
    plt.savefig(filename, dpi=200)
    plt.clf()

    return degree_list, R2_test_list

n = 100000
x_ray = np.linspace(0, 1, n)
y_ray = simplefunc(x_ray)

degree_list, R2_test_noscale = plot_MSE_R2_fig(x_ray, y_ray, 6, "imgs/MSE_R2_noscale.png")
_, R2_test_scale = plot_MSE_R2_fig(x_ray, y_ray, 6, "imgs/MSE_R2_scale.png", scaling=True)
plt.plot(degree_list, np.array(R2_test_noscale) - np.array(R2_test_scale), label="R2 no scale - R2 scale")
plt.legend()
plt.savefig("imgs/test.png", dpi=200)
