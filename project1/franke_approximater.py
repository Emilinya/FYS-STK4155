from franke_func import frankeFunction
import matplotlib.pyplot as plt
import numpy as np

def power_itr(d):
    for c in range(d+1):
        for y in range(c+1):
            for x in range(c+1):
                if x + y == c:
                    yield (x, y)

def test_power_itr(d):
    power_strs = []
    for (x, y) in power_itr(d):
        if x == 0 and y == 0:
            power_strs.append("1")
            continue
        xsym = f"x^{x}" if x > 1 else "x" if x == 1 else ""
        ysym = f"y^{y}" if y > 1 else "y" if y == 1 else ""
        power_strs.append(xsym+ysym)

    print(f"P_{d}(x, y) = " + " + ".join(power_strs))

def poly_generator(d, beta):
    if len(list(poly_generator(d))) != len(beta):
        print(">:(")
        exit(1)

    def p(x, y):
        return np.sum([beta[i] * x**xd * y**yd for i, (xd, yd) in enumerate(power_itr(d))])
    p = np.vectorize(p)
    return p

def bet_beta(X, z):
    return np.linalg.pinv(np.tensordot(X.T, X)) @ np.tensordot(X.T, z_grid)

def get_fit(X, beta):
    return np.tensordot(X, beta, axes=1)

def MSE_R2(z_dat, z_fit):
    n = len(z_dat.flat)
    average = np.sum(z_dat) / n
    squared_error = np.sum((z_dat - z_fit)**2)
    return squared_error / n, 1 - squared_error / np.sum((z_dat - average)**2)

def get_MSE_R2(poly_degree, x_ray, y_ray, z_grid):
    X = np.array(
        [[[
            x**xd * y**yd for (xd, yd) in power_itr(poly_degree)
        ] for x in x_ray 
        ] for y in y_ray]
    )


    beta = bet_beta(X, z_grid)
    z_approx = get_fit(X, beta)

    return MSE_R2(z_grid, z_approx)

n = 100
x_ray = np.linspace(0, 1, n)
y_ray = np.linspace(0, 1, n)
x_grid, y_grid = np.meshgrid(x_ray, y_ray)
z_grid = frankeFunction(x_grid, y_grid)

max_degree = 5
degree_list = list(range(max_degree+1))
MSE_list = []
R2_list = []
for poly_degree in degree_list:
    MSE, R2 = get_MSE_R2(poly_degree, x_ray, y_ray, z_grid)
    MSE_list.append(MSE)
    R2_list.append(R2)

plt.plot(degree_list, MSE_list, label="MSE")
plt.plot(degree_list, R2_list, label="R2")
plt.xlabel("polynomial degree []")
plt.ylabel("y []")
plt.legend()
plt.savefig("plot.png", dpi=200)
