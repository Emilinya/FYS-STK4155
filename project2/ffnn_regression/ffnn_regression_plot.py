import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt


def get_len(line):
    return int(line.split()[0])


def cost(y_ray, y_approx):
    return np.mean((y_ray - y_approx)**2)


x_ray, y_ray, sigmoid_ffnn_ray, RELU_ffnn_ray, leaky_RELU_ffnn_ray, ols_ray, ridge_ray = np.loadtxt("ffnn_regression/data.dat")

idxs = np.argsort(x_ray)

plt.figure(tight_layout=True)
plt.plot(x_ray[idxs], sigmoid_ffnn_ray[idxs],
    label=f"FFNN sigmoid [MSE={cost(y_ray, sigmoid_ffnn_ray)*1e4:.1f}$\\cdot 10^{{-4}}$]"
)
plt.plot(x_ray[idxs], RELU_ffnn_ray[idxs],
    label=f"FFNN RELU [MSE={cost(y_ray, RELU_ffnn_ray)*1e4:.1f}$\\cdot 10^{{-4}}$]"
)
plt.plot(x_ray[idxs], leaky_RELU_ffnn_ray[idxs],
    label=f"FFNN leaky RELU [MSE={cost(y_ray, leaky_RELU_ffnn_ray)*1e4:.1f}$\\cdot 10^{{-4}}$]"
)
plt.plot(x_ray[idxs], ols_ray[idxs],
    label=f"OLS, degree 10 [MSE={cost(y_ray, ols_ray)*1e4:.1f}$\\cdot 10^{{-4}}$]"
)
plt.plot(x_ray[idxs], ridge_ray[idxs],
    label=f"Ridge, degree 10 [MSE={cost(y_ray, ridge_ray)*1e4:.1f}$\\cdot 10^{{-4}}$]"
)
# plt.plot(x_ray[idxs], keras_nn_ray[idxs],
#     label=f"FFNN using tensorflow [MSE={cost(y_ray, keras_nn_ray)*1e4:.1f}$\\cdot 10^{{-4}}$]"
# )
plt.plot(x_ray[idxs], y_ray[idxs], "k--", label="funciton")

plt.xlabel("input []")
plt.ylabel("output []")
plt.legend()
plt.savefig("imgs/ffnn_regression/plot.svg")
plt.clf()

with open(f"ffnn_regression/data_grid.dat", "r") as datafile:
    error = np.loadtxt(
        datafile, max_rows=get_len(datafile.readline())).T
    learning_rate_ray = np.loadtxt(
        datafile, max_rows=get_len(datafile.readline())).T
    lda_ray = np.loadtxt(
        datafile, max_rows=get_len(datafile.readline())).T

plt.figure(tight_layout=True)
sb.heatmap(
    np.log10(error), cmap="viridis", annot=True, vmax=1,
    xticklabels=[f"{np.log10(lr):.2f}" for lr in learning_rate_ray],
    yticklabels=[f"{np.log10(lda):.2f}" for lda in lda_ray],
    cbar_kws={'label': 'log10(error) []'}
)
plt.yticks(rotation=0) 
plt.xlabel("log10(learning rate) []")
plt.ylabel("log10($\\lambda$) []")
plt.savefig(f"imgs/ffnn_regression/grid.svg")
plt.clf()
