import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt


def get_len(line):
    return int(line.split()[0])


def cost(y_ray, y_approx):
    return np.mean((y_ray - y_approx)**2)


def plot_heatmap(name: str):
    with open(f"ffnn_regression/data_{name.upper().replace(' ', '_')}_grid.dat", "r") as datafile:
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
    plt.savefig(f"imgs/ffnn_regression/{name.replace(' ', '_')}_grid.svg")
    plt.clf()

    min_idx = np.unravel_index(error.argmin(), error.shape)

    best_lda = lda_ray[min_idx[0]]
    best_learning_rate = learning_rate_ray[min_idx[1]]

    return best_learning_rate, best_lda


def plot_FFNN_type(name: str, x_ray, y_ray, ols_ray, ffnn_ray):
    mse = cost(y_ray, ffnn_ray)

    plt.figure(tight_layout=True)
    plt.plot(x_ray, y_ray, label="f(x)")
    plt.plot(x_ray, ols_ray,
        label=f"OLS, degree 10 [MSE={cost(y_ray, ols_ray)*1e4:.1f}$\\cdot 10^{{-4}}$]"
    )
    plt.plot(x_ray, ffnn_ray,
        label=f"FFNN {name} [MSE={mse*1e4:.1f}$\\cdot 10^{{-4}}$]"
    )
    plt.xlabel("input []")
    plt.ylabel("output []")
    plt.legend()
    plt.savefig(f"imgs/ffnn_regression/{name.replace(' ', '_')}_plot.svg")
    plt.clf()

    return mse, *plot_heatmap(name)


x_ray, y_ray, sigmoid_ffnn_ray, RELU_ffnn_ray, leaky_RELU_ffnn_ray, ols_ray = np.loadtxt(
    "ffnn_regression/data.dat")
idxs = np.argsort(x_ray)
best_list = [("OLS", cost(y_ray, ols_ray), "N/A", "N/A")]

def plotify(name, array): 
    best_list.append((name, *plot_FFNN_type(
        name, x_ray[idxs], y_ray[idxs], ols_ray[idxs], array[idxs]
    )))

plotify("sigmoid", sigmoid_ffnn_ray)
plotify("RELU", RELU_ffnn_ray)
plotify("Leaky RELU", leaky_RELU_ffnn_ray)

best_list.sort(key=lambda v: v[1])
print(f"{'Type':^10} │ {'error*10^4':^10} │ {'learning rate':^13} │ {'lda':^9}")
for name, error, learning_rate, lda in best_list:
    error = error*10**4
    if name == "OLS":
        print(f"{name:^10} │ {error:.8f} │ {f'{learning_rate}':^13} │ {lda:^9}")
    else:
        print(f"{name:^10} │ {error:.8f} │ {f'{learning_rate:.8f}':^13} │ {lda:.3e}")
