import os
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt


def f2typ(filename):
    """Find hidden neuron count and activation function from filename."""
    hidden_neurons, *activation_function = filename.split(".npz")[0].split("_")
    activation_function = "_".join(activation_function)

    return int(hidden_neurons.split("=")[1]), activation_function.split("=")[1]


def main():
    datafiles = [v[2] for v in os.walk("data/ffnn_solver")][0]
    data_pairs = [
        (f2typ(f), np.load("data/ffnn_solver/"+f)) for f in datafiles
    ]

    x_ray = data_pairs[0][1]["x_ray"]
    t_ray = data_pairs[0][1]["t_ray"]
    u_ana_grid = data_pairs[0][1]["u_ana_grid"]

    _, Nt = x_ray.size, t_ray.size

    # create plots of u(x, t) for three values of t
    for f in [0, np.log(2)/np.log(100), 1]:
        plt.figure(tight_layout=True)

        idx = int(f * (Nt-1))
        for (hn, tp), data in data_pairs:
            u_ffnn_grid = data["u_ffnn_grid"]
            plt.plot(x_ray, u_ffnn_grid[idx, :], ".-", label=f"{hn=} {tp=}")

        plt.plot(x_ray, u_ana_grid[idx, :], "k--", label="analytic")

        plt.legend()
        plt.xlabel("x []")
        plt.ylabel("u []")
        plt.savefig(f"imgs/ffnn_solver/t={t_ray[idx]:.3g}_comp.svg")
        plt.clf()

    # create plots of abs_err(t)
    plt.figure(tight_layout=True)
    plt.ticklabel_format(scilimits=(-2, 2))
    for (hn, tp), data in data_pairs:
        u_ffnn_grid = data["u_ffnn_grid"]

        abs_err_t = np.mean(np.abs(u_ana_grid - u_ffnn_grid), axis=1)
        plt.plot(t_ray, abs_err_t, ".-", label=f"{hn=} {tp=}")

    plt.legend()
    plt.yscale("log")
    plt.xlabel("t []")
    plt.ylabel("absolute error []")
    plt.savefig(f"imgs/ffnn_solver/abs_err.svg")
    plt.clf()

    # create plots of rel_err(t)
    plt.figure(tight_layout=True)
    plt.ticklabel_format(scilimits=(-2, 2))
    for (hn, tp), data in data_pairs:
        u_ffnn_grid = data["u_ffnn_grid"]

        rel_err_x = np.mean(np.abs(u_ana_grid - u_ffnn_grid) /
                            (np.abs(u_ana_grid) + 1e-14), axis=1)
        plt.plot(t_ray, rel_err_x, ".-", label=f"{hn=} {tp=}")

    plt.legend()
    plt.yscale("log")
    plt.xlabel("t []")
    plt.ylabel("relative error []")
    plt.savefig(f"imgs/ffnn_solver/rel_err.svg")
    plt.clf()

    # create hyperparameter heatmaps
    plt.figure(tight_layout=True)
    for (hn, tp), data in data_pairs:
        sb.heatmap(
            np.log10(data["loss_grid"]), fmt=".2f", cmap="viridis", annot=True, vmax=2,
            xticklabels=[
                f"{np.log10(lr):.2f}" for lr in data["learning_rate_ray"]
            ],
            yticklabels=[f"{np.log10(lda):.2f}" for lda in data["lda_ray"]],
            cbar_kws={'label': 'log10(error) []'}
        )
        plt.yticks(rotation=0)
        plt.xlabel("log10(learning rate) []")
        plt.ylabel("log10($\\lambda$) []")
        plt.savefig(f"imgs/ffnn_solver/{hn=}_{tp=}_loss.svg")
        plt.clf()


if __name__ == "__main__":
    plt.rcParams['font.size'] = '14'
    main()
