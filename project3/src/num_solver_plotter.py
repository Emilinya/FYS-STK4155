import matplotlib.pyplot as plt
import numpy as np
import os


def f2dx(filename):
    """Find dx from filename."""
    return float(filename.split("=")[1].split(".npz")[0])


def main():
    datafiles = [v[2] for v in os.walk("data/num_solver")][0]
    datafiles.sort(key=f2dx, reverse=True)
    data_pairs = [(f2dx(f), np.load("data/num_solver/"+f)) for f in datafiles]
    min_dx = 1e-2

    # create plots of u(x, t) for three values of t
    for f in [0, np.log(2)/np.log(100), 1]:
        plt.figure(tight_layout=True)

        # we do not have infinite resolution, find the
        # t-value that f corresponds to in the lowest
        # resolution data array
        min_t_ray = data_pairs[0][1]["t_ray"]
        t = min_t_ray[int(f * (min_t_ray.size-1))]

        for dx, data in data_pairs:
            t_ray = data["t_ray"]
            u_ana_grid = data["u_ana_grid"]
            u_num_grid = data["u_num_grid"]
            _, Nx = u_ana_grid.shape

            x_ray = np.linspace(0, 1, Nx)
            idx = np.argmin(t_ray < t)

            if dx >= 1e-1:
                plt.plot(x_ray, u_num_grid[idx, :], ".-", label=f"{dx=}")
            else:
                plt.plot(x_ray, u_num_grid[idx, :], label=f"{dx=}")
            if dx == min_dx:
                plt.plot(x_ray, u_ana_grid[idx, :], "k--", label="analytic")

        plt.legend()
        plt.xlabel("x []")
        plt.ylabel("u []")
        plt.savefig(f"imgs/num_solver/{t=:.3g}_comp.svg")
        plt.clf()

    # create plots of abs_err(t)
    plt.figure(tight_layout=True)
    plt.ticklabel_format(scilimits=(-2, 2))
    for dx, data in data_pairs:
        t_ray = data["t_ray"]
        u_ana_grid = data["u_ana_grid"]
        u_num_grid = data["u_num_grid"]
        _, Nx = u_ana_grid.shape

        abs_err_t = np.mean(np.abs(u_ana_grid - u_num_grid), axis=1)
        plt.plot(t_ray, abs_err_t, label=f"{dx=}")

    plt.legend()
    plt.yscale("log")
    plt.xlabel("t []")
    plt.ylabel("absolute error []")
    plt.savefig(f"imgs/num_solver/abs_err.svg")
    plt.clf()

    # create plots of rel_err(t)
    plt.figure(tight_layout=True)
    plt.ticklabel_format(scilimits=(-2, 2))
    for dx, data in data_pairs:
        t_ray = data["t_ray"]
        u_ana_grid = data["u_ana_grid"]
        u_num_grid = data["u_num_grid"]
        _, Nx = u_ana_grid.shape

        rel_err_x = np.mean(np.abs(u_ana_grid - u_num_grid) /
                            (np.abs(u_ana_grid) + 1e-14), axis=1)
        plt.plot(t_ray, rel_err_x, label=f"{dx=}")

    plt.legend()
    plt.yscale("log")
    plt.xlabel("t []")
    plt.ylabel("relative error []")
    plt.savefig(f"imgs/num_solver/rel_err.svg")
    plt.clf()


if __name__ == "__main__":
    plt.rcParams['font.size'] = '14'
    main()
