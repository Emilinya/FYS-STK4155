import matplotlib.pyplot as plt
import numpy as np
import os


def f2d(filename):
    """Find polynomial degree from filename."""
    return float(filename.split("=")[1].split(".npz")[0])


def main():
    datafiles = [v[2] for v in os.walk("data/poly_solver")][0]
    datafiles.sort(key=f2d, reverse=True)
    data_pairs = [(f2d(f), np.load("data/poly_solver/"+f)) for f in datafiles]

    t_ray = data_pairs[0][1]["t_ray"]
    u_ana_grid = data_pairs[0][1]["u_ana_grid"]
    Nt, Nx = u_ana_grid.shape

    # create plots of u(x, t) for three values of t
    for f in [0, np.log(2)/np.log(100), 1]:
        plt.figure(tight_layout=True)

        idx = int(f * (Nt-1))
        t = t_ray[idx]

        for d, data in data_pairs:
            u_poly_grid = data["u_poly_grid"]

            x_ray = np.linspace(0, 1, Nx)

            plt.plot(x_ray, u_poly_grid[idx, :], ".-", label=f"degree={d}")
        plt.plot(x_ray, u_ana_grid[idx, :], "k--", label="analytic")

        plt.legend()
        plt.xlabel("x []")
        plt.ylabel("u []")
        plt.savefig(f"imgs/poly_solver/{t=:.3g}_comp.svg")
        plt.clf()

    # create plots of abs_err(t)
    plt.figure(tight_layout=True)
    plt.ticklabel_format(scilimits=(-2, 2))
    for d, data in data_pairs:
        u_poly_grid = data["u_poly_grid"]

        abs_err_t = np.mean(np.abs(u_ana_grid - u_poly_grid), axis=1)
        plt.plot(t_ray, abs_err_t, label=f"{d=}")

    plt.legend()
    plt.yscale("log")
    plt.xlabel("t []")
    plt.ylabel("absolute error []")
    plt.savefig(f"imgs/poly_solver/abs_err.svg")
    plt.clf()

    # create plots of rel_err(t)
    plt.figure(tight_layout=True)
    plt.ticklabel_format(scilimits=(-2, 2))
    for d, data in data_pairs:
        u_poly_grid = data["u_poly_grid"]

        rel_err_x = np.mean(np.abs(u_ana_grid - u_poly_grid) /
                            (np.abs(u_ana_grid) + 1e-14), axis=1)
        plt.plot(t_ray, rel_err_x, label=f"{d=}")

    plt.legend()
    plt.yscale("log")
    plt.xlabel("t []")
    plt.ylabel("relative error []")
    plt.savefig(f"imgs/poly_solver/rel_err.svg")
    plt.clf()


if __name__ == "__main__":
    plt.rcParams['font.size'] = '14'
    main()
