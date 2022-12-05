import matplotlib.pyplot as plt
import numpy as np
import os


def get_data():
    files = [v[2] for v in os.walk("data/num_solver_data")][0]
    dxs = sorted([
        float(f.split("=")[1].split(".npy")[0])
        for f in files if "t_ray" in f
    ], reverse=True)

    return {
        dx:
        {
            "t_ray": np.load(f"data/num_solver_data/t_ray_dx={dx}.npy"),
            "ana_u_grid": np.load(f"data/num_solver_data/ana_u_grid_dx={dx}.npy"),
            "num_u_grid": np.load(f"data/num_solver_data/num_u_grid_dx={dx}.npy"),
        } for dx in dxs}


def main():
    data = get_data()
    for f in [0, 0.15, 1]:
        min_dx = list(data.keys())[-1]
        plt.figure(tight_layout=True)
        for dx, vals in data.items():
            ana_u_grid = vals["ana_u_grid"]
            num_u_grid = vals["num_u_grid"]
            Nt, Nx = ana_u_grid.shape
            idx = int(f * (Nt-1))

            x_ray = np.linspace(0, 1, Nx)

            if dx >= 1e-1:
                plt.plot(x_ray, num_u_grid[idx, :], ".-", label=f"{dx=}")
            else:
                plt.plot(x_ray, num_u_grid[idx, :], label=f"{dx=}")
            if dx == min_dx:
                plt.plot(x_ray, ana_u_grid[idx, :], "k--", label="analytic")
        plt.legend()
        plt.xlabel("x []")
        plt.ylabel("u []")
        plt.savefig(f"imgs/num_solver_imgs/{f=}_comp.svg")
        plt.clf()

        for dx, vals in data.items():
            ana_u_grid = vals["ana_u_grid"]
            num_u_grid = vals["num_u_grid"]
            Nt, Nx = ana_u_grid.shape
            idx = int(f * (Nt-1))

            x_ray = np.linspace(0, 1, Nx)
            rel_err = np.abs(
                ana_u_grid[idx, :] - num_u_grid[idx, :])/(np.abs(ana_u_grid[idx, :])+1e-14) * 100
            if dx >= 1e-1:
                plt.plot(x_ray[1:-1], rel_err[1:-1], ".-", label=f"{dx=}")
            else:
                plt.plot(x_ray[1:-1], rel_err[1:-1], label=f"{dx=}")

        plt.legend()
        plt.xlabel("x []")
        plt.ylabel("rel err [%]")
        plt.savefig(f"imgs/num_solver_imgs/{f=}_rel_err.svg")


if __name__ == "__main__":
    main()
