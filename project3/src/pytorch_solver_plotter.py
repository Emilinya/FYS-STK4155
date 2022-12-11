import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

def main():
    data = np.load(f"data/pytorch_solver/hn=10_tp=SIGMOID.npz")

    x_ray = data["x_ray"]
    t_ray = data["t_ray"]
    u_pytorch_grid = data["u_pytorch_grid"]
    u_ana_grid = data["u_ana_grid"]

    Nt, Nx = u_pytorch_grid.shape
    for f in [0, np.log(2)/np.log(100), 1]:
        idx = int(f * (Nt - 1))
        plt.figure(tight_layout=True)
        plt.plot(x_ray, u_pytorch_grid[idx, :], ".-", label="pytorch")
        plt.plot(x_ray, u_ana_grid[idx, :], "k--", label="analytical")
        plt.legend()
        plt.xlabel("x []")
        plt.ylabel("u []")
        plt.savefig(f"imgs/pytorch_solver/t={t_ray[idx]:.3g}_comp.svg")
        plt.clf()

    plt.figure(tight_layout=True)
    plt.ticklabel_format(scilimits=(-2, 2))
    plt.plot(t_ray, np.mean(np.abs(u_pytorch_grid - u_ana_grid), axis=1), ".-")
    plt.yscale("log")
    plt.xlabel("t []")
    plt.ylabel("absolute error []")
    plt.savefig(f"imgs/pytorch_solver/abs_err.svg")
    plt.clf()

    plt.figure(tight_layout=True)
    plt.ticklabel_format(scilimits=(-2, 2))
    plt.plot(t_ray, np.mean(np.abs(u_pytorch_grid - u_ana_grid)/(np.abs(u_ana_grid)+1e-14), axis=1), ".-")
    plt.yscale("log")
    plt.xlabel("t []")
    plt.ylabel("relative error []")
    plt.savefig(f"imgs/pytorch_solver/rel_err.svg")
    plt.clf()

if __name__ == "__main__":
    plt.rcParams['font.size'] = '14'
    main()
