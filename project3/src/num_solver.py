from scipy.sparse import diags
from utils import Timer
import numpy as np


def init(dx, L, T):
    dt = 0.5 * dx ** 2  # stability condition

    Nx = int(round(L / dx, 10)) + 1
    Nt = int(round(T / dt, 10)) + 1

    return dx, dt, Nx, Nt


def tridiag(n, a, b, c):
    k = [np.full(n-1, a), np.full(n, b), np.full(n-1, c)]
    return diags(k, [-1, 0, 1])


def heat_solver(dx, L, T, u0):
    dx, dt, Nx, Nt = init(dx, L, T)

    u_grid = np.zeros((Nt, Nx))
    t_ray = np.zeros(Nt)

    assert u0[0] == u0[-1] == 0

    u_grid[0, :] = u0

    alpha = dt / (dx**2)  # = 0.5
    A = tridiag(Nx-2, alpha, 1-2*alpha, alpha)
    if Nx < 90:
        # for small values of Nx, the dense matrix representation is faster
        A = A.toarray()

    for i in range(Nt-1):
        u_grid[i+1, 1:-1] = A.dot(u_grid[i, 1:-1])
        t_ray[i+1] = t_ray[i] + dt

    return t_ray, u_grid


def get_anasol(dx, T):
    def u(x, t):
        return np.exp(-np.pi**2 * t)*np.sin(np.pi * x)

    dx, _, Nx, Nt = init(dx, 1, T)

    x_ray = np.linspace(0, 1, Nx)
    t_ray = np.linspace(0, T, Nt)

    return u(*np.meshgrid(x_ray, t_ray))


def main():
    timer = Timer()
    T = np.log(100) / (np.pi**2)

    for dx in [1e-1, 1e-2]:
        print(f"dx={dx}")
        u0 = np.sin(np.pi*np.linspace(0, 1, int(1 / dx) + 1))
        u0[-1] = 0

        timer.restart()
        t_ray, u_num_grid = heat_solver(dx, 1, T, u0)
        print(f"  time: {timer.get_pretty()}")

        u_ana_grid = get_anasol(dx, T)

        np.savez(f"data/num_solver/dx={dx}.npz", t_ray=t_ray,
                 u_num_grid=u_num_grid, u_ana_grid=u_ana_grid)


if __name__ == "__main__":
    main()
