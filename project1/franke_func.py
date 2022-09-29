from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

def frankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def initialize_franke(n):
    x_ray = np.linspace(0, 1, n)
    x_grid, y_grid = np.meshgrid(x_ray, x_ray)
    return x_grid, y_grid, frankeFunction(x_grid, y_grid)

if __name__ == "__main__":
    fig = plt.figure(constrained_layout=True)
    ax = plt.axes(projection='3d')

    # make data
    x_grid, y_grid, z_grid = initialize_franke(1000)

    # plot the surface.
    surf = ax.plot_surface(x_grid, y_grid, z_grid, cmap="plasma", linewidth=0, antialiased=False)

    # set labels
    ax.set_xlabel("x []")
    ax.set_ylabel("y []")
    ax.set_zlabel("z []")

    # add a color bar which maps values to colors.
    cbar = fig.colorbar(surf)
    cbar.set_ticks(np.linspace(np.ceil(np.min(z_grid)*100)/100, np.floor(np.max(z_grid)*100)/100, 10))

    ax.view_init(azim=30)
    plt.savefig("imgs/franke/FrankeFunction.png", dpi=200)
