from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
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
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # Make data.
    x = np.linspace(0, 1, 1000)
    y = np.linspace(0, 1, 1000)

    x_grid, y_grid = np.meshgrid(x, y)
    z = frankeFunction(x_grid, y_grid)

    # Plot the surface.
    surf = ax.plot_surface(x, y, z, cmap="plasma", linewidth=0, antialiased=False)

    ax.set_xlabel("x []")
    ax.set_ylabel("y []")
    ax.set_zlabel("z []")

    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    cbar = fig.colorbar(surf)
    cbar.set_ticks(np.linspace(np.round(np.min(z), 1), np.round(np.max(z), 2), 10))

    ax.view_init(azim=30)
    plt.savefig("imgs/FrankeFunction.png", dpi=200)