from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from imageio.v2 import imread
import numpy as np

def get_terrain(sample_num):
    terrain = imread('SRTM_data.tif')
    n, m = terrain.shape

    if sample_num != -1:
        terrain = terrain[::n//(sample_num), ::m//sample_num]
        n, m = terrain.shape

    x = np.linspace(0, 1, m)
    y = np.linspace(0, 1, n)
    x_grid, y_grid = np.meshgrid(x,y)

    return x_grid, y_grid, terrain

if __name__ == "__main__":
    fig = plt.figure(constrained_layout=True)
    ax = plt.axes(projection='3d')

    # get data
    x_grid, y_grid, z_grid = get_terrain()

    # plot the surface
    surf = ax.plot_surface(x_grid, y_grid, z_grid, cmap="plasma", linewidth=0, antialiased=False)

    # set labels
    ax.set_xlabel("x []")
    ax.set_ylabel("y []")
    ax.set_zlabel("z []")

    # add a color bar which maps values to colors.
    cbar = fig.colorbar(surf)
    cbar.set_ticks(np.linspace(0, 1100, 11))

    # rotate view
    ax.view_init(azim=45)
    
    plt.savefig("imgs/terrain/terrain.png", dpi=200)
