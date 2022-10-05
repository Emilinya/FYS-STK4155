from sklearn.linear_model import Ridge
from polyUtils import poly_generator
from linfitUtils import SplitData
import matplotlib.pyplot as plt
from terrain import get_terrain
import numpy as np

np.random.seed(1337)
optimal_degree = 13
optimal_lda = 0.0283

lda_ray = np.logspace(-8, 4, 200)
x_grid, y_grid, z_grid = get_terrain(600)
split_data = SplitData.from_2d_polynomial(optimal_degree, x_grid, y_grid, z_grid, False)

model = Ridge(optimal_lda, fit_intercept=False)
model.fit(*split_data.train_data)
beta = model.coef_
poly = poly_generator(optimal_degree, beta)

fig = plt.figure(constrained_layout=True)
ax = plt.axes(projection='3d')
poly_z_grid = poly(x_grid, y_grid)

surf = ax.plot_surface(x_grid, y_grid, poly_z_grid, cmap="plasma", linewidth=0, antialiased=False)
ax.set_xlabel("x []")
ax.set_ylabel("y []")
ax.set_zlabel("z []")
cbar = fig.colorbar(surf)
# cbar.set_ticks(np.linspace(0, 1000, 11))
ax.view_init(azim=45)
plt.savefig("imgs/terrain/terrain_approximation.png", dpi=200)
ax.cla()
cbar.remove()


surf = ax.plot_surface(x_grid, y_grid, np.abs(z_grid - poly_z_grid), cmap="plasma", linewidth=0, antialiased=False)
ax.set_xlabel("x []")
ax.set_ylabel("y []")
ax.set_zlabel("z []")
cbar = fig.colorbar(surf)
# cbar.set_ticks(np.linspace(0, 1000, 11))
ax.view_init(azim=45)
plt.savefig("imgs/terrain/terrain_difference.png", dpi=200)
plt.clf()
