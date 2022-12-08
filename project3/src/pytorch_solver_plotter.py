import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

data = np.load(f"data/pytorch_solver/hn=10_tp=SIGMOID.npz")

x_ray = data["x_ray"]
t_ray = data["t_ray"]
u_ffnn_grid = data["u_ffnn_grid"]
u_ana_grid = data["u_ana_grid"]

# plt.figure(tight_layout=True)
# sb.heatmap(
#     np.log10(data["loss_grid"]), cmap="viridis", annot=True, vmax=2,
#     xticklabels=[f"{np.log10(lr):.2f}" for lr in data["learning_rate_ray"]],
#     yticklabels=[f"{np.log10(lda):.2f}" for lda in data["lda_ray"]],
#     cbar_kws={'label': 'log10(error) []'}
# )
# plt.yticks(rotation=0)
# plt.xlabel("log10(learning rate) []")
# plt.ylabel("log10($\\lambda$) []")
# plt.savefig(f"imgs/pytorch_solver/loss_grid.svg")
# plt.clf()

Nt, Nx = u_ffnn_grid.shape
for f in [0, np.log(2)/np.log(100), 1]:
    idx = int(f * (Nt - 1))
    plt.figure(tight_layout=True)
    plt.plot(x_ray, u_ffnn_grid[idx, :], ".-", label="ffnn")
    plt.plot(x_ray, u_ana_grid[idx, :], "k--", label="analytical")
    plt.legend()
    plt.xlabel("x []")
    plt.ylabel("u []")
    plt.savefig(f"imgs/pytorch_solver/t={t_ray[idx]:.3g}comp.svg")
    plt.clf()

plt.figure(tight_layout=True)
plt.plot(t_ray, np.mean(np.abs(u_ffnn_grid - u_ana_grid), axis=1), ".-")
plt.ticklabel_format(scilimits=(-2, 2))
plt.xlabel("t []")
plt.ylabel("abs_diff []")
plt.savefig(f"imgs/pytorch_solver/abs_diff.svg")
plt.clf()

plt.figure(tight_layout=True)
plt.plot(t_ray, np.mean(np.abs(u_ffnn_grid - u_ana_grid)/(np.abs(u_ana_grid)+1e-14), axis=1), ".-")
plt.ticklabel_format(scilimits=(-2, 2))
plt.xlabel("t []")
plt.ylabel("rel_diff []")
plt.savefig(f"imgs/pytorch_solver/rel_diff.svg")
plt.clf()
