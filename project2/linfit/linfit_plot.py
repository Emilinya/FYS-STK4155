import numpy as np
from linfit_data import f
import matplotlib.pyplot as plt
import seaborn as sb


def get_shape(line):
    if line[0] != "¤":
        print(f"problem reading shape from line: {line}")
        exit()
    else:
        shape = []
        for val in line[2:].split(", "):
            if val[-1] == ")":
                if val[-2] == ",":
                    shape.append(int(val[:-2]))
                else:
                    shape.append(int(val[:-1]))
                return shape
            shape.append(int(val))


def mse(y_ray, y_approx_ray):
    return np.sum((y_ray - y_approx_ray)**2) / len(y_ray)


def plot_schedule_type(schedule_type: str, error_list):
    with open(f"linfit/data_{schedule_type.upper()}.dat", "r") as datafile:
        best_beta_grid = np.loadtxt(
            datafile, max_rows=get_shape(datafile.readline())[0]).T
        best_step_size_ray = np.loadtxt(
            datafile, max_rows=get_shape(datafile.readline())[0]).T
        step_size_ray = np.loadtxt(
            datafile, max_rows=get_shape(datafile.readline())[0])
        mean_grid = np.loadtxt(
            datafile, max_rows=get_shape(datafile.readline())[0]).T
        std_grid = np.loadtxt(
            datafile, max_rows=get_shape(datafile.readline())[0]).T

    x_ray = np.linspace(0, 1, 1000)
    f_ray = f(x_ray)

    plt.figure(tight_layout=True)
    plt.plot(x_ray, f_ray, label="f(x)")

    label_list = [
        "A", "P", "M",
        "S", "M + S",
    ]

    best_label = None
    best_beta = None
    lowest_MSE = float("infinity")

    for i, (label, beta) in enumerate(zip(label_list, best_beta_grid)):
        poly_ray = np.polynomial.Polynomial(beta)(x_ray)
        MSE = mse(f_ray, poly_ray)
        if label == "A":
            plt.plot(x_ray, poly_ray, label=f"{label} [MSE={MSE:.3g}]")
            error_list.append(("OLS", schedule_type, label, MSE, "N/A"))
        else:
            plt.plot(
                x_ray, poly_ray, label=f"{label} [MSE={MSE:.3g}] [$\\eta$={best_step_size_ray[i-1]:.3g}]")
            error_list.append(
                ("OLS", schedule_type, label, MSE, best_step_size_ray[i-1]))

            if MSE < lowest_MSE:
                lowest_MSE = MSE
                best_label = label
                best_beta = beta

    plt.legend()
    plt.ylim(-0.25, 0.75)
    plt.savefig(f"imgs/linfit/{schedule_type.lower()}_plot.svg")
    plt.clf()

    idxs = list(range(0, len(step_size_ray), 10)) + [len(step_size_ray)-1]
    small_step_size_ray = step_size_ray[idxs]
    small_mean_grid = mean_grid[:, idxs]
    small_std_grid = std_grid[:, idxs]

    sb.heatmap(
        np.log10(small_mean_grid), cmap="viridis", annot=True, vmax=1,
        xticklabels=[f"{np.log10(lr):.2f}" for lr in small_step_size_ray],
        yticklabels=label_list[1:],
        cbar_kws={'label': 'log10(MSE) []'}
    )
    plt.xlabel("log10(step size) []")
    plt.ylabel("solver type []")
    plt.savefig(f"imgs/linfit/{schedule_type.lower()}_mean.svg")
    plt.clf()

    sb.heatmap(
        small_std_grid/small_mean_grid, cmap="viridis", annot=True, vmax=2,
        xticklabels=[f"{np.log10(lr):.2f}" for lr in small_step_size_ray],
        yticklabels=label_list[1:],
        cbar_kws={'label': 'std/MSE []'}
    )
    plt.xlabel("log10(step size) []")
    plt.ylabel("solver type []")
    plt.savefig(f"imgs/linfit/{schedule_type.lower()}_std.svg")
    plt.clf()

    return best_label, best_beta


def nice_str(regression_type, schedule_type, step_type):
    nice_regression = regression_type[0].upper() + regression_type[1:]
    nice_schedule = schedule_type[0].upper() + schedule_type[1:]
    nice_step = step_type[0].upper() + step_type[1:]
    return f"{nice_regression}, {nice_schedule}, {nice_step}"


def plot_best_OLS(best_list):
    x_ray = np.linspace(0, 1, 1000)
    f_ray = f(x_ray)

    plt.figure(tight_layout=True)
    plt.plot(x_ray, f_ray, label="f(x)")

    schedules = ["plain", "adagrad", "rms_prop", "adam"]
    for schedule, (label, beta) in zip(schedules, best_list):
        poly_ray = np.polynomial.Polynomial(beta)(x_ray)
        MSE = mse(f_ray, poly_ray)
        plt.plot(x_ray, poly_ray,
                 label=f"{schedule} [MSE={MSE:.3g}]")

    plt.legend()
    plt.ylim(-0.25, 0.75)
    plt.savefig(f"imgs/linfit/best_OLS_plot.svg")
    plt.clf()


def plot_best_Ridge(best_list, error_list):
    with open(f"linfit/data_Ridge.dat", "r") as datafile:
        best_beta_grid = np.loadtxt(
            datafile, max_rows=get_shape(datafile.readline())[0]).T
        best_step_size_ray = np.loadtxt(
            datafile, max_rows=get_shape(datafile.readline())[0]).T
        best_lda_ray = np.loadtxt(
            datafile, max_rows=get_shape(datafile.readline())[0]).T
        step_size_ray = np.loadtxt(
            datafile, max_rows=get_shape(datafile.readline())[0])
        lda_ray = np.loadtxt(
            datafile, max_rows=get_shape(datafile.readline())[0])

    x_ray = np.linspace(0, 1, 1000)
    f_ray = f(x_ray)

    plt.figure(tight_layout=True)
    plt.plot(x_ray, f_ray, label="f(x)")


    schedules = ["plain", "adagrad", "rms_prop", "adam"]
    for schedule, (label, _), beta, step_size, lda in zip(schedules, best_list, best_beta_grid, best_step_size_ray, best_lda_ray):
        poly_ray = np.polynomial.Polynomial(beta)(x_ray)
        MSE = mse(f_ray, poly_ray)
        plt.plot(
            x_ray, poly_ray, label=f"{schedule} [MSE={MSE:.3g}] [$\\eta$={step_size:.3g}] [$\\lambda$={lda:.3g}]")
        error_list.append(
            ("Ridge", schedule, label, MSE, step_size))

    plt.legend()
    plt.ylim(-0.25, 0.75)
    plt.savefig(f"imgs/linfit/best_Ridge_plot.svg")
    plt.clf()


error_list = []
best_list = []
best_list.append(plot_schedule_type("plain", error_list))
best_list.append(plot_schedule_type("adagrad", error_list))
best_list.append(plot_schedule_type("rms_prop", error_list))
best_list.append(plot_schedule_type("adam", error_list))

plot_best_OLS(best_list)
plot_best_Ridge(best_list, error_list)


analytic_error = [error for _, _, step_type, error, _ in error_list if step_type == "A"][0]

error_list = [v for v in error_list if v[2] != "A"]
error_list.sort(key=lambda e: e[3])

max_str = max([len(nice_str(re, sc, st)) for re, sc, st, _, _ in error_list])

print(f"{'Type':^{max_str}} │ {'MSE':^7} │ Step Size")
print(f"{'Analytic':<{max_str}} │ {analytic_error:.5f} │ N/A")
for regression_type, schedule_type, step_type, error, step_size in error_list:
    print(f"{nice_str(regression_type, schedule_type, step_type):<{max_str}} │ {error:.5f} │ {step_size:.5f}")
