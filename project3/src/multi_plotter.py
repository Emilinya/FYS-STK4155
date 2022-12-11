import os
import numpy as np
import matplotlib.pyplot as plt
from num_solver_plotter import f2dx
from ffnn_solver_plotter import f2typ


def rank_types(err_type, data_files):
    ranks = []
    for folder_files in data_files:
        folder_name = folder_files[0].split("/")[1].split("_")[0]

        for data_file in folder_files:
            filename = data_file.split("/")[2]
            data = np.load(data_file)

            u_calc_grid = data[f"u_{folder_name}_grid"]
            u_ana_grid = data[f"u_ana_grid"]

            if folder_name == "num":
                label = f"numeric dx={f2dx(filename)}"
            elif folder_name == "poly":
                label = f"poly,degree={f2dx(filename)}"
            else:
                hn, tp = f2typ(filename)
                label = f"{folder_name} {hn=} tp={tp}"

            if err_type == "abs":
                score = np.mean(np.abs(u_ana_grid - u_calc_grid))
            else:
                score = np.mean(np.abs(u_ana_grid - u_calc_grid)/(np.abs(u_ana_grid)+1e-14))*100

            ranks.append((label, score))
    ranks.sort(key=lambda v: v[1])

    max_label_len = max([len(label) for label, _ in ranks])
    if err_type == "abs":
        print(f"{f'label':^{max_label_len}} │ {f'abs.err []':^7}")
    else:
        print(f"{f'label':^{max_label_len}} │ {f'rel.err [%]':^7}")
    for label, score in ranks:
        print(f"{label:<{max_label_len}} │ {score:>11.7f}")

def plottify(err_type, data_files):
    plt.figure(tight_layout=True, figsize=(10, 6))
    plt.ticklabel_format(scilimits=(-2, 2))
    for folder_files in data_files:
        folder_name = folder_files[0].split("/")[1].split("_")[0]

        for data_file in folder_files:
            filename = data_file.split("/")[2]
            data = np.load(data_file)

            u_calc_grid = data[f"u_{folder_name}_grid"]
            u_ana_grid = data[f"u_ana_grid"]
            t_ray = data["t_ray"]

            if err_type == "abs":
                err_t = np.mean(np.abs(u_ana_grid - u_calc_grid), axis=1)
            else:
                err_t = np.mean(np.abs(u_ana_grid - u_calc_grid)/(np.abs(u_ana_grid)+1e-14), axis=1)

            if folder_name == "num":
                marker = "-"
            elif folder_name == "ffnn":
                marker = ".-"
            elif folder_name == "poly":
                marker = "+-"
            else:
                marker = "*-"

            if folder_name == "num":
                label = f"numeric,dx={f2dx(filename)}"
            elif folder_name == "poly":
                label = f"poly,degree={f2dx(filename)}"
            else:
                hn, tp = f2typ(filename)
                label = f"{folder_name},{hn=},tp={tp}"

            plt.plot(t_ray, err_t, marker, label=label)

    _, ymax = plt.ylim()
    plt.ylim(1e-6, ymax)
    plt.legend(loc='lower center', bbox_to_anchor=(0.36, -0.01),
          ncol=2, fancybox=True)
    plt.yscale("log")
    plt.xlabel("t []")
    if err_type == "abs":
        plt.ylabel("absolute error []")
    else:
        plt.ylabel("relative error []")
    plt.savefig(f"imgs/multi/{err_type}_err.svg")
    plt.clf()

def main():
    data_folders = ["data/ffnn_solver", "data/num_solver", "data/pytorch_solver", "data/poly_solver"]
    data_files = [
        [data_folder+"/"+f for f in [v[2] for v in os.walk(data_folder)][0]]
        for data_folder in data_folders
    ]

    plottify("rel", data_files)
    plottify("abs", data_files)
    rank_types("rel", data_files)
    rank_types("abs", data_files)

if __name__ == "__main__":
    plt.rcParams['font.size'] = '12'
    main()
