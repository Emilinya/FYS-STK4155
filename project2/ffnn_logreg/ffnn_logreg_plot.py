from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np

digits = load_digits()

inputs=digits.images
labels=digits.target

def plot_FFNN():
    with open("ffnn_logreg/data.dat", "r") as datafile:
        n_wrongs = int(datafile.readline())
        wrong_labels = np.loadtxt(datafile, max_rows=n_wrongs).astype(int)
        wrong_idxs = np.loadtxt(datafile).astype(int)

    acc = (1 - len(wrong_labels) / (0.1*len(labels)))*100

    fig, axs = plt.subplots(3, 3, figsize=(6, 8), tight_layout=True)
    plt.suptitle(f"""\
    Accuracy: {int(0.1*len(labels)) - len(wrong_labels)} / {int(0.1*len(labels))} = {acc:.1f} %

    Wrongly predicted characters:
    """)
    for i in range(3):
        for j in range(3):
            idx = i*3 + j
            axs[i,j].axis('off')
            if idx < len(wrong_idxs):
                wrong_idx = wrong_idxs[idx]
                axs[i,j].imshow(inputs[wrong_idx], cmap=plt.cm.gray_r, interpolation='nearest')
                axs[i,j].set_title(f"Label: {labels[wrong_idx]}, Predicted: {wrong_labels[idx]}")
    plt.savefig("imgs/ffnn_logreg/chars.png", dpi=200)

def plot_grad():
    def get_len(line):
        return int(line.split()[0])

    with open(f"ffnn_logreg/data_grad_grid_rms.dat", "r") as datafile:
        error = np.loadtxt(
            datafile, max_rows=get_len(datafile.readline())).T
        learning_rate_ray = np.loadtxt(
            datafile, max_rows=get_len(datafile.readline())).T
        lda_ray = np.loadtxt(
            datafile, max_rows=get_len(datafile.readline())).T

    plt.figure(tight_layout=True)
    sb.heatmap(
        np.log10(error), cmap="viridis", annot=True, vmax=1,
        xticklabels=[f"{np.log10(lr):.2f}" for lr in learning_rate_ray],
        yticklabels=[f"{np.log10(lda):.2f}" for lda in lda_ray],
        cbar_kws={'label': 'log10(error) []'}
    )
    plt.yticks(rotation=0)
    plt.xlabel("log10(learning rate) []")
    plt.ylabel("log10($\\lambda$) []")
    plt.savefig(f"imgs/ffnn_logreg/grad_error_grid.svg")
    plt.clf()

plot_FFNN()
plot_grad()
