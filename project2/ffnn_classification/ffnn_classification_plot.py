import numpy as np
import matplotlib.pyplot as plt

accuracy_ray = np.loadtxt("ffnn_classification/data.dat")

max_acc = np.max(accuracy_ray)
min_acc = np.min(accuracy_ray)
avg_acc = np.mean(accuracy_ray)
std_acc = np.std(accuracy_ray)

plt.figure(tight_layout=True)
plt.hist(accuracy_ray, density=True)
plt.text(.2, .9, f"Minimum accuracy: {min_acc*100:.1f} %", transform=plt.gca().transAxes)
plt.text(.2, .85, f"Maximum accuracy: {max_acc*100:.1f} %", transform=plt.gca().transAxes)
plt.text(.2, .8, f"Average accuracy:   {avg_acc*100:.1f} $\\pm$ {std_acc*100:.1f} %", transform=plt.gca().transAxes)
plt.xlabel("accuracy []")
plt.ylabel("probability density []")
plt.savefig("imgs/ffnn_classification/plot.svg")
