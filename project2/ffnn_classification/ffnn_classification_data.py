from matplotlib import test
from sklearn.datasets import load_breast_cancer
import numpy as np

# hack to include code from utils
import sys
sys.path.insert(1, 'utils')
from linfit_utils import Data, SplitData
from neural_network import FunctionType, NNType, NeuralNetwork


def accuracy(pred, true):
    return np.sum(pred == true) / len(true)


def gridsearch(neural_network: NeuralNetwork, split_data: SplitData, learning_rate_ray, lda_ray):
    optimal_lambda = 0
    optimal_learning_rate = 0
    min_error = float("infinity")
    for i, learning_rate in enumerate(learning_rate_ray):
        for j, lda in enumerate(lda_ray):
            print(f"\r{j + i*len(lda_ray)}/{len(learning_rate_ray)*len(lda_ray)} min loss = {min_error}", end="")
            neural_network.learning_rate = learning_rate
            neural_network.lda = lda

            neural_network.train(silent=True)
            pred = neural_network.predict(split_data.test_data.X).flatten()
            error = accuracy(pred, split_data.test_data.y.flatten())

            if error < min_error:
                min_error = error
                optimal_learning_rate = learning_rate
                optimal_lambda = lda

            neural_network.create_biases_and_weights()
    print(f"\nOptimal learning rate: {optimal_learning_rate}, lambda: {optimal_lambda}")

    neural_network.learning_rate = optimal_learning_rate
    neural_network.lda = optimal_lambda

cancer = load_breast_cancer()

inputs=cancer.data
outputs=cancer.target.astype("float")
labels=cancer.feature_names

# X = np.array([inputs[:,1], inputs[:,2], inputs[:,5], inputs[:,8]]).T
X = inputs
split_data = SplitData.from_data(Data(X, outputs.reshape(-1, 1)), test_size=0.1)

neuron_counts = [10, 10, 10]
functions = [FunctionType.SIGMOID]*len(neuron_counts)
neural_network = NeuralNetwork(
    *split_data.train_data, functions=functions,
    nn_type=NNType.CLASSIFICATION,
    learning_rate=0.005, lda=0,
    epochs=10, batch_size=20, hidden_neuron_counts=neuron_counts
)

pred = neural_network.predict(split_data.test_data.X).flatten()
print(f"initial accuracy: {accuracy(pred, split_data.test_data.y.flatten())*100:.1f} %")

# 
if False:
    gridsearch(
        neural_network, split_data, [0.005],
        np.append([0], np.logspace(-8, 0, 10)),
    )

neural_network.epochs=20

accuracy_ray = np.zeros(8) 
for i in range(len(accuracy_ray)):
    neural_network.create_biases_and_weights()
    neural_network.train()

    pred = neural_network.predict(split_data.test_data.X).flatten()
    accuracy_ray[i] = accuracy(pred, split_data.test_data.y.flatten())

np.savetxt("ffnn_classification/data.dat", accuracy_ray)
print(f"mean accuracy: {np.mean(accuracy_ray)*100:.1f} %")
print(f"min accuracy: {np.min(accuracy_ray)*100:.1f} %")
print(f"max accuracy: {np.max(accuracy_ray)*100:.1f} %")
