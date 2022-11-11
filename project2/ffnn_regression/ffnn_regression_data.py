# from tensorflow.keras.layers import Input
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense           
# from tensorflow.keras import optimizers          
# from tensorflow.keras import regularizers     
import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# hack to include code from utils
import sys
sys.path.insert(1, 'utils')
from neural_network import NeuralNetwork, FunctionType
from linfit_utils import SplitData, LinFit


def f(x):
    term1 = 0.5*np.exp(-(9*x-7)**2/4.0)
    term2 = -0.2*np.exp(-(9*x-4)**2)
    return term1 + term2


def cost(y_ray, y_approx):
    return np.mean((y_ray - y_approx)**2)


def ffnn_predict(
    x_train, y_train, x_test, y_test,
    function_type: FunctionType, do_gridsearch = False
):
    neuron_counts = [10, 10, 10]
    functions = [function_type]*len(neuron_counts)
    neural_network = NeuralNetwork(
        x_train, y_train, functions=functions,
        learning_rate=0.005994842503189409,
        lda=5.994842503189409e-07, epochs=200,
        batch_size=20, hidden_neuron_counts=neuron_counts
    )

    resolution = 10
    learning_rate_ray = np.logspace(-4, 0, resolution)
    lda_ray = np.append([0], np.logspace(-8, 0, resolution))

    # Only do grid search when other hyperparameters have changed
    if do_gridsearch:
        error_grid = np.zeros((len(learning_rate_ray), len(lda_ray)))

        optimal_lambda = 0
        optimal_learning_rate = 0
        min_error = float("infinity")
        for i, learning_rate in enumerate(learning_rate_ray):
            for j, lda in enumerate(lda_ray):
                print(f"\r{j + i*len(lda_ray)}/{error_grid.size}", end="")
                neural_network.learning_rate = learning_rate
                neural_network.lda = lda

                neural_network.train(silent=True)
                error = cost(
                    neural_network.predict(x_test),
                    y_test.reshape(-1, 1)
                )
                error_grid[i, j] = error

                if error < min_error:
                    min_error = error
                    optimal_learning_rate = learning_rate
                    optimal_lambda = lda

                neural_network.create_biases_and_weights()
        print(
            f"\nOptimal learning rate: {optimal_learning_rate}, optimal lambda: {optimal_lambda}")


        with open(f"ffnn_regression/data_{function_type.name}_grid.dat", "w") as datafile:
            datafile.write(f"{error_grid.shape[0]} Error grid\n")
            np.savetxt(datafile, error_grid)
            datafile.write(f"{learning_rate_ray.size} Learning rates\n")
            np.savetxt(datafile, learning_rate_ray)
            datafile.write(f"{lda_ray.size} Lambdas\n")
            np.savetxt(datafile, lda_ray)

        neural_network.learning_rate = optimal_learning_rate
        neural_network.lda = optimal_lambda

    neural_network.epochs = 1000
    neural_network.train()
    return neural_network.predict(x_test).reshape(-1)


x_ray = np.linspace(0, 1, 500)
y_ray = f(x_ray)

split_data = SplitData.from_1d_polynomial(10, x_ray, y_ray, test_size=0.8)
x_train = split_data.train_data.X[:, 1].reshape(-1, 1)
y_train = split_data.train_data.y.reshape(-1, 1)
x_test = split_data.test_data.X[:, 1].reshape(-1, 1)
y_test = split_data.test_data.y.reshape(-1, 1)

# neural network approximations

do_gridsearch = True
sigmoid_ffnn_pred = ffnn_predict(x_train, y_train, x_test, y_test, FunctionType.SIGMOID, do_gridsearch)
RELU_ffnn_pred = ffnn_predict(x_train, y_train, x_test, y_test, FunctionType.RELU, do_gridsearch)
leaky_RELU_ffnn_pred = ffnn_predict(x_train, y_train, x_test, y_test, FunctionType.LEAKY_RELU, do_gridsearch)

# polynomial approximation

linfit = LinFit(split_data)
ols_pred = linfit.get_fit()

# tensorflow approximation | Disabled because of problems when importing tensorflow
# model = Sequential()
# model.add(Dense(neuron_counts[0], activation='sigmoid', kernel_regularizer=regularizers.l2(lda_ray)))
# model.add(Dense(neuron_counts[1], activation='sigmoid', kernel_regularizer=regularizers.l2(lda_ray)))
# model.add(Dense(neuron_counts[2], activation='sigmoid', kernel_regularizer=regularizers.l2(lda_ray)))
# model.add(Dense(1, activation='linear'))
# sgd = optimizers.SGD(lr=neural_network.learning_rate)
# model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
# model.fit(x_train, y_train, epochs=1000, batch_size=20, verbose=False)

np.savetxt(
    "ffnn_regression/data.dat",
    np.array([
        x_test.reshape(-1), y_test.reshape(-1), sigmoid_ffnn_pred, RELU_ffnn_pred,
        leaky_RELU_ffnn_pred, ols_pred
    ])
)
