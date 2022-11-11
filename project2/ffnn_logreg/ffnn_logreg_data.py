from sklearn.datasets import load_digits
import numpy as np

# hack to include code from utils
import sys
sys.path.insert(1, 'utils')
from linfit_utils import Data, SplitData
from neural_network import NeuralNetwork, NNType, FunctionType
from gradient_descent import GradSolver, StepType, ScheduleType


def to_categorical(integer_vector):
    n_inputs = len(integer_vector)
    n_categories = np.max(integer_vector) + 1
    onehot_vector = np.zeros((n_inputs, n_categories))
    onehot_vector[range(n_inputs), integer_vector] = 1
    
    return onehot_vector


def get_accuracy(pred, true):
    return np.sum(pred == true) / len(true)

digits = load_digits()

inputs=digits.images
labels=digits.target

X = inputs.reshape(-1, 8*8)
y = to_categorical(labels)
idxs = np.arange(len(y))

split_data = SplitData.from_data(Data(X, idxs), test_size=0.1)
train_idxs = split_data.train_data.y
test_idxs = split_data.test_data.y

split_data.train_data.y = y[train_idxs]
split_data.test_data.y = y[test_idxs]

test_labels = np.argmax(split_data.test_data.y, axis=1)

# gradient descent regression

def cost_function_generator(X_grid, y_ray):
    N, ni = X_grid.shape
    no = y_ray.shape[1]
    def cost_function(beta, lda=0):
        beta = beta.reshape(no, ni)
        probabilities = np.zeros((no, N))
        for i in range(no):
            probabilities[i, :] = np.exp(X_grid @ beta[i, :])
        
        probabilities = probabilities / np.sum(probabilities, axis=0)

        cost = np.sum((y_ray.T - probabilities)**2) / (no*ni)
        if lda != 0:
            cost +=  lda*np.dot(beta, beta)
        return cost

    return cost_function


def cost_function_gradient_generator(X_grid, Y_grid):
    ni = X_grid.shape[1]
    no = Y_grid.shape[1]
    def cost_funciton_gradient(beta, lda=0, idxs=None):
        if idxs is not None:
            X = X_grid[idxs, :]
            Y = X_grid[idxs, :]
        else:
            X = X_grid
            Y = Y_grid

        beta = beta.reshape(no, ni)
        beta_grad = np.zeros_like(beta)
        for i in range(no):
            probabilities = np.exp(X @ beta[i, :])
            probabilities = probabilities / np.sum(probabilities)

            prob_derivative = probabilities*(1-probabilities)
            beta_grad[i, :] = -2 * X.T @ ((Y[:, i] - probabilities)*prob_derivative) / (no*ni)
            if lda != 0:
                beta_grad[i, :] += lda * beta[i, :]

        return beta_grad.flatten()

    return cost_funciton_gradient

def predict(X_grid, Y_grid, beta):
    N, ni = X_grid.shape
    no = Y_grid.shape[1]

    beta = beta.reshape(no, ni)
    probabilities = np.zeros((no, N))
    for i in range(no):
        probabilities[i, :] = np.exp(X_grid @ beta[i, :])
    
    probabilities = probabilities / np.sum(probabilities, axis=0)
    return np.argmax(probabilities, axis=0)

def get_accuracy(true_labels, pred_labels):
    return np.sum(true_labels == pred_labels) / len(true_labels)

softmax_cost = cost_function_generator(*split_data.train_data)
softmax_cost_gradient = cost_function_gradient_generator(*split_data.train_data)

N = split_data.train_data.X.shape[0]
beta0 = np.random.normal(size=64*10)
print(f"initial cost: {softmax_cost(beta0):.3f} ")
print(f"initial accuracy: {get_accuracy(labels, predict(*split_data.test_data, beta0))*100:.1f} %")

grad_solver = GradSolver(
    64*10, softmax_cost, softmax_cost_gradient,
    schedule_type=ScheduleType.RMS_PROP, step_type=StepType.MOMENTUM_STOCHASTIC
)
minibatch_size = 100

# reuse parameters from previous run, or set to None
optimal_params = (0.002154434690031882, 1)
optimal_mass = 0.7894736842105263

if optimal_params is None:
    if optimal_mass is None:
        mass = 0.8
    else:
        mass = optimal_mass

    resolution = 10
    step_size_ray = np.logspace(-4, 0, resolution)
    lda_ray = np.append([0], np.logspace(-8, 0, resolution))
    cost_grid = np.zeros((len(step_size_ray), len(lda_ray)))

    optimal_lambda = 0
    optimal_step_size = 0
    min_cost = float("infinity")
    for i, step_size in enumerate(step_size_ray):
        for j, lda in enumerate(lda_ray):
            print(f"\r{j + i*len(lda_ray)}/{cost_grid.size}, min cost: {min_cost:.3f} ", end="")
            beta = grad_solver.solve(
                lda=lda, beta0=beta0, step_size=step_size, max_steps=1000,
                minibatch_size=minibatch_size, minibatch_count=int(N/minibatch_size),
                mass=mass
            )

            cost = softmax_cost(beta0)
            cost_grid[i, j] = cost

            if cost < min_cost:
                min_cost = cost
                optimal_step_size = step_size
                optimal_lambda = lda
    print(
        f"\nOptimal step size: {optimal_step_size}, optimal lambda: {optimal_lambda}")
    optimal_params = (optimal_step_size, optimal_lambda)

    with open(f"ffnn_logreg/data_grad_grid.dat", "w") as datafile:
        datafile.write(f"{cost_grid.shape[0]} Error grid\n")
        np.savetxt(datafile, cost_grid)
        datafile.write(f"{step_size_ray.size} Step sizes\n")
        np.savetxt(datafile, step_size_ray)
        datafile.write(f"{lda_ray.size} Lambdas\n")
        np.savetxt(datafile, lda_ray)

if optimal_mass == None:
    optimal_mass = 0
    min_cost = float("infinity")
    for i, mass in enumerate(np.linspace(0, 1, 20)):
        print(f"\r{i+1}/{20}, min cost: {min_cost:.3f} ", end="")
        beta = grad_solver.solve(
            lda=optimal_params[1], beta0=beta0, step_size=optimal_params[0], max_steps=1000,
            minibatch_size=minibatch_size, minibatch_count=int(N/minibatch_size),
            mass=mass
        )

        cost = softmax_cost(beta)

        if cost < min_cost:
            min_cost = cost
            optimal_mass = mass
    print(f"\noptimal mass: {optimal_mass}")

beta = grad_solver.solve(
    lda=optimal_params[1], beta0=beta0, step_size=optimal_params[0], max_steps=10000,
    minibatch_size=minibatch_size, minibatch_count=int(N/minibatch_size),
    mass=optimal_mass
)

print(f"final cost: {softmax_cost(beta):.3f} ")
print(f"final accuracy: {get_accuracy(labels, predict(*split_data.test_data, beta))*100:.1f} %")


# neural network regression

neuron_counts = [50]
functions = [FunctionType.SIGMOID]*(len(neuron_counts) + 1)
neural_network = NeuralNetwork(
    *split_data.train_data, functions=functions,
    nn_type=NNType.LOGISTIC_REGRESSION,
    learning_rate=0.01, lda=0,
    epochs=20, batch_size=100, hidden_neuron_counts=neuron_counts
)

pred_labels = neural_network.predict(split_data.test_data.X)
print(f"initial accuracy: {get_accuracy(pred_labels, test_labels)*100:.1f} %")
neural_network.train()

pred_labels = neural_network.predict(split_data.test_data.X)
print(f"final accuracy: {get_accuracy(pred_labels, test_labels)*100:.1f} %")

wrong_labels = []
wrong_idxs = []
for i, (pred, true) in enumerate(zip(pred_labels, test_labels)):
    if pred != true:
        wrong_labels.append(pred)
        wrong_idxs.append(test_idxs[i])

with open("ffnn_logreg/data.dat", "w") as datafile:
    datafile.write(f"{len(wrong_labels)}\n")
    np.savetxt(datafile, np.array(wrong_labels, dtype=int))
    np.savetxt(datafile, np.array(wrong_idxs, dtype=int))

