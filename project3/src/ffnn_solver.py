from autograd import grad, elementwise_grad
import autograd.numpy.random as npr
import autograd.numpy as np
from enum import Enum, auto


class FunctionType(Enum):
    LEAKY_RELU = auto()
    SOFTMAX = auto()
    SIGMOID = auto()
    TANH = auto()
    UNIT = auto()
    RELU = auto()
    ELU = auto()


class ActivationFunction:
    def __init__(self, function_type: FunctionType):
        self.epsilon = 1e-2
        self.set_type(function_type)

    def set_type(self, function_type: FunctionType):
        if function_type == FunctionType.LEAKY_RELU:
            self.function = self.leaky_RELU
        elif function_type == FunctionType.SIGMOID:
            self.function = self.sigmoid
        elif function_type == FunctionType.TANH:
            self.function = self.tanh
        elif function_type == FunctionType.UNIT:
            self.function = self.unit
        elif function_type == FunctionType.RELU:
            self.function = self.RELU
        elif function_type == FunctionType.ELU:
            self.function = self.ELU
        else:
            print("ActivationFunction: unknown function type:", function_type)
            self.function = self.sigmoid

    def __call__(self, x):
        return self.function(x)

    def leaky_RELU(self, x):
        return np.where(x > 0., x, self.epsilon*x)

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def unit(self, x):
        return x

    def RELU(self, x):
        return np.where(x > 0., x, 0.)

    def ELU(self, x):
        return np.where(x > 0., x, self.epsilon*(np.exp(x)-1))


def neural_network(params, x_grid, t_grid, activation_function):
    w_hidden, b_hidden, w_out, b_out = params

    # input layer
    X_input = np.append(
        x_grid.flatten().reshape(-1, 1),
        t_grid.flatten().reshape(-1, 1),
        axis=1
    ).T

    # hidden layer
    z_hidden = w_hidden @ X_input + b_hidden
    a_hidden = activation_function(z_hidden)

    # output layer
    z_output = w_out @ a_hidden + b_out

    return z_output


# trial solution that satisfies initial conditions
def u_trial(x, t, params, activation_function):
    neural_network_out = neural_network(
        params, x, t, activation_function
    ).reshape(x.shape)

    return np.sin(np.pi * x) + x * (1 - x) * t * neural_network_out


def cost_function(P, x_grid, t_grid, u_t_diff_t, u_t_diff_x2, activation_function):
    err_sqr = (
        u_t_diff_t(x_grid, t_grid, P, activation_function)
        - u_t_diff_x2(x_grid, t_grid, P, activation_function)
    )**2
    return np.mean(err_sqr)


def solve_ode_neural_network(
    x_grid, t_grid, num_neurons_hidden, epochs, batch_size,
    learning_rate, activation_function, lda=0, silent=False
):
    # create random parameters
    w_hidden = npr.randn(num_neurons_hidden, 2)
    b_hidden = npr.randn(num_neurons_hidden, 1)
    w_out = npr.randn(1, num_neurons_hidden)
    b_out = npr.randn(1, 1)

    params = [w_hidden, b_hidden, w_out, b_out]

    # find gradients
    cost_function_grad = grad(cost_function, 0)
    u_t_diff_t = elementwise_grad(u_trial, 1)
    u_t_diff_x2 = elementwise_grad(elementwise_grad(u_trial, 0))

    #
    iterations = x_grid.size // batch_size
    data_indices = np.arange(x_grid.size)

    optimal_params = [p.copy() for p in params]
    min_loss = cost_function(
        params, x_grid, t_grid, u_t_diff_t,
        u_t_diff_x2, activation_function
    )

    if not silent:
        print(f" 0/10 [loss={min_loss:.5f}]")

    for i in range(epochs):
        for j in range(iterations):
            # pick datapoints with replacement
            chosen_datapoints = np.unravel_index(np.random.choice(
                data_indices, size=batch_size, replace=False
            ), x_grid.shape)

            # minibatch training data
            x_grid_batch = x_grid[chosen_datapoints]
            t_grid_batch = t_grid[chosen_datapoints]

            # update weights and biases
            cost_grad = cost_function_grad(
                params, x_grid, t_grid, u_t_diff_t,
                u_t_diff_x2, activation_function
            )

            w_hidden -= learning_rate * cost_grad[0]
            b_hidden -= learning_rate * cost_grad[1]
            w_out -= learning_rate * cost_grad[2]
            b_out -= learning_rate * cost_grad[3]

            if lda > 0:
                w_hidden += lda * w_hidden
                b_hidden += lda * b_hidden
                w_out += lda * w_out
                b_out += lda * b_out

        loss = cost_function(
            params, x_grid, t_grid, u_t_diff_t,
            u_t_diff_x2, activation_function
        )
        if not np.isnan(loss) and loss < min_loss:
            min_loss = loss
            optimal_params = [p.copy() for p in params]

        if not silent:
            p = int(10*(i)/epochs)
            if p != int(10*(i-1)/epochs):
                print(f"{p:2d}/10 [loss={loss:.5f}]")

        if loss > 1e10:
            print("\nloss diverging, abort")
            return None

    params = optimal_params

    if not silent:
        loss = cost_function(
            params, x_grid, t_grid, u_t_diff_t,
            u_t_diff_x2, activation_function
        )
        print(f"10/10 [loss={loss:.5f}]")

    return params


class NNDiffEqSolver:
    def __init__(
        self, num_neurons_hidden, function_type
    ):
        self.num_neurons_hidden = num_neurons_hidden
        self.function = ActivationFunction(function_type)
        self.params = None

    def train(self, x_grid, t_grid, epochs, batch_size, learning_rate, lda=0, silent=False):
        self.params = solve_ode_neural_network(
            x_grid, t_grid, self.num_neurons_hidden,
            epochs, batch_size, learning_rate, self.function, lda, silent
        )

    def predict(self, x_grid, t_grid):
        if self.params is None:
            print("you must train the network before predicting")
            return None
        return u_trial(x_grid, t_grid, self.params, self.function)

    def loss(self, x_grid, t_grid):
        if self.params is None:
            print("you must train the network before finding loss")
            return np.nan

        u_t_diff_t = elementwise_grad(u_trial, 1)
        u_t_diff_x2 = elementwise_grad(elementwise_grad(u_trial, 0))

        return cost_function(
            self.params, x_grid, t_grid, u_t_diff_t,
            u_t_diff_x2, self.function
        )


def gridsearch(
    x_grid_train, t_grid_train, x_grid_test, t_grid_test,
    learning_rate_ray, lda_ray, solver: NNDiffEqSolver,
    epochs, batch_size
):
    optimal_lda = None
    optimal_learning_rate = None
    min_loss = float("infinity")
    loss_grid = np.zeros((learning_rate_ray.size, lda_ray.size))

    for i, learning_rate in enumerate(learning_rate_ray):
        for j, lda in enumerate(lda_ray):
            solver.train(
                x_grid_train, t_grid_train, epochs, batch_size,
                learning_rate, lda, silent=True
            )
            loss = solver.loss(x_grid_test, t_grid_test)
            loss_grid[i, j] = loss

            if not np.isnan(loss) and loss < min_loss:
                optimal_lda = lda
                optimal_learning_rate = learning_rate
                min_loss = loss

            idx = i*lda_ray.size + j
            tot = learning_rate_ray.size*lda_ray.size
            print(
                f"\r{idx + 1}/{tot} [loss={loss:.5f}, min_loss={min_loss:.5f}]", end="")
    print("\n")

    return optimal_learning_rate, optimal_lda, loss_grid


def u_analytic(x, t):
    return np.exp(-np.pi**2 * t) * np.sin(np.pi * x)


def test_network(
    x_grid_train, t_grid_train, x_grid_test, t_grid_test,
    learning_rate_ray, lda_ray, search_epochs, train_epochs,
    batch_size, num_hidden_neurons, function_type: FunctionType
):
    print(
        f"testing NN with {num_hidden_neurons} hidden neurons "
        + f"and the {function_type.name} activation funcion"
    )
    solver = NNDiffEqSolver(num_hidden_neurons, function_type)

    learning_rate, lda, loss_grid = gridsearch(
        x_grid_train, t_grid_train, x_grid_test, t_grid_test,
        learning_rate_ray, lda_ray, solver, search_epochs, batch_size
    )
    print(f"Found {learning_rate=}, {lda=}")

    solver.train(
        x_grid_train, t_grid_train, train_epochs,
        batch_size, learning_rate, lda
    )

    u_ffnn_grid = solver.predict(x_grid_test, t_grid_test)
    u_ana_grid = u_analytic(x_grid_test, t_grid_test)

    np.savez(
        f"data/ffnn_solver/hn={num_hidden_neurons}_tp={function_type.name}.npz",
        x_ray=x_grid_test[0, :], t_ray=t_grid_test[:, 0],
        u_ffnn_grid=u_ffnn_grid, u_ana_grid=u_ana_grid,
        learning_rate_ray=learning_rate_ray, lda_ray=lda_ray,
        loss_grid=loss_grid
    )


def main():
    npr.seed(15)

    # create test and train data
    Nx = 20
    Nt = 20
    T = np.log(100) / (np.pi**2)

    x_ray_test = np.linspace(0, 1, Nx)
    x_ray_train = (np.linspace(0, 1, Nx) + 0.5/Nx)[:-1]

    t_ray_test = np.linspace(0, T, Nt)
    t_ray_train = (np.linspace(0, T, Nt) + 0.5*T/Nt)[:-1]

    x_grid_train, t_grid_train = np.meshgrid(x_ray_train, t_ray_train)
    x_grid_test, t_grid_test = np.meshgrid(x_ray_test, t_ray_test)

    # hyper parameters
    search_epochs = 25
    train_epochs = 500
    batch_size = 10
    learning_rate_ray = np.logspace(-4, 1, 10)
    lda_ray = np.append([0], np.logspace(-8, -1, 9))

    for hidden_neuron_count in [10, 100]:
        for function_type in [FunctionType.SIGMOID, FunctionType.LEAKY_RELU]:
            test_network(
                x_grid_train, t_grid_train, x_grid_test, t_grid_test,
                learning_rate_ray, lda_ray, search_epochs, train_epochs,
                batch_size, hidden_neuron_count, function_type
            )


if __name__ == '__main__':
    main()
