import numpy as np
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
        self.leaky_RELU_v = np.vectorize(self.leaky_RELU_f)
        self.softmax_v = np.vectorize(self.softmax_f)
        self.sigmoid_v = np.vectorize(self.sigmoid_f)
        self.tanh_v = np.vectorize(self.tanh_f)
        self.unit_v = np.vectorize(self.unit_f)
        self.RELU_v = np.vectorize(self.RELU_f)
        self.ELU_v = np.vectorize(self.ELU_f)

        self.leaky_RELU_derivative_v = np.vectorize(self.leaky_RELU_d)
        self.softmax_derivative_v = np.vectorize(self.softmax_d)
        self.sigmoid_derivative_v = np.vectorize(self.sigmoid_d)
        self.tanh_derivative_v = np.vectorize(self.tanh_d)
        self.unit_derivative_v = np.vectorize(self.unit_d)
        self.RELU_derivative_v = np.vectorize(self.RELU_d)
        self.ELU_derivative_v = np.vectorize(self.ELU_d)

        self.epsilon = 1e-2
        self.set_type(function_type)

    def set_type(self, function_type: FunctionType):
        if function_type == FunctionType.LEAKY_RELU:
            self.derivative = self.leaky_RELU_derivative_v
            self.function = self.leaky_RELU_v
        if function_type == FunctionType.SOFTMAX:
            self.derivative = self.softmax_d
            self.function = self.softmax_f
        elif function_type == FunctionType.SIGMOID:
            self.derivative = self.sigmoid_derivative_v
            self.function = self.sigmoid_v
        elif function_type == FunctionType.TANH:
            self.derivative = self.tanh_derivative_v
            self.function = self.tanh_v
        elif function_type == FunctionType.UNIT:
            self.derivative = self.unit_derivative_v
            self.function = self.unit_v
        elif function_type == FunctionType.RELU:
            self.derivative = self.RELU_derivative_v
            self.function = self.RELU_v
        elif function_type == FunctionType.ELU:
            self.derivative = self.ELU_derivative_v
            self.function = self.ELU_v
        else:
            print("ActivationFunction: unknown function type:", function_type)
            self.derivative = self.sigmoid_derivative_v
            self.function = self.sigmoid_v

    def __call__(self, x):
        return self.function(x)


    def leaky_RELU_f(self, x):
        if x > 0.:
            return x
        return self.epsilon*x

    def leaky_RELU_d(self, x):
        if x > 0.:
            return 1.
        return self.epsilon


    def softmax_f(self, x):
        exp = np.exp(x)
        return exp/np.sum(exp, axis=0, keepdims=True)

    def softmax_d(self, x):
        softmax = self.softmax_f(x)
        return softmax*(1-softmax)


    def sigmoid_f(self, x):
        return 1/(1 + np.exp(-x))

    def sigmoid_d(self, x):
        sigmoid = self.sigmoid_f(x)
        return sigmoid*(1-sigmoid)


    def tanh_f(self, x):
        return np.tanh(x)

    def tanh_d(self, x):
        tanh = self.tanh_f(x)
        return 1 - tanh**2


    def unit_f(self, x):
        return x

    def unit_d(self, _):
        return 1.



    def RELU_f(self, x):
        if x > 0.:
            return x
        return 0.

    def RELU_d(self, x):
        if x > 0:
            return 1.
        return 0.


    def ELU_f(self, x):
        if x > 0.:
            return x
        return self.epsilon*(np.exp(x)-1)

    def ELU_d(self, x):
        if x > 0.:
            return 1.
        return self.epsilon*np.exp(x)


class NeuralNetwork:
    def __init__(
        self, X_data, Y_data, hidden_neuron_counts=[50], functions=FunctionType.SIGMOID,
        epochs=10, batch_size=10, learning_rate=0.1, lda=0.0
    ):

        self.X_data = X_data
        self.Y_data = Y_data

        self.n_layers = len(hidden_neuron_counts)+1

        self.sigmas = []

        if hasattr(functions, "__iter__"):
            if len(functions) < self.n_layers:
                print(
                    f"NeuralNetwork: expected {self.n_layers} activation functions, but only got {len(functions)}!")
                exit()
            self.sigmas = [ActivationFunction(function) for function in functions]
        else:
            self.sigmas = [ActivationFunction(functions)]*self.n_layers

        self.hidden_neuron_counts = hidden_neuron_counts
        self.n_inputs, self.n_features = X_data.shape
        self.n_categories = Y_data.shape[1]

        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.learning_rate = learning_rate
        self.lda = lda

        self.create_biases_and_weights()

    def create_biases_and_weights(self):
        if self.n_layers == 1:
            self.weights = [np.random.randn(
                self.n_features, self.n_categories)]
            self.biases = [np.zeros(self.n_categories) + 0.01]
        else:
            self.weights = [np.random.randn(
                self.n_features, self.hidden_neuron_counts[0])]
            self.biases = [np.zeros(self.hidden_neuron_counts[0]) + 0.01]

            for i in range(len(self.hidden_neuron_counts)-1):
                nci, ncip = self.hidden_neuron_counts[i], self.hidden_neuron_counts[i+1]
                self.weights.append(np.random.randn(nci, ncip))
                self.biases.append(np.zeros(ncip) + 0.01)

            self.weights.append(np.random.randn(
                self.hidden_neuron_counts[-1], self.n_categories))
            self.biases.append(np.zeros(self.n_categories) + 0.01)

        self.weight_gradients = [np.zeros_like(w) for w in self.weights]
        self.weight_velocities = [np.zeros_like(w) for w in self.weights]

        self.bias_gradients = [np.zeros_like(b) for b in self.biases]
        self.bias_velocities = [np.zeros_like(b) for b in self.biases]

    def feed_forward(self, X_batch):
        # feed-forward for training
        self.z_vals = [X_batch @ self.weights[0] + self.biases[0]]
        self.a_vals = [self.sigmas[0](self.z_vals[0])]

        for i in range(1, self.n_layers):
            self.z_vals.append(
                self.a_vals[i-1] @ self.weights[i] + self.biases[i])
            self.a_vals.append(self.sigmas[i](self.z_vals[i]))

    def feed_forward_out(self, X):
        # feed-forward for output
        a = self.sigmas[0](X @ self.weights[0] + self.biases[0])

        for i in range(1, self.n_layers):
            a = self.sigmas[i](a @ self.weights[i] + self.biases[i])

        return a

    def backpropagation(self, X_batch, Y_batch):
        delta_out = self.sigmas[-1].derivative(
            self.z_vals[-1]) * (self.a_vals[-1] - Y_batch)

        if self.n_layers == 1:
            self.weight_gradients[0] = X_batch.T @ delta_out
            self.bias_gradients[0] = np.sum(delta_out, axis=0)
        else:
            self.weight_gradients[-1] = self.a_vals[-2].T @ delta_out
            self.bias_gradients[-1] = np.sum(delta_out, axis=0)

            delta_h = delta_out
            for i in range(self.n_layers-1, 0, -1):
                delta_h = (delta_h @ self.weights[i].T) * self.sigmas[i-1].derivative(self.z_vals[i-1])

                if i == 1:
                    self.weight_gradients[0] = X_batch.T @ delta_h
                else:
                    self.weight_gradients[i-1] = self.a_vals[i-2].T @ delta_h
                self.bias_gradients[i-1] = np.sum(delta_h, axis=0)

    
        # velocity = mass * velocity + step_size * grad

        # s = memlif1*s + (1 - memlif1)*np.outer(velocity, velocity)
        # sinv = step_size / (1e-8 + np.sqrt(np.diagonal(s)))

        # beta += -np.multiply(sinv, velocity)

        if self.lda > 0.0:
            for i in range(self.n_layers):
                self.weight_gradients[i] += self.lda * self.weights[i]
                self.bias_gradients[i] += self.lda * self.biases[i]

        mass = 0.8
        lifetime = 0.9
        for i in range(self.n_layers):
            self.weight_velocities[i] = mass * self.weight_velocities[i] + self.learning_rate * self.weight_gradients[i]
            self.weight_s[i] = lifetime*self.weight_s[i] + (1 - lifetime)*self.weight_velocities[i]**2
            weight_sinv = self.learning_rate / (1e-8 + np.sqrt(self.weight_s[i]))
            self.weights[i] -= weight_sinv * self.weight_velocities[i]
            
            self.bias_velocities[i] = mass * self.bias_velocities[i] + self.learning_rate * self.bias_gradients[i]
            self.bias_s[i] = lifetime*self.bias_s[i] + (1 - lifetime)*self.bias_velocities[i]**2
            bias_sinv = self.learning_rate / (1e-8 + np.sqrt(self.bias_s[i]))
            self.biases[i] -= bias_sinv * self.bias_velocities[i]

    def predict(self, X):
        return self.feed_forward_out(X)

    def loss(self):
        # is this loss?
        return np.sum((self.predict(self.X_data) - self.Y_data)**2)

    def train(self, silent=False):
        data_indices = np.arange(self.n_inputs)

        optimal_weights = [w.copy() for w in self.weights]
        optimal_biases = [b.copy() for b in self.biases]
        min_loss = self.loss()

        for i in range(self.epochs):
            if not silent:
                percentage = int(100*(i)/self.epochs)
                if percentage != int(100*(i-1)/self.epochs):
                    # print percentage and loss
                    print(f"\r{percentage} %, loss={self.loss():.5f}", end="")

            if self.loss() > 1e10:
                print("\nloss diverging, abort")
                return

            self.weight_s = [np.zeros_like(w) for w in self.weights]
            self.bias_s = [np.zeros_like(b) for b in self.biases]
            for j in range(self.iterations):
                # pick datapoints with replacement
                chosen_datapoints = np.random.choice(
                    data_indices, size=self.batch_size, replace=False
                )

                # minibatch training data
                X_batch = self.X_data[chosen_datapoints]
                Y_batch = self.Y_data[chosen_datapoints]

                self.feed_forward(X_batch)
                self.backpropagation(X_batch, Y_batch)

                loss = self.loss()
                if loss < min_loss:
                    min_loss = loss
                    optimal_weights = [w.copy() for w in self.weights]
                    optimal_biases = [b.copy() for b in self.biases]

        self.weights = optimal_weights
        self.biases = optimal_biases
        if not silent:
            print(f"\r100 %, loss={self.loss():.5f}")
