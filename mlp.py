import numpy as np


class Activation:
    def __init__(self, f, df):
        self.f = f
        self.df = df


class Loss:
    def __init__(self, f, grad):
        self.f = f
        self.grad = grad


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def _softmax(z):
    e = np.exp(z - z.max(axis=0, keepdims=True))
    return e / e.sum(axis=0, keepdims=True)


sigmoid = Activation(_sigmoid, lambda z: _sigmoid(z) * (1 - _sigmoid(z)))
tanh = Activation(np.tanh, lambda z: 1 - np.tanh(z) ** 2)
relu = Activation(lambda z: np.maximum(0, z), lambda z: (z > 0).astype(float))
linear = Activation(lambda z: z, lambda z: np.ones_like(z))
softmax = Activation(_softmax, None)

mse = Loss(lambda y_hat, y: 0.5 * np.sum((y_hat - y) ** 2) / y.shape[1],
           lambda y_hat, y: y_hat - y)
mae = Loss(lambda y_hat, y: np.sum(np.abs(y_hat - y)) / y.shape[1],
           lambda y_hat, y: np.sign(y_hat - y))
cross_entropy = Loss(lambda y_hat, y: -np.sum(y * np.log(y_hat + 1e-12)) / y.shape[1],
                     None)


class MLP:
    def __init__(self, layer_sizes, hidden_activation, output_activation, loss, seed, use_bias=True):
        if (output_activation is softmax) != (loss is cross_entropy):
            raise ValueError("softmax output must be paired with cross_entropy loss")
        self.rng = np.random.default_rng(seed)
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.loss = loss
        self.use_bias = use_bias
        self.weights = []
        self.biases = []

        for i in range(len(layer_sizes) - 1):
            n_in = layer_sizes[i]
            n_out = layer_sizes[i + 1]
            self.weights.append(self.rng.normal(0, 0.1, size=(n_out, n_in)))
            self.biases.append(np.zeros((n_out, 1)))

    def forward(self, X):
        zs = []
        activations = [X]
        A = X
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            Z = W @ A + b
            if i < len(self.weights) - 1:
                A = self.hidden_activation.f(Z)
            else:
                A = self.output_activation.f(Z)
            zs.append(Z)
            activations.append(A)

        return zs, activations

    def predict(self, X):
        return self.forward(X)[1][-1]

    def backward(self, zs, activations, Y):
        B = Y.shape[1]
        grads_W = [None] * len(self.weights)
        grads_b = [None] * len(self.weights)

        y_hat = activations[-1]
        if self.output_activation is softmax:
            delta = y_hat - Y
        else:
            delta = self.loss.grad(y_hat, Y) * self.output_activation.df(zs[-1])

        for l in reversed(range(len(self.weights))):
            grads_W[l] = delta @ activations[l].T / B
            grads_b[l] = delta.sum(axis=1, keepdims=True) / B
            if l > 0:
                delta = (self.weights[l].T @ delta) * self.hidden_activation.df(zs[l - 1])

        return grads_W, grads_b

    def update(self, grads_W, grads_b, lr):
        for l in range(len(self.weights)):
            self.weights[l] -= lr * grads_W[l]
            if self.use_bias:
                self.biases[l] -= lr * grads_b[l]

    def train(self, X, Y, epochs, lr, batch_size):
        history = {"loss": [], "weights": []}
        N = X.shape[1]
        for epoch in range(epochs):
            order = self.rng.permutation(N)
            for start in range(0, N, batch_size):
                batch = order[start:start + batch_size]
                zs, activations = self.forward(X[:, batch])
                grads_W, grads_b = self.backward(zs, activations, Y[:, batch])
                self.update(grads_W, grads_b, lr)
            history["loss"].append(self.loss.f(self.predict(X), Y))
            history["weights"].append([W.copy() for W in self.weights])

        return history
