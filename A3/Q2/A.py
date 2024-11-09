import numpy as np

class NeuralNetwork:
    def __init__(self, N, layer_sizes, lr, activation, weight_init, epochs, batch_size):
        self.N = N  # Number of layers
        self.layer_sizes = layer_sizes  # List of neurons in each layer
        self.lr = lr  # Learning rate
        self.activation = activation  # Activation function
        self.weight_init = weight_init  # Weight initialization method
        self.epochs = epochs  # Number of epochs
        self.batch_size = batch_size  # Batch size

        # Initialize weights and biases
        self.weights, self.biases = self.initialize_weights()

    def initialize_weights(self):
        weights = []
        biases = []

        for i in range(self.N - 1):
            if self.weight_init == "xavier":
                weight = np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1]) * np.sqrt(1 / self.layer_sizes[i])
            elif self.weight_init == "he":
                weight = np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1]) * np.sqrt(2 / self.layer_sizes[i])
            else:  # default random initialization
                weight = np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1]) * 0.01
            bias = np.zeros((1, self.layer_sizes[i + 1]))

            weights.append(weight)
            biases.append(bias)

        return weights, biases

    def activation_function(self, z):
        if self.activation == "relu":
            return np.maximum(0, z)
        elif self.activation == "tanh":
            return np.tanh(z)
        elif self.activation == "sigmoid":
            return 1 / (1 + np.exp(-z))
        else:
            raise ValueError("Unsupported activation function")

    def activation_derivative(self, z):
        if self.activation == "relu":
            return np.where(z > 0, 1, 0)
        elif self.activation == "tanh":
            return 1 - np.tanh(z) ** 2
        elif self.activation == "sigmoid":
            sig = self.activation_function(z)
            return sig * (1 - sig)
        else:
            raise ValueError("Unsupported activation function")

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X):
        activations = [X]
        zs = []

        for i in range(self.N - 2):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            zs.append(z)
            activation = self.activation_function(z)
            activations.append(activation)

        # Output layer with softmax for probabilities
        z = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        zs.append(z)
        activation = self.softmax(z)
        activations.append(activation)

        return activations, zs

    def backward(self, X, Y, activations, zs):
        grads_w = [None] * (self.N - 1)
        grads_b = [None] * (self.N - 1)

        # Output layer error
        delta = activations[-1] - Y  # Assuming Y is one-hot encoded
        grads_w[-1] = np.dot(activations[-2].T, delta) / X.shape[0]
        grads_b[-1] = np.sum(delta, axis=0, keepdims=True) / X.shape[0]

        # Backpropagation through hidden layers
        for i in range(self.N - 3, -1, -1):
            delta = np.dot(delta, self.weights[i + 1].T) * self.activation_derivative(zs[i])
            grads_w[i] = np.dot(activations[i].T, delta) / X.shape[0]
            grads_b[i] = np.sum(delta, axis=0, keepdims=True) / X.shape[0]

        return grads_w, grads_b

    def update_parameters(self, grads_w, grads_b):
        for i in range(self.N - 1):
            self.weights[i] -= self.lr * grads_w[i]
            self.biases[i] -= self.lr * grads_b[i]

    def fit(self, X, Y):
        epoch_loss = None  # Initialize loss variable
        for epoch in range(self.epochs):
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)

            for start in range(0, X.shape[0], self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                X_batch, Y_batch = X[batch_indices], Y[batch_indices]

                activations, zs = self.forward(X_batch)
                grads_w, grads_b = self.backward(X_batch, Y_batch, activations, zs)
                self.update_parameters(grads_w, grads_b)

            # Calculate epoch loss for the entire dataset at the end of each epoch
            full_activations, _ = self.forward(X)
            epoch_loss = -np.mean(np.sum(Y * np.log(full_activations[-1] + 1e-8), axis=1))
            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {epoch_loss:.4f}")
        
        # Return the final epoch loss to be stored in train_loss_history
        return epoch_loss


    def predict(self, X):
        activations, _ = self.forward(X)
        return np.argmax(activations[-1], axis=1)

    def predict_proba(self, X):
        activations, _ = self.forward(X)
        return activations[-1]

    def score(self, X, Y):
        predictions = self.predict(X)
        labels = np.argmax(Y, axis=1)  # Assuming Y is one-hot encoded
        accuracy = np.mean(predictions == labels)
        return accuracy