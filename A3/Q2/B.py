import numpy as np

class ActivationFunctions:
    
    @staticmethod
    def sigmoid(z):
        """
        Sigmoid activation function.
        """
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sigmoid_derivative(z):
        """
        Derivative of the sigmoid function.
        """
        sig = ActivationFunctions.sigmoid(z)
        return sig * (1 - sig)

    @staticmethod
    def tanh(z):
        """
        Tanh activation function.
        """
        return np.tanh(z)

    @staticmethod
    def tanh_derivative(z):
        """
        Derivative of the tanh function.
        """
        return 1 - np.tanh(z) ** 2

    @staticmethod
    def relu(z):
        """
        ReLU activation function.
        """
        return np.maximum(0, z)

    @staticmethod
    def relu_derivative(z):
        """
        Derivative of the ReLU function.
        """
        return np.where(z > 0, 1, 0)

    @staticmethod
    def leaky_relu(z, alpha=0.01):
        """
        Leaky ReLU activation function with a small slope for negative inputs.
        """
        return np.where(z > 0, z, alpha * z)

    @staticmethod
    def leaky_relu_derivative(z, alpha=0.01):
        """
        Derivative of the Leaky ReLU function.
        """
        return np.where(z > 0, 1, alpha)

    @staticmethod
    def softmax(z):
        """
        Softmax activation function for the output layer.
        """
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Stability improvement
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)