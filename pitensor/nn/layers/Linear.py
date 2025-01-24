import numpy as np
from .Layer import Layer

class Linear(Layer):
    """Fully connected linear layer, which applies a linear transformation to the input.
    """
    def __init__(self, input_dim, output_dim):
        """
        Initializes the linear layer with random weights and biases.

        Args:
            input_dim (int): The size of the input features.
            output_dim (int): The size of the output features.
        """
        # Initialize weights and biases with small random values (using Xavier initialization)
        # Xavier initialization helps keep the magnitude of the outputs of the neurons similar 
        # across all layers and this helps in converging the model faster and prevents exploding 
        # and vanishing gradients.
        # self.weights is a matrix of shape input_size x output_size
        # self.biases is a vector of shape output_size
        self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(1. / input_dim)
        self.biases = np.random.randn(output_dim) * np.sqrt(1. / output_dim)
        self.input = None
        self.grad_weights = None
        self.grad_biases = None

    def forward(self, input) -> np.ndarray:
        self.input = input # Save the input for use in the backward pass
        # Compute the output Y = XW + B, where X is input, W is weights, B is biases.
        # This operation transforms the input data linearly. The weights determine 
        # the transformation's scaling and rotation, while the biases shift the result.
        return np.dot(input, self.weights) + self.biases

    def backward(self, grad_output) -> np.ndarray:
        # Compute gradient of the weights with respect to the loss. 
        # This computation effectively captures how each element of 
        # the input vector influences the loss through each weight.
        self.grad_weights = np.dot(self.input.T, grad_output)
        # Compute gradient of the biases with respect to the loss.
        self.grad_biases = np.sum(grad_output, axis=0)
        # Compute gradient of the input to this layer with respect to the loss and
        # this calculates how much each input value contributed to the loss.
        grad_input = np.dot(grad_output, self.weights.T)
        return grad_input
    
    def get_parameters(self):
        return {
            'type': self.__class__.__name__,
            'weights': self.weights,
            'biases': self.biases
        }
    
    def update_parameters(self, params):
        self.weights = params['weights']
        self.biases = params['biases']