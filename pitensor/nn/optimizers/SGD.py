import numpy as np
from .Optimizer import Optimizer
from pitensor.nn.layers import Linear

class SGD(Optimizer):
    """Implements the Stochastic Gradient Descent (SGD) optimization algorithm.
    """
    def __init__(self, learning_rate=0.01):
        """
        Initializes the SGD optimizer.

        Args:
            learning_rate (float, optional): The learning rate used to update the layer parameters. Defaults to 0.01.
        """
        self.learning_rate = learning_rate

    def step(self, layers):
        """
        Updates the parameters of the layers using the gradients computed during backpropagation.

        Args:
            layers (list): A list of layers in the model.

        Notes:
            - Only layers that are instances of the `Linear` class are updated.
        """
        for layer in layers:
            if isinstance(layer, Linear):
                layer.weights -= self.learning_rate * layer.grad_weights
                layer.biases -= self.learning_rate * layer.grad_biases