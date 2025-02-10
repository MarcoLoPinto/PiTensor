import numpy as np
from .Layer import Layer

class Flatten(Layer):
    """Flattens the input tensor from (batch, channels, height, width) to (batch, features)."""
    def forward(self, input):
        self.input_shape = input.shape  # Save shape for backward pass
        return input.reshape(input.shape[0], -1)

    def backward(self, grad_output):
        return grad_output.reshape(self.input_shape)
    
    def get_parameters(self):
        return {
            'type': self.__class__.__name__
        }
    
    def update_parameters(self, params):
        """
        Updates the parameters of the ReLU layer.

        Args:
            params (dict): Placeholder for parameters, but ReLU has no parameters to update.
        """
        pass
