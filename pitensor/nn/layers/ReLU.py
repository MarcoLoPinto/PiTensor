import numpy as np
from .Layer import Layer

class ReLU(Layer):
    """Implements the Rectified Linear Unit (ReLU) activation function.
    """
    def forward(self, input):
        """
        Computes the forward pass of the ReLU activation function.

        Args:
            input (np.ndarray): Input array of any shape.

        Returns:
            np.ndarray: Output array where each element is the maximum of 0 and the corresponding input value.

        Notes:
            - ReLU outputs zero for all negative input values and keeps positive values unchanged.
        """
        self.input = input
        # The ReLU activation, outputs zero where input is less than zero
        return np.maximum(0, input)

    def backward(self, grad_output):
        """
        Computes the backward pass of the ReLU activation function.

        Args:
            grad_output (np.ndarray): Gradient of the loss with respect to the output of ReLU, 
                                      of the same shape as the input.

        Returns:
            np.ndarray: Gradient of the loss with respect to the input of ReLU, 
                        of the same shape as the input.

        Notes:
            - The gradient is zero for input values less than or equal to zero 
              and unchanged for positive input values.
        """
        grad_input = grad_output * (self.input > 0)
        return grad_input
    
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