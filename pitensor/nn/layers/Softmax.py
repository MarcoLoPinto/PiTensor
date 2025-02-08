import numpy as np
from .Layer import Layer

class Softmax(Layer):
    """Implements the softmax activation function for multi-class classification.
    """
    def forward(self, input) -> np.ndarray:
        """
        Computes the forward pass of the softmax function.

        Args:
            input (np.ndarray): Input array of shape (batch_size, num_classes).

        Returns:
            np.ndarray: The output probabilities of shape (batch_size, num_classes),
                        where each row represents a valid probability distribution.

        Notes:
            - The implementation includes a numerical stability improvement by subtracting
              the maximum value in each row of the input array to prevent overflow during exponentiation.
        """
        epsilon = 1e-8
        exps = np.exp(input - np.max(input, axis=1, keepdims=True)) # stability improvement
        self.output = exps / ( np.sum(exps, axis=1, keepdims=True) + epsilon ) # Avoid division by zero
        return self.output

    def backward(self, grad_output) -> np.ndarray:
        """
        Computes the backward pass for the softmax layer.

        Args:
            grad_output (np.ndarray): Gradient of the loss with respect to the softmax output,
                                      of shape (batch_size, num_classes).

        Returns:
            np.ndarray: Gradient of the loss with respect to the softmax input,
                        of shape (batch_size, num_classes).

        Notes:
            - The gradient computation involves the Jacobian matrix of the softmax function,
              which is implemented for each sample in the batch.
        """
        return self.output * (grad_output - np.sum(grad_output * self.output, axis=1, keepdims=True))
    
    def get_parameters(self):
        return {
            'type': self.__class__.__name__
        }
    
    def update_parameters(self, params):
        pass