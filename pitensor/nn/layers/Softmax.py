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
        exps = np.exp(input - np.max(input, axis=1, keepdims=True)) # stability improvement
        self.output = exps / np.sum(exps, axis=1, keepdims=True)
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
        # Compute the Jacobian matrix of the softmax
        batch_size = self.output.shape[0]
        grad_input = np.empty_like(grad_output)
        
        for i in range(batch_size):
            softmax_vector = self.output[i].reshape(-1, 1)
            jacobian = np.diagflat(softmax_vector) - np.dot(softmax_vector, softmax_vector.T)
            grad_input[i] = np.dot(jacobian, grad_output[i])

        return grad_input
    
    def get_parameters(self):
        return {
            'type': self.__class__.__name__
        }
    
    def update_parameters(self, params):
        pass