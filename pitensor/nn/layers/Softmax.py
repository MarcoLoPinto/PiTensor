from pitensor.Tensor import Tensor
from .Layer import Layer

class Softmax(Layer):
    """Implements the softmax activation function for multi-class classification.
    """
    def forward(self, input: Tensor) -> Tensor:
        """
        Computes the forward pass of the softmax function.

        Args:
            input (Tensor): Input array of shape (batch_size, num_classes).

        Returns:
            Tensor: The output probabilities of shape (batch_size, num_classes),
                        where each row represents a valid probability distribution.

        Notes:
            - The implementation includes a numerical stability improvement by subtracting
              the maximum value in each row of the input array to prevent overflow during exponentiation.
        """
        epsilon = 1e-8
        shifted: Tensor = input - input.max(axis=1, keepdims=True) # stability improvement
        exps = Tensor.exp(shifted)
        self.output = exps / (exps.sum(axis=1, keepdims=True) + epsilon) # Avoid division by zero
        return self.output

    def backward(self, grad_output: Tensor) -> Tensor:
        """
        Computes the backward pass for the softmax layer.

        Args:
            grad_output (Tensor): Gradient of the loss with respect to the softmax output,
                                      of shape (batch_size, num_classes).

        Returns:
            Tensor: Gradient of the loss with respect to the softmax input,
                        of shape (batch_size, num_classes).

        Notes:
            - The gradient computation involves the Jacobian matrix of the softmax function,
              which is implemented for each sample in the batch.
        """
        return self.output * (grad_output - (grad_output * self.output).sum(axis=1, keepdims=True))
    
    def get_parameters(self):
        return {
            'type': self.__class__.__name__
        }
    
    def update_parameters(self, params):
        pass
