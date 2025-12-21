from .Optimizer import Optimizer

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
            - Updates any layer exposing weights/biases and their gradients.
        """
        for layer in layers:
            if hasattr(layer, "weights") and hasattr(layer, "grad_weights") and layer.grad_weights is not None:
                layer.weights -= self.learning_rate * layer.grad_weights
            if hasattr(layer, "biases") and hasattr(layer, "grad_biases") and layer.grad_biases is not None:
                layer.biases -= self.learning_rate * layer.grad_biases
