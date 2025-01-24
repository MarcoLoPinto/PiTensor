from abc import ABC, abstractmethod

class Optimizer(ABC):
    """Abstract base class for all optimizers.
    """

    def __init__(self):
        """Initializes the optimizer.
        """

    @abstractmethod
    def step(self, layers: list):
        """
        Updates the parameters of the given layers.

        Args:
            layers (list): A list of layers in the model to update. Each layer must implement
                          attributes or methods to retrieve gradients and parameters.

        Notes:
            - This method must be implemented by subclasses.
        """
        pass