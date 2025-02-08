import numpy as np
from .Layer import Layer
from typing import Dict, Any, Iterator

class Sequential(Layer):
    """
    Container layer that applies a sequence of layers in order.
    """
    def __init__(self, *layers: Layer):
        super().__init__()
        self.layers = list(layers)

    def __iter__(self) -> Iterator[Layer]:
        """
        Allows iteration over the layers in the sequential model.

        Returns:
            Iterator[Layer]: An iterator over the layers in the model.
        """
        return iter(self.layers)
    
    def __getitem__(self, index: int) -> Layer:
        """Allows indexing into the layers like a list."""
        return self.layers[index]

    def __len__(self) -> int:
        """Returns the number of layers."""
        return len(self.layers)

    def forward(self, input) -> np.ndarray:
        """
        Performs the forward pass through all layers in sequence.

        Args:
            input (np.ndarray): The input data.

        Returns:
            np.ndarray: The final output after passing through all layers.
        """
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def backward(self, grad_output) -> np.ndarray:
        """
        Performs the backward pass through all layers in reverse order.

        Args:
            grad_output (np.ndarray): The gradient of the loss with respect to the output.

        Returns:
            np.ndarray: The gradient of the loss with respect to the input.
        """
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        return grad_output

    def get_parameters(self) -> list[Dict[str, Any]]:
        """
        Retrieves the parameters of all layers in the sequential model.

        Returns:
            list[Dict[str, Any]]: A list of parameter dictionaries for each layer.
        """
        return [layer.get_parameters() for layer in self.layers]

    def update_parameters(self, params: list[Dict[str, Any]]) -> None:
        """
        Updates the parameters of all layers in the sequential model.

        Args:
            params (list[Dict[str, Any]]): A list of parameter dictionaries for each layer.
        """
        for layer, params_layer in zip(self.layers, params):
            layer.update_parameters(params_layer)

    def add(self, layer: Layer):
        """Adds a new layer to the sequential model.

        Args:
            layer (Layer): The layer to add.
        """
        self.layers.append(layer)
