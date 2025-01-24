from typing import Any, Dict

class Layer:
    """
    Base class for all layers. Defines the required methods for forward and backward passes,
    as well as parameter management. All custom layers should inherit from this class.
    """
    def forward(self, input: any) -> Any:
        """
        Perform the forward pass through the layer.

        Args:
            input (Any): Input data to the layer.

        Returns:
            Any: The output after applying the layer's transformation.
        """
        raise NotImplementedError("Please implement the forward step for your custom layer.")

    def backward(self, grad_output: Any) -> Any:
        """
        Perform the backward pass through the layer.

        Args:
            grad_output (Any): Gradient of the loss with respect to the layer's output.

        Returns:
            Any: Gradient of the loss with respect to the layer's input.
        """
        raise NotImplementedError("Please implement the backward step for your custom layer.")
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Retrieve the parameters of the layer.

        Returns:
            Dict[str, Any]: Dictionary containing the layer's parameters.
        """
        raise NotImplementedError("Please implement the get_parameters function in order to return all needed trained parameters from your custom layer.")
    
    def update_parameters(self: Dict[str, Any]) -> None:
        """
        Update the parameters of the layer.

        Args:
            params (Dict[str, Any]): Dictionary containing the updated parameter values.
        """
        raise NotImplementedError("Please implement the update_parameters function in order to update all needed trained parameters for your custom layer.")