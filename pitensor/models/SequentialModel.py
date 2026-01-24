from typing import Any, Dict, Optional

from pitensor.Tensor import Tensor
from pitensor.nn.layers import Sequential
from pitensor.nn.losses import Loss
from pitensor.nn.optimizers import Optimizer


class SequentialModel:
    """Generic model wrapper around a Sequential layer container."""

    def __init__(self, layers: Sequential, loss: Optional[Loss] = None):
        self.layers = layers
        self.loss = loss

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers.forward(inputs)

    def compute_loss(self, predictions: Tensor, targets: Tensor) -> float:
        if self.loss is None:
            raise ValueError("Loss is not set for this model.")
        return self.loss.forward(predictions, targets)

    def backward(self) -> Tensor:
        if self.loss is None:
            raise ValueError("Loss is not set for this model.")
        grad = self.loss.backward()
        return self.layers.backward(grad)

    def train_step(
        self,
        inputs: Tensor,
        targets: Optional[Tensor] = None,
        optimizer: Optional[Optimizer] = None,
    ):
        predictions = self.forward(inputs)
        if self.loss is None or targets is None:
            return predictions, None
        loss = self.compute_loss(predictions, targets)
        if optimizer is not None:
            self.backward()
            optimizer.step(self.layers)
        return predictions, loss

    def predict(self, inputs: Tensor) -> Tensor:
        return self.forward(inputs)

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "layers": self.layers.get_parameters(),
        }

    def update_parameters(self, params: Dict[str, Any]) -> None:
        if "layers" in params:
            self.layers.update_parameters(params["layers"])

    def save_parameters(self, file_path: str) -> None:
        Tensor.save(file_path, self.get_parameters(), allow_pickle=True)

    def load_parameters(self, file_path: str) -> None:
        params = Tensor.load(file_path, allow_pickle=True).item()
        self.update_parameters(params)
