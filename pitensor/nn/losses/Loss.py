from abc import ABC, abstractmethod

from pitensor.Tensor import Tensor


class Loss(ABC):
    """Abstract base class for all losses."""

    @abstractmethod
    def forward(self, logits: Tensor, targets: Tensor) -> float:
        """Computes the loss value for a batch."""

    @abstractmethod
    def backward(self) -> Tensor:
        """Computes the gradient of the loss with respect to inputs."""
