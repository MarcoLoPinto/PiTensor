from abc import ABC, abstractmethod

import numpy as np


class Loss(ABC):
    """Abstract base class for all losses."""

    @abstractmethod
    def forward(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """Computes the loss value for a batch."""

    @abstractmethod
    def backward(self) -> np.ndarray:
        """Computes the gradient of the loss with respect to inputs."""
