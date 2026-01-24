from pitensor.Tensor import Tensor
from .Loss import Loss

class CrossEntropyLoss(Loss):
    """Computes the softmax cross-entropy loss and its gradient for multi-class classification.
    """
    def forward(self, logits: Tensor, targets: Tensor) -> float:
        """
        Computes the forward pass of the cross-entropy loss.

        Args:
            logits (Tensor): Raw, unnormalized scores for each class. Shape: (batch_size, num_classes).
            targets (Tensor): True class indices for each sample. Shape: (batch_size,). Must be 1D with integer values representing class indices.

        Returns:
            float: The average cross-entropy loss over the batch.

        Raises:
            ValueError: If the targets are not 1D or their size does not match the batch size of predictions.
            ValueError: If the targets are not of integer type.
        """
        if targets.ndim != 1 or logits.shape[0] != targets.shape[0]:
            raise ValueError("Targets should be a 1D array of size equal to the batch size.")
        if not Tensor.issubdtype(targets.dtype, Tensor.integer):
            raise ValueError("Targets should be an array of integers representing class indices.")
        # Stable log-softmax via log-sum-exp
        shifted = logits - logits.max(axis=1, keepdims=True)
        logsumexp = Tensor.log(Tensor.exp(shifted).sum(axis=1, keepdims=True))
        log_probs = shifted - logsumexp
        self.probs = Tensor.exp(log_probs)
        self.targets = targets
        self.batch_size = logits.shape[0]
        log_likelihood = -log_probs[range(self.batch_size), targets]
        return log_likelihood.sum() / self.batch_size

    def backward(self) -> Tensor:
        """
        Computes the backward pass of the cross-entropy loss.

        Returns:
            Tensor: The gradient of the loss with respect to the predictions. Shape: (batch_size, num_classes).
        """
        grad = self.probs.copy()
        grad[range(self.batch_size), self.targets] -= 1
        grad /= self.batch_size
        return grad
