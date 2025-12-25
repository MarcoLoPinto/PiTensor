import numpy as np

class CrossEntropyLoss:
    """Computes the softmax cross-entropy loss and its gradient for multi-class classification.
    """
    def forward(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """
        Computes the forward pass of the cross-entropy loss.

        Args:
            logits (np.ndarray): Raw, unnormalized scores for each class. Shape: (batch_size, num_classes).
            targets (np.ndarray): True class indices for each sample. Shape: (batch_size,). Must be 1D with integer values representing class indices.

        Returns:
            float: The average cross-entropy loss over the batch.

        Raises:
            ValueError: If the targets are not 1D or their size does not match the batch size of predictions.
            ValueError: If the targets are not of integer type.
        """
        if targets.ndim != 1 or logits.shape[0] != targets.shape[0]:
            raise ValueError("Targets should be a 1D array of size equal to the batch size.")
        if not np.issubdtype(targets.dtype, np.integer):
            raise ValueError("Targets should be an array of integers representing class indices.")
        # Stable log-softmax via log-sum-exp
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        logsumexp = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
        log_probs = shifted - logsumexp
        self.probs = np.exp(log_probs)
        self.targets = targets
        self.batch_size = logits.shape[0]
        log_likelihood = -log_probs[range(self.batch_size), targets]
        return np.sum(log_likelihood) / self.batch_size

    def backward(self) -> np.ndarray:
        """
        Computes the backward pass of the cross-entropy loss.

        Returns:
            np.ndarray: The gradient of the loss with respect to the predictions. Shape: (batch_size, num_classes).
        """
        grad = self.probs.copy()
        grad[range(self.batch_size), self.targets] -= 1
        grad /= self.batch_size
        return grad
