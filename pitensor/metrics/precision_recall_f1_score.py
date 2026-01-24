from typing import Literal

from pitensor.Tensor import Tensor

def precision_score(
    y_true: Tensor, 
    y_pred: Tensor, 
    average: Literal['binary', 'micro', 'macro', 'weighted', 'samples'] = 'binary'
) -> float:
    """
    Computes precision with different averaging methods.

    Args:
        y_true (Tensor): Ground truth labels (1D or 2D array).
        y_pred (Tensor): Predicted labels (1D or 2D array).
        average (str): Averaging method - 'binary', 'micro', 'macro', 'weighted', 'samples'. Defaults to 'binary'.

    Returns:
        float: Precision score.
    """
    y_true, y_pred = Tensor.asarray(y_true), Tensor.asarray(y_pred)

    if average == 'binary':
        TP = ((y_true == 1) & (y_pred == 1)).sum()
        FP = ((y_true == 0) & (y_pred == 1)).sum()
        return TP / (TP + FP) if (TP + FP) > 0 else 0.0

    num_classes = int(y_true.max()) + 1 if y_true.ndim == 1 else y_true.shape[1]
    precision_per_class = Tensor.zeros(num_classes)
    support_per_class = Tensor.zeros(num_classes)

    for c in range(num_classes):
        TP = ((y_true == c) & (y_pred == c)).sum()
        FP = ((y_true != c) & (y_pred == c)).sum()
        support = (y_true == c).sum()

        precision_per_class[c] = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        support_per_class[c] = support

    if average == 'micro':
        TP = (y_true == y_pred).sum()
        total = y_true.size
        return TP / total if total > 0 else 0.0

    if average == 'macro':
        return precision_per_class.mean()

    if average == 'weighted':
        return (precision_per_class * support_per_class).sum() / support_per_class.sum()

    raise ValueError("Invalid average method. Choose from 'binary', 'micro', 'macro', 'weighted', 'samples'.")

def recall_score(
    y_true: Tensor, 
    y_pred: Tensor, 
    average: Literal['binary', 'micro', 'macro', 'weighted', 'samples'] = 'binary'
) -> float:
    """
    Computes recall with different averaging methods.

    Args:
        y_true (Tensor): Ground truth labels (1D or 2D array).
        y_pred (Tensor): Predicted labels (1D or 2D array).
        average (str): Averaging method - 'binary', 'micro', 'macro', 'weighted', 'samples'. Defaults to 'binary'.

    Returns:
        float: Recall score.
    """
    y_true, y_pred = Tensor.asarray(y_true), Tensor.asarray(y_pred)

    if average == 'binary':
        TP = ((y_true == 1) & (y_pred == 1)).sum()
        FN = ((y_true == 1) & (y_pred == 0)).sum()
        return TP / (TP + FN) if (TP + FN) > 0 else 0.0

    num_classes = int(y_true.max()) + 1 if y_true.ndim == 1 else y_true.shape[1]
    recall_per_class = Tensor.zeros(num_classes)
    support_per_class = Tensor.zeros(num_classes)

    for c in range(num_classes):
        TP = ((y_true == c) & (y_pred == c)).sum()
        FN = ((y_true == c) & (y_pred != c)).sum()
        support = (y_true == c).sum()

        recall_per_class[c] = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        support_per_class[c] = support

    if average == 'micro':
        TP = (y_true == y_pred).sum()
        total = y_true.size
        return TP / total if total > 0 else 0.0

    if average == 'macro':
        return recall_per_class.mean()

    if average == 'weighted':
        return (recall_per_class * support_per_class).sum() / support_per_class.sum()

    raise ValueError("Invalid average method. Choose from 'binary', 'micro', 'macro', 'weighted', 'samples'.")

def f1_score(
    y_true: Tensor, 
    y_pred: Tensor, 
    average: Literal['binary', 'micro', 'macro', 'weighted', 'samples'] = 'binary'
) -> float:
    """
    Computes F1-score with different averaging methods.

    Args:
        y_true (Tensor): Ground truth labels (1D or 2D array).
        y_pred (Tensor): Predicted labels (1D or 2D array).
        average (str): Averaging method - 'binary', 'micro', 'macro', 'weighted', 'samples'. Defaults to 'binary'.

    Returns:
        float: F1 score.
    """
    precision = precision_score(y_true, y_pred, average)
    recall = recall_score(y_true, y_pred, average)

    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
