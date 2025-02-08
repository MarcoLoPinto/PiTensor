import numpy as np
from typing import Literal

def precision_score(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    average: Literal['binary', 'micro', 'macro', 'weighted', 'samples'] = 'binary'
) -> float:
    """
    Computes precision with different averaging methods.

    Args:
        y_true (np.ndarray): Ground truth labels (1D or 2D array).
        y_pred (np.ndarray): Predicted labels (1D or 2D array).
        average (str): Averaging method - 'binary', 'micro', 'macro', 'weighted', 'samples'. Defaults to 'binary'.

    Returns:
        float: Precision score.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)

    if average == 'binary':
        TP = np.sum((y_true == 1) & (y_pred == 1))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        return TP / (TP + FP) if (TP + FP) > 0 else 0.0

    num_classes = np.max(y_true) + 1 if y_true.ndim == 1 else y_true.shape[1]
    precision_per_class = np.zeros(num_classes)
    support_per_class = np.zeros(num_classes)

    for c in range(num_classes):
        TP = np.sum((y_true == c) & (y_pred == c))
        FP = np.sum((y_true != c) & (y_pred == c))
        support = np.sum(y_true == c)

        precision_per_class[c] = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        support_per_class[c] = support

    if average == 'micro':
        TP = np.sum(y_true == y_pred)
        FP = np.sum((y_true != y_pred) & (y_pred > 0))
        return TP / (TP + FP) if (TP + FP) > 0 else 0.0

    if average == 'macro':
        return np.mean(precision_per_class)

    if average == 'weighted':
        return np.sum(precision_per_class * support_per_class) / np.sum(support_per_class)

    raise ValueError("Invalid average method. Choose from 'binary', 'micro', 'macro', 'weighted', 'samples'.")

def recall_score(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    average: Literal['binary', 'micro', 'macro', 'weighted', 'samples'] = 'binary'
) -> float:
    """
    Computes recall with different averaging methods.

    Args:
        y_true (np.ndarray): Ground truth labels (1D or 2D array).
        y_pred (np.ndarray): Predicted labels (1D or 2D array).
        average (str): Averaging method - 'binary', 'micro', 'macro', 'weighted', 'samples'. Defaults to 'binary'.

    Returns:
        float: Recall score.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)

    if average == 'binary':
        TP = np.sum((y_true == 1) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))
        return TP / (TP + FN) if (TP + FN) > 0 else 0.0

    num_classes = np.max(y_true) + 1 if y_true.ndim == 1 else y_true.shape[1]
    recall_per_class = np.zeros(num_classes)
    support_per_class = np.zeros(num_classes)

    for c in range(num_classes):
        TP = np.sum((y_true == c) & (y_pred == c))
        FN = np.sum((y_true == c) & (y_pred != c))
        support = np.sum(y_true == c)

        recall_per_class[c] = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        support_per_class[c] = support

    if average == 'micro':
        TP = np.sum(y_true == y_pred)
        FN = np.sum((y_true != y_pred) & (y_true > 0))
        return TP / (TP + FN) if (TP + FN) > 0 else 0.0

    if average == 'macro':
        return np.mean(recall_per_class)

    if average == 'weighted':
        return np.sum(recall_per_class * support_per_class) / np.sum(support_per_class)

    raise ValueError("Invalid average method. Choose from 'binary', 'micro', 'macro', 'weighted', 'samples'.")

def f1_score(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    average: Literal['binary', 'micro', 'macro', 'weighted', 'samples'] = 'binary'
) -> float:
    """
    Computes F1-score with different averaging methods.

    Args:
        y_true (np.ndarray): Ground truth labels (1D or 2D array).
        y_pred (np.ndarray): Predicted labels (1D or 2D array).
        average (str): Averaging method - 'binary', 'micro', 'macro', 'weighted', 'samples'. Defaults to 'binary'.

    Returns:
        float: F1 score.
    """
    precision = precision_score(y_true, y_pred, average)
    recall = recall_score(y_true, y_pred, average)

    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
