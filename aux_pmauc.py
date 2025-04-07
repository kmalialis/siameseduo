import numpy as np
from itertools import combinations


def calculate_A_pairs(y_window, probs):
    """
    Calculate A(i|j) pairs for a multi-class classification problem.

    Parameters:
    - X_window: ndarray of shape [n_examples, n_features], feature matrix.
    - y_window: 1D array of shape [n_examples], ground truth labels starting from 0, 1, 2, etc.
    - probs: ndarray of shape [n_examples, n_classes], predicted probabilities for each class.

    Returns:
    - A_pairs: Dictionary with keys as tuples (i, j) representing class pairs
               and values as A(i|j) values.
    """

    # Unique classes in the dataset
    classes = np.unique(y_window)

    # Initialize a dictionary to store A(i|j) pairs
    A_pairs = {}

    # Iterate over each pair of classes (i, j) where i != j
    for (i, j) in combinations(classes, 2):
        # Get indices for class i and class j
        indices_i = np.where(y_window == i)[0]
        indices_j = np.where(y_window == j)[0]

        # Compare probabilities for class j between samples of class i and class j
        win_count_j = 0
        total_count_j = 0

        # For A(j | i): Compare class j's probabilities
        for idx_i in indices_i:
            for idx_j in indices_j:
                # Compare the probability of class j for both class i and class j samples
                if probs[idx_j, j] > probs[idx_i, j]:
                    win_count_j += 1
                total_count_j += 1

        # Calculate A(j | i)
        A_j_given_i = win_count_j / total_count_j if total_count_j > 0 else 0
        A_pairs[(j, i)] = A_j_given_i

        # Compare probabilities for class i between samples of class j and class i
        win_count_i = 0
        total_count_i = 0

        # For A(i | j): Compare class i's probabilities
        for idx_j in indices_j:
            for idx_i in indices_i:
                # Compare the probability of class i for both class i and class j samples
                if probs[idx_i, i] > probs[idx_j, i]:
                    win_count_i += 1
                total_count_i += 1

        # Calculate A(i | j)
        A_i_given_j = win_count_i / total_count_i if total_count_i > 0 else 0
        A_pairs[(i, j)] = A_i_given_j

    return A_pairs


def average_A_pairs(A_pairs):
    """
    Calculate the average of A(i|j) and A(j|i) for each unique class pair.

    Parameters:
    - A_pairs: Dictionary with keys as tuples (i, j) representing class pairs
               and values as A(i|j) values.

    Returns:
    - A_pairs_avg: Dictionary with keys as tuples (i, j) where i < j,
                   and values as the average of A(i|j) and A(j|i).
    """
    A_pairs_avg = {}

    # Get all unique pairs (i, j) where i < j
    unique_pairs = set((min(i, j), max(i, j)) for i, j in A_pairs.keys())

    for i, j in unique_pairs:
        # Average A(i | j) and A(j | i)
        A_avg = (A_pairs.get((i, j), 0) + A_pairs.get((j, i), 0)) / 2
        A_pairs_avg[(i, j)] = A_avg

    return A_pairs_avg


def calculate_M(A_pairs_avg, n_classes):
    """
    Calculate the overall separability metric M based on A(i, j) pairs.

    Parameters:
    - A_pairs_avg: Dictionary with keys as tuples (i, j) representing class pairs,
                   and values as the average of A(i|j) and A(j|i).
    - n_classes: Total number of classes (c).

    Returns:
    - M: The overall separability metric.
    """
    # Sum all A(i, j) values
    A_sum = sum(A_pairs_avg.values())

    # Calculate M
    M = (2 / (n_classes * (n_classes - 1))) * A_sum
    return M
