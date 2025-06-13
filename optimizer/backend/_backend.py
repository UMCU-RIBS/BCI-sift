# -------------------------------------------------------------
# BCI-FeaST
# Copyright (c) 2024
#       Dirk Keller,
#       Elena Offenberg,
#       Nick Ramsey's Lab, University Medical Center Utrecht, University Utrecht
# Licensed under the MIT License [see LICENSE for detail]
# -------------------------------------------------------------

import re
from typing import Tuple, List, Union, Dict, Any

import numpy
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold


class SimulatedAnnealingReporter:
    """
    Custom callback to monitor the annealing process and log the performance
    metrics during optimization iterations.

    This callback function is particularly useful in optimization scenarios
    where tracking the progression of the search through parameter space and
    responding to specific performance thresholds is crucial.

    Parameters:
    -----------
    :param verbose: bool, default = False
        If True, enables the output of progress and performance metrics
        during the optimization process.

    Methods:
    --------
    __call__:
        Creates logs and terminating the optimization early.

    Returns:
    --------
    :return: None
    """

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
        self.iteration = 1

    def __call__(
        self, x: numpy.ndarray, f: float, context: Dict[str, Any]
    ) -> Union[bool, None]:
        """
        Invoked during each iteration of the optimization process. Logs performance
        metrics and checks if the optimization criteria have been met, potentially
        terminating the optimization early if a sufficient performance level is achieved.

        Parameters:
        :param x: array-like
            Current solution vector in the parameter space.
        :param f: float
            Current value of the objective function, which is being minimized.
        :param context: dict
            Additional information about the current state of the optimization
            process (not used in the current implementation).

        Returns:
        :return: Union[bool, None]
            Returns True if the optimization should be terminated early,
            otherwise None to continue.
        """

        if -f >= 1.0:
            if self.verbose:
                print(
                    f"Local Minimum {self.iteration} detected: Performance Metric = {np.round(-f, 4)}"
                )
                print(f"Reached maximum performance")
            return True

        if self.verbose:
            print(
                f"Local Minimum {self.iteration} detected: Performance Metric = {np.round(-f, 4)}"
            )
        self.iteration += 1


def pso_maxp_stopper(x, f):
    """
    Callback function to stop the PSO if performance reaches -1.

    Parameters:
    -----------
    x : np.ndarray
        The current particle's position.
    f : float
        The current particle's objective function value.

    Returns:
    --------
    bool
        If True, PSO continues; if False, PSO stops early.
    """
    if f <= -1:
        return False  # Return False to stop the optimization
    return True  # Continue optimization


def to_string(value: Union[list, int, float, bool]) -> str:
    """
    Convert a list or integers, floats and bools to a string, with integers
    sorted and joined by dashes.

    Parameters:
    -----------
    :param arr: numpy.ndarray
        A list of integers representing channel IDs.

    Return:
    -------
    :return: List[str]
        A string representation of the list, sorted and joined by dashes,
        suitable for use as a dictionary key.
    """
    return "-".join(map(str, value)) if isinstance(value, list) else str(value)


def to_str(value: Union[list, numpy.ndarray, int, float, bool]) -> str:
    """
    Convert a list of integers to strings, with integers sorted and joined by dashes.

    Parameters:
    -----------
    :param value: Union[list, numpy.ndarray, int, float, bool]
        A list, or ndarray or primitives.

    Return:
    -------
    :return: str
        A string representation of the list, numpy array or primitives, sorted and
        joined by dashes, suitable for use as a dictionary key.
    """
    return (
        "-".join(map(str, value))
        if isinstance(value, (list, numpy.ndarray))
        else str(value)
    )


def channel_id_to_int(l: List[str]) -> List[int]:
    """
    Convert a list of strings to integers.

    Parameters:
    -----------
    :param l: List[str]
        A list of strings representing channel IDs.

    Return:
    -------
    :return: List[int]
        A list of integer representing channel IDs.
    """
    return [int(re.search(r"\d+", s).group()) for s in l if re.search(r"\d+", s)]


def grid_to_channel_id(grid: List[List[Any]]) -> Dict[Any, Tuple[int, int]]:
    """
    Maps each value in a 2D grid to its corresponding (row, column) indices.

    This utility is useful for scenarios where the spatial location of
    each element within a structured grid is significant, such as in certain
    optimization problems or spatial analyses.

    Parameters:
    -----------
    :param grid: List[List[Any]]
        A 2D list where each sublist represents a row, and each element within
        those sublists can be any hashable value that represents data points in the grid.

    Returns:
    --------
    :return: Dict[Any, Tuple[int, int]]
        A dictionary mapping each unique value in the grid to a tuple
        of its (row, column) indices.

    Example:
    --------
    >>> grid_to_channel_id([[1, 2], [3, 4]])
    {1: (0, 0), 2: (0, 1), 3: (1, 0), 4: (1, 1)}
    """
    return {val: (i, j) for i, row in enumerate(grid) for j, val in enumerate(row)}


def compute_subgrid_dimensions(mask: numpy.ndarray) -> Tuple[int, int]:
    """
    Calculates the height and width of the smallest rectangle that encompasses
    all the True values within a 2D mask.

    This function is useful for identifying the dimensions of a rectangular
    region within a grid or image where a certain condition (True value) is met.

    Parameters:
    -----------
    :param mask: numpy.ndarray
        A 2D boolean array (mask) where True values indicate the region of interest.

    Returns:
    --------
    :return: Tuple[int, int]
        The height and width of the rectangle that includes all the True values.

    Example:
    --------
    mask = np.array([
        [False, False, False, False],
        [False, True, True, False],
        [False, True, True, False],
        [False, False, False, False]
    ])
    calculate_height_and_width(mask)
    (2, 2)
    """
    true_indices = np.argwhere(mask)

    # Check if there are any True values in the mask
    if true_indices.size == 0:
        return 0, 0

    # Check if the mask is 2D and handle accordingly
    if true_indices.shape[1] < 2:
        raise ValueError(
            "The mask is not a 2D array or does not contain valid True values."
        )

    min_row_idx, max_row_idx = true_indices[:, 0].min(), true_indices[:, 0].max()
    min_col_idx, max_col_idx = true_indices[:, 1].min(), true_indices[:, 1].max()

    length = max_row_idx - min_row_idx + 1
    width = max_col_idx - min_col_idx + 1
    return length, width


class FlattenTransformer(BaseEstimator, TransformerMixin):
    """
    A scikit-learn transformer that flattens a multidimensional array into a 2D matrix.
    All dimensions after the first are collapsed into a single dimension.
    """

    def fit(self, X, y=None):
        # No fitting necessary, so we just return self
        return self

    def transform(self, X):
        """
        Transforms the input array X into a 2D matrix by flattening all dimensions after the first.

        Parameters:
        X (np.ndarray): A multidimensional array of shape (n_samples, ...).

        Returns:
        np.ndarray: A 2D matrix of shape (n_samples, -1).
        """
        # Ensure X is a numpy array
        X = np.asarray(X)

        # Check if X has at least 2 dimensions
        if X.ndim < 2:
            raise ValueError("Input array must have at least 2 dimensions")

        # Reshape X to (n_samples, -1), collapsing all dimensions after the first
        n_samples = X.shape[0]
        return X.reshape(n_samples, -1)


class SafeVarianceThreshold(BaseEstimator, TransformerMixin):
    """
    A wrapper around scikit-learn's VarianceThreshold that handles the case where no features meet the threshold.
    If an exception is thrown (e.g., because all features have non-zero variance), the original input is returned.
    """

    def __init__(self, threshold=0.0):
        self.threshold = threshold

    def fit(self, X, y=None):
        self.variance_threshold = VarianceThreshold(threshold=self.threshold)
        try:
            self.variance_threshold.fit(X, y)
        except Exception as e:
            # Handle any exceptions that occur during fitting
            self.error_ = e
        return self

    def transform(self, X):
        try:
            # Attempt to transform using VarianceThreshold
            return self.variance_threshold.transform(X)
        except Exception:
            # If an exception occurs, return the original input X
            return X

def divide_into_parts(number, parts):
    # Calculate the base size of each part and the remainder
    base_size = number // parts
    remainder = number % parts
    # Create the list of parts
    result = [base_size + 1] * remainder + [base_size] * (parts - remainder)
    return result