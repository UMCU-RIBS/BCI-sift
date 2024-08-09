# -------------------------------------------------------------
# Channel Elimination
# Copyright (c) 2024
#       Dirk Keller,
#       Nick Ramsey's Lab, University Medical Center Utrecht, University Utrecht
# Licensed under the MIT License [see LICENSE for detail]
# -------------------------------------------------------------


import re

from typing import Tuple, List, Union, Dict, Any, Optional, Type

import numpy
import numpy as np


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

    def __init__(
            self,
            verbose: bool = False
    ) -> None:
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
                print(f"Local Minimum {self.iteration} detected: Performance Metric = {np.round(-f, 4)}")
                print(f"Reached maximum performance")
            return True

        if self.verbose:
            print(
                f"Local Minimum {self.iteration} detected: Performance Metric = {np.round(-f, 4)}")
        self.iteration += 1


def to_dict_keys(
        arr: numpy.ndarray
) -> str:
    """
    Convert a list of integers to a dictionary key string, with integers
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
    return '-'.join(map(str, sorted(arr)))


def channel_id_to_int(
        l: List[str]
) -> List[int]:
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
    return [int(re.search(r'\d+', s).group()) for s in l if re.search(r'\d+', s)]


def grid_to_channel_id(
        grid: List[List[Any]]
) -> Dict[Any, Tuple[int, int]]:
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


def compute_subgrid_dimensions(
        mask: numpy.ndarray
) -> Tuple[int, int]:
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

    min_row_idx, max_row_idx = true_indices[:, 0].min(), true_indices[:, 0].max()
    min_col_idx, max_col_idx = true_indices[:, 1].min(), true_indices[:, 1].max()

    length = max_row_idx - min_row_idx + 1
    width = max_col_idx - min_col_idx + 1
    return length, width
