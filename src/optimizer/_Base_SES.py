# -------------------------------------------------------------
# BCI-FeaST
# Copyright (c) 2024
#       Dirk Keller,
#       Elena Offenberg,
#       Nick Ramsey's Lab, University Medical Center Utrecht, University Utrecht
# Licensed under the MIT License [see LICENSE for detail]
# -------------------------------------------------------------


from copy import copy
from typing import Tuple, List

import numpy
import numpy as np
from sklearn.base import TransformerMixin, MetaEstimatorMixin
from tqdm import tqdm


class SpatialExhaustiveSearch(MetaEstimatorMixin, TransformerMixin):
    """
    Performs an exhaustive search over all possible subgrids in a given channel grid,
    evaluating each using a specified objective function.

    Parameters:
    -----------
    :param channel_grid: numpy.ndarray
        The grid of channel IDs.
    :param func: callable
        The objective function, which takes a mask and a list of included channels and returns a metric score.
    :param verbose: bool, default = False
        If set to True, outputs progress messages during the optimization process.

    Returns:
    --------
    :return: None
    """

    def __init__(
            self,
            channel_grid,
            func,
            verbose=False
    ) -> None:
        self.channel_grid = channel_grid
        self.objective = func
        self.verbose = verbose

    def run(
            self
    ) -> Tuple[int, numpy.ndarray]:
        """
        Executes the exhaustive search to find the best subgrid spatially constrained
        channel combination.

        Iteratively searches through the whole subgrid space.
        Records and returns the evaluation history.

        Returns:
        --------
        :return: tuple
            A tuple containing the best score achieved, the corresponding mask
            for the best subgrid, and a DataFrame detailing the evaluation results
            for each configuration.
        """
        best_score = 0.0
        best_mask = None

        # Main loop over the number of starting positions
        height, width = self.channel_grid.shape
        subgrids = self.generate_subgrids(height, width)

        mask_template = np.zeros_like(self.channel_grid, dtype=bool)

        progress_bar = tqdm(range(len(subgrids)), desc=self.__class__.__name__, disable=not self.verbose, leave=True)

        for idx in progress_bar:
            start_row, start_col, end_row, end_col = subgrids[idx]
            mask = copy(mask_template)

            mask[start_row:end_row, start_col:end_col] = True

            # Calculate the score for the current subgrid
            score = self.objective(mask)

            # Check if this is the best score so far
            if score > best_score:
                best_score = score
                best_mask = mask

            progress_bar.set_postfix({'score': f"{best_score:.6f}"})

        return best_score, best_mask

    @staticmethod
    def generate_subgrids(
            grid_height: int, grid_width: int
    ) -> List[Tuple[int, int, int, int]]:
        """
        Generates all possible subgrids within a given grid height and width.
        Each subgrid is defined by its starting and ending coordinates.

        Parameters:
        -----------
        :param grid_height: int
            The height of the grid.
        :param grid_width: int
            The width of the grid.

        Returns:
        --------
        :return: List[Tuple[int, int, int, int]]
            A list of tuples, where each tuple contains the coordinates of a subgrid in the format
            (start_row, start_col, end_row, end_col).
        """
        subgrids = []
        for start_row in range(grid_height + 1):
            for start_col in range(grid_width + 1):
                for end_row in range(start_row, grid_height + 1):
                    for end_col in range(start_col, grid_width + 1):
                        if start_row < end_row and start_col < end_col:
                            subgrids.append((start_row, start_col, end_row, end_col))
        return subgrids
