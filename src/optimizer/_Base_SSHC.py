# -------------------------------------------------------------
# BCI-FeaST
# Copyright (c) 2024
#       Dirk Keller,
#       Elena Offenberg,
#       Nick Ramsey's Lab, University Medical Center Utrecht, University Utrecht
# Licensed under the MIT License [see LICENSE for detail]
# -------------------------------------------------------------


import random
from copy import copy
from typing import Tuple, List, Union, Optional

import numpy
import numpy as np
from tqdm import tqdm

from src.optimizer.backend._backend import to_dict_keys, grid_to_channel_id


class RectangleSubgridExpansion:
    """
    Manages the expansion of a rectangular subgrid within a channel grid,
    starting from a given position.

    Parameters:
    -----------
    :param channel_grid: numpy.ndarray
        The grid representing channels, where each cell's value
        corresponds to a unique channel ID.
    :param start_pos: int
        The channel ID from which the subgrid expansion starts.

    Methods:
    --------
    - generate_boolean_mask:
        Creates a boolean mask from the starting position.
    - get_subgrid_corners:
        Identifies corners of the subgrid based on the mask.
    - adjacent_channel_corners:
        Calculates corners of channels adjacent to a given subgrid.
    - get_adjacent_channels:
        Identifies channels adjacent to the subgrid using the current mask.
    - determine_directions:
        Determines possible expansion directions based on adjacent channels.
    - expand_subgrid:
        Expands the subgrid to include new channels.

    Returns:
    --------
    returns: None
    """

    def __init__(
            self,
            channel_grid: numpy.ndarray,
            start_pos: int
    ) -> None:
        self.rows, self.cols = channel_grid.shape
        self.mask, self.incl_channels = self.generate_boolean_mask(channel_grid, start_pos)
        self.subgrid_corners = self.get_subgrid_corners(self.mask)

    @staticmethod
    def generate_boolean_mask(
            channel_grid: numpy.ndarray, start: int
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """
        Generate a mask indicating the subgrid starting from a given channel,
        along with the indices of included channels.

        Parameters:
        -----------
        :param channel_grid: np.ndarray
            The channel grid with the channel ID's to operate on.
        :param start: int
            The starting channel ID for subgrid expansion.

        Return:
        --------
        :return: Tuple[numpy.ndarray, numpy.ndarray]
            A tuple containing the boolean mask of the subgrid and the
            indices of included channels.
        """
        subgrid = (channel_grid == start)
        return subgrid, np.argwhere(subgrid)

    @staticmethod
    def get_subgrid_corners(
            subgrid: Union[bool, numpy.ndarray]
    ) -> numpy.ndarray:
        """
        Identifies the corners of the current subgrid based on the boolean mask.

        Parameters:
        -----------
        :param subgrid: numpy.ndarray
            The subgrid mask to operate on.

        Returns:
        --------
        :return: numpy.ndarray
            An array of coordinates for the corners of the subgrid.
        """
        # Get the coordinates of True values in subgrid
        coordinates = np.argwhere(subgrid == True)

        # Calculate the subgrid_corners (top-left, top-right, bottom-left, bottom-right)
        row_min, col_min = coordinates.min(axis=0)
        row_max, col_max = coordinates.max(axis=0)
        return np.array([(row_min, col_min), (row_min, col_max), (row_max, col_min), (row_max, col_max)])

    @staticmethod
    def adjacent_channel_corners(
            corners: numpy.ndarray
    ) -> numpy.ndarray:
        """
        Calculates the corners of adjacent channels based on the input corners.

        Parameters:
        -----------
        :param corners: numpy.ndarray
            An array of coordinates for the corners of a subgrid.

        Returns:
        --------
        :return: numpy.ndarray
            An array of coordinates for the corners of adjacent channels.
        """
        return corners + np.array([(-1, -1), (-1, 1), (1, -1), (1, 1)])

    def get_adjacent_channels(
            self
    ) -> numpy.ndarray:
        """
        Identifies the channels adjacent to the current subgrid, considering
        the mask's current state.

        This function pads the current mask to handle edge cases and uses
        the padded version to identify channels that are directly adjacent
        to the subgrid. It relies on `self.get_subgrid_corners` to determine
        the subgrid's bounds and on `self.adjacent_channel_corners` to identify
        adjacent areas. The results of this function are used by `determine_directions`
        to identify potential expansion directions.

        Returns:
        --------
        :return: numpy.ndarray
            The coordinates of channels adjacent to the subgrid.
        """
        # Pad the mask and mark the adjacent area outside the subgrid
        mask_pad = np.pad(self.mask.astype(int), pad_width=1, constant_values=-1)
        adj_pad_corners = self.adjacent_channel_corners(self.get_subgrid_corners(mask_pad))
        mask_pad[adj_pad_corners[0][0]:adj_pad_corners[2][0] + 1, adj_pad_corners[0][1]:adj_pad_corners[1][1] + 1] = 2

        # Adjust mask_pad to mark included channels as part of the subgrid
        # Account for the padding offset when setting included channels
        candidate_mask = mask_pad[1:-1, 1:-1]
        candidate_mask[self.incl_channels[:, 0], self.incl_channels[:, 1]] = 1

        # Remove corner candidates
        adj_corners = np.array(self.adjacent_channel_corners(self.subgrid_corners))
        adj_corners = adj_corners[
            ~np.any(np.logical_or(adj_corners < 0, adj_corners >= [self.rows, self.cols]), axis=1)]
        if len(adj_corners):
            candidate_mask[adj_corners[:, 0], adj_corners[:, 1]] = 0

        # Extract candidate coordinates from the adjusted padded mask
        return np.argwhere(candidate_mask == 2)

    def determine_directions(
            self
    ) -> List[numpy.ndarray]:
        """
        Determines potential expansion directions based on the positions
        of adjacent channels.

        Updates `self.subgrid_corners` with the current subgrid's corners
        and evaluates adjacent channels to determine in which directions
        the subgrid can potentially expand.

        Returns:
        --------
        :return: List[numpy.ndarray]
            A list containing the coordinates of candidates for each
            viable expansion direction (up, down, left, right).
        """
        # Calculate corners of the current subgrid
        self.subgrid_corners = self.get_subgrid_corners(self.mask)

        # Identify adjacent channels
        candidates = self.get_adjacent_channels()

        # Iterate through each candidate and corner using NumPy operations
        expand_directions = [np.empty((0, 2), dtype=int) for _ in range(4)]
        expand_directions[0] = candidates[np.where(candidates[:, 0] < self.subgrid_corners[0][0])]  # up direction
        expand_directions[1] = candidates[np.where(candidates[:, 0] > self.subgrid_corners[3][0])]  # down direction
        expand_directions[2] = candidates[np.where(candidates[:, 1] < self.subgrid_corners[2][1])]  # left direction
        expand_directions[3] = candidates[np.where(candidates[:, 1] > self.subgrid_corners[3][1])]  # right direction

        # Clean empty directions from list
        return [direction for direction in expand_directions if direction.shape[0] > 0]

    def expand_subgrid(
            self, new_chan: np.ndarray
    ) -> None:
        """
        Expands the current subgrid to include new channels.

        Parameters:
        -----------
        :param new_chan: numpy.ndarray
            Indices of new channels to include in the subgrid.

        Returns:
        --------
        returns: None
        """
        self.mask[new_chan[:, 0], new_chan[:, 1]] = True
        self.incl_channels = np.vstack([self.incl_channels, new_chan])


class SpatialStochasticHillClimbing:
    """
    Optimizes to find the best channel combinations within a grid
    for a given metric, using a stochastic hill-climbing algorithm
    with exploration-exploitation balance.

    Parameters:
    -----------
    :param channel_grid: Union[List[List[int]], numpy.ndarray]
        The grid of channels to be considered for optimization,
        where each element is a channel ID.
    :param func : callable
        The objective function mapping from channel combination
        to the metric. The objective function must have access to an estimator,
        a metric, the matrix X and target vector y.
    :param epsilon: Tuple[float, float], default = (0.75, 0.25)
        A tuple representing the initial and final exploration factors,
        balancing exploration and exploitation over the course of the optimization.
    :param n_iter: int, default = 100
        The number of iterations for the optimization process.
        Each iteration starts from a random position in the channel grid.
    :param tol: float, default = 1e-5
        The function tolerance; if the change in the best objective value
        is below this for `patience` iterations, the optimization will stop early.
    :param patience: int, default = int(1e5)
        The number of iterations for which the objective function
        improvement must be below `tol` to stop optimization.
    :param prior: Optional[numpy.ndarray], default = None
        Explicitly initialize the optimizer state.
        If set to None if population characteristics are initialized randomly.
    :param seed: Optional[int], default = None
        Setting a seed to fix randomness (for reproduceability).
        Default does not use a seed.
    :param verbose: bool, default = False
        If set to True, enables the output of progress messages
         during the optimization process.


    Methods:
    --------
    - run:
        Executes the optimization process, iterating over potential subgrids.
    - step:
        Performs a single step of the stochastic hill-climbing,
        choosing between exploration and exploitation.
    - set_start:
        Samples potential start positions, excluding bad channels if specified.

    Returns:
    --------
    :return: None
    """

    def __init__(
            self,
            channel_grid: Union[List[List[int]], np.ndarray],
            func: callable,
            epsilon: Tuple[float, float] = (0.75, 0.25),
            n_iter: int = 100,
            tol: float = 1e-5,
            patience: int = 1e5,
            prior: Optional[numpy.ndarray] = None,
            seed: Optional[int] = None,
            verbose: bool = False
    ) -> None:

        self.objective = func
        self.channel_grid = channel_grid
        self.e_init = max(epsilon)
        self.e_min = min(epsilon)
        self.n_iter = n_iter
        self.tol = tol
        self.patience = patience
        self.prior = prior
        self.seed = seed
        self.verbose = verbose

        self.eval_hist = {}

        self.channel_id_map = grid_to_channel_id(channel_grid)
        self.start_pos = list(channel_grid.flatten())
        self.e_decay = (max(epsilon) - min(epsilon)) / n_iter

    # Function to find the best channel combination
    def run(
            self
    ) -> Tuple[int, numpy.ndarray]:
        """
        Executes the optimization process to find the best spatially constrained
        channel combination.

        Iteratively expands subgrids from random starting positions,
        evaluates candidate expansions, and updates the exploration factor.
        Records and returns the evaluation history.

        Returns:
        --------
        :return: tuple
            A tuple containing the best score achieved, the corresponding mask
            for the best subgrid, and a DataFrame detailing the evaluation results
            for each configuration.
        """
        # Set the seeds
        random.seed(self.seed)
        np.random.seed(self.seed)

        init_pos = self.channel_grid.flatten()
        if self.prior is not None:
            prior_mask = np.where(self.prior > .5, False, True).reshape(self.channel_grid.shape)
            init_pos = self.channel_grid[prior_mask].flatten()

        wait = 0
        best_score, best_mask = 0.0, None

        progress_bar = tqdm(range(self.n_iter), desc=self.__class__.__name__, disable=not self.verbose, leave=True)

        # Main loop over the number of starting positions
        starts = self.set_start(init_pos, self.n_iter)
        for _ in progress_bar:
            start = int(np.random.choice(starts, size=1, replace=False))
            rse = RectangleSubgridExpansion(self.channel_grid, start)

            # Evaluate start position
            # Evaluate each combination and store the results
            candidate_directions = list(map(lambda x: x.reshape((1, 2)), rse.incl_channels))
            results = self.local_neighbourhood_search(rse, candidate_directions=candidate_directions)

            self.eval_hist[str(self.channel_grid[rse.incl_channels[0, 0], rse.incl_channels[0, 1]])] = np.round(
                np.mean(results[0][1]), 8)

            # Expansion process using stochastic hill-climbing
            while len(rse.incl_channels) < rse.mask.size:
                new_chan, score = self.step(rse)
                rse.expand_subgrid(new_chan)

                channel_ids = to_dict_keys(
                    self.channel_grid[rse.incl_channels[:, 0], rse.incl_channels[:, 1]].flatten())
                self.eval_hist[channel_ids] = score

                if score > best_score:
                    best_score = score
                    best_mask = copy(rse.mask)

                progress_bar.set_postfix({'score': f"{best_score:.6f}"})
                if abs(best_score - score) > self.tol:
                    wait = 0
                else:
                    wait += 1
            if wait > self.patience:  # or best_score >= 1.0:
                break

            self.e_init -= self.e_decay if self.e_init > self.e_min else 0
        return best_score, best_mask

    def step(
            self, rse: RectangleSubgridExpansion
    ) -> Tuple[numpy.ndarray, float]:
        """
        Executes a single step in the hill-climbing process,
        choosing between exploration and exploitation.

        Parameters:
        -----------
        :param rse: RectangleSubgridExpansion
            The current state of the subgrid expansion.

        Returns:
        --------
        :return Tuple[numpy.ndarray, str]
            The chosen new channel to include and its evaluation score.
        """
        candidate_directions = rse.determine_directions()

        # Exploitation: Evaluate all candidates and choose the best
        if random.uniform(0, 1) > self.e_init:
            results = self.local_neighbourhood_search(rse, candidate_directions)
            # results = self.objective(rse.mask, candidate_directions, self.eval_hist)
            new_chan, score = max(results, key=lambda x: x[1])

        # Exploration: Randomly choose a direction and evaluate
        else:
            random_choice = random.randint(0, len(candidate_directions) - 1)
            candidate_directions = [candidate_directions[random_choice]]
            [(new_chan, score)] = self.local_neighbourhood_search(rse, candidate_directions)
            # new_chan, score = self.objective(rse.mask, candidate_directions, self.eval_hist)[0]
        return new_chan, score

    def local_neighbourhood_search(self, rse, candidate_directions):

        results = []
        # mask = np.full(shape=self.channel_grid.shape, fill_value=False)
        for candidate_id in candidate_directions:
            candidate_mask = copy(rse.mask)
            candidate_mask[candidate_id[:, 0], candidate_id[:, 1]] = True  # Temporarily include the candidate

            # Generate a key for eval_hist to check if this configuration was already evaluated
            channel_ids = to_dict_keys(self.channel_grid[candidate_mask].flatten())
            if channel_ids in self.eval_hist:
                results.append((candidate_id, self.eval_hist[channel_ids]))
                continue

            score = self.objective(candidate_mask)
            results.append((candidate_id, score))
        return results

    @staticmethod
    def set_start(
            channels: numpy.ndarray, num_samples: int, bad_channels: Union[List[int], None] = None
    ) -> numpy.ndarray:
        """
        Uniformly samples start positions from the available channels,
        excluding any specified bad channels.

        Parameters:
        -----------
        :param channels: np.ndarray
            An array of channels from which to sample the start positions.
        :param num_samples: int
            The number of samples to draw for start positions.
        :param bad_channels: Union[List[int], None]
            A list of channels to exclude from the start position sampling.

        Returns:
        --------
        :return: numpy.ndarray
            An array of sampled start positions.
        """
        # Remove the bad channels to get an array of good channels
        if bad_channels is not None:
            channels = np.setdiff1d(channels, np.array(bad_channels))
        return np.random.choice(channels, size=num_samples, replace=True)
