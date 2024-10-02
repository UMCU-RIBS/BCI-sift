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
from typing import Tuple, List, Union, Dict, Any, Optional, Callable, Type

import numpy
import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from optimizer.backend._backend import to_str, compute_subgrid_dimensions
from .Base_Optimizer import BaseOptimizer


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

    def __init__(self, channel_grid: numpy.ndarray, start_pos: int) -> None:
        self.channel_grid = channel_grid
        self.mask, self.incl_channels = self.generate_boolean_mask(
            channel_grid, start_pos
        )
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
        subgrid = channel_grid == start
        return subgrid, np.argwhere(subgrid)

    @staticmethod
    def get_subgrid_corners(subgrid: Union[bool, numpy.ndarray]) -> numpy.ndarray:
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
        return np.array(
            [
                (row_min, col_min),
                (row_min, col_max),
                (row_max, col_min),
                (row_max, col_max),
            ]
        )

    @staticmethod
    def adjacent_channel_corners(corners: numpy.ndarray) -> numpy.ndarray:
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

    def get_adjacent_channels(self) -> numpy.ndarray:
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
        adj_pad_corners = self.adjacent_channel_corners(
            self.get_subgrid_corners(mask_pad)
        )
        mask_pad[
            adj_pad_corners[0][0] : adj_pad_corners[2][0] + 1,
            adj_pad_corners[0][1] : adj_pad_corners[1][1] + 1,
        ] = 2

        # Adjust mask_pad to mark included channels as part of the subgrid
        # Account for the padding offset when setting included channels
        candidate_mask = mask_pad[1:-1, 1:-1]
        candidate_mask[self.incl_channels[:, 0], self.incl_channels[:, 1]] = 1

        # Remove corner candidates
        adj_corners = np.array(self.adjacent_channel_corners(self.subgrid_corners))
        adj_corners = adj_corners[
            ~np.any(
                np.logical_or(
                    adj_corners < 0, adj_corners >= [*self.channel_grid.shape]
                ),
                axis=1,
            )
        ]
        if len(adj_corners):
            candidate_mask[adj_corners[:, 0], adj_corners[:, 1]] = 0

        # Extract candidate coordinates from the adjusted padded mask
        return np.argwhere(candidate_mask == 2)

    def determine_directions(self) -> List[numpy.ndarray]:
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
        expand_directions[0] = candidates[
            np.where(candidates[:, 0] < self.subgrid_corners[0][0])
        ]  # up direction
        expand_directions[1] = candidates[
            np.where(candidates[:, 0] > self.subgrid_corners[3][0])
        ]  # down direction
        expand_directions[2] = candidates[
            np.where(candidates[:, 1] < self.subgrid_corners[2][1])
        ]  # left direction
        expand_directions[3] = candidates[
            np.where(candidates[:, 1] > self.subgrid_corners[3][1])
        ]  # right direction

        # Clean empty directions from list
        return [direction for direction in expand_directions if direction.shape[0] > 0]

    def expand_subgrid(self, new_chan: np.ndarray) -> None:
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


class SpatialStochasticHillClimbing(BaseOptimizer):
    """
    Implements a stochastic hill climbing algorithm optimized for finding
    the best channel combinations within a grid based on a given metric.
    This optimization technique incorporates exploration-exploitation
    balance, effectively searching through the channel configuration space.

    Parameters:
    -----------
    :param dimensions: Tuple[int, ...]
        A tuple of dimension indices to apply the feature selection onto.
        Any combination of dimensions can be specified, except for
        dimension 'zero', which represents the samples.
    :param estimator: Union[Any, Pipeline]
        The machine learning model or pipeline to evaluate feature sets.
    :param estimator_params: Optional[Dict[str, any]], default = None
         Optional parameters to adjust the estimator parameters.
    :param scoring: str, default = 'f1_weighted'
        The metric to optimize. Must be scikit-learn compatible.
    :param cv: Union[BaseCrossValidator, int], default = 10
        The cross-validation strategy or number of folds.
        If an integer is passed, train_test_split() for 1 and
        StratifiedKFold() is used for >1 as default.
    :param groups: Optional[numpy.ndarray], default = None
        Groups for LeaveOneGroupOut-crossvalidator
    :param n_iter: int, default=100
        Number of reinitialization for random starting positions of the algorithm.
    :param epsilon: Tuple[float, float], default = (0.75, 0.25)
        Exploration factor, a tuple indicating the starting
        and final exploration values.
    :param tol: float, default = 1e-5
        The function tolerance; if the change in the best objective value
        is below this for `patience` iterations, the optimization will stop early.
    :param patience: int, default = 1e5
        The number of iterations for which the objective function
        improvement must be below `tol` to stop optimization.
    :param bounds: Tuple[float, float], default = (0.0, 1.0)
        Has no effect but is kept for consistency.
    :param prior: Optional[numpy.ndarray], default = None
        Explicitly initialize the optimizer state.
        If set to None if the to be optimized features are
        initialized randomly within the bounds.
    :param callback: Optional[Union[Callable, Type]], default = None, #TODO adjust and add design
    :param n_jobs: int, default = 1
        The number of parallel jobs to run during cross-validation.
    :param random_state: Optional[int], default = None
        Setting a seed to fix randomness (for reproducibility).
        Default does not use a seed.
    :param verbose: Union[bool, int], default = False
         If set to True, enables the output of progress status
         during the optimization process.

    Methods:
    --------
    - fit:
        Fit the model to the data, optimizing the channel combinations.
    - transform:
        Apply the mask obtained from the optimization to transform the data.
    - run:
        Execute the spatial stochastic hill climbing optimization process.
    - evaluate_candidates:
        Evaluates the selected features using cross-validation or train-test split.
    - objective_function:
        Evaluate each candidate configuration and return their scores.
    - elimination_plot:
        Generates and saves a plot visualizing the maximum and all scores across different subgrid sizes.
    - importance_plot:
        Generates and saves a heatmap visualizing the importance of each channel within the grid.
    Notes:
    ------
    This implementation is semi-compatible with the scikit-learn
    framework, which builds around two-dimensional feature matrices.
    To use this transformation within a scikit-learn Pipeline, the
    four dimensional data must eb flattened after the first dimension
    [samples, features]. For example, scikit-learn's FunctionTransformer can
    achieve this.

    Examples:
    ---------
    The following example shows how to retrieve a feature mask for
    a synthetic data set.

    # >>> import numpy as np
    # >>> from sklearn.svm import SVC
    # >>> from sklearn.pipeline import Pipeline
    # >>> from sklearn.preprocessing import MinMaxScaler
    # >>> from sklearn.datasets import make_classification
    # >>> from FingersVsGestures.src.channel_elimination import StochasticHillClimbing # TODO adjust
    # >>> X, y = make_classification(n_samples=100, n_features=8 * 4 * 100)
    # >>> X = X.reshape((100, 8, 4, 100))
    # >>> grid = np.arange(1, 33).reshape(X.shape[1:3])
    # >>> estimator = Pipeline([('scaler', MinMaxScaler()), ('svc', SVC())])

    # >>> shc = StochasticHillClimbing(grid, estimator, verbose=True)
    # >>> shc.fit(X, y)
    # >>> print(shc.mask_)
    array([[False  True False False], [False False False False], [ True  True False False], [False False False  True],
           [False False False False], [False False False False], [False False  True False], [False False False False]])
    # >>> print(shc.score_)
     26.966666666666672

    Returns:
    --------
    :return: None
    """

    def __init__(
        self,
        # General and Decoder
        dimensions: Tuple[int, ...],
        estimator: Union[Any, Pipeline],
        estimator_params: Optional[Dict[str, any]] = None,
        scoring: str = "f1_weighted",
        cv: Union[BaseCrossValidator, int] = 10,
        groups: Optional[numpy.ndarray] = None,
        # Spatial Stochastic Search Settings
        n_iter: int = 100,
        epsilon: Tuple[float, float] = (0.75, 0.25),
        # Training Settings
        tol: float = 1e-5,
        patience: int = int(1e5),
        bounds: Tuple[float, float] = (0.0, 1.0),
        prior: Optional[numpy.ndarray] = None,
        callback: Optional[Union[Callable, Type]] = None,
        # Misc
        n_jobs: int = 1,
        random_state: Optional[int] = None,
        verbose: Union[bool, int] = False,
    ) -> None:

        super().__init__(
            dimensions,
            estimator,
            estimator_params,
            scoring,
            cv,
            groups,
            tol,
            patience,
            bounds,
            prior,
            callback,
            n_jobs,
            random_state,
            verbose,
        )

        # Spatial Stochastic Search Settings
        self.n_iter = n_iter
        self.epsilon = epsilon

    def _run(self) -> Tuple[numpy.ndarray, numpy.ndarray, float]:
        """
        Executes the Spatial Stochastic Hill Climbing algorithm.

        Parameters:
        --------
        :return: Tuple[np.ndarray, float]
            The best channel configuration and its score.

        Returns:
        --------
        :return: Tuple[numpy.ndarray, numpy.ndarray, float, pandas.DataFrame]
            A tuples with the solution, mask, the evaluation scores and the optimization history.
        """
        if len(self.dimensions) > 2:
            raise ValueError(
                f"Only two dimensions are allowed. Got {len(self.dimensions)}."
            )

        wait = 0
        best_score = 0.0
        best_state = None
        eval_hist = {}
        e = self.epsilon[0]
        e_min = self.epsilon[1]
        e_decay = (e - e_min) / self.n_iter

        grid_dimensions = np.array(self._X.shape)[np.array(self.dimensions)]
        grid = np.arange(1, np.prod(grid_dimensions) + 1).reshape(grid_dimensions)
        init_pos = grid.flatten()
        if self.prior:
            prior_mask = np.where(self.prior > 0.5, False, True).reshape(grid.shape)
            init_pos = grid[prior_mask].flatten()
        starts = self._set_start(init_pos, self.n_iter)

        # Run Search
        progress_bar = tqdm(
            range(self.n_iter),
            desc=self.__class__.__name__,
            postfix=f"{best_score:.6f}",
            disable=not self.verbose,
            leave=True,
        )
        for _ in progress_bar:
            start = int(np.random.choice(starts, size=1, replace=False))
            rse = RectangleSubgridExpansion(grid, start)

            # Evaluate start position
            # Evaluate each combination and store the results
            candidate_directions = list(
                map(lambda x: x.reshape((1, 2)), rse.incl_channels)
            )
            results = self._local_neighbourhood_search(
                eval_hist, rse, candidate_directions
            )

            eval_hist[str(grid[rse.incl_channels[0, 0], rse.incl_channels[0, 1]])] = (
                np.round(np.mean(results[0][1]), 8)
            )

            # Expansion process using stochastic hill-climbing
            while len(rse.incl_channels) < rse.mask.size:
                new_chan, score = self._step(eval_hist, rse, e)
                rse.expand_subgrid(new_chan)

                channel_ids = to_str(
                    grid[rse.incl_channels[:, 0], rse.incl_channels[:, 1]].flatten()
                )
                eval_hist[channel_ids] = score
                if score > best_score:
                    best_score, best_state = score, copy(rse.mask)
                    progress_bar.set_postfix(best_score=f"{best_score:.6f}")
                    if abs(best_score - score) > self.tol:
                        wait = 0
            if wait > self.patience or best_score >= 1.0:
                progress_bar.write(f"\nMaximum score reached")
                break

            wait += 1
            e -= e_decay if e > e_min else 0

        solution = best_state.reshape(-1).astype(float)
        best_score *= 100
        return solution, best_state, best_score

    def _step(
        self, eval_hist: Dict[str, float], rse: RectangleSubgridExpansion, e: float
    ) -> Tuple[numpy.ndarray, float]:
        """
        Executes a single step in the hill-climbing process,
        choosing between exploration and exploitation.

        Parameters:
        -----------
        :param eval_hist: Dict[str, float]
            A probability to determine the stochastic-greedy strategy.
        :param rse: RectangleSubgridExpansion
            The current state of the subgrid expansion.
        :param e: float
            A probability to determine the stochastic-greedy strategy.

        Returns:
        --------
        :return Tuple[numpy.ndarray, str]
            The chosen new channel to include and its evaluation score.
        """
        candidate_directions = rse.determine_directions()
        # Exploitation: Evaluate all candidates and choose the best
        if random.uniform(0, 1) > e:
            results = self._local_neighbourhood_search(
                eval_hist, rse, candidate_directions
            )
            # results = self.objective(rse.mask, candidate_directions, self.eval_hist)
            new_chan, score = max(results, key=lambda x: x[1])
        # Exploration: Randomly choose a direction and evaluate
        else:
            random_choice = random.randint(0, len(candidate_directions) - 1)
            candidate_directions = [candidate_directions[random_choice]]
            [(new_chan, score)] = self._local_neighbourhood_search(
                eval_hist, rse, candidate_directions
            )
        return new_chan, score

    def _local_neighbourhood_search(self, eval_hist, rse, candidate_directions):
        """
        Executes a local neigbourhood search on the edges of the mask.

        Parameters:
        -----------
        :param eval_hist: Dict[str, float]
            A probability to determine the stochastic-greedy strategy.
        :param rse: RectangleSubgridExpansion
            The current state of the subgrid expansion.
        :param candidate_directions: List[numpy.ndarray]
            A list containing the coordinates of candidates for each
            viable expansion direction (up, down, left, right).

        Returns:
        --------
        :return List[Tuple[numpy.ndarray, float]
            A list of tuples indicating the channel ids and their performance.
        """
        results = []
        # mask = np.full(shape=self.channel_grid.shape, fill_value=False)
        for candidate_id in candidate_directions:
            candidate_mask = copy(rse.mask)
            candidate_mask[candidate_id[:, 0], candidate_id[:, 1]] = (
                True  # Temporarily include the candidate
            )
            # Generate a key for eval_hist to check if this configuration was already evaluated
            channel_ids = to_str(rse.channel_grid[candidate_mask].flatten())
            if channel_ids in eval_hist:
                results.append((candidate_id, eval_hist[channel_ids]))
                continue
            score = self._objective_function(candidate_mask)
            results.append((candidate_id, score))
        return results

    @staticmethod
    def _set_start(
        channels: numpy.ndarray,
        num_samples: int,
        bad_channels: Union[List[int], None] = None,
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

    def _handle_bounds(self) -> None:
        """
        Placeholder method for handling bounds.

        Returns:
        -------
        :returns: None
        """

    def _handle_prior(self) -> None:
        """
        Placeholder method for handling prior.

        Returns:
        -------
        :returns: None
        """

    def _prepare_result_grid(self) -> None:
        """
        Finalizes the result grid. For the Spatial Stochastic Hill Climbing, the height
        and width of the included area is added.

        Returns:
        --------
        returns: None
        """

        # Concatenate the result grid along the rows (axis=0) and reset the index
        self.result_grid_ = pd.concat(self.result_grid_, axis=0, ignore_index=True)

        # Compute the height and width for each mask and assign them to the result grid
        self.result_grid_[["Height", "Width"]] = self.result_grid_["Mask"].apply(
            lambda mask: pd.Series(
                compute_subgrid_dimensions(
                    mask.reshape(
                        tuple(np.array(self._X.shape)[np.array(self.dimensions)])
                    )
                )
            )
        )

        # Reorder the columns to place 'Height' and 'Width' after 'Size'
        columns = list(self.result_grid_.columns)
        size_index = columns.index("Size")
        new_order = (
            columns[: size_index + 1]
            + ["Height", "Width"]
            + columns[size_index + 1 : -2]
        )
        self.result_grid_ = self.result_grid_[new_order]
