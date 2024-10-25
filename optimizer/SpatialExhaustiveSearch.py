# -------------------------------------------------------------
# BCI-FeaST
# Copyright (c) 2024
#       Dirk Keller,
#       Elena Offenberg,
#       Nick Ramsey's Lab, University Medical Center Utrecht, University Utrecht
# Licensed under the MIT License [see LICENSE for detail]
# -------------------------------------------------------------
import multiprocessing
from functools import partial
from typing import Tuple, Union, Dict, Any, Optional, List, Callable

import numpy
import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from .Base_Optimizer import BaseOptimizer
from .backend._backend import compute_subgrid_dimensions


class SpatialExhaustiveSearch(BaseOptimizer):
    """
    Implements a spatially constrained exhaustive search for finding
    global best channel combinations within a grid based on a given metric.

    Parameters:
    -----------
    Parameters:
    -----------
    :param dimensions: Tuple[int, ...]
        A tuple of dimensions indies tc apply the feature selection onto. Any
        combination of dimensions can be specified, except for dimension 'zero', which
        represents the samples.
    :param estimator: Union[Any, Pipeline]
        The machine learning model or pipeline to evaluate feature sets.
    :param estimator_params: Dict[str, any], optional
        Optional parameters to adjust the estimator parameters.
    :param scoring: str, default = 'f1_weighted'
        The metric to optimize. Must be scikit-learn compatible.
    :param cv: Union[BaseCrossValidator, int, float], default = 10
        The cross-validation strategy or number of folds. If an integer is passed,
        :code:`train_test_split` for <1 and :code:`BaseCrossValidator` is used for >1 as
        default. A float below 1 represents the percentage of training samples for the
        train-test split ratio.
    :param groups: Optional[numpy.ndarray], optional
        Groups for a LeaveOneGroupOut generator.
    :param strategy: str, default = "joint"
        The strategy of optimization to apply. Valid options are: 'joint' and
        'conditional'.
        * Joint Optimization: Optimizes all features simultaneously. Should be only
          selected for small search spaces.
        * Conditional Optimization: Optimizes each feature dimension iteratively,
          building on previous results. Generally, yields better performance for large
          search spaces.
    :param patience: Union[int Tuple[int, ...], default = int(1e5)
        Patience parameter has no effect but is kept for consistency.
    :param bounds: Tuple[float, float], default = (0.0, 1.0)
        Bounds parameter has no effect but is kept for consistency.
    :param prior: numpy.ndarray, optional
        Prior parameter has no effect but is kept for consistency.
    :param callback: Callable, optional
        A callback function of the structure :code: `callback(x, f, context)`, which
        will be called at each iteration. :code: `x` and :code: `f` are the solution and
        function value, and :code: `context` contains the diagnostics of the current
        iteration.
    :param n_jobs: Union[int, float], default = -1
        The number of parallel jobs to run during cross-validation; -1 uses all cores.
    :param random_state: int, optional
        Random State parameter has no effect but is kept for consistency.
    :param verbose: Union[bool, int], default = False
         If set to True, enables the output of progress status during the optimization
         process.

    Methods:
    --------
    - fit:
        Fit the model to the data, search through the spatial
        constrained channel combinations.
    - transform:
        Apply the mask obtained from the search to transform the data.
    - run:
        Execute the spatial exhaustive search.
    - evaluate_candidates:
        Evaluates the selected features using cross-validation or train-test split.
    - objective_function:
        Evaluate each candidate configuration and return their scores.
    - elimination_plot:
        Generate and save a plot visualizing the performance across different subgrid sizes.
    - importance_plot:
        Generate and save a heatmap visualizing the importance of each channel.

    Methods:
    --------
    - fit:
        Fit the optimizer to the data.
    - transform:
        Transform the input data using the mask from the optimization process.

    Notes:
    ------
    This implementation is semi-compatible with the scikit-learn framework, which builds
    around two-dimensional feature matrices. To use this transformation within a
    scikit-learn Pipeline, the four dimensional data must be flattened after the first
    dimension [samples, features]. For example, scikit-learn's
    :code: `FunctionTransformer` can achieve this.


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

    # >>> shc = SpatialExhaustiveSearch(grid, estimator, verbose=True)
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
        cv: Union[BaseCrossValidator, int, float] = 10,
        groups: Optional[numpy.ndarray] = None,
        strategy: str = "joint",
        # Training Settings
        tol: Union[Tuple[int, ...], float] = 1e-5,
        patience: Union[Tuple[int, ...], int] = int(1e5),
        bounds: Tuple[float, float] = (0.0, 1.0),
        prior: Optional[numpy.ndarray] = None,
        callback: Optional[Callable] = None,
        # Misc
        n_jobs: int = -1,
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
            strategy,
            tol,
            patience,
            bounds,
            prior,
            callback,
            n_jobs,
            random_state,
            verbose,
        )

    @staticmethod
    def compute_objective_function(
        mask: numpy.ndarray, objective_func: Callable, pool: multiprocessing.Pool = None
    ):
        """
        Evaluate particles using the objective function

        This method evaluates each particle in the swarm according to the
        objective function passed.

        If a pool is passed, then the evaluation of the particles is done in
        parallel using multiple processes.

        Parameters
        ----------
        :param swarm : Swarm
            A Swarm instance
        :param objective_func : Callable
            Objective function to be evaluated
        :param pool: multiprocessing.Pool, optional
            The pool to be used for parallel particle evaluation

        Returns
        -------
        :return: numpy.ndarray
            Cost-matrix for the given swarm
        """
        if pool is None:
            return objective_func(mask)
        else:
            results = pool.map(
                partial(objective_func),
                numpy.array_split(mask, pool._processes),
            )
            return numpy.concatenate(results)

    def _run(self) -> Tuple[numpy.ndarray, numpy.ndarray, float]:
        """
        Runs the exhaustive search algorithm to optimize the feature configuration, by
        evaluating the objective function :code:`f` for all possible feature
        combinations.

        Returns:
        --------
        Tuple[numpy.ndarray, numpy.ndarray, float]
            The best solution, mask found and their score.
        """
        if len(self.dimensions) > 2:
            raise ValueError(
                f"Only two dimensions are allowed. Got {len(self.dimensions)}."
            )
        if self.strategy == "conditional":
            raise ValueError(
                f"Spatial exhaustive Search requires access to all dimensions at the"
                f" same time, hence only joint is allowed. Got {len(self.strategy)}."
            )

        wait = 0
        best_score = 0.0
        best_state = None

        # Main loop over the number of starting positions
        grid = np.arange(1, np.prod(self._dim_size) + 1).reshape(self._dim_size)
        subgrids = self._generate_subgrids(*grid.shape)

        total_iterations = len(subgrids)
        chunk_size = total_iterations // self.n_jobs
        chunks = [
            subgrids[i : i + chunk_size] for i in range(0, total_iterations, chunk_size)
        ]

        pool = None
        if self.n_jobs > 1:
            self.result_grid_ = multiprocessing.Manager().list()
            pool = multiprocessing.Pool(self.n_jobs)

        progress_bar = tqdm(
            range(chunk_size),
            desc=self.__class__.__name__,
            postfix=f"{best_score:.6f}",
            disable=not self.verbose,
            leave=True,
        )
        for iteration in progress_bar:
            mask = np.zeros((chunk_size, *grid.shape), dtype=bool)

            for i in range(chunk_size):
                start_row, start_col, end_row, end_col = chunks[iteration][i]
                mask[i, start_row:end_row, start_col:end_col] = True

            scores = self.compute_objective_function(
                mask, self._objective_function, pool
            )

            # Calculate the score for the current subgrid and update if it's the best score
            if (score := self._objective_function(mask)) > best_score:
                best_score, best_state = score, mask
                progress_bar.set_postfix(best_score=f"{best_score:.6f}")
                if abs(best_score - score) > self.tol:
                    wait = 0
            if wait > self.patience or score >= 1.0:
                progress_bar.write(f"\nMaximum score reached")
                break
            wait += 1

        solution = best_state.reshape(-1).astype(float)
        best_state = self._prepare_mask(best_state)
        best_score *= 100
        return solution, best_state, best_score

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

    @staticmethod
    def _generate_subgrids(
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

    def _prepare_result_grid(self) -> None:
        """
        Finalizes the result grid. For the Spatial Exhaustive Search, the height
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
