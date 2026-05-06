# -------------------------------------------------------------
# BCI-FeaST
# Copyright (c) 2025
#       Dirk Keller,
#       Elena Offenberg,
#       Nick Ramsey's Lab, University Medical Center Utrecht, University Utrecht
# Licensed under the MIT License [see LICENSE for detail]
# -------------------------------------------------------------
import multiprocessing
from functools import partial
from typing import Tuple, Union, Dict, Any, Optional, List, Callable

import numpy
import ray
import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from .Base_Optimizer import BaseOptimizer
from .backend._backend import compute_subgrid_dimensions


class ContiguousExhaustiveSearch(BaseOptimizer):
    """
    Implements a contiguous constrained exhaustive search based on a given metric, 
    for example for finding the global best channel combinations within a grid .

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
        Fit the model to the data, search through the contiguous
        constrained channel combinations.
    - transform:
        Apply the mask obtained from the search to transform the data.
    - run:
        Execute the contiguous exhaustive search.
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

    # >>> shc = ContiguousExhaustiveSearch(grid, estimator, verbose=True)
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
        feature_space: str,
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
            feature_space,
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

    #TODO: fix naming & documentation of this method
    def _compute_objective(
        self,
        masks: numpy.ndarray,
    ) -> numpy.ndarray:
        """
        Computes their objective function scores of a set of masks. This method
        allows the exhaustive search algorithm to interface correctly with the objective function by
        converting the input mask tensor into individuals and evaluating them.

        If more than one cpu core is passed, then the evaluation of the particles is
        done in parallel using multiple processes.

        Parameters
        ----------
        :param masks: numpy.ndarray
            The masks for which to compute the objective function scores.

        Returns
        -------
        numpy.ndarray
            The performance-matrix for a collection of masks.
        """
        positions_split = numpy.array_split(
            masks, masks.shape[0]
        )

        # fmt: off
        # Use multiprocessing to compute scores in parallel
        if self.n_jobs > 1:
            return ray.get(
                [self._objective_function_wrapper.remote(self, pos) for pos in positions_split]
            )
        # Alternatively, compute scores sequentially
        else:
            return [self._objective_function(pos) for pos in positions_split]
        # fmt: on

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
                f"Only one or two dimensions are allowed. Got {len(self.dimensions)}."
            )
        if self.strategy == "conditional":
            raise ValueError(
                f"Contiguous Exhaustive Search requires access to all dimensions at the"
                f" same time, hence only joint is allowed. Got {len(self.strategy)}."
            )

        wait = 0
        best_score = 0.0
        best_state = None

        # Main loop over the number of starting positions
        grid = np.arange(1, np.prod(self._dim_size) + 1).reshape(self._dim_size)
        if grid.ndim==1:
            grid_shape = (grid.shape[0],1)
            subgrids = self._generate_subgrids(*grid_shape) #add empty dimension for compatibility
        else:
            grid_shape = grid.shape
            subgrids = self._generate_subgrids(*grid_shape)

        total_iterations = len(subgrids)
        chunk_size = max(total_iterations // self.n_jobs, self.n_jobs) # Ensure at least n_jobs iterations per chunk
        print(chunk_size)
        chunks = [
            subgrids[i : i + chunk_size] for i in range(0, total_iterations, chunk_size) #total_iterations
        ]
        
        progress_bar = tqdm(
            range(len(chunks)), #range(chunk_size)
            desc=self.__class__.__name__,
            postfix=f"{best_score:.6f}",
            disable=not self.verbose,
            leave=True,
        )
        for iteration in progress_bar:
            mask = np.zeros((chunk_size, *grid.shape), dtype=bool)

            for i in range(len(chunks[iteration])):
                start_row, start_col, end_row, end_col = chunks[iteration][i]
                mask[i, start_row:end_row, start_col:end_col] = True

            scores = self._compute_objective(
                mask
            )

            # Update logs and early stopping
            wait += 1
            score = numpy.max(scores)
            if best_score < score:
                if score - best_score > self._tol:
                    wait = 0
                best_score = score
                best_state = mask[scores == score]
                best_state = best_state[numpy.random.choice(best_state.shape[0])]
            progress_bar.set_postfix(best_score=f"{best_score:.6f}")
            if wait > self._patience:
                progress_bar.set_postfix(
                    best_score=f"Early Stopping Criteria reached: {best_score:.6f}"
                )
                break
            elif score >= 1.0:
                progress_bar.set_postfix(
                    best_score=f"Maximum score reached: {best_score:.6f}"
                )
                break
            elif self.callback is not None:
                if self.callback(best_score, best_state, self.result_grid_):
                    progress_bar.set_postfix(
                        best_score=f"Stopped by callback: {best_score:.6f}"
                    )
                    break

        solution = best_state.reshape(-1).astype(float)
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
        Finalizes the result grid. For the Contiguous Exhaustive Search, the height
        and width of the included area is added.

        Returns:
        --------
        returns: None
        """
        if self.n_jobs > 1:
            self.result_grid_ = pd.concat(
                ray.get(self.result_grid_.get.remote()), axis=0, ignore_index=True
            )
        else:
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


    @ray.remote
    def _objective_function_wrapper(self, mask: numpy.ndarray) -> float:
        """
        Wraps the objective function to adapt it for compatibility with ray's cpu
        parallelization.

        Parameters:
        -----------
        mask : numpy.ndarray
            An individual mask representing potential solutions.

        Returns:
        --------
        :return: numpy.ndarray
            The mask performanc score.
        """
        return self._objective_function(mask)
