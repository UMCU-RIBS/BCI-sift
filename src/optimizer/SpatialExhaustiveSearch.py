# -------------------------------------------------------------
# BCI-FeaST
# Copyright (c) 2024
#       Dirk Keller,
#       Elena Offenberg,
#       Nick Ramsey's Lab, University Medical Center Utrecht, University Utrecht
# Licensed under the MIT License [see LICENSE for detail]
# -------------------------------------------------------------
from typing import Tuple, Union, Dict, Any, Optional, List, Callable, Type

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
    :param dims: Tuple[int, ...]
        A tuple of dimensions indies tc apply the feature selection onto.
        Any combination of dimensions can be specified, except for
        dimension 'zero', which represents the samples.
    :param estimator: Union[Any, Pipeline]
        The machine learning model or pipeline to evaluate feature sets.
    :param estimator_params: Optional[Dict[str, any]], default = None
         Optional parameters to adjust the estimator parameters.
    :param metric: str, default = 'f1_weighted'
        The metric to optimize. Must be scikit-learn compatible.
    :param cv: Union[BaseCrossValidator, int], default = 10
        The cross-validation strategy or number of folds.
        If an integer is passed, train_test_split() for 1 and
        StratifiedKFold() is used for >1 as default.
    :param groups: Optional[numpy.ndarray], default = None
        Groups for LeaveOneGroupOut-crossvalidator
    :param patience: int, default = 1e5
        The number of iterations for which the objective function
        improvement must be below tol to stop optimization.
    :param tol: float, default = 1e-5
        The function tolerance; if the change in the best objective value
        is below this for 'patience' iterations, the optimization will stop early.
    :param bounds: Tuple[float, float], default = (0.0, 1.0)
        Has no effect but is kept for consistency.
    :param prior: Optional[numpy.ndarray], default = None
        Has no effect but is kept for consistency.
    :param callback: Optional[Union[Callable, Type]], default = None, #TODO add description and callback design
    :param n_jobs: int, default = 1
        The number of parallel jobs to run during cross-validation.
    :param seed: Optional[int], default = None
        Setting a seed to fix randomness (for reproducibility).
        Default does not use a seed.
    :param verbose: Union[bool, int], default = False
         If set to True, enables the output of progress status
         during the optimization process.

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
            dims: Tuple[int, ...],
            estimator: Union[Any, Pipeline],
            estimator_params: Optional[Dict[str, any]] = None,
            metric: str = 'f1_weighted',
            cv: Union[BaseCrossValidator, int] = 10,
            groups: Optional[numpy.ndarray] = None,

            # Training Settings
            tol: float = 1e-5,
            patience: int = 1e5,
            bounds: Tuple[float, float] = (0.0, 1.0),
            prior: Optional[numpy.ndarray] = None,
            callback: Optional[Union[Callable, Type]] = None,

            # Misc
            n_jobs: int = 1,
            seed: Optional[int] = None,
            verbose: Union[bool, int] = False
    ) -> None:

        super().__init__(
            dims, estimator, estimator_params, metric, cv, groups, tol,
            patience, bounds, prior, callback, n_jobs, seed, verbose
        )

    def _run(
            self
    ) -> Tuple[numpy.ndarray, numpy.ndarray, float]:
        """
        Executes the Spatial Exhaustive Search.

        Returns:
        --------
        :return: Tuple[numpy.ndarray, numpy.ndarray, float, pandas.DataFrame]
            A tuple with the solution, mask, the evaluation scores and the optimization history.
        """
        if len(self.dims) > 2:
            raise ValueError(f'Only two dimensions are allowed. Got {len(self.dims)}.')
        wait = 0
        best_score = 0.0
        best_state = None

        # Main loop over the number of starting positions
        grid_dimensions = np.array(self.X_.shape)[np.array(self.dims)]
        grid = np.arange(1, np.prod(grid_dimensions) + 1).reshape(grid_dimensions)
        subgrids = self._generate_subgrids(*grid.shape)

        progress_bar = tqdm(range(len(subgrids)), desc=self.__class__.__name__, postfix=f'{best_score:.6f}',
                            disable=not self.verbose, leave=True)
        for iteration in progress_bar:
            mask = np.zeros_like(grid, dtype=bool)
            start_row, start_col, end_row, end_col = subgrids[iteration]
            mask[start_row:end_row, start_col:end_col] = True
            # Calculate the score for the current subgrid and update if it's the best score
            if (score := self._objective_function(mask)) > best_score:
                best_score, best_state = score, mask
                progress_bar.set_postfix(best_score=f'{best_score:.6f}')
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

    def _handle_bounds(
            self
    ) -> None:
        """
        Placeholder method for handling bounds.

        Returns:
        -------
        :returns: None
        """

    def _handle_prior(
            self
    ) -> None:
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

    def _prepare_result_grid(
            self
    ) -> None:
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
        self.result_grid_[['Height', 'Width']] = self.result_grid_['Mask'].apply(
            lambda mask: pd.Series(compute_subgrid_dimensions(mask.reshape(
                tuple(np.array(self.X_.shape)[np.array(self.dims)]))))
        )

        # Reorder the columns to place 'Height' and 'Width' after 'Size'
        columns = list(self.result_grid_.columns)
        size_index = columns.index('Size')
        new_order = columns[:size_index + 1] + ['Height', 'Width'] + columns[size_index + 1:-2]
        self.result_grid_ = self.result_grid_[new_order]
