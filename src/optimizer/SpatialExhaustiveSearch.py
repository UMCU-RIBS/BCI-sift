# -------------------------------------------------------------
# BCI-FeaST
# Copyright (c) 2024
#       Dirk Keller,
#       Elena Offenberg,
#       Nick Ramsey's Lab, University Medical Center Utrecht, University Utrecht
# Licensed under the MIT License [see LICENSE for detail]
# -------------------------------------------------------------

from typing import Tuple, Union, Dict, Any, Optional, Type

import numpy
import pandas as pd
from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted as sklearn_is_fitted
from sklearn.base import BaseEstimator

from src.optimizer.backend._backend import compute_subgrid_dimensions

from ._Base_SES import SpatialExhaustiveSearch as SES
from ._Base_Optimizer import BaseOptimizer

class SpatialExhaustiveSearch(BaseOptimizer):
    """
    Implements a spatially constrained exhaustive search for finding
    global best channel combinations within a grid based on a given metric.

    Parameters:
    -----------
    :param grid: numpy.ndarray
        The grid structure specifying how channels are arranged.
    :param estimator: Union[BaseEstimator, Pipeline]
        The machine learning estimator or pipeline to evaluate
        channel combinations.
    :param estimator_params: Dict[str, any], default = {}
        Optional parameters to adjust the estimator parameters.
    :param metric: str, default = 'f1_weighted'
        The metric to optimize, compatible with scikit-learn metrics.
    :param cv: Union[BaseCrossValidator, int], default = 10
        Cross-validation splitting strategy, can be a fold number
        or a scikit-learn cross-validator.
    :param groups: numpy.ndarray, default = None
        Groups for LeaveOneGroupOut-crossvalidator
    :param n_jobs: int, default = 1
        Number of parallel jobs to run during cross-validation.
         '-1' uses all available cores.
    :param seed: Optional[int], default = None
        Seed for randomness, ensuring reproducibility.
    :param verbose: Union[bool, int], default = False
        Enables verbose output during the optimization process.
    :param **kwargs: Dict[str, any]
        Optional parameters to adjust the estimator parameters.

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
            grid: numpy.array,
            estimator: Union[BaseEstimator, Pipeline],
            estimator_params: Dict[str, Any] = {},
            metric: str = 'f1_weighted',
            cv: Union[BaseCrossValidator, int] = 10,
            groups: numpy.ndarray = None,

            # Misc
            n_jobs: int = 1,
            seed: Optional[int] = None,
            verbose: Union[bool, int] = False
    ) -> None:
        super().__init__(grid, estimator, estimator_params, metric, cv, groups, n_jobs, seed, verbose)

    def fit(
            self, X: numpy.ndarray, y: numpy.ndarray = None
    ) -> Type['SpatialExhaustiveSearch']:
        """
        Fit method optimizes the channel combination with
        Spatial Exhaustive Search.

        Parameters:
        -----------
        :param X: numpy.ndarray
            Array-like with dimensions [samples, channel_height, channel_width, time]
        :param y: numpy.ndarray, default = None
            Array-like with dimensions [targets].

        Return:
        -----------
        :return: Type['SpatialExhaustiveSearch']
        """
        self.X_, self.y_ = self._validate_data(
            X, y, reset=False, **{'ensure_2d': False, 'allow_nd': True}
        )


        self.iter_ = int(0)
        self.result_grid_ = []

        self.set_estimator_params()

        self.solution_, self.mask_, self.score_ = self._run()

        # Conclude the result grid (Calculate the Size and Height)
        self.result_grid_ = pd.concat(self.result_grid_, axis=0, ignore_index=True)
        self.result_grid_[['Height', 'Width']] = self.result_grid_['Mask'].apply(
            lambda mask: pd.Series(compute_subgrid_dimensions(mask))
        )
        columns = list(self.result_grid_.columns)
        size_index = columns.index('Size')
        new_order = columns[:size_index + 1] + ['Height', 'Width'] + columns[size_index + 1:-2]
        self.result_grid_ = self.result_grid_[new_order]

        return self

    def transform(
            self, X: numpy.ndarray, y: numpy.ndarray = None
    ) -> numpy.ndarray:
        """
        Transforms the input with the mask obtained from
        the solution of Spatial Exhaustive Search.

        Parameters:
        -----------
        :param X: numpy.ndarray
            Array-like with dimensions [samples, channel_height, channel_width, time]
        :param y: numpy.ndarray, default = None
            Array-like with dimensions [targets].

        Return:
        -----------
        :return: numpy.ndarray
            Returns a filtered array-like with dimensions
            [samples, channel_height, channel_width, time]
        """
        sklearn_is_fitted(self)

        return X[:, self.mask_, :]

    def _run(
            self
    ) -> Tuple[numpy.ndarray, numpy.ndarray, float]:
        """
        Executes the Spatial Exhaustive Search.

        Parameters:
        --------
        :return: Tuple[numpy.ndarray, float]
            The best channel configuration and its score.

        Returns:
        --------
        :return: Tuple[numpy.ndarray, numpy.ndarray, float, pandas.DataFrame]
            A tuple with the solution, mask, the evaluation scores and the optimization history.
        """
        # Initialize and run the SSHC optimizer
        es = SES(
            func=self._objective_function,
            channel_grid=self.grid,
            verbose=self.verbose
        )
        score, mask = es.run()

        solution = mask.reshape(-1).astype(float)
        best_state = mask
        best_score = score * 100
        return solution, best_state, best_score
