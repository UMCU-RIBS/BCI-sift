# -------------------------------------------------------------
# Channel Elimination
# Copyright (c) 2024
#       Dirk Keller,
#       Nick Ramsey's Lab, University Medical Center Utrecht, University Utrecht
# Licensed under the MIT License [see LICENSE for detail]
# -------------------------------------------------------------

import random
from copy import copy
from typing import Tuple, List, Union, Dict, Any, Optional, Type

import numpy
import numpy as np
import pandas as pd

from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import Pipeline
# from sklearn.utils._metadata_requests import _RoutingNotSupportedMixin
from sklearn.utils.validation import check_is_fitted as sklearn_is_fitted
from sklearn.base import BaseEstimator

from ._utils import to_dict_keys, compute_subgrid_dimensions

from ._Base_SSHC import SpatialStochasticHillClimbing as SSHC
from ._Base_Optimizer import BaseOptimizer

class SpatialStochasticHillClimbing(BaseOptimizer):
    """
    Implements a stochastic hill climbing algorithm optimized for finding
    the best channel combinations within a grid based on a given metric.
    This optimization technique incorporates exploration-exploitation
    balance, effectively searching through the channel configuration space.

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
    :param n_iter: int, default=100
        Number of reinitializations for random starting positions of the algorithm.
    :param epsilon: Tuple[float, float], default = (0.75, 0.25)
        Exploration factor, a tuple indicating the starting
        and final exploration values.
    :param tol: float, default = 1e-5
        The function tolerance; if the change in the best objective value
        is below this for `patience` iterations, the optimization will stop early.
    :param patience: int, default = int(1e5)
        The number of iterations for which the objective function
        improvement must be below `tol` to stop optimization.
    :param prior: Optional[numpy.ndarray], default = None
        Explicitly initialize the optimizer state.
        If set to None if particles positions are initialized randomly.
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
            grid: numpy.ndarray,
            estimator: Union[BaseEstimator, Pipeline],
            estimator_params: Dict[str, Any],
            metric: str = 'f1_weighted',
            cv: Union[BaseCrossValidator, int] = 10,
            groups: numpy.ndarray = None,
            # Spatial Stochastic Search Settings
            n_iter: int = 100,
            epsilon: Tuple[float, float] = (0.75, 0.25),
            prior: Optional[numpy.ndarray] = None,

            # Training Settings
            tol: float = 1e-5,
            patience: int = int(1e5),

            # Misc
            n_jobs: int = 1,
            seed: Optional[int] = None,
            verbose: Union[bool, int] = False,
    ) -> None:

        super().__init__(grid, estimator, estimator_params, metric, cv, groups, n_jobs, seed, verbose)

        # Spatial Stochastic Search Settings
        self.n_iter = n_iter
        self.epsilon = epsilon

        # Training Settings
        self.tol = tol
        self.patience = patience
        self.prior = prior

    def fit(
            self, X: numpy.ndarray, y: numpy.ndarray = None
    ) -> Type['StochasticHillClimbing']:
        """
        Fit method optimizes the channel combination with
        Spatial Stochastic Hill Climbing.

        Parameters:
        -----------
        :param X: numpy.ndarray
            Array-like with dimensions [samples, channel_height, channel_width, time]
        :param y: numpy.ndarray, default = None
            Array-like with dimensions [targets].

        Return:
        -----------
        :return: Type['StochasticHillClimbing']
        """
        self.X_, self.y_ = self._validate_data(
            X, y, reset=False, **{'ensure_2d': False, 'allow_nd': True}
        )

        self.iter_ = int(0)
        self.result_grid_ = []

        self.set_estimator_params()

        # Set the seeds
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.prior_ = self.prior
        if self.prior is not None:
            if self.prior.shape != self.grid.reshape(-1).shape:
                raise RuntimeError(
                    f'The argument prior {self.prior.shape} must match '
                    f'the number of cells of grid {self.grid.reshape(-1).shape}.')

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
        the solution of Spatial Stochastic Hill Climbing algorithm.

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
        # Initialize and run the SSHC optimizer
        sshc = SSHC(
            func=self._objective_function,
            channel_grid=self.grid,
            epsilon=self.epsilon,
            n_iter=self.n_iter,
            tol=self.tol,
            patience=self.patience,
            prior=self.prior,
            seed=self.seed,
            verbose=self.verbose
        )
        score, mask = sshc.run()

        solution = mask.reshape(-1).astype(float)
        best_state = mask
        best_score = score * 100
        return solution, best_state, best_score

    def _objective_function(
            self, mask: Union[bool, numpy.ndarray], candidate_directions: List[numpy.ndarray],
            eval_hist: Dict[str, float]
    ) -> List[Tuple[numpy.ndarray, float]]:
        """
        Evaluates each candidate channel expansion and computes their scores.

        Parameters:
        -----------
        :param mask : numpy.ndarray
            The current mask of included channels.
        :param candidate_directions : List[np.ndarray]
            The directions in which the subgrid could potentially expand.

        Returns:
        --------
        :return: List[Tuple[numpy.ndarray, float]]
            A list of tuples with candidate channels and their evaluation scores.
        """
        # Evaluate each combination and store the results
        results = []
        for candidate_id in candidate_directions:
            candidate_mask = copy(mask)
            candidate_mask[candidate_id[:, 0], candidate_id[:, 1]] = True  # Temporarily include the candidate

            # Generate a key for eval_hist to check if this configuration was already evaluated
            channel_ids = to_dict_keys(self.grid[candidate_mask].flatten())
            if channel_ids in eval_hist:
                results.append((candidate_id, eval_hist[channel_ids]))
                continue

            self.iter_ += 1

            # If not previously evaluated, perform evaluation on the data
            X_sub = self.X_[:, candidate_mask].reshape(self.X_.shape[0], -1)
            scores = self._evaluate_candidates(X_sub)

            score = scores.mean()
            results.append((candidate_id, score))

            self._save_statistics(candidate_mask.reshape(self.grid.shape), scores)

        return results

