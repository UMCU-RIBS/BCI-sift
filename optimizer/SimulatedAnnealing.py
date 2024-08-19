# -------------------------------------------------------------
# Channel Elimination
# Copyright (c) 2024
#       Dirk Keller,
#       Nick Ramsey's Lab, University Medical Center Utrecht, University Utrecht
# Licensed under the MIT License [see LICENSE for detail]
# -------------------------------------------------------------

import random
from typing import Tuple, List, Union, Dict, Any, Optional, Type

import numpy
import numpy as np
import pandas as pd

from scipy.optimize import dual_annealing
from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted as sklearn_is_fitted

from .utils import SimulatedAnnealingReporter
from ._Base_Optimizer import BaseOptimizer

class SimulatedAnnealing(BaseOptimizer):
    """
    This class implements an Simulated Annealing algorithm for optimizing
    EEG/MEG/ECOG channels within a structured grid. The class rests on the
    shoulders of scipy's dual annealing implementation (see scipys dual
    annealing implementation for more information). This stochastic
    approach derived from [3]_ combines the generalization of CSA (Classical
    Simulated Annealing) and FSA (Fast Simulated Annealing) [1]_ [2]_ coupled
    to a strategy for applying a local search on accepted locations [4]_.
    An alternative implementation of this same algorithm is described in [5]_
    and benchmarks are presented in [6]_. This approach introduces an advanced
    method to refine the solution found by the generalized annealing
    process. This algorithm uses a distorted Cauchy-Lorentz visiting
    distribution, with its shape controlled by the parameter :math:`q_{v}`

    .. math::

        g_{q_{v}}(\\Delta x(t)) \\propto \\frac{ \\
        \\left[T_{q_{v}}(t) \\right]^{-\\frac{D}{3-q_{v}}}}{ \\
        \\left[{1+(q_{v}-1)\\frac{(\\Delta x(t))^{2}} { \\
        \\left[T_{q_{v}}(t)\\right]^{\\frac{2}{3-q_{v}}}}}\\right]^{ \\
        \\frac{1}{q_{v}-1}+\\frac{D-1}{2}}}

    Where :math:`t` is the artificial time. This visiting distribution is used
    to generate a trial jump distance :math:`\\Delta x(t)` of variable
    :math:`x(t)` under artificial temperature :math:`T_{q_{v}}(t)`.

    From the starting point, after calling the visiting distribution
    function, the acceptance probability is computed as follows:

    .. math::

        p_{q_{a}} = \\min{\\{1,\\left[1-(1-q_{a}) \\beta \\Delta E \\right]^{ \\
        \\frac{1}{1-q_{a}}}\\}}

    Where :math:`q_{a}` is a acceptance parameter. For :math:`q_{a}<1`, zero
    acceptance probability is assigned to the cases where

    .. math::

        [1-(1-q_{a}) \\beta \\Delta E] < 0

    The artificial temperature :math:`T_{q_{v}}(t)` is decreased according to

    .. math::

        T_{q_{v}}(t) = T_{q_{v}}(1) \\frac{2^{q_{v}-1}-1}{\\left( \\
        1 + t\\right)^{q_{v}-1}-1}

    Where :math:`q_{v}` is the visiting parameter.

    Parameters:
    -----------
    :param grid: numpy.ndarray
        The grid structure specifying how channels (e.g., EEG sensors)
         are arranged.
    :param estimator: Union[Any, Pipeline]
        The machine learning estimator to evaluate channel combinations.
    :param metric: str, default = 'f1_weighted'
        The metric name to optimize, must be compatible with scikit-learn.
    :param cv: Union[BaseCrossValidator, int], default = 10
        The cross-validation strategy or number of folds.
    :param groups: numpy.ndarray, default = None
        Groups for LeaveOneGroupOut-crossvalidator
    :param bounds: Optional[list of tuple(float, float)], default = None
        Bounds for the variables during optimization. If None, defaults to
        (0, 1) for each variable.
    :param n_iter: int, default = 1000
        The number of iterations for the simulated annealing process.
    :param initial_temp: float, default = 5230.0
        The initial temperature for the annealing process.
    :param restart_temp_ratio: float, default = 2.e-5
        The ratio of the restart temperature to the initial temperature.
    :param visit: float, default = 2.62
        The visiting parameter for the annealing process.
    :param accept: float, default = -5.0
        The acceptance parameter for the annealing process.
    :param maxfun: float, default = 1e7
        The maximum function evaluations.
    :param tol: float, default = 1e-5
        The function tolerance; if the change in the best objective value
        is below this for `patience` iterations, the optimization will stop early.
    :param prior: Optional[numpy.ndarray], default = None
        Explicitly initialize the optimizer state.
        If set to None if coordinates are initialized randomly.
    :param n_jobs: int, default = 1
        The number of parallel jobs to run during cross-validation.
    :param seed: Optional[int], default = None
        The random seed for initializing the random number generator.
    :param verbose: bool, default = False
        Enables verbose output during the optimization process.

    Methods:
    --------
    - fit:
        Fit the optimizer to the data.
    - transform:
        Transform the input data using the mask from the optimization process.
    - run:
        Execute the simulated annealing optimization algorithm.
    - evaluate_candidates:
        Evaluates the selected features using cross-validation or train-test split.
    - objective_function:
        Calculates the score to maximize or minimize based on the provided mask.
    - elimination_plot:
        Generates and saves a plot visualizing the maximum and all scores across different subgrid sizes.
    - importance_plot:
        Generates and saves a heatmap visualizing the importance of each channel within the grid.

    Notes
    --------
    This implementation is semi-compatible with the scikit-learn framework,
    which builds around two-dimensional feature matrices. To use this
    transformation within a scikit-learn Pipeline, the four dimensional data
    must eb flattened after the first dimension [samples, features]. For example,
    scikit-learn's FunctionTransformer can achieve this.

    References
    --------
    .. [1] Tsallis C. Possible generalization of Boltzmann-Gibbs
        statistics. Journal of Statistical Physics, 52, 479-487 (1998).
    .. [2] Tsallis C, Stariolo DA. Generalized Simulated Annealing.
        Physica A, 233, 395-406 (1996).
    .. [3] Xiang Y, Sun DY, Fan W, Gong XG. Generalized Simulated
        Annealing Algorithm and Its Application to the Thomson Model.
        Physics Letters A, 233, 216-220 (1997).
    .. [4] Xiang Y, Gong XG. Efficiency of Generalized Simulated
        Annealing. Physical Review E, 62, 4473 (2000).
    .. [5] Xiang Y, Gubian S, Suomela B, Hoeng J. Generalized
        Simulated Annealing for Efficient Global Optimization: the GenSA
        Package for R. The R Journal, Volume 5/1 (2013).
    .. [6] Mullen, K. Continuous Global Optimization in R. Journal of
        Statistical Software, 60(6), 1 - 45, (2014).
        :doi:`10.18637/jss.v060.i06`

    Examples
    --------
    The following example shows how to retrieve a feature mask for
    a synthetic data set.

    # >>> import numpy as np
    # >>> from sklearn.svm import SVC
    # >>> from sklearn.pipeline import Pipeline
    # >>> from sklearn.preprocessing import MinMaxScaler
    # >>> from sklearn.datasets import make_classification
    # >>> from FingersVsGestures.src.channel_elimination import SimulatedAnnealing # TODO adjust
    # >>> X, y = make_classification(n_samples=100, n_features=8 * 4 * 100)
    # >>> X = X.reshape((100, 8, 4, 100))
    # >>> grid = np.arange(1, 33).reshape(X.shape[1:3])
    # >>> estimator = Pipeline([('scaler', MinMaxScaler()), ('svc', SVC())])

    # >>> sa = SimulatedAnnealing(grid, estimator)
    # >>> sa.fit(X, y)
    # >>> print(sa.mask_)
    array([[False False False False], [False False False False], [False  True False False], [False False False False],
          [False False False False], [False False False False], [False  True False False], [False False False False]])
    # >>> print(sa.score_)
    0.29307936507936505

    Returns:
    --------
    :return: None
    """

    def __init__(
            self,

            # General and Decoder
            grid: numpy.ndarray,
            estimator: Union[Any, Pipeline],
            metric: str = 'f1_weighted',
            cv: Union[BaseCrossValidator, int] = 10,
            groups: numpy.ndarray = None,

            # Simulated Annealing Settings
            bounds: Optional[List[Tuple[float, float]]] = None,
            n_iter: int = 1000,
            initial_temp: float = 5230.0,
            restart_temp_ratio: float = 2.e-5,
            visit: float = 2.62,
            accept: float = -5.0,
            maxfun: float = 1e7,

            # Training Settings
            tol: float = 1e-5,
            prior: Optional[numpy.ndarray] = None,
            # patience: int = int(1e5),

            # Misc
            n_jobs: int = 1,
            seed: Optional[int] = None,
            verbose: bool = False
    ) -> None:

        super().__init__(grid, estimator, metric, cv, groups, n_jobs, seed, verbose)

        # Simulated Annealing Settings
        self.bounds = bounds
        self.n_iter = n_iter
        self.initial_temp = initial_temp
        self.restart_temp_ratio = restart_temp_ratio
        self.visit = visit
        self.accept = accept
        self.maxfun = maxfun

        # Training Settings
        self.tol = tol
        self.prior = prior
        # self.patience = patience

    def fit(self, X: numpy.ndarray, y: numpy.ndarray = None) -> Type['SimulatedAnnealing']:
        """
        Fit method optimizes the channel combination with Simulated Annealing.

        Parameters:
        -----------
        :param X: numpy.ndarray
            Array-like with dimensions [samples, channel_height, channel_width, time]
        :param y: numpy.ndarray, default = None
            Array-like with dimensions [targets].

        Return:
        -----------
        :return: Type['SimulatedAnnealing']
        """
        self.X_ = X
        self.y_ = y

        self.iter_ = int(0)
        self.result_grid_ = []

        self.prior_ = self.prior
        if self.prior is not None:
            if self.prior.shape != self.grid.reshape(-1).shape:
                raise RuntimeError(
                    f'The argument prior {self.prior.shape} must match '
                    f'the number of cells of grid {self.grid.reshape(-1).shape}.')

            self.prior_ = np.where(self.prior.astype(float) > 0.5, 0.51 + np.random.normal(loc=0, scale=0.06125),
                                   0.49 - np.random.normal(loc=0, scale=0.06125))

        self.bounds_ = self.bounds if self.bounds else [(0, 1) for _ in range(self.grid.size)]

        np.random.seed(self.seed)
        random.seed(self.seed)

        self.solution_, self.mask_, self.score_ = self.run()

        # Conclude the result grid
        self.result_grid_ = pd.concat(self.result_grid_, axis=0, ignore_index=True)
        return self

    def transform(self, X: numpy.ndarray, y: numpy.ndarray = None) -> numpy.ndarray:
        """
        Transforms the input with the mask obtained from the solution
        of Simulated Annealing.

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

    def run(self) -> Tuple[numpy.ndarray, numpy.ndarray, float]:
        """
        Executes the simulated annealing optimization to find the best feature subset.

        Returns:
        --------
        Tuple[numpy.ndarray, numpy.ndarray, float]
            The best solution and state found and their corresponding fitness score.
        """
        callback = SimulatedAnnealingReporter(verbose=self.verbose)

        method_args = {
            'bounds': self.bounds_, 'maxiter': self.n_iter, 'minimizer_kwargs': {'tol': self.tol},
            'initial_temp': self.initial_temp, 'restart_temp_ratio': self.restart_temp_ratio,
            'visit': self.visit, 'accept': self.accept, 'maxfun': self.maxfun, 'seed': self.seed,
            'callback': callback, 'x0': self.prior_
        }

        result = dual_annealing(lambda x: -self.objective_function(x), **method_args)

        best_state = (result.x > 0.5).reshape(self.grid.shape)
        best_score = -result.fun * 100
        return result.x, best_state, best_score
