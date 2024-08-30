# -------------------------------------------------------------
# BCI-FeaST
# Copyright (c) 2024
#       Dirk Keller,
#       Elena Offenberg,
#       Nick Ramsey's Lab, University Medical Center Utrecht, University Utrecht
# Licensed under the MIT License [see LICENSE for detail]
# -------------------------------------------------------------

from typing import Tuple, List, Union, Dict, Any, Optional, Type, Callable

import numpy
import numpy as np
from scipy._lib._util import check_random_state
from scipy.optimize import Bounds
from scipy.optimize import OptimizeResult
from scipy.optimize import dual_annealing
from scipy.optimize._constraints import new_bounds_to_old
from scipy.optimize._dual_annealing import ObjectiveFunWrapper, LocalSearchWrapper, EnergyState, VisitingDistribution, \
    StrategyChain
from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from .Base_Optimizer import BaseOptimizer

__all__ = ['dual_annealing']


class SimulatedAnnealing(BaseOptimizer):
    """
    This class implements an Simulated Annealing algorithm for optimizing
    EEG/MEG/ECOG channels within a structured grid. The Class rests on the
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
    :param n_iter: int, default = 1000
        The number of iterations for the simulated annealing process.
    :param optimizer_method: str, default = 'L-BFGS-B'
        The tye of optimization method used. Valid options are:
        'Nelder-Mead', 'Powell’, 'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B',
        'TNC', 'COBYLA', 'COBYQA', 'SLSQP', 'trust-constr’', 'dogleg',
        'trust-ncg', 'trust-exact', 'trust-krylov'.
    :param local_search: bool, default = True
        If `local_search` is set to False, a traditional Generalized
        Simulated Annealing will be performed with no local search
        strategy applied.
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
    :param patience: int, default = 1e5
        The number of iterations for which the objective function
        improvement must be below tol to stop optimization.
    :param tol: float, default = 1e-5
        The function tolerance; if the change in the best objective value
        is below this for 'patience' iterations, the optimization will stop early.
    :param bounds: Tuple[float, float], default = (0.0, 1.0)
        Bounds for the algorithm's parameters to optimize. Since
        it is a binary selection task, bounds are set to (0.0, 1.0).
    :param prior: Optional[numpy.ndarray], default = None
        Explicitly initialize the optimizer state.
        If set to None if the to be optimized features are
        initialized randomly within the bounds.
    :param callback: Optional[Union[Callable, Type]], default = None
        A callback function with signature ``callback(x, f, context)``,
        which will be called for all minima found.
        ``x`` and ``f`` are the coordinates and function value of the
        latest minimum found, and ``context`` has value in [0, 1, 2], with the
        following meaning:

            - 0: minimum detected in the annealing process.
            - 1: detection occurred in the local search process.
            - 2: detection done in the dual annealing process.

        If the callback implementation returns True, the algorithm will stop.
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
            dims: Tuple[int, ...],
            estimator: Union[Any, Pipeline],
            estimator_params: Optional[Dict[str, any]] = None,
            metric: str = 'f1_weighted',
            cv: Union[BaseCrossValidator, int] = 10,
            groups: Optional[numpy.ndarray] = None,

            # Simulated Annealing Settings
            n_iter: int = 1000,
            optimizer_method: str = 'L-BFGS-B',
            local_search: bool = True,
            initial_temp: float = 5230.0,
            restart_temp_ratio: float = 2.e-5,
            visit: float = 2.62,
            accept: float = -5.0,
            maxfun: float = 1e7,

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

        # Simulated Annealing Settings
        self.n_iter = n_iter
        self.optimizer_method = optimizer_method
        self.local_search = local_search
        self.initial_temp = initial_temp
        self.restart_temp_ratio = restart_temp_ratio
        self.visit = visit
        self.accept = accept
        self.maxfun = maxfun

        # Training Settings
        self.tol = tol
        self.patience = patience
        self.callback = callback

    def _run(self) -> Tuple[numpy.ndarray, numpy.ndarray, float]:
        """
        Executes the simulated annealing optimization to find the best feature subset.

        Returns:
        --------
        Tuple[numpy.ndarray, numpy.ndarray, float]
            The best solution and state found and their corresponding fitness score.
        """
        if isinstance(self.bounds_, Bounds):
            self.bounds_ = new_bounds_to_old(self.bounds_.lb, self.bounds_.ub, len(self.bounds_.lb))

        # noqa: E501
        if self.prior_ is not None and not len(self.prior_) == len(self.bounds_):
            raise ValueError('Bounds size does not match x0')

        lu = list(zip(*self.bounds_))
        lower = np.array(lu[0])
        upper = np.array(lu[1])
        # Check that restart temperature ratio is correct
        if self.restart_temp_ratio <= 0. or self.restart_temp_ratio >= 1.:
            raise ValueError('Restart temperature ratio has to be in range (0, 1)')
        # Checking bounds are valid
        if (np.any(np.isinf(lower)) or np.any(np.isinf(upper)) or np.any(
                np.isnan(lower)) or np.any(np.isnan(upper))):
            raise ValueError('Some bounds values are inf values or nan values')
        # Checking that bounds are consistent
        if not np.all(lower < upper):
            raise ValueError('Bounds are not consistent min < max')
        # Checking that bounds are the same length
        if not len(lower) == len(upper):
            raise ValueError('Bounds do not have the same dimensions')

        minimizer_kwargs = {'method': self.optimizer_method}

        # Wrapper for the objective function
        func_wrapper = ObjectiveFunWrapper(lambda x: -self._objective_function(x), self.maxfun)

        minimizer_wrapper = LocalSearchWrapper(
            self.bounds_, func_wrapper, **minimizer_kwargs)

        # Initialization of random Generator for reproducible runs if seed provided
        rand_state = check_random_state(self.seed)
        # Initialization of the energy state
        energy_state = EnergyState(lower, upper)
        energy_state.reset(func_wrapper, rand_state, self.prior_)
        # Minimum value of annealing temperature reached to perform
        # re-annealing
        temperature_restart = self.initial_temp * self.restart_temp_ratio
        # VisitingDistribution instance
        visit_dist = VisitingDistribution(lower, upper, self.visit, rand_state)
        # Strategy chain instance
        strategy_chain = StrategyChain(self.accept, visit_dist, func_wrapper,
                                       minimizer_wrapper, rand_state, energy_state)
        need_to_stop = False
        iteration = 0
        best_score = 0.
        wait = 0
        message = []
        # OptimizeResult object to be returned
        optimize_res = OptimizeResult()
        optimize_res.success = True
        optimize_res.status = 0

        t1 = np.exp((self.visit - 1) * np.log(2.0)) - 1.0
        # Run the search loop
        while not need_to_stop:
            progress_bar = tqdm(range(self.n_iter), desc=self.__class__.__name__, postfix=f'{best_score:.6f}',
                                disable=not self.verbose, leave=True)
            for i in progress_bar:
                # Compute temperature for this step
                s = float(i) + 2.0
                t2 = np.exp((self.visit - 1) * np.log(s)) - 1.0
                temperature = self.initial_temp * t1 / t2
                # Update logs and early stopping
                score = energy_state.ebest
                if best_score > score:
                    best_score = score
                    progress_bar.set_postfix(best_score=f'{best_score:.6f}')
                    if abs(best_score - score) > self.tol:
                        wait = 0
                if wait > self.patience or score <= -1.0:
                    progress_bar.write(f"\nMaximum score reached")
                    message.append("Maximum score reached")
                    need_to_stop = True
                    break
                if iteration >= self.n_iter:
                    message.append("Maximum number of iteration reached")
                    need_to_stop = True
                    break
                # Need a re-annealing process?
                if temperature < temperature_restart:
                    energy_state.reset(func_wrapper, rand_state)
                    break
                # starting strategy chain
                val = strategy_chain.run(i, temperature)
                if val is not None:
                    message.append(val)
                    need_to_stop = True
                    optimize_res.success = False
                    break
                # Possible local search at the end of the strategy chain
                if self.local_search:
                    val = strategy_chain.local_search()
                    if val is not None:
                        message.append(val)
                        need_to_stop = True
                        optimize_res.success = False
                        break
                iteration += 1
                wait += 1

        # Setting the OptimizeResult values
        optimize_res.x = energy_state.xbest
        optimize_res.fun = energy_state.ebest
        optimize_res.nit = iteration
        optimize_res.nfev = func_wrapper.nfev
        optimize_res.njev = func_wrapper.ngev
        optimize_res.nhev = func_wrapper.nhev
        optimize_res.message = message

        best_solution = optimize_res.x
        best_state = self._prepare_mask((optimize_res.x > 0.5).reshape(self.dim_size_))
        best_score = -optimize_res.fun * 100
        return best_solution, best_state, best_score

    def _handle_bounds(
            self
    ) -> List[Tuple[float, float]]:
        """
        Returns the bounds for the SA optimizer. If bounds are not set, it returns default
        bounds of (0, 1) for each dimension.

        :return: List[Tuple[float, float]]
            A list of tuples representing the lower and upper bounds for each dimension.
        """
        return [self.bounds for _ in range(np.prod(self.dim_size_))]

    def _handle_prior(
            self
    ) -> Optional[np.ndarray]:
        """
        Generates a list of prior individuals based on a provided
        prior mask. The function checks the validity of the prior
        mask's shape, applies a Gaussian perturbation to the mask
        values, and creates an array of bounds.

        Returns:
        --------
        Optional[np.ndarray]:
        A numpy array of transformed prior values, or None if no prior is provided.

        """
        # Determine the prior values from the mask if provided
        prior = self.prior
        if self.prior is not None:
            if self.prior.shape != self.dim_size_:  # self.grid.reshape(-1).shape:
                raise RuntimeError(
                    f'The argument prior must match the size of the dimensions to be considered.'
                    f'Got {self.prior.shape} but expected {self.dim_size_}.')  # {self.grid.reshape(-1).shape}.')

            prior = np.where(self.prior.astype(float) > 0.5, 0.51 + np.random.normal(loc=0, scale=0.06125),
                             0.49 - np.random.normal(loc=0, scale=0.06125))
        return prior
