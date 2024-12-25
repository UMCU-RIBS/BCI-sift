# -------------------------------------------------------------
# BCI-FeaST
# Copyright (c) 2024
#       Dirk Keller,
#       Elena Offenberg,
#       Nick Ramsey's Lab, University Medical Center Utrecht
# Licensed under the MIT License [see LICENSE for detail]
# -------------------------------------------------------------
from numbers import Real, Integral
from typing import Tuple, List, Union, Dict, Any, Optional, Callable

import numpy
from scipy._lib._util import check_random_state
from scipy.optimize import Bounds
from scipy.optimize._constraints import new_bounds_to_old
from scipy.optimize._dual_annealing import (
    ObjectiveFunWrapper,
    LocalSearchWrapper,
    EnergyState,
    VisitingDistribution,
    StrategyChain,
)
from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import Pipeline
from sklearn.utils._param_validation import StrOptions, Interval
from tqdm import tqdm

from .Base_Optimizer import BaseOptimizer

__all__ = ["SimulatedAnnealing"]


class SimulatedAnnealing(BaseOptimizer):
    """
    This class implements an Simulated Annealing algorithm to optimize the
    selection of feature combinations by iteratively trying to improve a
    quality. The Class rests on the shoulders :code: `dual_annealing`
    implementation from of :code: `scipy's`(see scipys dual annealing
    implementation for more information). This stochastic approach derived
    from [3]_ combines the generalization of CSA (Classical Simulated
    Annealing) and FSA (Fast Simulated Annealing) [1]_ [2]_ coupled to a
    strategy for applying a local search on accepted locations [4]_. An
    alternative implementation of this same algorithm is described in [5]_
    and benchmarks are presented in [6]_. This approach introduces an
    advanced method to refine the solution found by the generalized
    annealing process. This algorithm uses a distorted Cauchy-Lorentz
    visiting distribution, with its shape controlled by the parameter
    :math:`q_{v}`

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
    :param dimensions: Tuple[int, ...]
        A tuple of dimensions indies tc apply the feature selection onto.
        Any combination of dimensions can be specified, except for dimension
        'zero', which represents the samples.
    :param feature_space: str
        The type of feature space required for the model architecture. Valid options
        are: 'tensor' and 'tabular'.
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
    :param strategy: str, default = "conditional"
        The strategy of optimization to apply. Valid options are: 'joint' and
        'conditional'.
        * Joint Optimization: Optimizes all features simultaneously. Should be only
          selected for small search spaces.
        * Conditional Optimization: Optimizes each feature dimension iteratively,
          building on previous results. Generally, yields better performance for large
          search spaces.
    :param n_iter: int, default = 1000
        The number of iterations for the simulated annealing process.
    :param optimizer_method: str, default = 'L-BFGS-B'
        The tye of optimization method used. Valid options are:
        'Nelder-Mead', 'Powell’, 'CG', 'BFGS','Newton-CG', 'L-BFGS-B',
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
    :param tol: float, default = 1e-5
        The function tolerance; if the change in the best objective value is below this
        for 'patience' iterations, the optimization will stop early. If 'conditional'
        optimization is chosen, multiple stopping criteria can be passed; one for each
        dimension.
    :param patience: Union[int Tuple[int, ...], default = int(1e5)
        The number of iterations for which the objective function improvement must be
        below tol to stop optimization. If 'conditional' optimization is chosen, multiple
        stopping criteria can be passed; one for each dimension.
    :param bounds: Tuple[float, float], default = (0.0, 1.0)
        Bounds for the algorithm's parameters to optimize. Since it is a binary
        selection task, bounds are set to (0.0, 1.0).
    :param prior: numpy.ndarray, optional
        Explicitly initialize the optimizer state.
        If set to None if the to be optimized features are initialized randomly
        within the bounds.
    :param callback: Callable, optional
        A callback function of the structure :code: `callback(x, f, context)`,
        which will be called at each iteration. :code: `x` and :code: `f` are
        the solution and function value, and :code: `context` contains the
        diagnostics of the current iteration.
    :param n_jobs: int, default = -1
        The number of parallel jobs to run during cross-validation; -1 uses all cores.
    :param hof_size: int, default = 1
        The number of seats in the hall of fame (best solutions).
    :param random_state: int, optional
        Setting a seed to fix randomness (for reproducibility).
    :param verbose: Union[bool, int], default = False
         If set to True, enables the output of progress status during the
         optimization process.

    Methods:
    --------
    - fit:
        Fit the optimizer to the data.
    - transform:
        Transform the input data using the mask from the optimization process.

    Notes:
    ------
    This implementation is semi-compatible with the scikit-learn framework,
    which builds around two-dimensional feature matrices. To use this
    transformation within a scikit-learn Pipeline, the four dimensional data
    must be flattened after the first dimension [samples, features].
    For example, scikit-learn's :code:`FunctionTransformer` can achieve this.

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

    .. code-block:: python

        import numpy
        from sklearn.svm import SVC
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.datasets import make_classification
        from FingersVsGestures.src.channel_elimination import SimulatedAnnealing # TODO adjust

        X, y = make_classification(n_samples=100, n_features=8 * 4 * 100)
        X = X.reshape((100, 8, 4, 100))
        grid = (2, 3)
        estimator = Pipeline([('scaler', MinMaxScaler()), ('svc', SVC())])

        sa = SimulatedAnnealing(grid, estimator)
        sa.fit(X, y)
        print(sa.score_)
        0.29307936507936505

    Returns:
    --------
    :return: None
    """

    # fmt: off
    _parameter_constraints: dict = {**BaseOptimizer._parameter_constraints}
    _parameter_constraints.update(
        {
            "n_iter": [Interval(Integral, 1, None, closed="left")],
            "optimizer_method": [StrOptions({"Nelder-Mead", "Powell","CG", "BFGS", "Newton-CG", "L-BFGS-B",
                                             "TNC", "COBYLA", "COBYQA", "SLSQP", "trust-constr", "dogleg",
                                             "trust-ncg", "trust-exact", "trust-krylov"})],
            "local_search": [bool],
            "initial_temp": [Interval(Real, 0.0, None, closed="left")],
            "restart_temp_ratio": [Interval(Real, 0.0, None, closed="right")],
            "visit": [Interval(Real, 0.0, None, closed="right")],
            "accept": [Interval(Real, None, 0.0, closed="left")],
            "maxfun": [Interval(Real, 1.0, None, closed="left")],
        }
    )
    # fmt: on

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
        strategy: str = "conditional",
        # Simulated Annealing Settings
        n_iter: int = 1000,
        optimizer_method: str = "L-BFGS-B",
        local_search: bool = True,
        initial_temp: float = 5230.0,
        restart_temp_ratio: float = 2.0e-5,
        visit: float = 2.62,
        accept: float = -5.0,
        maxfun: float = 1e7,
        # Training Settings
        tol: Union[Tuple[int, ...], float] = 1e-5,
        patience: Union[Tuple[int, ...], int] = int(1e5),
        bounds: Tuple[float, float] = (0.0, 1.0),
        prior: Optional[numpy.ndarray] = None,
        callback: Optional[Callable] = None,
        # Misc
        n_jobs: int = -1,
        hof_size: int = 1,
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
            hof_size,
            random_state,
            verbose,
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


    def _run(self) -> Tuple[numpy.ndarray, numpy.ndarray, float]:
        """
        Executes the simulated annealing algorithm to optimize the feature
        configuration, by evaluating the proposed candidate solutions in the objective
        function :code:`f` for a number of iterations :code:`n_iter.`

        Returns:
        --------
        Tuple[numpy.ndarray, numpy.ndarray, float]
            The best solution, mask found and their score.
        """
        if isinstance(self._bounds, Bounds):
            self._bounds = new_bounds_to_old(
                self._bounds.lb, self._bounds.ub, len(self._bounds.lb)
            )

        if self._prior is not None and not len(self._prior) == len(self._bounds):
            raise ValueError("Bounds size does not match prior")

        lu = list(zip(*self._bounds))
        lower = numpy.array(lu[0])
        upper = numpy.array(lu[1])

        # Check that restart temperature ratio is correct
        if self.restart_temp_ratio <= 0.0 or self.restart_temp_ratio >= 1.0:
            raise ValueError("Restart temperature ratio has to be in range (0, 1)")

        # Checking bounds are valid
        if (
            numpy.any(numpy.isinf(lower))
            or numpy.any(numpy.isinf(upper))
            or numpy.any(numpy.isnan(lower))
            or numpy.any(numpy.isnan(upper))
        ):
            raise ValueError("Some bounds values are inf values or nan values")

        # Checking that bounds are consistent
        if not numpy.all(lower < upper):
            raise ValueError("Bounds are not consistent min < max")

        # Checking that bounds are the same length
        if not len(lower) == len(upper):
            raise ValueError("Bounds do not have the same dimensions")

        wait = 0
        self.iter_ = 0
        need_to_stop = False
        best_score = 0.0

        minimizer_kwargs = {"method": self.optimizer_method, "tol": self._tol}

        # Wrapper for the objective function
        func_wrapper = ObjectiveFunWrapper(
            lambda x: -self._objective_function(x), self.maxfun
        )

        minimizer_wrapper = LocalSearchWrapper(
            self._bounds, func_wrapper, **minimizer_kwargs
        )

        # Initialization of random Generator for reproducible runs if seed provided
        rand_state = check_random_state(self.random_state)

        # Initialization of the energy state
        energy_state = EnergyState(lower, upper)
        energy_state.reset(func_wrapper, rand_state, self._prior)

        # Minimum value of annealing temperature reached to perform
        # re-annealing
        temperature_restart = self.initial_temp * self.restart_temp_ratio

        # VisitingDistribution instance
        visit_dist = VisitingDistribution(lower, upper, self.visit, rand_state)

        # Strategy chain instance
        strategy_chain = StrategyChain(
            self.accept,
            visit_dist,
            func_wrapper,
            minimizer_wrapper,
            rand_state,
            energy_state,
        )

        t1 = numpy.exp((self.visit - 1) * numpy.log(2.0)) - 1.0

        # Run the search loop
        idtr = f"{self._dims_incl}: " if isinstance(self._dims_incl, int) else ""
        progress_bar = tqdm(
            range(self._update_n_iter(self.n_iter)),
            desc=f"{idtr}{self.__class__.__name__}",
            postfix=f"{best_score:.6f}",
            disable=not self.verbose,
            leave=True,
        )
        while not need_to_stop:
            for i in progress_bar:
                # Compute temperature for this step
                s = float(i) + 2.0
                t2 = numpy.exp((self.visit - 1) * numpy.log(s)) - 1.0
                temperature = self.initial_temp * t1 / t2

                # Update logs and early stopping
                wait += 1
                score = -energy_state.ebest
                if best_score < score:
                    if score - best_score > self._tol:
                        wait = 0
                    best_score = score
                progress_bar.set_postfix(best_score=f"{best_score:.6f}")
                if wait > self._patience:
                    progress_bar.set_postfix(
                        best_score=f"Early Stopping Criteria reached: {best_score:.6f}"
                    )
                    need_to_stop = True
                    break
                elif score >= 1.0:
                    progress_bar.set_postfix(
                        best_score=f"Maximum score reached: {best_score:.6f}"
                    )
                    need_to_stop = True
                    break
                elif self.callback is not None:
                    if self._callback():
                        progress_bar.set_postfix(
                            best_score=f"Stopped by callback: {best_score:.6f}"
                        )
                        need_to_stop = True
                        break
                elif self.iter_ >= self.n_iter:
                    need_to_stop = True
                    break

                # Need a re-annealing process?
                if temperature < temperature_restart:
                    energy_state.reset(func_wrapper, rand_state)
                    break

                # starting strategy chain
                val = strategy_chain.run(i, temperature)
                if val is not None:
                    need_to_stop = True
                    break

                # Possible local search at the end of the strategy chain
                if self.local_search:
                    val = strategy_chain.local_search()
                    if val is not None:
                        need_to_stop = True
                        break
                self.iter_ += 1
                wait += 1

        # Obtain the final best_solution, best_state and best_score
        best_solution = energy_state.xbest
        best_state = energy_state.xbest > 0.5
        best_score = -energy_state.ebest * 100
        return best_solution, best_state, best_score

    def _handle_bounds(self) -> List[Tuple[float, float]]:
        """
        Returns the bounds for the SA optimizer. If bounds are not set, it returns
        default bounds of (0, 1) for each dimension.

        :return: List[Tuple[float, float]]
            A list of tuples representing the lower and upper bounds for each dimension.
        """
        return [self.bounds for _ in range(numpy.prod(self._dim_size))]

    def _handle_prior(self) -> Optional[numpy.ndarray]:
        """
        This function checks the validity of the 'prior'; the function accepts an array
        (features, ) specifying the energy state. Otherwise, if a mask (features, ) is
        provided, Gaussian perturbation to the values are generated.

        Returns:
        --------
        return: numpy.ndarray, optional
            A numpy array of transformed prior values, or None if no prior is provided.

        """
        if self.prior is None:
            return self.prior

        # If energy state is provided
        if isinstance(self.prior, numpy.ndarray) and self.prior.size == numpy.prod(
            self._dim_size
        ):
            if self.prior.dtype == float:
                return self.prior
            gaus = numpy.abs(numpy.random.normal(0, 0.06125, self.prior.size))
            return numpy.where(
                self.prior.flatten() < 0.5,
                0.49 - gaus,
                0.51 + gaus,
            )

    def _callback(self) -> Union[None, bool]:
        """
        Handles the callbacks provided to the class.

        Returns:
        --------
        :return Union[None, bool]:
            Returns None, True or False depending on the callback function provided.
        """
        return self.callback(self.iter_, 1, self.result_grid_)