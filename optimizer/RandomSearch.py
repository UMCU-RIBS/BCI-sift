# -------------------------------------------------------------
# BCI-FeaST
# Copyright (c) 2024
#       Dirk Keller,
#       Elena Offenberg,
#       Nick Ramsey's Lab, University Medical Center Utrecht, University Utrecht
# Licensed under the MIT License [see LICENSE for detail]
# -------------------------------------------------------------
from numbers import Integral
from typing import Optional, Union, Any, Dict, Tuple, Callable

import numpy
import ray
from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import Pipeline
from sklearn.utils._param_validation import Interval
from tqdm import tqdm

from optimizer.Base_Optimizer import BaseOptimizer

__all__ = ["RandomSearch"]


class RandomSearch(BaseOptimizer):
    """
    This class implements a Random Search algorithm to optimize the selection of feature
     combinations by randomly generating solutions with regard to a predefined measure
     of quality.

    Parameters:
    -----------
    :param dimensions: Tuple[int, ...]
        A tuple of dimensions indies tc apply the feature selection onto. Any
        combination of dimensions can be specified, except for dimension 'zero', which
        represents the samples.
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
    :param n_iter: int, default = 100
        The number of iterations for the rand search process.
    :param n_perturbations : int, default = 128
        The number of perturbations to be executed per iteration.
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
        Has no effect but is kept for consistency.
    :param callback: Callable, optional
        A callback function of the structure :code: `callback(x, f, context)`, which
        will be called at each iteration. :code: `x` and :code: `f` are the solution and
        function value, and :code: `context` contains the diagnostics of the current
        iteration.
    :param n_jobs: Union[int, float], default = -1
        The number of parallel jobs to run during cross-validation; -1 uses all cores.
    :param random_state: int, optional
        Setting a seed to fix randomness (for reproducibility).
    :param verbose: Union[bool, int], default = False
         If set to True, enables the output of progress status during the optimization
         process.

    Examples:
    ---------
    The following example shows how to retrieve a feature mask for a synthetic data set.

    .. code-block:: python

        import numpy
        from sklearn.svm import SVC
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.datasets import make_classification
        from FingersVsGestures.src.channel_elimination import RandomSearch # TODO adjust

        X, y = make_classification(n_samples=100, n_features=8 * 4 * 100)
        X = X.reshape((100, 8, 4, 100))
        grid = (2, 3)
        estimator = Pipeline([('scaler', MinMaxScaler()), ('svc', SVC())])

        rs = RandomSearch(grid, estimator)
        rs.fit(X, y)
        print(rs.score_)
        20.49120472562842

    Return:
    -------
    :return: None
    """

    # fmt: off
    _parameter_constraints: dict = {**BaseOptimizer._parameter_constraints}
    _parameter_constraints.update(
        {
            "n_iter": [Interval(Integral, 1, None, closed="left")],
            "n_pertubations": [Interval(Integral, 1, None, closed="left")],
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
        # Random Search Settings
        n_iter: int = 100,
        n_perturbations: int = 128,
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

        # Random Search Settings
        self.n_iter = n_iter
        self.n_perturbations = n_perturbations

    def _run(self) -> Tuple[numpy.ndarray, numpy.ndarray, float]:
        """
        Runs the random search process to optimize the feature configuration, by
        evaluating the proposed candidate solutions in the objective function
        :code:`f` for a number of iterations :code:`n_iter.`

        Returns:
        --------
        :return: Tuple[numpy.ndarray, numpy.ndarray, float]
            The best solution, mask found and their score.
        """
        wait = 0
        best_score = 0.0
        best_state = None

        mask_dim = (
            (self._dim_size,)
            if isinstance(self._dim_size, numpy.int64)
            else self._dim_size
        )

        # Run search
        idtr = f"{self._dims_incl}: " if isinstance(self._dims_incl, int) else ""
        progress_bar = tqdm(
            range(self._update_n_iter(self.n_iter)),
            desc=f"{idtr}{self.__class__.__name__}",
            postfix=f"{best_score:.6f}",
            disable=not self.verbose,
            leave=True,
        )
        for self.iter_ in progress_bar:
            mask = numpy.random.uniform(
                self._bounds[0],
                self.bounds[1],
                size=(self.n_perturbations, *mask_dim),
            )

            scores = self._compute_objective(mask)

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

        best_solution = best_state
        best_state = best_state > 0.5
        best_score = best_score * 100
        return best_solution, best_state, best_score

    def _compute_objective(
        self,
        random_perturbations: numpy.ndarray,
    ) -> numpy.ndarray:
        """
        Generate random masks and compute their objective function scores. This method
        allows the RS algorithm to interface correctly with the objective function by
        converting the input mask tensor into individuals and evaluating them.

        If more than one cpu core is passed, then the evaluation of the particles is
        done in parallel using multiple processes.

        Parameters
        ----------
        :param random_perturbations: numpy.ndarray
            The random generated perturbations.

        Returns
        -------
        numpy.ndarray
            The performance-matrix for a collection of masks.
        """
        positions_split = numpy.array_split(
            random_perturbations, random_perturbations.shape[0]
        )

        # fmt: off
        # Use multiprocessing pool to compute scores in parallel
        if self.n_jobs > 1:
            return ray.get(
                [self._objective_function_wrapper.remote(self, pos) for pos in positions_split]
            )
        # If no pool is provided, compute scores sequentially
        else:
            return [self._objective_function(pos) for pos in positions_split]
        # fmt: on

    def _handle_bounds(self):
        """
        Returns the bounds for the random search. If bounds are not set, default bounds
        of [0, 1] for each dimension are used.

        Returns:
        --------
        :return: Tuple[float, float]
            A tuple of two floats representing the lower and upper bounds.
        """
        return self.bounds

    def _handle_prior(self):
        """Placeholder Function"""
        return None

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
