from typing import Optional, Union, Any, Dict, Tuple, Callable, Type

import numpy
import numpy as np
from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from src.optimizer.Base_Optimizer import BaseOptimizer


class RandomSearch(BaseOptimizer):
    """
    RandomSearch

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
    :param scoring: str, default = 'f1_weighted'
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
    :param callback: Optional[Union[Callable, Type]], default = None, #TODO adjust and add design
    :param n_jobs: int, default = 1
        The number of parallel jobs to run during cross-validation.
    :param random_state: Optional[int], default = None
        Setting a seed to fix randomness (for reproducibility).
        Default does not use a seed.
    :param verbose: Union[bool, int], default = False
         If set to True, enables the output of progress status
         during the optimization process.
    """

    def __init__(
            self,

            # General and Decoder
            dims: Tuple[int, ...],
            estimator: Union[Any, Pipeline],
            estimator_params: Optional[Dict[str, any]] = None,
            scoring: str = 'f1_weighted',
            cv: Union[BaseCrossValidator, int] = 10,
            groups: Optional[numpy.ndarray] = None,

            # Random Search Settings
            n_iter: int = 100,

            # Training Settings
            tol: float = 1e-5,
            patience: int = 1e5,
            bounds: Tuple[float, float] = (0.0, 1.0),
            prior: Optional[numpy.ndarray] = None,
            callback: Optional[Union[Callable, Type]] = None,

            # Misc
            n_jobs: int = 1,
            random_state: Optional[int] = None,
            verbose: Union[bool, int] = False
    ) -> None:

        super().__init__(
            dims, estimator, estimator_params, scoring, cv, groups, tol,
            patience, bounds, prior, callback, n_jobs, random_state, verbose
        )

        # Random Search Settings
        self.n_iter = n_iter

    def _handle_bounds(self):
        """Method to handle bounds for feature selection."""
        return self.bounds

    def _handle_prior(self):
        """Initialize the feature mask with the prior if provided."""
        if self.prior is not None:
            return self.prior
        else:
            return np.random.uniform(*self.bounds, size=np.prod(self.dim_size_)).reshape(self.dim_size_)

    def _run(
            self
    ) -> Tuple[numpy.ndarray, numpy.ndarray, float]:
        """
        Runs the multi-armed bandit optimization process.

        Returns:
        --------
        :return: Tuple[numpy.ndarray, numpy.ndarray, float, pandas.DataFrame]
            A tuple with the solution, mask, the evaluation scores and the optimization history.
        """
        wait = 0
        best_score = 0.0
        best_state = None

        # Run search
        progress_bar = tqdm(range(self.n_iter), desc=self.__class__.__name__, postfix=f'{best_score:.6f}',
                            disable=not self.verbose, leave=True)
        for _ in progress_bar:
            mask = np.random.uniform(self.bounds_[0], self.bounds[1], size=np.prod(self.dim_size_))
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

        best_solution = best_state
        best_state = self._prepare_mask((best_state > 0.5).reshape(self.dim_size_))
        return best_solution, best_state, best_score
