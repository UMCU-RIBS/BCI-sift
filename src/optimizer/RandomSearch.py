from typing import Optional, Union, Any, Dict, Tuple

import numpy
import numpy as np
from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from src.optimizer._Base_Optimizer import BaseOptimizer


class RandomSearch(BaseOptimizer):
    """
    RandomSearch

    Parameters:
    -----------
    :param dims: Tuple[int, ...]
        A tuple of dimensions indies tc apply the feature selection onto.
        Any combination of dimensions can be specified, except for
        dimension 'zero', which represents the samples.
    :param estimator: Union[BaseEstimator, Pipeline]
        The machine learning estimator to evaluate channel combinations.
    :param estimator_params: Optional[Dict[str, any]], default = None
         Optional parameters to adjust the estimator parameters.
    :param metric: str, default = 'f1_weighted'
        The metric name to optimize, must be compatible with scikit-learn.
    :param cv: Union[BaseCrossValidator, int], default = 10
        The cross-validation strategy or number of folds.
        If an integer is passed, train_test_split() for 1 and
        StratifiedKFold() is used for >1 as default.
    :param groups: numpy.ndarray, default = None
        Groups for LeaveOneGroupOut-crossvalidator
    :param bounds: Tuple[float, float]], default = (0.0, 1.0)
        Has no effect but is kept for consistency.
    :param prior: Optional[numpy.ndarray], default = None
        Has no effect but is kept for consistency.
    :param n_jobs: int, default = 1
        The number of parallel jobs to run during cross-validation.
    :param seed: Optional[int], default = None
        The seed for random number generation.
    :param verbose: bool, default = False
        Enables verbose output during the optimization process.
    """

    def __init__(
            self,

            # General and Decoder
            dims: Tuple[int, ...],
            estimator: Union[Any, Pipeline],
            estimator_params: Union[Dict[str, any], None] = None,
            metric: str = 'f1_weighted',
            cv: Union[BaseCrossValidator, int] = 10,
            groups: numpy.ndarray = None,

            # Random Search Settings
            n_iter: int = 100,

            # Training Settings
            bounds: Tuple[float, float] = (0.0, 1.0),
            prior: Optional[numpy.ndarray] = None,

            # Misc
            n_jobs: int = 1,
            seed: Optional[int] = None,
            verbose: Union[bool, int] = False

    ) -> None:
        super().__init__(
            dims, estimator, estimator_params, metric, cv, groups, bounds, prior, n_jobs, seed, verbose
        )

        # Random Search properties
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

    def _run(self):
        """Runs the multi-armed bandit optimization process."""

        best_mask = None
        best_score = 0.0

        # Set up tqdm progress bar
        progress_bar = tqdm(range(self.n_iter), desc=self.__class__.__name__, disable=not self.verbose, leave=True)

        # Random generations
        for _ in progress_bar:

            # Set up the mask to operate on
            # mask = np.random.randint(0, 2, size=np.prod(self.dim_size_))
            mask = np.random.uniform(self.bounds_[0], self.bounds[1], size=np.prod(self.dim_size_))

            # Evaluate the selected subset
            score = self._objective_function(mask)

            # Track the best score
            if score > best_score:
                best_score = score
                best_mask = mask

            progress_bar.set_postfix(best_score=best_score)

        best_solution = best_mask
        best_state = best_mask.astype(bool)

        return best_solution, best_state, best_score
