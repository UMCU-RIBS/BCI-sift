# -------------------------------------------------------------
# BCI-FeaST
# Copyright (c) 2024
#       Dirk Keller,
#       Elena Offenberg,
#       Nick Ramsey's Lab, University Medical Center Utrecht, University Utrecht
# Licensed under the MIT License [see LICENSE for detail]
# -------------------------------------------------------------

import random
import warnings
from copy import copy
from typing import Tuple, Union, Dict, Any, Optional, Type

import matplotlib
import numpy
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import TransformerMixin, MetaEstimatorMixin, BaseEstimator
from sklearn.metrics import get_scorer
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
# from sklearn.utils._metadata_requests import _RoutingNotSupportedMixin
from sklearn.utils.validation import check_is_fitted

from src.optimizer.backend._backend import FlattenTransformer, SafeVarianceThreshold

matplotlib.use('Agg')  # Use the 'Agg' backend for PNG rendering


class BaseOptimizer(MetaEstimatorMixin, TransformerMixin, BaseEstimator):  # _RoutingNotSupportedMixin
    """
    Base class for all channel optimizers that provides framework
    functionalities such as estimator serialization, cross-validation
    strategy setup, parameter and data validation.

    Optimizes channel combinations within a grid for a given performance
    metric using a specified machine learning model or pipeline.

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
    :param bounds: Tuple[float, float], default = (0.0, 1.0)
        Bounds for the algorithm's parameters to optimize. Since
        it is a binary selection task, bounds are set to (0.0, 1.0).
    :param prior: Optional[numpy.ndarray], default = None
        Explicitly initialize the optimizer state.
        If set to None if the to be optimized features are
        initialized randomly within the bounds.
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
        Abstract method that must be implemented by subclasses, defining
        the algorithm design and fit to the data.
    - transform:
        Abstract method that must be implemented by subclasses, executing
        the transformation for the data with the optimizer result.
    - run:
        Abstract method that must be implemented by subclasses, defining
        the specific steps of the optimization process.
    - evaluate_candidates:
        Evaluates the selected features using cross-validation or train-test split.
    - objective_function:
        Calculates the score to maximize or minimize based on the provided mask.
    - elimination_plot:
        Generates and saves a plot visualizing the maximum and all scores across different subgrid sizes.
    - importance_plot:
        Generates and saves a heatmap visualizing the importance of each channel within the grid.

    Notes:
    ------
    This implementation is semi-compatible with the scikit-learn framework,
    which builds around two-dimensional feature matrices. To use this
    transformation within a scikit-learn Pipeline, the four dimensional data
    must be flattened after the first dimension [samples, features]. For example,
    scikit-learn's FunctionTransformer can achieve this.

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
            bounds: Tuple[float, float] = (0.0, 1.0),
            prior: Optional[numpy.ndarray] = None,

            # Misc
            n_jobs: int = 1,
            seed: Optional[int] = None,
            verbose: Union[bool, int] = False,
    ) -> None:

        # General and Decoder
        self.dims = dims
        self.estimator = estimator
        self.estimator_params = estimator_params
        self.metric = metric
        self.cv = cv
        self.groups = groups

        # Training Settings
        self.bounds = bounds
        self.prior = prior

        # Misc
        self.n_jobs = n_jobs
        self.seed = seed
        self.verbose = verbose

    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.

        __init__ parameters are not touched.
        """
        # Checking one attribute is enough, because they are all set together
        # in partial_fit
        if hasattr(self, "X_"):
            del self.X_
            del self.y_
            del self.iter_
            del self.result_grid_
            del self.n_cv_
            del self.dims_incl_
            del self.dim_size_
            del self.prior_
            del self.bounds_
            del self.solution_
            del self.mask_
            del self.score_

    def fit(
            self, X: numpy.ndarray, y: numpy.ndarray = None
    ) -> Type['BaseOptimizer']:
        """
        Fit method to fit the optimizer to the data.

        Parameters:
        -----------
        :param X: numpy.ndarray
            Array-like with dimensions (e.g.
            [samples, channel_height, channel_width, time])
        :param y: numpy.ndarray, default = None
            Array-like with dimensions [targets].

        Return:
        -------
        :return: Type['BaseOptimizer']
        """
        self._reset()
        random.seed(self.seed)
        np.random.seed(self.seed)

        self.X_, self.y_ = self._validate_data(
            X, y, reset=False, ensure_2d=False, allow_nd=True
        )
        if self.estimator_params:
            self.set_estimator_params()
        self._check_estimator_data_requirements()

        self.iter_ = 0
        self.result_grid_ = []
        self.n_cv_ = self.cv if isinstance(self.cv, int) else self.cv.get_n_splits()

        # Make dimensions, size and slicing accessible to the objective function.
        # Slicing is used for broadcasting the mask (included dims) to all dimensions.
        self.dims_incl_ = sorted(self.dims)
        self.dim_size_ = tuple(np.array(self.X_.shape)[list(self.dims)])
        self.slices_ = [np.newaxis if dim not in self.dims_incl_
                        else slice(None) for dim in range(self.X_.ndim)]

        self.bounds_ = self._handle_bounds()
        self.prior_ = self._handle_prior()

        self.solution_, self.mask_, self.score_ = self._run()
        self._prepare_result_grid()

        return self

    def transform(
            self, X: numpy.ndarray, y: numpy.ndarray = None
    ) -> numpy.ndarray:
        """
        Transforms the input with the mask obtained from
        the solution of Optimization process.

        Parameters:
        -----------
        :param X: numpy.ndarray
            Array-like with dimensions [samples, channel_height, channel_width, time]
        :param y: numpy.ndarray, default = None
            Array-like with dimensions [targets].

        Return:
        -----------
        :return: numpy.ndarray
            Returns a sparse representation of the input tensor.
        """
        check_is_fitted(self)

        return X * self.mask_

    def _run(
            self
    ) -> NotImplementedError:
        """
        Abstract method defining the main optimization procedure.

        This method must be implemented by subclasses.

        Raises:
        -------
        :raises NotImplementedError:
            If the method is not implemented by the subclass.
        """
        raise NotImplementedError('The run method must be implemented by subclasses')

    def _objective_function(
            self, mask: numpy.ndarray, **kwargs
    ) -> float:
        """
        Objective function that calculates the score to maximize/minimize.

        Parameters:
        -----------
        :param mask: numpy.ndarray
            The boolean mask indicating selected features.

        Returns:
        --------
        :return: float
            The evaluation score for the selected features,
            or -inf if no features are selected.
        """
        self.iter_ += 1

        # Convert mask to a boolean array if necessary
        if np.array(mask).dtype != bool:
            mask = np.array(mask) > 0.5

        full_mask = self._prepare_mask(mask)
        selected_features = self.X_ * full_mask

        scores = self._evaluate_candidates(selected_features)
        self._save_statistics(mask, scores)
        return scores.mean()

    def _prepare_mask(
            self, mask: np.ndarray
    ) -> np.ndarray:
        """
        Reshapes and broadcasts a mask to match the full shape of a data tensor.

        Parameters:
        -----------
        :params mask : np.ndarray
            The mask to be reshaped and broadcasted.

        Returns:
        --------
        :return: np.ndarray
            The full mask is broadcasted to match the shape of the data tensor.
        """
        reshaped_mask = mask.reshape(self.dim_size_)[tuple(self.slices_)]
        full_mask = np.zeros(self.X_.shape, dtype=bool)
        full_mask[np.broadcast_to(reshaped_mask, full_mask.shape)] = True
        return full_mask

    def _evaluate_candidates(
            self, selected_features: numpy.ndarray
    ) -> numpy.ndarray:
        """
        Evaluate the given features using cross-validation or train-test split.

        Parameters:
        -----------
        :param selected_features: numpy.ndarray
            The selected features to evaluate.

        Returns:
        --------
        :return: numpy.ndarray
            The evaluation scores for the selected features.
        """
        scorer = get_scorer(self.metric)

        if self.n_cv_ <= 1:
            if selected_features.size == 0 or not np.any(selected_features):
                return np.full((1,), fill_value=0.0)
            X_train, X_test, y_train, y_test = train_test_split(
                selected_features, self.y_, test_size=self.n_cv_
                if self.n_cv_ < 1 else 0.2, random_state=self.seed
            )
            scores = scorer(copy(self.estimator).fit(X_train, y_train), X_test, y_test)
        else:
            if selected_features.size == 0 or not np.any(selected_features):
                return np.full(self.n_cv_, 0.0)
            scores = cross_val_score(
                self.estimator, selected_features, self.y_, scoring=scorer,
                cv=self.cv, groups=self.groups, n_jobs=self.n_jobs
            )
        return scores

    def _save_statistics(
            self, mask: numpy.ndarray, scores: numpy.ndarray
    ) -> None:
        """
        Saves the diagnostics at a given iteration. The
        diagnostics include: Method, Iteration, Mask,
        Channel IDs, Size, Mean (Score), Median (Score),
        SD (Score), CI Lower, CI Upper and the score from
        the crossvaldiation folds (if selected)

        Parameters:
        -----------
        :param mask: numpy.ndarray
            The boolean mask indicating selected features.
        :param scores: numpy.ndarray
            The array of cross-validation scores.

        Returns:
        --------
        :return: None
        """
        if self.n_cv_ > 1:
            ci = stats.t.interval(0.95, len(scores) - 1, loc=np.mean(scores), scale=stats.sem(scores))
            cv_stats = {
                'Mean (Score)': np.round(np.mean(scores), 5),
                'Median (Score)': np.round(np.median(scores), 5),
                'Std (Score)': np.round(np.std(scores), 5),
                'CI Lower': np.round(ci[0], 5),
                'CI Upper': np.round(ci[1], 5),
            }
        else:
            cv_stats = {'Mean (Score)': scores[0]}

        for i, score in enumerate(scores):
            cv_stats[f'Fold {i + 1}'] = score

        self.result_grid_.append(pd.DataFrame({
            **{'Method': self.__class__.__name__,
               'Iteration': self.iter_,
               'Mask': [mask.reshape(self.dim_size_)],
               'Size': np.sum(mask),
               'Metric': self.metric},
            **cv_stats}))

    def _handle_bounds(
            self
    ) -> NotImplementedError:
        """
        Placeholder method for handling bounds. This method must be implemented by subclasses.

        Raises:
        -------
        :raises
            NotImplementedError: When the method is not overridden by a subclass.
        """
        raise NotImplementedError('The _handle_bounds method must be implemented by subclasses.')

    def _handle_prior(
            self
    ) -> NotImplementedError:
        """
        Handles the prior values by validating their shape and applying transformations if provided.

        Raises:
        -------
        :raises
            NotImplementedError: When the method is not overridden by a subclass.
        """
        raise NotImplementedError('The _handle_priors method must be implemented by subclasses.')

    def set_estimator_params(
            self
    ) -> None:
        """
        Validate that all kwargs correspond to valid parameters in the given
        scikit-learn estimator or pipeline.

        Raises:
        -------
        :raise ValueError: If any key in kwargs is not a valid parameter.

        Returns:
        --------
        None
        """
        # Estimator Check
        estimator = self.estimator.steps[-1][1] if isinstance(self.estimator, Pipeline) else self.estimator
        if not isinstance(estimator, BaseEstimator):
            raise TypeError("The provided estimator is not a valid scikit-learn estimator.")

        # Parameter Check
        invalid_params = [param for param in self.estimator_params if param not in estimator.get_params()]
        if invalid_params:
            raise ValueError(f"Invalid parameter(s) for estimator: {', '.join(invalid_params)}")
        estimator.set_params(**self.estimator_params)

        # Set the parameters
        if isinstance(self.estimator, Pipeline):
            self.estimator.steps[-1][1].set_params(**self.estimator_params)
        else:
            self.estimator.set_params(**self.estimator_params)

    def allocate_cpu_resources(
            self, cv: int, n: int
    ) -> Tuple[int, int]:
        """
        Choose the best (ea_cores, cv_cores) pair.

        Parameters:
        -----------
        :param cv: int
            Number of cross-validation folds.
        :param n: int
            Number of generations in the evolutionary algorithm.

        Returns:
        --------
        returns: Tuple[int, int]
            The best (ea_cores, cv_cores) pair satisfying the constraints.
        """
        pairs = [(algo_cores, self.n_jobs // algo_cores)
                 for algo_cores in range(1, self.n_jobs + 1) if self.n_jobs % algo_cores == 0]
        valid_pairs = [(algo_cores, cv_cores) for algo_cores, cv_cores in pairs if
                       algo_cores <= n and cv_cores <= cv and cv % cv_cores == 0]

        if not valid_pairs:
            raise ValueError(
                f"CPU resource allocation failed with constraints: n_jobs={self.n_jobs}, cv={cv}, n={n}.")

        return max(valid_pairs, key=lambda x: x[1])

    def _check_estimator_data_requirements(
            self
    ) -> None:
        """
        Performs a check on whether the passed estimator expects 2-dimensional
        data representation. Automatically adjusts the estimator to ensure compatibility.

        Raises:
        -------
        :raise ValueError:
            If the estimator raises a `ValueError` related to dimensionality
            that cannot be resolved by automatically inserting a
            `FlattenTransformer` and `SafeVarianceThreshold` into the pipeline,
            the ValueError is passed to the User.

        Returns:
        --------
        :returns: None
        """
        try:
            self.estimator.fit(self.X_[0], self.y_[0])
        except ValueError as e:
            cases = ['2D', 'dim', 'dims', 'dimensions', 'dimension',
                     'dimensional', 'input data', 'input', 'data']
            if any(term in str(e).lower() for term in cases):
                try:
                    test_estimator = Pipeline([
                        ('flatten', FlattenTransformer()),
                        ('clean', SafeVarianceThreshold(threshold=0.0)),
                        ('estimator', self.estimator)
                    ])
                    test_estimator.fit(self.X_[:2], self.y_[:2])
                    warnings.warn(f'Estimator adjusted for ND data compatibility.', UserWarning)
                    self.estimator = test_estimator
                except e as e:
                    raise e

    def _prepare_result_grid(
            self
    ) -> None:
        """
        Finalizes the result grid. For the Spatial Algorithm, the height
        and width of the included area is added.

        Returns:
        --------
        returns: None
        """
        self.result_grid_ = pd.concat(self.result_grid_, axis=0, ignore_index=True)

# Apply the sigmoid function to the mask
# mask = 1 / (1 + np.exp(-np.array(mask)))
#
# # Flatten the mask if it has more than 1 dimension
# if len(mask.shape) > 1:
#     mask = mask.ravel()
#
# # Reshape the mask to align with the selected dimensions
# reshaped_mask = mask.reshape(self.dim_size_)[tuple(self.slices_)]
#
# # Broadcast the mask of the considered dimensions to the full mask matching the data tensor
# full_mask = np.broadcast_to(reshaped_mask, self.X_.shape)
# mm = MinMaxScaler(feature_range=(self.bounds_[0], self.bounds_[1]))
# X = self.X_.reshape(self.X_.shape[0], -1)
# selected_features = mm.fit_transform(X) * full_mask.reshape(self.X_.shape[0], -1)
