# -------------------------------------------------------------
# BCI-FeaST
# Copyright (c) 2024
#       Dirk Keller,
#       Elena Offenberg,
#       Nick Ramsey's Lab, University Medical Center Utrecht, University Utrecht
# Licensed under the MIT License [see LICENSE for detail]
# -------------------------------------------------------------
import multiprocessing
import random
import warnings
from abc import ABC, abstractmethod
from numbers import Real
from typing import Tuple, Union, Dict, Any, Optional, Type, Callable

import numpy
import pandas as pd
from scipy import stats
from sklearn.base import (
    TransformerMixin,
    MetaEstimatorMixin,
    BaseEstimator,
    clone,
    _fit_context,
)
from sklearn.metrics import get_scorer, get_scorer_names
from sklearn.model_selection import BaseCrossValidator, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils._param_validation import (
    Integral,
    Interval,
    StrOptions,
    HasMethods,
)
from sklearn.utils.validation import check_is_fitted

from optimizer.backend._backend import FlattenTransformer, SafeVarianceThreshold
from utils.hp_tune import PerfTimer

__all__ = []


class BaseOptimizer(ABC, MetaEstimatorMixin, TransformerMixin, BaseEstimator):
    """
    Base class for all channel optimizers that provides framework functionalities
    such as estimator serialization, cross-validation strategy setup, parameter and
    data validation.

    Optimizes channel combinations within a grid for a given performance metric using a
    specified machine learning model or pipeline.

    Parameters:
    -----------
    :param dims: Tuple[int, ...]
        A tuple of dimensions indies tc apply the feature selection onto. Any
        combination of dimensions can be specified, except for dimension 'zero', which
        represents the samples.
    :param estimator: Union[Any, Pipeline]
        The machine learning model or pipeline to evaluate feature sets.
    :param estimator_params: Dict[str, any], optional
        Optional parameters to adjust the estimator parameters.
    :param scoring: str, default = 'f1_weighted'
        The metric to optimize. Must be scikit-learn compatible.
    :param cv: Union[BaseCrossValidator, int, float], default = 10
        The cross-validation strategy or number of folds. If an integer is passed,
        :code:`train_test_split` for 1 and :code:`StratifiedKFold` is used for >1 as
        default. A float below 1 represents the percentage of training samples for the
        train-test split ratio.
    :param groups: Optional[numpy.ndarray], optional
        Groups for a LeaveOneGroupOut generator.
    :param patience: int, default = int(1e5)
        The number of iterations for which the objective function improvement must be
        below tol to stop optimization.
    :param tol: float, default = 1e-5
        The function tolerance; if the change in the best objective value is below this
        for 'patience' iterations, the optimization will stop early.
    :param bounds: Tuple[float, float], default = (0.0, 1.0)
        Bounds for the algorithm's parameters to optimize. Since it is a binary
        selection task, bounds are set to (0.0, 1.0).
    :param prior: numpy.ndarray, optional
        Explicitly initialize the optimizer state. If set to None, the to be optimized
        features are initialized randomly within the bounds.
    :param callback: Callable, optional
        A callback function of the structure :code: `callback(x, f, context)`, which
        will be called at each iteration. :code: `x` and :code: `f` are the solution and
        function value, and :code: `context` contains the diagnostics of the current
        iteration.
    :param n_jobs: int, default = 1
        The number of parallel jobs to run during cross-validation.
    :param random_state: int, optional
        Setting a seed to fix randomness (for reproducibility).
    :param verbose: Union[bool, int], default = False
         If set to True, enables the output of progress status during the optimization
         process.

    Methods:
    --------
    - fit:
        Fit the optimizer to the data.
    - transform:
        Transform the input data using the mask from the optimization process.

    Notes:
    ------
    This implementation is semi-compatible with the scikit-learn framework, which builds
    around two-dimensional feature matrices. To use this transformation within a
    scikit-learn Pipeline, the four dimensional data must be flattened after the first
    dimension [samples, features]. For example, scikit-learn's
    :code: `FunctionTransformer` can achieve this.

    Returns:
    --------
    :return: None
    """

    _parameter_constraints: dict = {
        "dims": [tuple],
        "estimator": [HasMethods(["fit"])],
        "estimator_params": [dict, None],
        "scoring": [StrOptions(set(get_scorer_names()))],
        "cv": [
            "cv_object",
            Interval(Integral, 1, None, closed="left"),
            Interval(Real, 0, 1, closed="both"),
        ],
        "groups": [numpy.ndarray, None],
        "tol": [Interval(Real, 0, None, closed="left")],
        "patience": [Interval(Integral, 0, None, closed="left")],
        "bounds": [tuple],
        "prior": [numpy.ndarray, None],
        "callback": [callable, None],
        "n_jobs": [Integral],
        "seed": ["random_state"],
        "verbose": ["verbose"],
    }

    def __init__(
        self,
        # General and Decoder
        dims: Tuple[int, ...],
        estimator: Union[Any, Pipeline],
        estimator_params: Optional[Dict[str, any]] = None,
        scoring: str = "f1_weighted",
        cv: Union[BaseCrossValidator, int, float] = 10,
        groups: Optional[numpy.ndarray] = None,
        # Training Settings
        tol: float = 1e-5,
        patience: int = int(1e5),
        bounds: Tuple[float, float] = (0.0, 1.0),
        prior: Optional[numpy.ndarray] = None,
        callback: Optional[Callable] = None,
        # Misc
        n_jobs: int = 1,
        random_state: Optional[int] = None,
        verbose: Union[bool, int] = False,
    ) -> None:

        # General and Decoder
        self.dims = dims
        self.estimator = estimator
        self.estimator_params = estimator_params
        self.scoring = scoring
        self.cv = cv
        self.groups = groups
        # Training Settings
        self.tol = tol
        self.patience = patience
        self.bounds = bounds
        self.prior = prior
        self.callback = callback
        # Misc
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    def _reset(self) -> None:
        """
        Reset internal data-dependent state of the scaler, if necessary.

        __init__ parameters are not touched.

        Return:
        -------
        :return: None
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

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(
        self, X: numpy.ndarray, y: Optional[numpy.ndarray] = None
    ) -> Type["BaseOptimizer"]:
        """
        Fit method to fit the optimizer to the data.

        Parameters:
        -----------
        :param X: numpy.ndarray
            Array-like with dimensions (e.g. [samples, channel_height,
            channel_width, time])
        :param y: numpy.ndarray, optional
            Array-like with dimensions [targets].

        Return:
        -------
        :return: Type['BaseOptimizer']
        """
        self._reset()

        self.X_, self.y_ = self._validate_data(
            X, y, reset=False, ensure_2d=False, allow_nd=True
        )
        del X, y

        if isinstance(self.cv, int):
            self.n_cv_ = self.cv
            self.split_ = 1 - self.n_cv_ if self.n_cv_ < 1 else 0.2
        else:
            self.n_cv_ = self.cv.get_n_splits(groups=self.groups)
            self.split_ = None

        if self.estimator_params:
            self._set_estimator_params()
        self._check_estimator_data_requirements()

        random.seed(self.random_state)
        numpy.random.seed(self.random_state)

        self.result_grid_ = []
        if self.n_jobs > 1:
            manager = multiprocessing.Manager()
            self.result_grid_ = manager.list()

        self.iter_ = 0
        self.dims_incl_ = sorted(self.dims)
        self.dim_size_ = tuple(numpy.array(self.X_.shape)[list(self.dims)])
        self.slices_ = tuple(
            [
                numpy.newaxis if dim not in self.dims_incl_ else slice(None)
                for dim in range(self.X_.ndim)
            ]
        )
        # self.algo_cores_, self.cv_cores_ = 0, self.n_jobs
        self.bounds_ = self._handle_bounds()
        self.prior_ = self._handle_prior()

        self.solution_, self.state_, self.score_ = self._run()
        self.mask_ = self._prepare_mask(self.state_.reshape(self.dim_size_))

        self._prepare_result_grid()
        return self

    def transform(
        self, X: numpy.ndarray, y: Optional[numpy.ndarray] = None
    ) -> numpy.ndarray:
        """
        Transforms the input with the mask obtained from the solution of the
        optimization process.

        Parameters:
        -----------
        :param X: numpy.ndarray
            Array-like with dimensions [samples, channel_height, channel_width, time]
        :param y: numpy.ndarray, optional
            Array-like with dimensions [targets].

        Return:
        -----------
        :return: numpy.ndarray
            Returns a sparse representation of the input tensor.
        """
        check_is_fitted(self)

        return X * self.mask_

    @abstractmethod
    def _run(self) -> NotImplementedError:
        """
        Abstract method defining the main optimization procedure. This method must be
        implemented by subclasses.

        Raises:
        -------
        :raises NotImplementedError:
            If the method is not implemented by the subclass.
        """
        raise NotImplementedError("The run method must be implemented by subclasses")

    def _objective_function(self, mask: numpy.ndarray) -> float:
        """
        Objective function that calculates the score to maximize.

        Parameters:
        -----------
        :param mask: numpy.ndarray
            The boolean mask indicating selected features.

        Returns:
        --------
        :return: float
            The evaluation score for the selected features, or 0 if no features
            are selected.
        """
        # Convert mask to a boolean array if necessary
        if numpy.array(mask).dtype != bool:
            mask = numpy.array(mask) > 0.5

        full_mask = self._prepare_mask(mask)
        selected_features = self.X_ * full_mask.astype(int)

        scores, train_times, infer_times = self._evaluate_candidates(selected_features)
        score, train_time, infer_time = (
            scores.mean(),
            train_times.mean(),
            infer_times.mean(),
        )
        self._save_statistics(mask, scores, train_time, infer_time)
        return score

    def _prepare_mask(
        self,
        mask: numpy.ndarray,
    ) -> numpy.ndarray:
        """
        Reshapes and broadcasts a mask to match the full shape of a data tensor.

        Parameters:
        -----------
        :param mask: numpy.ndarray
            The mask to be reshaped and broadcasted.

        Returns:
        --------
        :return: numpy.ndarray
            The full mask is broadcasted to match the shape of the data tensor.
        """
        reshaped_mask = mask.reshape(self.dim_size_)[self.slices_]
        full_mask = numpy.zeros(self.X_.shape, dtype=bool)
        full_mask[numpy.broadcast_to(reshaped_mask, full_mask.shape)] = True
        return full_mask

    def _evaluate_candidates(
        self, selected_features: numpy.ndarray
    ) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        """
        Evaluate the given features using cross-validation or train-test split.

        Parameters:
        -----------
        :param selected_features: numpy.ndarray
            The selected features to evaluate.

        Returns:
        --------
        :return: Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
            The evaluation scores, training and inference time for the selected
            features.
        """
        scorer = get_scorer(self.scoring)

        # Handle the edge case where no features are selected
        if selected_features.size == 0 or not numpy.any(selected_features):
            return (
                numpy.full((int(self.n_cv_),), fill_value=0.0),
                numpy.array([0]),
                numpy.array([0]),
            )

        # Handle train-test split where n_cv_ <= 1
        if self.n_cv_ <= 1:
            # Split the data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                selected_features,
                self.y_,
                test_size=self.split_,
                random_state=self.random_state,
            )
            estimator_clone = clone(self.estimator)

            with PerfTimer() as train_timer:
                estimator_clone.fit(X_train, y_train)
            with PerfTimer() as inference_timer:
                scores = scorer(estimator_clone, X_test, y_test)
            return (
                numpy.array(scores),
                numpy.array(train_timer.duration),
                numpy.array(inference_timer.duration),
            )

        # Handle cross-validation where n_cv_ > 1
        result = cross_validate(
            self.estimator,
            selected_features,
            self.y_,
            scoring=scorer,
            cv=self.cv,
            groups=self.groups,
            n_jobs=self.n_jobs,
        )
        return result["test_score"], result["fit_time"], result["score_time"]

    def _save_statistics(
        self,
        mask: numpy.ndarray,
        scores: numpy.ndarray,
        train_time: float,
        infer_time: float,
    ) -> None:
        """
        Saves the diagnostics at a given iteration. The diagnostics include: Method,
        Iteration, Mask, Size, Metric, Mean, Median, SD, CI Lower, CI Upper, Train Time,
        Infer Time and the score from the crossvaldiation folds (if selected).

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
        diagnostics = {
            "Method": self.__class__.__name__,
            "Iteration": self.iter_,
            "Mask": [mask.reshape(self.dim_size_)],
            "Size": numpy.sum(mask),
            "Metric": self.scoring,
            "Mean": numpy.round(numpy.mean(scores), 6),
            "Median": numpy.round(numpy.median(scores), 6),
            "SD": numpy.round(numpy.std(scores), 6),
            "Train Time": train_time,
            "Infer Time": infer_time,
        }

        if self.n_cv_ > 1:
            ci = stats.t.interval(
                0.95, len(scores) - 1, loc=numpy.mean(scores), scale=stats.sem(scores)
            )
            diagnostics["CI Lower"] = (numpy.round(ci[0], 6),)
            diagnostics["CI Upper"] = (numpy.round(ci[1], 6),)
            for i, score in enumerate(scores):
                diagnostics[f"Fold {i + 1}"] = score

        self.result_grid_.append(pd.DataFrame(diagnostics))

    @abstractmethod
    def _handle_bounds(self) -> NotImplementedError:
        """
        Placeholder method for handling bounds. This method must be implemented by
        subclasses.

        Raises:
        -------
        :raises NotImplementedError:
            When the method is not overridden by a subclass.
        """
        raise NotImplementedError(
            "The _handle_bounds method must be implemented by subclasses."
        )

    @abstractmethod
    def _handle_prior(self) -> NotImplementedError:
        """
        Handles the prior values by validating their shape and applying transformations
        if provided.

        Raises:
        -------
        :raises NotImplementedError:
            When the method is not overridden by a subclass.
        """
        raise NotImplementedError(
            "The _handle_priors method must be implemented by subclasses."
        )

    def _set_estimator_params(self) -> None:
        """
        Validate that all kwargs correspond to valid parameters in the given
        scikit-learn estimator or pipeline.

        Raises:
        -------
        :raise TypeError:
            If the estimator is not a scikit-learn estimator.
        :raise ValueError:
            If any key in kwargs is not a valid parameter.

        Returns:
        --------
        None
        """
        # Estimator Check
        estimator = (
            self.estimator.steps[-1][1]
            if isinstance(self.estimator, Pipeline)
            else self.estimator
        )
        if not isinstance(estimator, BaseEstimator):
            raise TypeError(
                "The provided estimator is not a valid scikit-learn estimator."
            )

        # Parameter Check
        invalid_params = [
            param
            for param in self.estimator_params
            if param not in estimator.get_params()
        ]
        if invalid_params:
            raise ValueError(
                f"Invalid parameter(s) for estimator: " f"{', '.join(invalid_params)}"
            )
        estimator.set_params(**self.estimator_params)

        # Set the parameters
        if isinstance(self.estimator, Pipeline):
            self.estimator.steps[-1][1].set_params(**self.estimator_params)
        else:
            self.estimator.set_params(**self.estimator_params)

    # def _allocate_cpu_resources(self, cv: int, n: int) -> Tuple[int, int]:
    #     """
    #     Choose the best (ea_cores, cv_cores) pair.
    #
    #     Parameters:
    #     -----------
    #     :param cv: int
    #         Number of cross-validation folds.
    #     :param n: int
    #         Number of generations in the evolutionary algorithm.
    #
    #     Raise:
    #     ______
    #     :raise ValueError:
    #         Unable to allocate CPU resources with the specified cross-validation folds,
    #         number of iterations, and the available free cores.
    #
    #     Returns:
    #     --------
    #     returns: Tuple[int, int]
    #         The best combination of cores (algo, cv) allocated to the algorithm
    #         and the cross-validation satisfying the constraints.
    #     """
    #     pairs = [
    #         (algo_cores, self.n_jobs // algo_cores)
    #         for algo_cores in range(1, self.n_jobs + 1)
    #         if self.n_jobs % algo_cores == 0
    #     ]
    #     valid_pairs = [
    #         (algo_cores, cv_cores)
    #         for algo_cores, cv_cores in pairs
    #         if algo_cores <= n and cv_cores <= cv and cv % cv_cores == 0
    #     ]
    #
    #     if not valid_pairs:
    #         raise ValueError(
    #             f"CPU resource allocation failed with constraints: n_jobs={self.n_jobs}, cv={cv}, n={n}."
    #         )
    #
    #     return max(valid_pairs, key=lambda x: x[1])

    def _check_estimator_data_requirements(self) -> None:
        """
        Performs a check on whether the passed estimator expects 2-dimensional data
        representation. Automatically adjusts the estimator to ensure compatibility.

        Raises:
        -------
        :raise ValueError:
            If the estimator raises a `ValueError` related to dimensionality that cannot
            be resolved by automatically inserting a :code: `FlattenTransformer` and
            :code: `SafeVarianceThreshold` into the pipeline, the ValueError is passed
            to the User.

        Returns:
        --------
        :returns: None
        """
        try:
            self.estimator.fit(self.X_[0], self.y_[0])
        except ValueError as e:
            cases = [
                "2D",
                "dim",
                "dims",
                "dimensions",
                "dimension",
                "dimensional",
                "input data",
                "input",
                "data",
            ]
            if any(term in str(e).lower() for term in cases):
                try:
                    test_estimator = Pipeline(
                        [
                            ("flatten", FlattenTransformer()),
                            ("clean", SafeVarianceThreshold(threshold=0.0)),
                            ("estimator", self.estimator),
                        ]
                    )
                    test_estimator.fit(self.X_[:2], self.y_[:2])
                    warnings.warn(
                        f"Estimator adjusted for ND data compatibility.", UserWarning
                    )
                    self.estimator = test_estimator
                except e as e:
                    raise e

    def _prepare_result_grid(self) -> None:
        """
        Finalizes the result grid.

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
