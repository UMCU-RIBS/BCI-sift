# -------------------------------------------------------------
# BCI-FeaST
# Copyright (c) 2024
#       Dirk Keller,
#       Elena Offenberg,
#       Nick Ramsey's Lab, University Medical Center Utrecht, University Utrecht
# Licensed under the MIT License [see LICENSE for detail]
# -------------------------------------------------------------
import warnings
from copy import copy
from numbers import Integral
from operator import attrgetter
from typing import Tuple, Union, Dict, Optional, Any, Callable

import numpy
from sklearn import clone
from sklearn.metrics import get_scorer
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils._param_validation import Interval, RealNotInt
# from sklearn.utils._metadata_requests import _RoutingNotSupportedMixin
from tqdm import tqdm

from optimizer.backend._trainer import cross_validate
from .Base_Optimizer import BaseOptimizer

__all__ = ["RecursiveFeatureElimination"]

from .backend._backend import PerfTimer


class RecursiveFeatureElimination(BaseOptimizer):
    """
    This class implements Feature ranking with recursive feature elimination (RFE) for
    optimizing the selection of feature combinations by iteratively improving a
    candidate solution according to a predefined measure of quality. Given an external
    estimator that assigns weights to features (e.g., the coefficients of a linear
    model), the goal of RFE is to select features by recursively considering smaller and
    smaller sets of features.

    First, the estimator is trained on the initial set of features and the importance of
    each feature is obtained either through any specific attribute. Then, the least
    important features are pruned from current set of features. That procedure is
    recursively repeated on the pruned set until the desired number of features to
    select is eventually reached. The implementation is similar to scikit-leanr's RFE.
    For further details, consult the scikit-learn documentation.

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
    :param n_features_to_select: Union[float, int], default = 1
        Proportion of features that should be reached through elimination
    :param step: Union[float, int], default = 1
        Number of features to be eliminated in each step of the algorithm
    :param importance_getter: str, default = "named_steps.classifier.coef_" #TODO: change to auto see scikitlearn RFE
        String that specifies an attribute name/path for extracting feature importance
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
        Explicitly initialize the optimizer state. If set to None, the to be optimized
        features are initialized randomly within the bounds.
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

    Examples:
    ---------
    The following example shows how to retrieve a feature mask for a synthetic data set.

    .. code-block:: python

        import numpy
        from sklearn.svm import SVC
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.datasets import make_classification
        from FingersVsGestures.src.channel_elimination import RecursiveFeatureElimination # TODO adjust

        X, y = make_classification(n_samples=100, n_features=8 * 4 * 100)
        X = X.reshape((100, 8, 4, 100))
        grid = (2, 3)
        estimator = Pipeline([('scaler', MinMaxScaler()), ('svc', SVC())])

        rfe = RecursiveFeatureElimination(grid, estimator)
        rfe.fit(X, y)
        print(rfe.score_)
        28.679234564345677

    Returns:
    --------
    :return: None
    """

    # fmt: off
    _custom_store: dict = {**BaseOptimizer._custom_store}

    _parameter_constraints: dict = {**BaseOptimizer._parameter_constraints}
    _parameter_constraints.update(
        {
            "n_features_to_select": [Interval(RealNotInt, 0, 1, closed="right"),
                                     Interval(Integral, 0, None, closed="neither"), ],
            "step": [Interval(Integral, 0, None, closed="neither"),
                     Interval(RealNotInt, 0, 1, closed="neither"), ],
            "importance_getter": [str],
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
        # Recursive Feature Elimination Settings
        n_features_to_select: Union[int, float] = 1,
        step: Union[str, float] = 1,  # TODO automatical step size
        importance_getter: str = "named_steps.classifier.coef_",  # TODO Hardcoded mus be determined automatical as default
        # Training Settings
        tol: Union[Tuple[int, ...], float] = 1e-5,
        patience: Union[Tuple[int, ...], int] = int(1e5),
        bounds: Tuple[float, float] = (0.0, 1.0),
        prior: Optional[numpy.ndarray] = None,
        callback: Optional[Callable] = None,
        # Misc
        n_jobs: int = 1,
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

        # Recursive Feature Elimination Settings
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.importance_getter = importance_getter

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
        # Handle train-test split where n_cv_ <= 1
        if self._n_cv <= 1:
            X_train, X_test, y_train, y_test = train_test_split(
                selected_features,
                self._y,
                test_size=self._n_cv,
                random_state=self.random_state,
            )
            estimator_clone = clone(self._estimator)

            with PerfTimer() as train_timer:
                estimator_clone.fit(X_train, y_train)
            with PerfTimer() as inference_timer:
                scores = self._scorer(estimator_clone, X_test, y_test)
            self._custom_store["fitted_estimators"] = [estimator_clone]
            return (
                numpy.array(scores),
                numpy.array(train_timer.duration),
                numpy.array(inference_timer.duration),
            )

        # Handle cross-validation where n_cv_ > 1
        result = cross_validate(
            self._estimator,
            selected_features,
            self._y,
            scoring=get_scorer(self.scoring),
            cv=self.cv,
            groups=self.groups,
            n_jobs=self.n_jobs,
            return_estimator=True,
        )
        self._custom_store["fitted_estimators"] = result["estimator"]
        return result["test_score"], result["fit_time"], result["score_time"]

    # TODO: ask Dirk how to implement this
    def _handle_bounds(self):
        """Method to handle bounds for feature selection."""
        return self.bounds

    # TODO: ask Dirk how to implement this
    def _handle_prior(self):
        """Initialize the feature mask with the prior if provided."""
        if self.prior is not None:
            return self.prior
        else:
            return None

    def _run(self) -> Tuple[numpy.ndarray, numpy.ndarray, float]:
        """
        Executes the Recursive Feature Elimination

        Parameters:
        --------
        :return: Tuple[numpy.ndarray, float]
            The best channel configuration and its score.

        Returns:
        --------
        :return: Tuple[numpy.ndarray, numpy.ndarray, float, pandas.DataFrame]
            A tuple with the solution, mask, the evaluation scores and the optimization history.
        """

        wait = 0
        self.iter_ = 0
        best_score = 0.0
        best_state = None

        mask = numpy.ones(self._dim_size, dtype=bool)
        # getter = self.check_importance_getter()
        getter = attrgetter(self.importance_getter)

        # Determine number of features to select
        n_features = mask.size
        if isinstance(self.n_features_to_select, Integral):  # int
            n_features_to_select = self.n_features_to_select
            if n_features_to_select > n_features:
                warnings.warn(
                    (
                        f"Found {n_features_to_select=} > {n_features=}. There will be"
                        " no feature selection and all features will be kept."
                    ),
                    UserWarning,
                )
        else:  # float
            n_features_to_select = int(n_features * self.n_features_to_select)

        if 0.0 < self.step < 1.0:
            step = int(max(1, self.step * n_features))
        else:
            step = int(self.step)

        support = numpy.ones(shape=n_features, dtype=bool)

        # Run search
        idtr = f"{self._dims_incl}: " if isinstance(self._dims_incl, int) else ""
        progress_bar = tqdm(
            range(int((n_features - n_features_to_select) / step + 1)),
            desc=f"{idtr}{self.__class__.__name__}",
            postfix=f"{best_score:.6f}",
            disable=not self.verbose,
            leave=True,
        )
        for self.iter_ in progress_bar:
            features = numpy.arange(support.size)[support]
            score = self._objective_function(mask)
            fit_est = self._custom_store["fitted_estimators"]

            # Fetch and process coefficients
            coefs = numpy.stack([getter(est) for est in fit_est]).reshape(
                (len(fit_est), -1, n_features - self.iter_ * step)
            )
            ranks = numpy.argsort(
                numpy.mean(coefs**2, axis=tuple(range(coefs.ndim - 1)))
            )

            weights = numpy.zeros((n_features,), dtype=int)
            weights[support] = ranks.argsort()  # the larger the better

            # make sure step wouldn't reduce number of features below n_target_features
            threshold = min(step, numpy.sum(support) - n_features_to_select)

            # remove the least important features from result
            support[features[ranks][:threshold]] = False
            mask = support.reshape(mask.shape)

            # Update logs and early stopping
            wait += 1
            if best_score < score:
                if score - best_score > self._tol:
                    wait = 0
                best_score = score
                best_state = copy(mask)
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
                if self.callback(self.iter_, 1, self.result_grid_):
                    progress_bar.set_postfix(
                        best_score=f"Stopped by callback: {best_score:.6f}"
                    )
                    break

        best_solution = mask.reshape(-1).astype(float)
        best_score = best_score * 100
        return best_solution, best_state, best_score

    # def check_importance_getter(self) -> None:
    #     """
    #     Automatically retrieve the correct getter to return the coefficient of the
    #     machine learning model.
    #
    #     Raises:
    #     -------
    #     :raise ValueError:
    #         When importance_getter is not a callable it must be `coef_` or
    #         `feature_importances_`,otherwise a ValueError is passed to the User.
    #
    #     Returns:
    #     --------
    #     :returns: None
    #     """
    #     estimator = (
    #         self.estimator[-1]
    #         if isinstance(self.estimator, Pipeline)
    #         else self.estimator
    #     )
    #     if isinstance(self.importance_getter, str):
    #         if self.importance_getter == "auto":
    #             if hasattr(estimator, "coef_"):
    #                 getter = attrgetter("coef_")
    #             elif hasattr(estimator, "feature_importances_"):
    #                 getter = attrgetter("feature_importances_")
    #             else:
    #                 raise ValueError(
    #                     "when `importance_getter=='auto'`, the underlying "
    #                     f"estimator {estimator.__class__.__name__} should have "
    #                     "`coef_` or `feature_importances_` attribute. Either "
    #                     "pass a fitted estimator to feature selector or call fit "
    #                     "before calling transform."
    #                 )
    #         else:
    #             getter = attrgetter(self.importance_getter)
    #     elif not callable(self.importance_getter):
    #         raise ValueError("`importance_getter` has to be a string or `callable`")
    #     return getter

    #
    # def _check_estimator_data_requirements(self) -> None:
    #     """
    #     Update of :code: `_check_estimator_data_requirements`, assuming that only
    #     algorithms are used with RFE that rely on tabular data (e.g. scikit-learn-like).
    #     Automatically adjusts the estimator to ensure compatibility.
    #
    #     Raises:
    #     -------
    #     :raise ValueError:
    #         Automatically inserting a :code: `FlattenTransformer` and
    #         :code: `SafeVarianceThreshold` into the pipeline, the ValueError is passed
    #         to the User.
    #
    #     Returns:
    #     --------
    #     :returns: None
    #     """
    #     try:
    #         dim_comp = Pipeline(
    #             [
    #                 ("flatten", FlattenTransformer()),
    #                 ("clean", SafeVarianceThreshold(threshold=0.0)),
    #             ]
    #         )
    #         self._validate_data(dim_comp.fit_transform(self._X), self._y)
    #         warnings.warn(f"Estimator adjusted for ND data compatibility.", UserWarning)
    #         self._estimator = Pipeline(dim_comp.steps + self._estimator.steps)
    #     except ValueError as e:
    #         raise e
