# -------------------------------------------------------------
# BCI-FeaST
# Copyright (c) 2024
#       Dirk Keller,
#       Elena Offenberg,
#       Nick Ramsey's Lab, University Medical Center Utrecht, University Utrecht
# Licensed under the MIT License [see LICENSE for detail]
# -------------------------------------------------------------

from copy import copy
from operator import attrgetter
from typing import Tuple, Union, Dict, Optional

import numpy
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import get_scorer
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
# from sklearn.utils._metadata_requests import _RoutingNotSupportedMixin
from sklearn.utils.validation import check_is_fitted as sklearn_is_fitted
from tqdm import tqdm

from .Base_Optimizer import BaseOptimizer


class RecursiveFeatureElimination(BaseOptimizer):
    """
    Implements a recursive feature elimination search for finding
    global best channel combinations within a grid based on a given metric.

    Parameters:
    -----------
    :param dimensions: Tuple[int, ...]
        A tuple of dimensions indies tc apply the feature selection onto.
        Any combination of dimensions can be specified, except for
        dimension 'zero', which represents the samples.
    :param estimator: Union[BaseEstimator, Pipeline]
        The machine learning estimator or pipeline to evaluate
        channel combinations.
    :param estimator_params: Optional[Dict[str, any]], default = None
         Optional parameters to adjust the estimator parameters.
    :param scoring: str, default = 'f1_weighted'
        The metric to optimize, compatible with scikit-learn metrics.
    :param cv: Union[BaseCrossValidator, int], default = 10
        Cross-validation splitting strategy, can be a fold number
        or a scikit-learn cross-validator.
    :param groups: numpy.ndarray, default = None
        Groups for LeaveOneGroupOut-crossvalidator
    :params feature_retention_ratio: float or string, default = "auto"
        Proportion of features that should be reached through elimination; "auto": reduce to just one feature
    :param step: int, default = 1
        Number of features to be eliminated in each step of the algorithm
    :param importance_getter: str, default = "named_steps.svc.coef_" #TODO: change default?
        String that specifies an attribute name/path for extracting feature importance
    :param n_jobs: int, default = 1
        Number of parallel jobs to run during cross-validation.
         '-1' uses all available cores.
    :param random_state: Optional[int], default = None
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
        Apply the mask obtained from the search to transform the data.
    - run:
        Execute the recursive feature elimination.
    - evaluate_candidates:
        Evaluates the selected features using cross-validation or train-test split.
    - objective_function:
        Evaluate each candidate configuration and return their scores.

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

    Returns:
    --------
    :return: None
    """

    def __init__(
        self,
        # General and Decoder
        dimensions: Tuple[int, ...],
        estimator: Union[BaseEstimator, Pipeline],
        estimator_params: Optional[Dict[str, any]] = None,
        scoring: str = "f1_weighted",
        cv: Union[BaseCrossValidator, int] = 10,
        groups: numpy.ndarray = None,
        # Recursive Feature Elimination Settings
        feature_retention_ratio: Union[str, float] = "auto",
        step: int = 1,
        importance_getter: str = "named_steps.svc.coef_",
        # Misc
        n_jobs: int = 1,
        random_state: Optional[int] = None,
        verbose: Union[bool, int] = False,
    ) -> None:

        super().__init__(
            dimensions,
            estimator,
            estimator_params,
            scoring,
            cv,
            groups,
            n_jobs,
            random_state,
            verbose,
        )

        # Recursive Feature Elimination Settings
        self.feature_retention_ratio = feature_retention_ratio
        self.step = step
        self.importance_getter = importance_getter

    # def fit(
    #         self, X: numpy.ndarray, y: numpy.ndarray = None
    # ) -> Type['RecursiveFeatureElimination']:
    #     """
    #     Fit method optimizes the channel combination with
    #     Recursive Feature Elimination.

    #     Parameters:
    #     -----------
    #     :param X: numpy.ndarray
    #         Array-like with dimensions [samples, channel_height, channel_width, time]
    #     :param y: numpy.ndarray, default = None
    #         Array-like with dimensions [targets].

    #     Return:
    #     -----------
    #     :return: Type['RecursiveFeatureElimination']
    #     """
    #     self.X_, self.y_ = self._validate_data(
    #         X, y, reset=False, **{'ensure_2d': False, 'allow_nd': True}
    #     )

    #     self.iter_ = int(0)
    #     self.result_grid_ = []
    #     self.ranks_ = []

    #     self.solution_, self.mask_, self.score_ = self._run()

    #     # Conclude the result grid
    #     self.result_grid_ = pd.concat(self.result_grid_, axis=0, ignore_index=True)

    #     return self

    def _objective_function(self, mask: numpy.ndarray) -> float:
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
        X_sub = self.X_[:, mask].reshape(self.X_.shape[0], -1)

        scores = self._evaluate_candidates(X_sub)
        self._save_statistics(copy(mask).reshape(self.dim_size_), scores)

        return scores.mean()

    def _evaluate_candidates(self, selected_features: numpy.ndarray) -> numpy.ndarray:
        """
        Evaluate the given features using cross-validation or train-test split.

        Parameters:
        -----------
        :param selected_features: numpy.ndarray
            The selected features to evaluate.

        Returns:
        --------
        :return: numpy.ndarray #TODO: document coefs being returned
            The evaluation scores for the selected features.
        """
        if self.cv == 1:
            # Use train-test split instead of cross-validation
            X_train, X_test, y_train, y_test = train_test_split(
                selected_features,
                self.y_,
                test_size=0.2,
                random_state=self.random_state,
            )
            self.estimator.fit(X_train, y_train)
            scorer = get_scorer(self.scoring)
            # get feature weights
            getter = attrgetter(self.importance_getter)
            coefs = getter(self.estimator).reshape(
                (getter(self.estimator).shape[0], -1, self.X_.shape[3])
            )
            self.ranks_ = np.argsort(np.mean(coefs**2, axis=(0, 2)))
            scores = scorer(self.estimator, X_test, y_test)

        else:
            # Use cross-validation
            results = cross_validate(
                self.estimator,
                selected_features,
                self.y_,
                scoring=get_scorer(self.scoring),
                cv=self.cv,
                groups=self.groups,
                n_jobs=self.n_jobs,
                return_estimator=True,
            )
            res_ests = results["estimator"]
            getter = attrgetter(self.importance_getter)
            coefs = []
            for est in res_ests:
                coefs.append(getter(est[-1]))
            coefs = np.stack(coefs).reshape(
                (len(res_ests), coefs[0].shape[0], -1, self.X_.shape[3])
            )
            self.ranks_ = np.argsort(np.mean(coefs**2, axis=(0, 1, 3)))
            scores = results["test_score"]

        return scores

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

    def transform(self, X: numpy.ndarray, y: numpy.ndarray = None) -> numpy.ndarray:
        """
        Transforms the input with the mask obtained from
        the solution of Recursive Feature Elimination

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
        mask = np.ones(
            self.dim_size_, dtype=bool
        )  # TODO: exclude bad channels here (_handle_prior?)
        best_score = 0.0
        best_mask = None

        n_features = np.sum(mask)
        if self.feature_retention_ratio == "auto":
            self.n_target_features = 1
        else:
            self.n_target_features = int(self.feature_retention_ratio * n_features)
        support_ = np.ones(shape=np.sum(mask), dtype=bool)
        pbar = range(int((np.sum(mask) - self.n_target_features) / self.step + 1))
        if self.verbose:
            pbar = tqdm(
                pbar,
                desc="Recursive Feature Elimination",
                postfix={"best score": f"{best_score:.6f}"},
            )
        for _ in pbar:
            features = np.arange(support_.size)[support_]
            score = self._objective_function(mask)
            weights = np.zeros((n_features,), dtype=int)
            weights[support_] = self.ranks_.argsort()  # the larger the better
            # make sure step wouldn't reduce number of features below n_target_features
            threshold = min(self.step, np.sum(support_) - self.n_target_features)
            # remove least important features from result
            support_[features[self.ranks_][:threshold]] = False
            mask = support_.reshape(mask.shape)
            if (
                score >= best_score
            ):  # now: best mask is minimum # of electrodes with best score
                best_score = score
                best_mask = copy(mask)
            if self.verbose:
                pbar.set_postfix({"score": f"{best_score:.6f}"})

        solution = mask.reshape(-1).astype(float)
        best_score = best_score * 100
        return solution, best_mask, best_score
