import numpy
import ray
from sklearn.base import is_classifier, clone
from sklearn.metrics import check_scoring
from sklearn.metrics._scorer import _check_multimetric_scoring
from sklearn.model_selection import check_cv

from sklearn.model_selection._validation import (
    _warn_or_raise_about_fit_failures,
    _insert_error_scores,
    _aggregate_score_dicts,
    _normalize_score_results,
    _fit_and_score,
)
from sklearn.utils import indexable


# Define a remote function to wrap _fit_and_score
@ray.remote
def _remote_fit_and_score(
    estimator,
    X,
    y,
    scorers,
    train,
    test,
    verbose,
    parameters,
    fit_params,
    return_train_score,
    return_times,
    return_estimator,
    error_score,
):
    return _fit_and_score(
        estimator,
        X,
        y,
        scorers,
        train,
        test,
        verbose,
        parameters,
        fit_params,
        return_train_score=return_train_score,
        return_times=return_times,
        return_estimator=return_estimator,
        error_score=error_score,
    )


def cross_validate(
    estimator,
    X,
    y=None,
    *,
    groups=None,
    scoring=None,
    cv=None,
    n_jobs=None,
    verbose=0,
    fit_params=None,
    pre_dispatch="2*n_jobs",
    return_train_score=False,
    return_estimator=False,
    return_indices=False,
    error_score=numpy.nan,
):
    """Evaluate metric(s) by cross-validation and also record fit/score times.

    Read more in the :ref:`User Guide <multimetric_cross_validation>`.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The data to fit. Can be for example a list, or an array.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
        The target variable to try to predict in the case of
        supervised learning.

    groups : array-like of shape (n_samples,), default=None
        Group labels for the samples used while splitting the dataset into
        train/test set. Only used in conjunction with a "Group" :term:`cv`
        instance (e.g., :class:`GroupKFold`).

    scoring : str, callable, list, tuple, or dict, default=None
        Strategy to evaluate the performance of the cross-validated model on
        the test set.

        If `scoring` represents a single score, one can use:

        - a single string (see :ref:`scoring_parameter`);
        - a callable (see :ref:`scoring`) that returns a single value.

        If `scoring` represents multiple scores, one can use:

        - a list or tuple of unique strings;
        - a callable returning a dictionary where the keys are the metric
          names and the values are the metric scores;
        - a dictionary with metric names as keys and callables a values.

        See :ref:`multimetric_grid_search` for an example.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross validation,
        - int, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For int/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. versionchanged:: 0.22
            ``cv`` default value if None changed from 3-fold to 5-fold.

    n_jobs : int, default=None
        Number of jobs to run in parallel. Training the estimator and computing
        the score are parallelized over the cross-validation splits.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    verbose : int, default=0
        The verbosity level.

    fit_params : dict, default=None
        Parameters to pass to the fit method of the estimator.

    pre_dispatch : int or str, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - An int, giving the exact number of total jobs that are
              spawned

            - A str, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    return_train_score : bool, default=False
        Whether to include train scores.
        Computing training scores is used to get insights on how different
        parameter settings impact the overfitting/underfitting trade-off.
        However computing the scores on the training set can be computationally
        expensive and is not strictly required to select the parameters that
        yield the best generalization performance.

        .. versionadded:: 0.19

        .. versionchanged:: 0.21
            Default value was changed from ``True`` to ``False``

    return_estimator : bool, default=False
        Whether to return the estimators fitted on each split.

        .. versionadded:: 0.20

    return_indices : bool, default=False
        Whether to return the train-test indices selected for each split.

        .. versionadded:: 1.3

    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised.
        If a numeric value is given, FitFailedWarning is raised.

        .. versionadded:: 0.20

    Returns
    -------
    scores : dict of float arrays of shape (n_splits,)
        Array of scores of the estimator for each run of the cross validation.

        A dict of arrays containing the score/time arrays for each scorer is
        returned. The possible keys for this ``dict`` are:

            ``test_score``
                The score array for test scores on each cv split.
                Suffix ``_score`` in ``test_score`` changes to a specific
                metric like ``test_r2`` or ``test_auc`` if there are
                multiple scoring metrics in the scoring parameter.
            ``train_score``
                The score array for train scores on each cv split.
                Suffix ``_score`` in ``train_score`` changes to a specific
                metric like ``train_r2`` or ``train_auc`` if there are
                multiple scoring metrics in the scoring parameter.
                This is available only if ``return_train_score`` parameter
                is ``True``.
            ``fit_time``
                The time for fitting the estimator on the train
                set for each cv split.
            ``score_time``
                The time for scoring the estimator on the test set for each
                cv split. (Note time for scoring on the train set is not
                included even if ``return_train_score`` is set to ``True``
            ``estimator``
                The estimator objects for each cv split.
                This is available only if ``return_estimator`` parameter
                is set to ``True``.
            ``indices``
                The train/test positional indices for each cv split. A dictionary
                is returned where the keys are either `"train"` or `"test"`
                and the associated values are a list of integer-dtyped NumPy
                arrays with the indices. Available only if `return_indices=True`.
    """
    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))

    if callable(scoring):
        scorers = scoring
    elif scoring is None or isinstance(scoring, str):
        scorers = check_scoring(estimator, scoring)
    else:
        scorers = _check_multimetric_scoring(estimator, scoring)

    indices = cv.split(X, y, groups)
    if return_indices:
        # materialize the indices since we need to store them in the returned dict
        indices = list(indices)

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.

    if n_jobs > 1:
        results = ray.get(
            [
                _remote_fit_and_score.remote(
                    clone(estimator),
                    X,
                    y,
                    scorers,
                    train,
                    test,
                    verbose,
                    None,
                    fit_params,
                    return_train_score=return_train_score,
                    return_times=True,
                    return_estimator=return_estimator,
                    error_score=error_score,
                )
                for train, test in indices
            ]
        )
    else:
        results = [
            _fit_and_score(
                clone(estimator),
                X,
                y,
                scorers,
                train,
                test,
                verbose,
                None,
                fit_params,
                return_train_score=return_train_score,
                return_times=True,
                return_estimator=return_estimator,
                error_score=error_score,
            )
            for train, test in indices
        ]

    _warn_or_raise_about_fit_failures(results, error_score)

    # For callable scoring, the return type is only know after calling. If the
    # return type is a dictionary, the error scores can now be inserted with
    # the correct key.
    if callable(scoring):
        _insert_error_scores(results, error_score)

    results = _aggregate_score_dicts(results)

    ret = {}
    ret["fit_time"] = results["fit_time"]
    ret["score_time"] = results["score_time"]

    if return_estimator:
        ret["estimator"] = results["estimator"]

    if return_indices:
        ret["indices"] = {}
        ret["indices"]["train"], ret["indices"]["test"] = zip(*indices)

    test_scores_dict = _normalize_score_results(results["test_scores"])
    if return_train_score:
        train_scores_dict = _normalize_score_results(results["train_scores"])

    for name in test_scores_dict:
        ret["test_%s" % name] = test_scores_dict[name]
        if return_train_score:
            key = "train_%s" % name
            ret[key] = train_scores_dict[name]

    return ret
