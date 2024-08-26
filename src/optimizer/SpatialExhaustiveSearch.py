# -------------------------------------------------------------
# BCI-FeaST
# Copyright (c) 2024
#       Dirk Keller,
#       Elena Offenberg,
#       Nick Ramsey's Lab, University Medical Center Utrecht, University Utrecht
# Licensed under the MIT License [see LICENSE for detail]
# -------------------------------------------------------------

from typing import Tuple, Union, Dict, Any, Optional

import numpy
import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import Pipeline

from ._Base_Optimizer import BaseOptimizer
from ._Base_SES import SpatialExhaustiveSearch as SES
from .backend._backend import compute_subgrid_dimensions


class SpatialExhaustiveSearch(BaseOptimizer):
    """
    Implements a spatially constrained exhaustive search for finding
    global best channel combinations within a grid based on a given metric.

    Parameters:
    -----------
    :param dims: Tuple[int, ...]
        A tuple of dimensions indies tc apply the feature selection onto.
        Any combination of dimensions can be specified, except for
        dimension 'zero', which represents the samples.
    :param estimator: Union[BaseEstimator, Pipeline]
        The machine learning estimator or pipeline to evaluate
        channel combinations.
    :param estimator_params: Optional[Dict[str, any]], default = None
         Optional parameters to adjust the estimator parameters.
    :param metric: str, default = 'f1_weighted'
        The metric to optimize, compatible with scikit-learn metrics.
    :param cv: Union[BaseCrossValidator, int], default = 10
        Cross-validation splitting strategy, can be a fold number
        or a scikit-learn cross-validator.
    :param groups: numpy.ndarray, default = None
        Groups for LeaveOneGroupOut-crossvalidator.
    :param n_jobs: int, default = 1
        Number of parallel jobs to run during cross-validation.
         '-1' uses all available cores.
    :param seed: Optional[int], default = None
        Seed for randomness, ensuring reproducibility.
    :param verbose: Union[bool, int], default = False
        Enables verbose output during the optimization process.
    :param **kwargs: Dict[str, any]
        Optional parameters to adjust the estimator parameters.

    Methods:
    --------
    - fit:
        Fit the model to the data, search through the spatial
        constrained channel combinations.
    - transform:
        Apply the mask obtained from the search to transform the data.
    - run:
        Execute the spatial exhaustive search.
    - evaluate_candidates:
        Evaluates the selected features using cross-validation or train-test split.
    - objective_function:
        Evaluate each candidate configuration and return their scores.
    - elimination_plot:
        Generate and save a plot visualizing the performance across different subgrid sizes.
    - importance_plot:
        Generate and save a heatmap visualizing the importance of each channel.

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
    The following example shows how to retrieve a feature mask for
    a synthetic data set.

    # >>> import numpy as np
    # >>> from sklearn.svm import SVC
    # >>> from sklearn.pipeline import Pipeline
    # >>> from sklearn.preprocessing import MinMaxScaler
    # >>> from sklearn.datasets import make_classification
    # >>> from FingersVsGestures.src.channel_elimination import StochasticHillClimbing # TODO adjust
    # >>> X, y = make_classification(n_samples=100, n_features=8 * 4 * 100)
    # >>> X = X.reshape((100, 8, 4, 100))
    # >>> grid = np.arange(1, 33).reshape(X.shape[1:3])
    # >>> estimator = Pipeline([('scaler', MinMaxScaler()), ('svc', SVC())])

    # >>> shc = SpatialExhaustiveSearch(grid, estimator, verbose=True)
    # >>> shc.fit(X, y)
    # >>> print(shc.mask_)
    array([[False  True False False], [False False False False], [ True  True False False], [False False False  True],
           [False False False False], [False False False False], [False False  True False], [False False False False]])
    # >>> print(shc.score_)
     26.966666666666672

    Returns:
    --------
    :return: None
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

    def _run(
            self
    ) -> Tuple[numpy.ndarray, numpy.ndarray, float]:
        """
        Executes the Spatial Exhaustive Search.

        Parameters:
        --------
        :return: Tuple[numpy.ndarray, float]
            The best channel configuration and its score.

        Returns:
        --------
        :return: Tuple[numpy.ndarray, numpy.ndarray, float, pandas.DataFrame]
            A tuple with the solution, mask, the evaluation scores and the optimization history.
        """

        if len(self.dims) > 2:
            raise ValueError(
                f"{self.__class__.__name__} algorithm requires 'dims' to have"
                f"exactly 2 dimensions. Got {len(self.dims)}."
            )

        grid_dimensions = np.array(self.X_.shape)[np.array(self.dims)]
        grid = np.arange(1, np.prod(grid_dimensions) + 1).reshape(grid_dimensions)

        # Initialize and run the SSHC optimizer
        ses = SES(
            func=self._objective_function,
            channel_grid=grid,
            verbose=self.verbose
        )
        score, mask = ses.run()

        solution = mask.reshape(-1).astype(float)
        best_state = mask
        best_score = score * 100
        return solution, best_state, best_score

    def _handle_bounds(
            self
    ) -> None:
        """
        Placeholder method for handling bounds.

        Returns:
        -------
        :returns: None
        """

    def _handle_prior(
            self
    ) -> None:
        """
        Placeholder method for handling prior.

        Returns:
        -------
        :returns: None
        """

    def _prepare_result_grid(
            self
    ) -> None:
        """
        Finalizes the result grid. For the Spatial Exhaustive Search, the height
        and width of the included area is added.

        Returns:
        --------
        returns: None
        """

        # Concatenate the result grid along the rows (axis=0) and reset the index
        self.result_grid_ = pd.concat(self.result_grid_, axis=0, ignore_index=True)

        # Compute the height and width for each mask and assign them to the result grid
        self.result_grid_[['Height', 'Width']] = self.result_grid_['Mask'].apply(
            lambda mask: pd.Series(compute_subgrid_dimensions(mask.reshape(
                tuple(np.array(self.X_.shape)[np.array(self.dims)]))))
        )

        # Reorder the columns to place 'Height' and 'Width' after 'Size'
        columns = list(self.result_grid_.columns)
        size_index = columns.index('Size')
        new_order = columns[:size_index + 1] + ['Height', 'Width'] + columns[size_index + 1:-2]
        self.result_grid_ = self.result_grid_[new_order]
