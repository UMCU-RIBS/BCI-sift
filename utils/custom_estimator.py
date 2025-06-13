from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from typing import Callable, Optional, Union
import numpy as np


class CustomEstimator(BaseEstimator, ClassifierMixin):
    """
    A custom scikit-learn estimator that wraps a callable model,
    allowing it to be used within the scikit-learn ecosystem,
     even if it doesn't adhere to scikit-learn conventions.

    Parameters:
    -----------
    model : Callable
        The model to wrap. It should have `fit` and `predict`
        methods compatible with scikit-learn.
    model_params : dict, optional
        Dictionary of parameters to pass to the model upon initialization.
    """

    def __init__(
            self,
            model: Callable,
            model_params: Optional[dict] = None
    ):

        self.model = model
        self.model_params = model_params

    def fit(
            self, X: np.ndarray, y: np.ndarray
    ):
        """
        Fits the model to the data.

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Target vector.

        Returns:
        --------
        self : object
            Fitted estimator.
        """
        # Validate data
        self.X_, self.y_ = self._validate_data(
            X, y, reset=False, **{'ensure_2d': False, 'allow_nd': True}
        )

        self.model_params_ = self.model_params
        if not self.model_params:
            self.model_params_ = {}

        # Initialize the model with the parameters
        if callable(self.model):
            self.model_instance = self.model(**self.model_params_)
        else:
            raise ValueError("Model must be callable.")

        # Fit the model
        self.model_instance.fit(X, y)

        # Store fitted status
        self.is_fitted_ = True
        return self

    def predict(
            self, X: np.ndarray
    ) -> np.ndarray:
        """
        Predicts the target for the given feature matrix.

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix.

        Returns:
        --------
        y_pred : np.ndarray
            Predicted target vector.
        """
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X)

        # Make predictions
        return self.model_instance.predict(X)
