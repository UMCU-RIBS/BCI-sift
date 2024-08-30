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
from pyswarms.discrete import BinaryPSO
from pyswarms.single import GlobalBestPSO, LocalBestPSO
# from sklearn.utils._metadata_requests import _RoutingNotSupportedMixin
from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import Pipeline

from .Base_Optimizer import BaseOptimizer


class ParticleSwarmOptimization(BaseOptimizer):
    """
    Implements Particle Swarm Optimization (PSO) to optimize
    the selection of channel combinations within a grid based on
    a specified performance metric. The implementation uses pyswarms
    global and local PSO implementation PSO is a computational method
    that optimizes a problem by iteratively trying to improve a
    candidate solution with regard to a given measure of quality.

    Implements both global and local variants of Particle Swarm
    Optimization (PSO) [1]_ [2]_ from the pyswarm library. The PSO
    optimizes channel combinations within a structured grid. PSO is
    a bio-inspired optimization algorithm that simulates the social
    behavior of birds or fish to find optimal solutions in a search
    space through position-velocity updates.

    The global PSO Uses a star-topology where each particle is attracted
    to the best-performing particle in the swarm. The position and velocity
    updates are defined as:
        Position: xi(t+1) = xi(t) + vi(t+1)
        Velocity: vij(t+1) = w * vij(t) + c1 * r1j(t) *
                             [yij(t) - xij(t)] + c2 * r2j(t) * [y^j(t) - xij(t)]
    Here, 'w' is the inertia weight, 'c1' and 'c2' are cognitive and
    social parameters respectively, controlling the particle's adherence
    to  personal best and swarm's global best. This method is explorative
    or exploitative based on the relative weighting of these parameters.


    The Local PSO employs a ring topology, where each particle is influenced
    by its local neighborhood's best performance rather than the global best.
    The position and velocity updates are similar to the global PSO:
        Position: xi(t+1) = xi(t) + vi(t+1)
        Velocity: vij(t+1) = m * vij(t) + c1 * r1j(t) *
                             [yij(t) - xij(t)] + c2 * r2j(t) * [y^j(t) - xij(t)]
    However, each particle compares itself to the best in its neighborhood,
    determined using a k-D tree to manage spatial queries based on L1 or
    L2 distances. Local PSO generally converges slower than global PSO but
    promotes more exploration, potentially escaping local optima more effectively.

    Both methods update each particle's position based on its velocity at
    each time step, with the ultimate goal of finding the optimal configuration
    of channels within the grid based on the specified performance metric.

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
    :param method: str, default = 'global'
        The variant of PSO to use ('global' for global best PSO, 'local'
        for local best PSO).
    :param n_particles: int, default = 80
        The number of particles in the swarm.
    :param n_iter: int, default = 100
        The number of iterations for the PSO process.
    :param c1: float, default = 0.5
        Cognitive parameter.
    :param c2: float, default = 0.3
        Social parameter.
    :param w: float, default = 0.9
        Inertia weight of the PSO.
    :param k: int, default = 20
        Number of neighbors to consider in local best PSO.
    :param p: int, default = 2
        The power parameter for local best PSO.
    :param oh_strategy: Optional[str], default = None
        The strategy for handling boundary conditions of the particles.
    :param bh_strategy: str, default = 'periodic'
        The boundary handling strategy in PSO.
    :param velocity_clamp: Optional[Tuple[float, float]], default = None
        A tuple specifying the minimum and maximum velocity of particles.
    :param vh_strategy: str, default = 'unmodified'
        The velocity handling strategy in PSO.
    :param center: float, default = 1.0
        The center point influence in the topology of the swarm.
    :param tol: float, default = 1e-5
        The function tolerance; if the change in the best objective value
        is below this for `patientce` iterations, the optimization will stop early.
    :param patience: int, default = 1e5
        The number of iterations for which the objective function
        improvement must be below `tol` to stop optimization.
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
        Fit the optimizer to the data.
    - transform:
        Transform the input data using the mask from the optimization process.
    - run:
        Execute the PSO optimization algorithm.
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
    This implementation is semi-compatible with the scikit-learn
    framework, which builds around two-dimensional feature matrices.
    To use this transfortmation within a scikit-learn Pipeline, the
    four dimensional data must eb flattened after the first dimension
    [samples, features]. For example, scikit-learn's FunctionTransformer can
    achieve this.

    References:
    -----------
    .. [1] J. Kennedy and R.C. Eberhart, Particle Swarm Optimization,
           Proceedings of the IEEE International Joint Conference on
            Neural Networks, 1995, pp. 1942-1948.
    .. [2] TJ. Kennedy and R.C. Eberhart, A New Optimizer using Particle
           Swarm Theory, in Proceedings of the Sixth International Symposium
           on Micromachine and Human Science, 1995, pp. 3943.

    Examples:
    ---------
    The following example shows how to retrieve a feature mask for
    a synthetic data set.

    # >>> import numpy as np
    # >>> from sklearn.svm import SVC
    # >>> from sklearn.pipeline import Pipeline
    # >>> from sklearn.preprocessing import MinMaxScaler
    # >>> from sklearn.datasets import make_classification
    # >>> from FingersVsGestures.src.channel_elimination import ParticleSwarmOptimization # TODO adjust
    # >>> X, y = make_classification(n_samples=100, n_features=8 * 4 * 100)
    # >>> X = X.reshape((100, 8, 4, 100))
    # >>> grid = np.arange(1, 33).reshape(X.shape[1:3])
    # >>> estimator = Pipeline([('scaler', MinMaxScaler()), ('svc', SVC())])

    # >>> pso = ParticleSwarmOptimization(grid, estimator)
    # >>> pso.fit(X, y)
    # >>> print(pso.mask_)
    array([[False  True False False], [False False False False], [False False False False], [ True False False False]
           [False False False False], [False False False False], [False False False False]])
    # >>> print(pso.score_)
    35.275396825396825

    Return:
    -------
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

            # Particle Swarm Optimization Settings
            method: str = 'global',
            n_particles: int = 80,
            n_iter: int = 100,

            # Particle parameters
            c1: float = 0.5,
            c2: float = 0.3,
            w: float = 0.9,
            k: int = 20,
            p: int = 2,
            oh_strategy: Optional[str] = None,
            bh_strategy: str = 'periodic',
            velocity_clamp: Optional[Tuple[float, float]] = None,
            vh_strategy: str = 'unmodified',
            center: float = 1.0,

            # Training Settings
            tol: float = 1e-5,
            patience: int = 1e5,
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

        # Particle Swarm Optimization Settings
        self.method = method
        self.bounds = bounds
        self.n_particles = n_particles
        self.n_iter = n_iter

        # Particle parameters
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.k = k
        self.p = p
        self.oh_strategy = oh_strategy
        self.bh_strategy = bh_strategy
        self.velocity_clamp = velocity_clamp
        self.vh_strategy = vh_strategy
        self.center = center

        # Training Settings
        self.tol = tol
        self.patience = patience

    def _run(
            self
    ) -> Tuple[numpy.ndarray, numpy.ndarray, float]:
        """
        Run the PSO algorithm to optimize the channel configuration.

        Returns:
        --------
        Tuple[numpy.ndarray, numpy.ndarray, float]
            The best solution, mask found and their score.
        """
        # Initialize PSO algorithm
        method = self._init_method()

        # Optimize Feature Selection
        cost, pos = method.optimize(
            self._objective_function_wrapper, iters=self.n_iter, verbose=self.verbose  # , n_processes=self.algo_cores_
        )

        best_state = self._mask_to_input_dims((pos > 0.5).reshape(self.dim_size_))  # (self.grid.shape)
        best_score = -cost * 100
        return pos, best_state, best_score

    def _handle_bounds(
            self
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the bounds for the PSO optimizer. If bounds are not set, default bounds
        of [0, 1] for each dimension are used.

        Returns:
        --------
        :return: Tuple[np.ndarray, np.ndarray]
            A tuple of two numpy arrays representing the lower and upper bounds.
        """
        return (
            np.full(np.prod(self.dim_size_), self.bounds[0]),  # np.array([self.bounds[0]] * np.prod(self.dim_size_)),
            np.full(np.prod(self.dim_size_), self.bounds[1])  # np.array(self.bounds[1] * np.prod(self.dim_size_))
        )

    def _handle_prior(
            self
    ) -> Optional[np.ndarray]:
        """
        Generates a list of prior individuals based on a provided
        prior mask. The function checks the validity of the prior
        mask's shape, applies a Gaussian perturbation to the mask
        values, and creates an array of bounds for each particles.

        Returns:
        -------
        :return: Optional[np.ndarray]
            A numpy array of transformed prior values or None if no prior is provided.
        """
        prior = self.prior
        if prior is not None:
            if self.prior.shape != self.dim_size_:  # self.grid.reshape(-1).shape:
                raise RuntimeError(
                    f'The argument prior must match the size of the dimensions to be considered.'
                    f'Got {self.prior.shape} but expected {self.dim_size_}.')  # {self.grid.reshape(-1).shape}.')

            particle_pos = np.tile(self.prior.astype(float), (self.n_particles, 1))
            prior = np.array(
                [np.where(x > 0.5, 0.51 + np.random.normal(loc=0, scale=0.06125),
                          0.49 - np.random.normal(loc=0, scale=0.06125))
                 for i, x in enumerate(list(particle_pos))]
            )
        return prior

    def _objective_function_wrapper(
            self, x: numpy.ndarray
    ) -> numpy.ndarray:
        """
        Wraps the objective function to adapt it for compatibility with the PSO algorithm. This method allows
        the PSO algorithm to interface correctly with the objective function by converting the input particle
        positions into a suitable format and evaluating them.

        Parameters:
        -----------
        x : numpy.ndarray
            An array of particle positions representing potential solutions.

        Returns:
        --------
        numpy.ndarray
            An array of fitness values for each particle in the swarm.
        """
        return np.array([-self._objective_function(x[i]) for i in range(x.shape[0])])

    def _init_method(
            self
    ) -> Union[GlobalBestPSO, LocalBestPSO]:
        """
        Initializes the PSO optimizer based on the specified method
        and parameters.

        :return: Union[GlobalBestPSO, LocalBestPSO]
            The initialized PSO optimizer object.
        """
        # Prepare the arguments for the PSO optimizer
        method_args = {
            'n_particles': self.n_particles,
            'dimensions': np.prod(self.dim_size_),
            'options': {'c1': self.c1, 'c2': self.c2, 'w': self.w, 'k': self.k, 'p': self.p},
            'bounds': self.bounds_,
            'oh_strategy': self.oh_strategy,
            'bh_strategy': self.bh_strategy,
            'velocity_clamp': self.velocity_clamp,
            'vh_strategy': self.vh_strategy,
            'center': self.center,
            'init_pos': self.prior_,
            'ftol': self.tol,
            'ftol_iter': self.patience
        }

        # Define the method library
        method_lib = {
            'global': GlobalBestPSO(**method_args),
            'local': LocalBestPSO(**method_args),
            'binary': BinaryPSO(**method_args)
        }

        # Return the appropriate optimizer based on the selected method
        return method_lib[self.method]
