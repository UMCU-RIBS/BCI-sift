# -------------------------------------------------------------
# BCI-FeaST
# Copyright (c) 2024
#       Dirk Keller,
#       Elena Offenberg,
#       Nick Ramsey's Lab, University Medical Center Utrecht, University Utrecht
# Licensed under the MIT License [see LICENSE for detail]
# -------------------------------------------------------------
from collections import deque
from numbers import Real
from typing import Tuple, Union, Dict, Any, Optional, Callable

import numpy
import ray
from pyswarms.backend import (
    BoundaryHandler,
    VelocityHandler,
    OptionsHandler,
    compute_pbest,
    create_swarm,
    Swarm,
)
from pyswarms.backend.topology import Ring, Star
from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import Pipeline
from sklearn.utils._param_validation import (
    Integral,
    Interval,
    StrOptions,
)
from tqdm import tqdm

from .Base_Optimizer import BaseOptimizer

__all__ = ["ParticleSwarmOptimization"]


class ParticleSwarmOptimization(BaseOptimizer):
    """
    This class implements a Particle Swarm Optimization (PSO) algorithm to optimize
    the selection of feature combinations by iteratively trying to improve a
    candidate solution with regard to a predefined measure of quality. PSO is a
    bio-inspired optimization algorithm that simulates the social behavior of birds
    or fish to find optimal solutions in a search space through position-velocity
    updates. The implementation can leverage two topologies - a global [1]_ and a
    local [2]_ variant from the PySwarms library. For more information consult the
    documentation of PySwarms.

    It takes a set of candidate solutions, and tries to find the best solution using
    a position-velocity update method. The position update can be defined as:

    .. math::

        x_{i}(t+1) = x_{i}(t) + v_{i}(t+1)

    Where the position at the current timestep :math:`t` is updated using the
    computed velocity at :math:`t+1`. Furthermore, the velocity update is defined as:

    .. math::
        v_{ij}(t + 1) = w * v_{ij}(t) + c_{1}r_{1j}(t)[y_{ij}(t) − x_{ij}(t)]
                        + c_{2}r_{2j}(t)[\\hat{y}_{j}(t) − x_{ij}(t)]

    Here, :math:`c1` and :math:`c2` are the cognitive and social parameters
    respectively. They control the particle's behavior given two choices: (1) to
    follow its *personal best* or (2) follow the swarm's  *global best* position.
    Overall, this dictates if the swarm is explorative or exploitative in nature. In
    addition, a parameter :math:`w` controls the inertia of the swarm's movement.

    In local-best PSO, a particle doesn't compare itself to the overall performance
    of the swarm using a star-topology. Instead, it looks at the performance of its
    nearest-neighbours, using a ring-topology. In general, this kind of topology
    takes much more time to converge, but has a more powerful explorative feature.

    In this implementation, a neighbor is selected via a k-D tree imported from
    :code:`scipy`. Distance are computed with either the L1 or L2 distance. The
    nearest-neighbours are then queried from this k-D tree. They are computed for
    every iteration.

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
    :param topology: str, default = 'global'
        Valid ovptions are 'global and 'local'.
            * Global: Uses a star-topology, where every particle compares itself with
              the best-performing particle in the swarm, whereas
            * 'Local': Uses a ring topology, where every particle compares itself only
              with its nearest-neighbours as computed by a distance metric.
    :param n_particles: int, default = 128
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
        The number of neighbors to be considered. Must be a positive integer less than
        :code:`n_particles`. For 'local' topology only.
    :param p: int, default = 2
        The power parameter uses the Minkowski p-norm to use. 1 is the sum-of-absolute
        values (or L1 distance) while 2 is the Euclidean (or L2) distance. For 'local'
        topology only.
    :param oh_strategy: Optional[str], optional
        The update schedule to adjust 'c1', 'c2', 'w', 'k' and 'p' during the
        optimization (for each n_iter). Valid options are: 'constant', 'exp_decay',
        'lin_variation', 'random' and 'nonlin_mod'.
            * Constant: The parameter does not change.
            * Exponential decay: Decreases the parameter exponentially between limits.
            * Linear variation: Decreases/increases the parameter linearly between limits.
            * Random: takes a uniform random value between (0.5,1)
            * Nonlinear modulation: Decreases/increases the parameter between limits
              according to a nonlinear modulation index.
    :param bh_strategy: str, default = 'periodic'
        The strategy for the handeling of the out-of-bounds particles. Valid options
        are: 'nearest', 'random', 'shrink', 'reflective', 'intermediate' and 'periodic'.
            * Nearest: Reposition the particle to the nearest bound.
            * Random: Reposition the particle randomly in between the bounds.
            * Shrink: Shrink the velocity of the particle such that it lands on the
              bounds.
            * Reflective: Mirror the particle position from outside the bounds to inside
              the bounds.
            * Intermediate: Reposition the particle to the midpoint between its current
              position on the bound surpassing axis and the bound itself. This only
              adjusts the axes that surpass the boundaries.
            * Periodic: Resets the particles using the modulo function to cut down the
              position. This creates a virtual, periodic plane which is tiled with the
              search space.
    :param velocity_clamp: Optional[Tuple[float, float]], optional
        A tuple specifying the minimum and maximum velocity of particles.
    :param vh_strategy: str, default = 'unmodified'
        The strategy for the handeling of the velocity out-of-bounds particles. Valid
        options are 'unmodified', 'adjust', 'invert' and 'zero'.
            * Unmodified: Returns the unmodified velocites.
            * Adjust: Returns the velocity that is adjusted to be the distance between
              the current and the previous position.
            * Invert: Inverts and shrinks the velocity by a factor.
            * Zero: Sets the velocity of out-of-bounds particles to zero.
    :param center: float, default = 1.0
        The center point influence in the topology of the swarm.
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

    References:
    -----------
    .. [1] J. Kennedy and R.C. Eberhart, Particle Swarm Optimization,
        Proceedings of the IEEE International Joint Conference on Neural
        Networks, 1995, pp. 1942-1948.
    .. [2] TJ. Kennedy and R.C. Eberhart, A New Optimizer using Particle Swarm
        Theory, in Proceedings of the Sixth International Symposium on
        Micromachine and Human Science, 1995, pp. 3943.

    Examples:
    ---------
    The following example shows how to retrieve a feature mask for a synthetic data set.

    .. code-block:: python

        import numpy
        from sklearn.svm import SVC
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.datasets import make_classification
        from FingersVsGestures.src.channel_elimination import ParticleSwarmOptimization # TODO adjust

        X, y = make_classification(n_samples=100, n_features=8 * 4 * 100)
        X = X.reshape((100, 8, 4, 100))
        grid = (2, 3)
        estimator = Pipeline([('scaler', MinMaxScaler()), ('svc', SVC())])

        pso = ParticleSwarmOptimization(grid, estimator)
        pso.fit(X, y)
        print(pso.score_)
        35.275396825396825

    Return:
    -------
    :return: None
    """

    # fmt: off
    _parameter_constraints: dict = {**BaseOptimizer._parameter_constraints}
    _parameter_constraints.update(
        {
            "topology": [StrOptions({"global", "local"})],
            "n_particles": [Interval(Integral, 1, None, closed="left")],
            "n_iter": [Interval(Integral, 1, None, closed="left")],
            "c1": [Interval(Real, 0, None, closed="left")],
            "c2": [Interval(Real, 0, None, closed="left")],
            "w": [Interval(Real, 0, 1, closed="both")],
            "k": [Interval(Integral, 1, None, closed="left")],
            "p": [Interval(Integral, 1, None, closed="left")],
            "oh_strategy": [StrOptions({"exp_decay", "lin_variation", "random", "nonlin_mod"}),None,],
            "bh_strategy": [StrOptions({"nearest", "random", "shrink", "reflective", "intermediate","periodic",})],
            "velocity_clamp": ["array-like", None],
            "vh_strategy": [StrOptions({"unmodified", "adjust", "invert", "zero"})],
            "center": [Interval(Real, 0, None, closed="left")],
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
        # Particle Swarm Optimization Settings
        topology: str = "global",
        n_particles: int = 128,
        n_iter: int = 100,
        # Particle parameters
        c1: float = 0.5,
        c2: float = 0.3,
        w: float = 0.9,
        k: int = 20,
        p: int = 2,
        oh_strategy: Optional[str] = None,
        bh_strategy: str = "periodic",
        velocity_clamp: Optional[Tuple[float, float]] = None,
        vh_strategy: str = "unmodified",
        center: float = 1.0,
        # Training Settings
        tol: Union[Tuple[int, ...], float] = 1e-5,
        patience: Union[Tuple[int, ...], int] = int(1e5),
        bounds: Tuple[float, float] = (0.0, 1.0),
        prior: Optional[numpy.ndarray] = None,
        callback: Optional[Callable] = None,
        # Misc
        n_jobs: int = -1,
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

        # Particle Swarm Optimization Settings
        self.topology = topology
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

    def _run(self) -> Tuple[numpy.ndarray, numpy.ndarray, float]:
        """
        Runs the particle swarm optimziation algorithm to optimize the feature
        configuration, by evalauting the proposed candidate solutions in the objective
        function :code:`f` for a number of iterations :code:`n_iter.`

        Returns:
        --------
        Tuple[numpy.ndarray, numpy.ndarray, float]
            The best solution, mask found and their score.
        """
        if self.oh_strategy is None:
            self.oh_strategy = {}

        # Assign k-neighbors and p-value as attributes
        options = {"c1": self.c1, "c2": self.c2, "w": self.w}
        if self.topology == "local":
            options = {**options, "k": self.k, "p": self.p}

        # Initialize the topology
        top = Ring(static=False) if self.topology == "local" else Star()
        bh = BoundaryHandler(strategy=self.bh_strategy)
        vh = VelocityHandler(strategy=self.vh_strategy)
        oh = OptionsHandler(strategy=self.oh_strategy)

        # Initialize the swarm
        swarm = create_swarm(
            n_particles=self.n_particles,
            dimensions=int(numpy.prod(self._dim_size)),
            bounds=self._bounds,
            center=self.center,
            init_pos=self._prior,
            clamp=self.velocity_clamp,
            options=options,
        )
        swarm_size = (self.n_particles, numpy.prod(self._dim_size))
        swarm.pbest_cost = numpy.full(swarm_size[0], numpy.inf)

        # Populate memory of the handlers
        bh.memory = swarm.position
        vh.memory = swarm.position

        ftol_history = deque(maxlen=self.patience)

        # Run the search loop
        idtr = f"{self._dims_incl}: " if isinstance(self._dims_incl, int) else ""
        progress_bar = tqdm(
            range(self._update_n_iter(self.n_iter)),
            desc=f"{idtr}{self.__class__.__name__}",
            postfix=f"{0.000000:.6f}",
            disable=not self.verbose,
            leave=True,
        )
        for self.iter_ in progress_bar:
            # Compute cost for current position and personal best
            swarm.current_cost = self.compute_objective_function(swarm)
            swarm.pbest_pos, swarm.pbest_cost = compute_pbest(swarm)
            best_cost_yet_found = numpy.min(swarm.best_cost)

            # Update gbest from the neighborhood
            if self.topology == "local":
                swarm.best_pos, swarm.best_cost = top.compute_gbest(
                    swarm, p=self.p, k=self.k
                )
            else:
                swarm.best_pos, swarm.best_cost = top.compute_gbest(swarm)

            # Update logs and early stopping
            best_score = -swarm.best_cost
            progress_bar.set_postfix(best_score=f"{best_score:.6f}")
            if -swarm.best_cost >= 1.0:
                progress_bar.set_postfix(
                    best_score=f"Maximum score reached: {best_score:.6f}"
                )
                break
            elif self.callback is not None:
                if self.callback(swarm.best_cost, swarm.position, self.result_grid_):
                    progress_bar.set_postfix(
                        best_score=f"Stopped by callback: {best_score:.6f}"
                    )
                    break

            # Verify stop criteria based on the relative acceptable cost tol
            relative_measure = self._tol * (1 + numpy.abs(best_cost_yet_found))
            delta = numpy.abs(swarm.best_cost - best_cost_yet_found) < relative_measure
            if self.iter_ < self._patience:
                ftol_history.append(delta)
            else:
                ftol_history.append(delta)
                if all(ftol_history):
                    progress_bar.set_postfix(
                        best_score=f"Early Stopping Criteria reached: {best_score:.6f}"
                    )
                    break

            # Perform options update
            swarm.options = oh(options, iternow=self.iter_, itermax=self.n_iter)

            # Perform velocity and position updates
            swarm.velocity = top.compute_velocity(
                swarm, self.velocity_clamp, vh, self._bounds
            )
            swarm.position = top.compute_position(swarm, self._bounds, bh)

        # Obtain the final best_solution, best_state and best_score
        best_solution = swarm.pbest_pos  # [swarm.pbest_cost.argmin()].copy()
        best_state = swarm.pbest_pos[swarm.pbest_cost.argmin()] > 0.5
        best_score = -swarm.best_cost * 100

        return best_solution, best_state, best_score

    def compute_objective_function(
        self,
        swarm: Swarm,  # objective_func: Callable, pool: multiprocessing.Pool = None
    ):
        """
        Evaluate particles using the objective function. This method allows the PSO
        algorithm to interface correctly with the objective function by converting the
        input particle positions into individuals and evaluating them.

        If more than one cpu core is passed, then the evaluation of the particles is
        done in parallel using multiple processes.

        Parameters
        ----------
        :param swarm : Swarm
            A Swarm instance

        Returns
        -------
        :return: numpy.ndarray
            Cost-matrix for the given swarm.
        """
        positions_split = numpy.array_split(
            swarm.position, swarm.position.shape[0]
        )

        # fmt: off
        # Use multiprocessing pool to compute scores in parallel
        if self.n_jobs > 1:
            results = ray.get(
                [self._objective_function_wrapper.remote(self, pos) for pos in positions_split]
            )
        # If no pool is provided, compute scores sequentially
        else:
            results = [self._objective_function(pos) for pos in positions_split]
        # fmt: on

        return numpy.array(results) * -1

    def _handle_bounds(self) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """
        Returns the bounds for the PSO optimizer. If bounds are not set, default bounds
        of [0, 1] for each dimension are used.

        Returns:
        --------
        :return: Tuple[numpy.ndarray, numpy.ndarray]
            A tuple of two numpy arrays representing the lower and upper bounds for each
            particle.
        """
        return (
            numpy.full(numpy.prod(self._dim_size), self.bounds[0]),
            numpy.full(numpy.prod(self._dim_size), self.bounds[1]),
        )

    def _handle_prior(self) -> Optional[numpy.ndarray]:
        """
        This function checks the validity of the 'prior'; the function accepts a
        position-matrix (particles, features); otherwise if a one dimensional array
        (features, ) is provided Gaussian perturbation to the values are generated.

        Raises:
        -------
        :raise ValueError:
            If 'prior' is not None and does not match any expected type or format.

        Returns:
        -------
        :return: numpy.ndarray, optional
            A numpy array of transformed prior values, if the prior is a boolean array
            or None if no prior is provided.
        """
        if self.prior is None:
            return self.prior

        # If position-matrix is provided
        if isinstance(self.prior, numpy.ndarray) and self.prior.shape == (
            self.n_particles,
            numpy.prod(self._dim_size),
        ):
            return self.prior

        # if a simple ndarray mask is provided
        if isinstance(self.prior, numpy.ndarray) and self.prior.size == numpy.prod(
            self._dim_size
        ):
            gaus = numpy.abs(
                numpy.random.normal(0, 0.06125, (self.n_particles, self.prior.size))
            )
            return numpy.where(
                numpy.tile(self.prior.flatten(), (self.n_particles, 1)) < 0.5,
                0.49 - gaus,
                0.51 + gaus,
            )

        raise ValueError(
            f"The argument 'prior' must either be a weight matrix, matching"
            f"the size of the dimensions to be considered or be the"
            f" position-matrix of the particles (n_particles, features)."
        )

    @ray.remote
    def _objective_function_wrapper(self, particle: numpy.ndarray) -> float:
        """
        Wraps the objective function to adapt it for compatibility with ray's cpu
        parallelization.

        Parameters:
        -----------
        particle : numpy.ndarray
            An individual particle positions representing potential solutions.

        Returns:
        --------
        :return: numpy.ndarray
            The particle performance score.
        """
        return self._objective_function(particle)
        # numpy.array([-self._objective_function(x[i]) for i in range(x.shape[0])])
