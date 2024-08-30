# -------------------------------------------------------------
# BCI-FeaST
# Copyright (c) 2024
#       Dirk Keller,
#       Elena Offenberg,
#       Nick Ramsey's Lab, University Medical Center Utrecht, University Utrecht
# Licensed under the MIT License [see LICENSE for detail]
# -------------------------------------------------------------

import random
from typing import Tuple, List, Union, Dict, Any, Optional, Callable, Type

import numpy
import numpy as np
from deap import base, creator, tools, algorithms
from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from .Base_Optimizer import BaseOptimizer


class EvolutionaryAlgorithms(BaseOptimizer):
    """
    This class implements an evolutionary algorithm-based optimizer
    for channel selection within a structured grid. It uses genetic
    algorithm techniques from the deap library such as crossover,
    mutation, and selection to evolve a population of potential
    solutions to find the best channel combinations for a given metric.
    The class supports multiple evolutionary strategies (e.g. simple [1]_,
    MuPlusLambda, MuCommaLambda, GenerateUpdate [2]_), customizable genetic
    operations, and parallel island evolution. For more information consult
    the documentation of deap.

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
    :param population_size: int, default = 120
        The size of the population in each generation.
    :param n_gen: int, default = 100
        The number of generations over which the population evolves.
    :param islands: int, default = 1
        The number of separate populations (islands) used in
        parallel evolutionary processes.
    :param method: str, default = 'simple'
        The evolutionary algorithm to use. Options include 'simple',
        'mu_plus_lambda', 'mu_lambda', 'generate_update'.
    :param crossover: str, default = 'two_point'
        The method used for crossing over individuals in the population.
        Options include 'one_point', 'two_point', 'uniform', etc.
    :param mutate: str, default = 'flip'
        The mutation method applied to offspring.
        Options include 'gaussian', 'shuffle', 'flip', etc.
    :param selection: str, default = 'tournament'
        The method used to select individuals for the next generation.
         Options include 'tournament', 'roulette', 'nsga2', etc.
    :param mu: int, default = 30
        The number of individuals to select for the next generation
        in 'mu_plus_lambda' and 'mu_lambda' methods.
    :param lmbda: int, default = 60
        The number of children to produce in 'mu_plus_lambda'
        and 'mu_lambda' methods.
    :param migration_chance: float, default = 0.1
        The probability of migrating individuals among islands per generation.
    :param migration_size: int, default = 5
        The number of individuals to migrate between islands
        when migration occurs.
    :param cxpb: float, default = 0.5
        The probability of mating two individuals (crossover probability).
    :param mutpb: float, default = 0.2
        The probability of mutating an individual (mutation probability).
    :param cx_indpb: float, default = 0.05
        The independent probability of each attribute being
        exchanged during crossover.
    :param cx_alpha: float, default = 0.3
        The alpha value for blend crossover.
    :param cx_eta: float, default = 5
        The eta value for simulated binary crossover.
    :param mut_sigma: float, default = 0.1
        The standard deviation of the Gaussian distribution
        used for Gaussian mutation.
    :param mut_mu: float, default = 0
        The mean of the Gaussian distribution used for Gaussian mutation.
    :param mut_indpb: float, default = 0.2
        The independent probability of each attribute being mutated.
    :param mut_c: float, default = 0.01
        The c value for ES log-normal mutation.
    mut_eta: float, default = 40
        The eta value for polynomial mutation.
    sel_tournsize: int, default = 3
        The tournament size for tournament selection method.
    sel_nd: str, default = 'standard'
        The non-dominated sorting type for NSGA-II selection.
    :param patience: int, default = 1e5
        The number of iterations for which the objective function
        improvement must be below tol to stop optimization.
    :param tol: float, default = 1e-5
        The function tolerance; if the change in the best objective value
        is below this for 'patience' iterations, the optimization will stop early.
    :param bounds: Tuple[float, float], default = (0.0, 1.0)
        Bounds for the algorithm's parameters to optimize. Since
        it is a binary selection task, bounds are set to (0.0, 1.0).
    :param prior: Optional[numpy.ndarray], default = None
        Explicitly initialize the optimizer state.
        If set to None if the to be optimized features are
        initialized randomly within the bounds.
    :param callback: Optional[Union[Callable, Type]], default = None, #TODO add description and callback design
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
        Fits the evolutionary algorithm on the data and calculates the mask by calling run.
    - transform:
        Transforms the input data by filtering the channels with the mask.
    - run:
        Executes the evolutionary optimization process, managing both single
        and multi-island configurations.
    - evaluate_candidates:
        Evaluates the selected features using cross-validation or train-test split.
    - objective_function:
        Calculates the score to maximize or minimize based on the provided mask.
    - elimination_plot:
        Generates and saves a plot visualizing the maximum and all scores across different subgrid sizes.
    - importance_plot:
        Generates and saves a heatmap visualizing the importance of each channel within the grid.
    - objective_function_wrapper:
        A wrapper for the objective function to adapt it to the optimization process.
    - migrate:
        Migrates individuals between islands based on the specified topology.

    Notes:
    ------
    This implementation is semi-compatible with the scikit-learn framework,
    which builds around two-dimensional feature matrices. To use this
    transfortmation within a scikit-learn Pipeline, the four dimensional data
    must eb flattened after the first dimension [samples, features]. For example,
    scikit-learn's FunctionTransformer can achieve this.

    Care must be taken when the lambda:mu ratio is 1 to 1 for the MuPlusLambda and
    MuCommaLambda algorithms as a non-stochastic selection will result in no  selection
    at all as the operator selects lambda individuals from a pool of mu.

    References:
    -----------
    .. [1] Back, Fogel and Michalewicz, Evolutionary Computation 1 :
           Basic Algorithms and Operators, 2000.
    .. [2] Collette, Y., N. Hansen, G. Pujol, D. Salazar Aponte and R. Le Riche (2010).
           On Object-Oriented Programming of Optimizers - Examples in Scilab.
           In P. Breitkopf and R. F. Coelho, eds.: Multidisciplinary Design Optimization
           in Computational Mechanics, Wiley, pp. 527-565;

    Examples:
    ---------
    The following example shows how to retrieve a feature mask for
    a synthetic data set.

    # >>> import numpy as np
    # >>> from sklearn.svm import SVC
    # >>> from sklearn.pipeline import Pipeline
    # >>> from sklearn.preprocessing import MinMaxScaler
    # >>> from sklearn.datasets import make_classification
    # >>> from FingersVsGestures.src.channel_elimination import EvolutionaryAlgorithms # TODO adjust
    # >>> X, y = make_classification(n_samples=100, n_features=8 * 4 * 100)
    # >>> X = X.reshape((100, 8, 4, 100))
    # >>> grid = np.arange(1, 33).reshape(X.shape[1:3])
    # >>> estimator = Pipeline([('scaler', MinMaxScaler()), ('svc', SVC())])

    # >>> ea = EvolutionaryAlgorithms(grid, estimator)
    # >>> ea.fit(X, y)
    # >>> print(ea.mask_)
    array([[False  True False False], [False False False False], [ True  True False False], [False False False  True],
           [False False False False], [False False False False], [False False  True False], [False False False False]])
    # >>> print(ea.score_)
    26.545670995670996

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

            # Genetic Algorithm Settings
            population_size: int = 120,
            n_gen: int = 100,
            islands: int = 1,
            method: str = 'simple',
            crossover: str = 'two_point',
            mutate: str = 'flip',
            selection: str = 'tournament',
            mu: int = 30,
            lmbda: int = 60,

            # Crossover, Mutation, Selection adn Migration Parameters
            migration_chance: float = 0.1,
            migration_size: int = 5,
            cxpb: float = 0.5,
            mutpb: float = 0.2,
            cx_indpb: float = 0.05,
            cx_alpha: float = 0.3,
            cx_eta: float = 5,
            mut_sigma: float = 0.1,
            mut_mu: float = 0,
            mut_indpb: float = 0.2,
            mut_c: float = 0.01,
            mut_eta: float = 40,
            sel_tournsize: int = 3,
            sel_nd: str = 'standard',

            # Training Settings
            tol: float = 1e-5,
            patience: int = 1e5,
            bounds: Tuple[float, float] = (0.0, 1.0),
            prior: Optional[numpy.ndarray] = None,
            callback: Optional[Union[Callable, Type]] = None,

            # Misc
            n_jobs: int = 1,
            seed: Optional[int] = None,
            verbose: Union[bool, int] = False
    ) -> None:

        super().__init__(
            dims, estimator, estimator_params, metric, cv, groups, tol,
            patience, bounds, prior, callback, n_jobs, seed, verbose
        )

        # Genetic Algorithm Settings
        self.population_size = population_size
        self.n_gen = n_gen
        self.islands = islands
        self.method = method
        self.crossover = crossover
        self.mutate = mutate
        self.selection = selection

        # Crossover, Mutation, Selection adn Migration Parameters
        self.migration_chance = migration_chance
        self.migration_size = migration_size
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.mu = mu
        self.lmbda = lmbda
        self.cx_indpb = cx_indpb
        self.cx_alpha = cx_alpha
        self.cx_eta = cx_eta
        self.mut_sigma = mut_sigma
        self.mut_mu = mut_mu
        self.mut_indpb = mut_indpb
        self.mut_c = mut_c
        self.mut_eta = mut_eta
        self.sel_tournsize = sel_tournsize
        self.sel_nd = sel_nd

        # Training Settings
        self.tol = tol
        self.patience = patience

    def _run(
            self
    ) -> Tuple[numpy.ndarray, numpy.ndarray, float]:
        """
        Executes the evolutionary optimization process,
        managing both single and multi-island configurations.

        Returns:
        --------
        :return Tuple[np.ndarray, numpy.ndarray, float]:
            The best found solution, mask, and their fitness score.
        """

        method, method_params = self._init_method()
        toolbox = self._init_toolbox()

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        hof = tools.HallOfFame(1)
        best_score, wait = 0.0, 0

        populations = self._initialize_population(toolbox)
        progress_bar = tqdm(range(self.n_gen), desc=self.__class__.__name__, post_fix=f'{best_score:.6f}',
                            disable=not self.verbose, leave=True)

        # Evolution process with or without migration
        for _ in progress_bar:
            for i, island in enumerate(populations):
                # Optimize Feature Selection
                populations[i], _ = method(populations[i], toolbox, ngen=1, stats=stats, halloffame=hof, verbose=False,
                                           **method_params)

            # Perform migration if applicable
            if self.islands > 1 and self.migration_chance >= random.random():
                populations = self._migrate(populations, self.migration_size, topology='ring')

            progress_bar.set_postfix(best_score=f'{hof.items[0].fitness.values[0]:.6f}')

            # Early stopping criteria
            stop, wait = self._early_stopping(hof, best_score, wait)
            if stop:
                break

        best_solution = np.array(hof.items[0]).reshape(-1)
        best_state = self._prepare_mask(np.array(hof.items[0]).reshape(self.dim_size_) > 0.5)
        best_score = hof.items[0].fitness.values[0] * 100
        return best_solution, best_state, best_score

    def _initialize_population(
            self, toolbox: base.Toolbox
    ) -> List[List[Any]]:
        """
        Initialize populations for each island with prior individuals.

        Parameters:
        -----------
        :param toolbox: base.Toolbox
            The toolbox object that describes the specified
            evolutionary process.


        Returns:
        --------
        :return: List[List[Any]]
            A population of several individuals.
        """
        populations = [toolbox.population(n=self.population_size - len(self.prior_))
                       for _ in range(self.islands)]
        for pop in populations:
            pop.extend(self.prior_)
        return populations

    def _early_stopping(
            self, hof: tools.HallOfFame, current_score: float, wait: int
    ) -> Tuple[bool, int]:
        """
        Determines if early stopping should occur based
        on the performance of the Hall of Fame individual.

        Parameters:
        -----------
        :param hof: tools.HallOfFame
            The Hall of Fame to keep track of the best individuals.
        :param gen: int
            The current generation number.
        :param current_score : float
            The current best score.
        :param wait: int
            The current count of generations without improvement.

        Returns:
        --------
        :return: Tuple[bool, int]
            A bool indicating whether early stopping should occur (True/False)
            and the updated wait count.
        """
        best_score = hof.items[0].fitness.values[0]
        stop = best_score < current_score and abs(best_score - current_score) > self.tol
        wait = 0 if stop else wait + 1
        return wait > self.patience or current_score >= 1, wait

    def _migrate(
            self, islands: List[List[Any]], k: int, topology: str = 'ring'
    ) -> List[List[Any]]:
        """
        Migrates individuals among islands based on a specified
        topology.

        Parameters:
        -----------
            :param islands: List[List[Any]]
                A list of populations (islands).
            :param k: int
                The number of individuals to migrate from each island.
            :param topology: str, default = 'ring'
                The structure of the migration network.

        Returns:
        --------
        return: List[List[Any]]
            A list of populations (islands).
        """
        if topology == 'ring':
            # Select k individuals from island i to migrate to island (i+1) % num_islands
            for i in range(self.islands):
                emigrants = tools.selBest(islands[i], self.migration_size)
                islands[(i + 1) % self.islands].extend(emigrants)
                islands[i] = [ind for ind in islands[i] if ind not in emigrants]
        return islands

    def _handle_bounds(
            self
    ) -> Tuple[float, float]:
        """
        Returns the bounds for the EA optimizer. If bounds are not set, default bounds
        of [0, 1] for each dimension are used.

        Returns:
        --------
        :return: Tuple[float, float]
            A tuple of two numpy arrays representing the lower and upper bounds.
        """
        return self.bounds

    def _handle_prior(
            self
    ) -> Optional[List[Any]]:
        """
        Generates a list of prior individuals based on a provided
        prior mask. The function checks the validity of the prior
        mask's shape, applies a Gaussian perturbation to the mask
        values, and creates DEAP individuals.

        Returns:
        --------
        :returns: Optional[List[Any]]:
            A list of DEAP Individual objects generated from the
            prior mask. If no prior is provided, an empty list is returned.
        """
        prior = []
        if self.prior:
            if self.prior.shape != self.dim_size_:  # self.grid.reshape(-1).shape:
                raise RuntimeError(
                    f'The argument prior must match the size of the dimensions to be considered.'
                    f'Got {self.prior.shape} but expected {self.dim_size_}.')
            gaus = abs(np.random.normal(loc=0, scale=0.06125, size=self.prior.size))
            prior = creator.Individual(
                [0.49 - gaus[i] if x < 0.5 else 0.51 + gaus[i]
                 for i, x in enumerate(([self.prior.astype(float)] * int(self.population_size * 0.2)))],
                self.toolbox_.attr_bool, n=1
            )
        return prior

    def _init_method(
            self
    ) -> Tuple[Callable, Dict[str, Any]]:
        """
        Returns the evolutionary algorithm method and its parameters.

        Returns:
        --------
        :return: Tuple[Callable, Dict[str, Any]
            A tuple containing the method function and a dictionary
            of its parameters.
        """

        return {
            'simple': (
                algorithms.eaSimple,
                {'cxpb': self.cxpb, 'mutpb': self.mutpb}),
            'mu_plus_lambda': (
                algorithms.eaMuPlusLambda,
                {'cxpb': self.cxpb, 'mutpb': self.mutpb,
                 'mu': self.mu, 'lambda_': self.lmbda}),
            'mu_lambda': (
                algorithms.eaMuCommaLambda,
                {'cxpb': self.cxpb, 'mutpb': self.mutpb,
                 'mu': self.mu, 'lambda_': self.lmbda}),
            'generate_update': (
                algorithms.eaGenerateUpdate,
                {})
        }[self.method]

    def _init_crossover(
            self,
    ) -> Tuple[Callable, Dict[str, Any]]:
        """
        Returns the crossover function and its parameters.

        Returns:
        --------
        :return: Tuple[Callable, Dict[str, Any]
            A tuple containing the crossover function and a dictionary
            of its parameters.
        """

        return {
            'one_point': (tools.cxOnePoint, {}),
            'two_point': (tools.cxTwoPoint, {}),
            'uniform': (tools.cxUniform, {'indpb': self.cx_indpb}),
            'part_matched': (tools.cxPartialyMatched, {}),
            'uni_part_matched': (tools.cxUniformPartialyMatched, {}),
            'ordered': (tools.cxOrdered, {}),
            'blend': (tools.cxBlend, {'alpha': self.cx_alpha}),
            'es_two_point': (tools.cxESTwoPoint, {'alpha': self.cx_alpha}),
            'sim_binary': (tools.cxSimulatedBinary, {'eta': self.cx_eta}),
            'messy_one_point': (tools.cxMessyOnePoint, {})
        }[self.crossover]

    def _init_mutation(
            self,
    ) -> Tuple[Callable, Dict[str, Any]]:
        """
        Returns the mutation function and its parameters.

        Returns:
        --------
        :return: Tuple[Callable, Dict[str, Any]
            A tuple containing the mutation function and a dictionary
            of its parameters.
        """
        return {
            'gaussian': (tools.mutGaussian,
                         {'mu': self.mut_mu, 'sigma': self.mut_sigma, 'indpb': self.mut_indpb}),
            'shuffle': (tools.mutShuffleIndexes, {'indpb': self.mut_indpb}),
            'flip': (tools.mutFlipBit, {'indpb': self.mut_indpb}),
            'es_log_normal': (tools.mutESLogNormal, {'c': self.mut_c, 'indpb': self.mut_indpb})
        }[self.mutate]

    def _init_selection(
            self,
    ) -> Tuple[Callable, Dict[str, Any]]:
        """
        Returns the selection function and its parameters.

        Returns:
        --------
        :return: Tuple[Callable, Dict[str, Any]
            A tuple containing the selection function and a dictionary
            of its parameters.
        """
        return {
            'tournament': (tools.selTournament, {'tournsize': self.sel_tournsize}),
            'roulette': (tools.selRoulette, {}),
            'nsga2': (tools.selNSGA2, {'nd': self.sel_nd}),
            'spea2': (tools.selSPEA2, {}),
            'best': (tools.selBest, {}),
            'tournament_dcd': (tools.selTournamentDCD, {'nd': self.sel_nd}),
            'stochastic_uni': (tools.selStochasticUniversalSampling, {}),
            'lexicase': (tools.selLexicase, {}),
            'epsilon_lexicase': (tools.selEpsilonLexicase, {}),
            'auto_epsilon_lexicase': (tools.selAutomaticEpsilonLexicase, {})
        }[self.selection]

    def _init_toolbox(self) -> base.Toolbox:
        """
        Initializes the DEAP toolbox with fitness, individual,
        and population registration, as well as the genetic operators:
        crossover, mutation, and selection.

        Returns:
        --------
        :return: base.Toolbox
            The toolbox object that describes the specified
            evolutionary process.
        """
        # Step 1: Create fitness and individual types
        creator.create("FitnessMax", base.Fitness, weights=[1.0])
        creator.create("Individual", list, fitness=creator.FitnessMax)

        # Step 2: Initialize the toolbox
        toolbox = base.Toolbox()

        # Step 2: Register the attribute, individual, and population creation functions
        toolbox.register("attr_bool", np.random.uniform, self.bounds_[0], self.bounds_[1])
        toolbox.register("individual", tools.initRepeat, creator.Individual,
                         toolbox.attr_bool, n=np.prod(self.dim_size_))
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Step 3: Register the evaluation, crossover, mutation, and selection functions
        toolbox.register("evaluate", lambda x: [self._objective_function(x)])
        crossover, crossover_params = self._init_crossover()
        toolbox.register("mate", crossover, **crossover_params)
        mutation, mutation_params = self._init_mutation()
        toolbox.register("mutate", mutation, **mutation_params)
        selection, selection_params = self._init_selection()
        toolbox.register("select", selection, **selection_params)

        def clip_individual(individual):
            """Clip the individual's values within the specified bounds."""
            array = np.array(individual)
            np.clip(array, self.bounds_[0], self.bounds_[1], out=array)
            individual[:] = array.tolist()
            return individual

        # def clip_individual(individual):
        #     """Clip the individual's values within the specified bounds."""
        #     np.clip(individual, self.bounds_[0], self.bounds_[1], out=individual)
        #     return individual

        # Decorate the mate and mutate functions with clipping
        toolbox.decorate("mate", lambda func: lambda ind1, ind2, **kwargs: (
            clip_individual(func(ind1, ind2, **kwargs)[0]), clip_individual(func(ind1, ind2, **kwargs)[1])
        ))
        toolbox.decorate("mutate", lambda func: lambda ind, **kwargs: (
            clip_individual(func(ind, **kwargs)[0]),
        ))

        return toolbox
