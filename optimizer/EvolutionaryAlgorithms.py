# -------------------------------------------------------------
# Channel Elimination
# Copyright (c) 2024
#       Dirk Keller,
#       Nick Ramsey's Lab, University Medical Center Utrecht, University Utrecht
# Licensed under the MIT License [see LICENSE for detail]
# -------------------------------------------------------------


import random
from typing import Tuple, List, Union, Dict, Any, Optional, Type

import numpy
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms

from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted as sklearn_is_fitted

from ._Base_Optimizer import BaseOptimizer

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
    :param grid: numpy.ndarray
        Grid structure specifying how features (e.g., channels)
         are arranged.
    :param estimator: Union[Any, Pipeline]
        The machine learning model or pipeline to evaluate feature sets.
    :param metric: str, default = 'f1_weighted'
        The performance metric to optimize. Must be a
        valid scikit-learn scorer string.
    :param cv: Union[BaseCrossValidator, int], default = 10
        The cross-validation strategy or number of folds
        for cross-validation.
    :param groups: numpy.ndarray, default = None
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
    :param migration_chance: float, default = 0.1
        The probability of migrating individuals among islands per generation.
    :param migration_size: int, default = 5
        The number of individuals to migrate between islands
        when migration occurs.
    :param cxpb: float, default = 0.5
        The probability of mating two individuals (crossover probability).
    :param mutpb: float, default = 0.2
        The probability of mutating an individual (mutation probability).
    :param mu: int, default = 30
        The number of individuals to select for the next generation
        in 'mu_plus_lambda' and 'mu_lambda' methods.
    :param lmbda: int, default = 60
        The number of children to produce in 'mu_plus_lambda'
        and 'mu_lambda' methods.
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
    :param tol: float, default = 1e-5
        The function tolerance; if the change in the best objective value
        is below this for `patientce` iterations, the optimization will stop early.
    :param patience: int, default = int(1e5)
        The number of iterations for which the objective function
        improvement must be below `tol` to stop optimization.
    :param prior: Optional[numpy.ndarray], default = None
        Explicitly initialize the optimizer state.
        If set to None if population characteristics are initialized randomly.
    :param n_jobs: int, default = 1
        The number of parallel jobs to run during cross-validation.
    :param seed: Optional[int], default = None
        Random seed for reproducibility of the evolutionary process.
    :param verbose: bool, default = False
        Enables detailed progress messages during the optimization process.

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

    Notes
    --------
    This implementation is semi-compatible with the scikit-learn framework,
    which builds around two-dimensional feature matrices. To use this
    transfortmation within a scikit-learn Pipeline, the four dimensional data
    must be flattened after the first dimension [samples, features]. For example,
    scikit-learn's FunctionTransformer can achieve this.

    Care must be taken when the lambda:mu ratio is 1 to 1 for the MuPlusLambda and
    MuCommaLambda algorithms as a non-stochastic selection will result in no selection
    at all, since the operator selects lambda individuals from a pool of mu.

    References
    --------
    .. [1] Back, Fogel and Michalewicz, Evolutionary Computation 1 :
           Basic Algorithms and Operators, 2000.
    .. [2] Collette, Y., N. Hansen, G. Pujol, D. Salazar Aponte and R. Le Riche (2010).
           On Object-Oriented Programming of Optimizers - Examples in Scilab.
           In P. Breitkopf and R. F. Coelho, eds.: Multidisciplinary Design Optimization
           in Computational Mechanics, Wiley, pp. 527-565;

    Examples
    --------
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
            grid: numpy.ndarray,
            estimator: Union[Any, Pipeline],
            metric: str = 'f1_weighted',
            cv: Union[BaseCrossValidator, int] = 10,
            groups: numpy.ndarray = None,

            # Genetic Algorithm Settings
            population_size: int = 120,
            n_gen: int = 100,
            islands: int = 1,
            method: str = 'simple',
            crossover: str = 'two_point',
            mutate: str = 'flip',
            selection: str = 'tournament',

            # Crossover, Mutation, Selection adn Migration Parameters
            migration_chance: float = 0.1,
            migration_size: int = 5,
            cxpb: float = 0.5,
            mutpb: float = 0.2,
            mu: int = 30,
            lmbda: int = 60,
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
            patience: int = int(1e5),
            prior: Optional[numpy.ndarray] = None,

            # Misc
            n_jobs: int = 1,
            seed: Optional[int] = None,
            verbose: bool = False
    ) -> None:

        super().__init__(grid, estimator, metric, cv, groups, n_jobs, seed, verbose)

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
        self.prior = prior

    def fit(
            self, X: numpy.ndarray, y: numpy.ndarray = None
    ) -> Type['EvolutionaryAlgorithms']:
        """
        Optimizes the channel combination with Evolutionary Algorithms.

        Parameters:
        -----------
        :param X: numpy.ndarray
            Array-like with dimensions [samples, channel_height, channel_width, time]
        :param y: numpy.ndarray, default = None
            Array-like with dimensions [targets].

        Return:
        -------
        :return: Type['EvolutionaryAlgorithms']
        """
        self.X_ = X
        self.y_ = y

        self.iter_ = int(0)
        self.result_grid_ = []

        # Set the seeds
        random.seed(self.seed)
        np.random.seed(self.seed)

        method_lib = {
            'simple': (algorithms.eaSimple, {'cxpb': self.cxpb, 'mutpb': self.mutpb}),
            'mu_plus_lambda': (algorithms.eaMuPlusLambda,
                               {'cxpb': self.cxpb, 'mutpb': self.mutpb,
                                'mu': self.mu, 'lambda_': self.lmbda}),
            'mu_lambda': (algorithms.eaMuCommaLambda,
                          {'cxpb': self.cxpb, 'mutpb': self.mutpb,
                           'mu': self.mu, 'lambda_': self.lmbda}),
            'generate_update': (algorithms.eaGenerateUpdate, {})
        }
        crossover_lib = {
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
        }
        mutation_lib = {
            'gaussian': (tools.mutGaussian,
                         {'mu': self.mut_mu, 'sigma': self.mut_sigma, 'indpb': self.mut_indpb}),
            'shuffle': (tools.mutShuffleIndexes, {'indpb': self.mut_indpb}),
            'flip': (tools.mutFlipBit, {'indpb': self.mut_indpb}),
            'es_log_normal': (tools.mutESLogNormal, {'c': self.mut_c, 'indpb': self.mut_indpb})
        }
        selection_lib = {
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
        }
        self.method_ = method_lib[self.method]

        crossover = crossover_lib[self.crossover]
        mutation = mutation_lib[self.mutate]
        selection = selection_lib[self.selection]

        creator.create("FitnessMax", base.Fitness, weights=[1.0])
        creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox_ = base.Toolbox()  # TODO can be parallelized from scoop import futures toolbox.register("map", futures.map)
        self.toolbox_.register("attr_bool", np.random.randint, 0, 2)
        self.toolbox_.register("individual", tools.initRepeat, creator.Individual, self.toolbox_.attr_bool,
                               n=self.grid.size)
        self.toolbox_.register("population", tools.initRepeat, list, self.toolbox_.individual)

        self.toolbox_.register("evaluate", self.objective_function_wrapper)
        self.toolbox_.register("mate", crossover[0], **crossover[1])  # tools.cxTwoPoint)
        self.toolbox_.register("mutate", mutation[0], **mutation[1])  # tools.mutFlipBit, indpb=0.05)
        self.toolbox_.register("select", selection[0], **selection[1])  # tools.selTournament, tournsize=3)

        self.prior_ = []
        if self.prior is not None:
            if self.prior.shape != self.grid.reshape(-1).shape:
                raise RuntimeError(
                    f'The argument prior {self.prior.shape} must match '
                    f'the number of cells of grid {self.grid.reshape(-1).shape}.')
            gaus = abs(np.random.normal(loc=0, scale=0.06125, size=self.prior.size))
            self.prior_ = creator.Individual(
                [0.49 - gaus[i] if x < 0.5 else 0.51 + gaus[i]
                 for i, x in enumerate(([self.prior.astype(float)] * int(self.population_size * 0.2)))]
            )

        if self.islands == 1:
            np.random.seed(self.seed)

        self.solution_, self.mask_, self.score_ = self.run()

        # Conclude the result grid
        self.result_grid_ = pd.concat(self.result_grid_, axis=0, ignore_index=True)
        return self

    def transform(self, X: numpy.ndarray, y: numpy.ndarray = None) -> numpy.ndarray:
        """
        Transforms the input with the mask obtained from the solution
        of the evolutionary algorithm.

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

    def run(
            self
    ) -> Tuple[numpy.ndarray, numpy.ndarray, float]:
        """
        Executes the evolutionary optimization process,
        managing both single and multi-island configurations.

        Returns:
        --------
        :return Tuple[numpy.ndarray, numpy.ndarray, float]
            The best found solution, mask and their fitness score.
        """
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        hof = tools.HallOfFame(1)
        best_score = 0
        wait = 0

        if self.islands == 1:

            population = self.toolbox_.population(n=self.population_size - len(self.prior_))
            population.extend(self.prior_)

            # Evolution process with single population
            for gen in range(self.n_gen):
                population, logbook = self.method_[0](population, self.toolbox_, ngen=1, stats=stats,
                                                      halloffame=hof, verbose=False, **self.method_[1])
                if self.verbose:
                    if gen > 0:
                        del logbook[0]
                        logbook[0]['gen'] = gen + 1
                        logbook.log_header = False
                    print(f'{logbook.stream}')
                if abs(best_score - hof.items[0].fitness.values[0]) > self.tol:
                    best_score = hof.items[0].fitness.values[0]
                    wait = 0
                else:
                    wait += 1
                if wait > self.patience or hof.items[0].fitness.values[0] >= 1:
                    print(f"Early stopping on generation {gen}")
                    break
            #
            # best_individual = hof.items[0]
        else:

            populations = [self.toolbox_.population(n=self.population_size - len(self.prior_)) for _ in
                           range(self.islands)]
            populations = [pop.extend(self.prior_) for pop in populations]

            # Evolution process with migration across multiple populations
            for gen in range(self.n_gen):
                # Evolve each population
                for i, island in enumerate(populations):
                    island, logbook = self.method_[0](island, self.toolbox_, ngen=1, stats=stats, halloffame=hof,
                                                      verbose=False, **self.method_[1])

                    if self.verbose:
                        logbook[0]['island'], logbook[1]['island'] = i, i
                        if gen > 0:
                            del logbook[0]
                            logbook[0]['gen'] = gen + 1
                        if 'island' not in logbook.header:
                            logbook.header.insert(2, 'island')
                        if i > 0 or gen > 0:
                            logbook.log_header = False
                        print(f'{logbook.stream}')

                    if abs(best_score - hof.items[0].fitness.values[0]) > self.tol:
                        best_score = hof.items[0].fitness.values[0]
                        wait = 0
                    else:
                        wait += 1
                    if wait > self.patience or hof.items[0].fitness.values[0] >= 1:
                        print(f"Early stopping on generation {gen}")
                        break

                    populations[i] = island

                # Perform migration at specified intervals
                if self.migration_chance >= random.random() and gen > 0:
                    populations = self.migrate(populations, self.migration_size, topology='ring')

                    if self.verbose:
                        print(f'\nMigration occurred at generation {gen + 1}!\n')

        best_solution = np.array(hof.items[0]).reshape(-1)
        best_state = np.array(hof.items[0]).reshape(self.grid.shape).astype(bool)
        best_score = hof.items[0].fitness.values[0] * 100
        return best_solution, best_state, best_score

    def objective_function_wrapper(
            self, mask: numpy.ndarray
    ) -> List[float]:
        """
        Wraps the objective function to adapt it for
        compatibility with the genetic algorithm framework.

        Parameters:
        -----------
        :params mask: numpy.ndarray
            A boolean array indicating which features are included
            in the subset.

        Returns:
        --------
        :return List[float]
            A list containing the objective function's score for
            the provided mask. Must have the same size as weights
            of the fitness function (e.g. size of one).
        """
        return [self.objective_function(mask)]

    def migrate(
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
            for i in range(self.islands):
                # Select k individuals from island i to migrate to island (i+1) % num_islands
                emigrants = tools.selBest(islands[i], k)
                islands[(i + 1) % self.islands].extend(emigrants)  # Send emigrants to next island

                # Remove emigrants from original population
                for emigrant in emigrants:
                    islands[i].remove(emigrant)
        return islands
