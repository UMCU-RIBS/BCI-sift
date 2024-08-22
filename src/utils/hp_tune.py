# -------------------------------------------------------------
# HandDecoding
# Copyright (c) 2023
#       Dirk Keller,
#       Nick Ramsey's Lab, University Medical Center Utrecht, University Utrecht
# Licensed under the MIT License [see LICENSE for detail]
# -------------------------------------------------------------

import os
import time
import multiprocessing
import math
from typing import Dict, Any, Union, Tuple

import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler, MedianStoppingRule
from ray.tune.search.hebo import HEBOSearch
from ray.tune.search.skopt import SkOptSearch
from ray.tune.search.ax import AxSearch

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import get_scorer

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

os.environ['RAY_AIR_NEW_OUTPUT'] = '1'

trial_num = 0


def get_trial_name(trial: ray.tune.experiment.trial.Trial) -> str:
    """
    Generates a unique trial name based on the trial number.

    Args:
        trial (ray.tune.experiment.trial.Trial): The trial object.

    Returns:
        str: The unique trial name.
    """
    global trial_num
    trial_num += 1
    trial_name = f"{trial.trainable_name}_{trial_num}"
    return trial_name


class PerfTimer:
    """
    High-resolution timer for measuring execution duration.
    """

    def __init__(self) -> None:
        self.start: float = 0.0
        self.duration: float = 0.0

    def __enter__(self) -> 'PerfTimer':
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self.duration = time.perf_counter() - self.start


class TrainTransformer(tune.Trainable):
    """
    A Ray Tune Trainable class that handles training and evaluation of a model pipeline.

    Attributes:
        estimator (BaseEstimator): The estimator to be trained.
        X (Any): The input features.
        y (Any): The target labels.
        metric (str): The metric used for evaluation.
        scorer (Callable): The scorer function based on the metric.
        cv_indices (Iterator): The cross-validation indices.
        mean (float): The running mean of the performance metric.
        iter (int): The current iteration count.
        _estimator_params (Dict[str, Any]): Parameters for the estimator.
        _global_best_model (BaseEstimator): The best model found during training.
        _global_best_score (float): The best score achieved.
    """

    def setup(self, config: Dict[str, Any], object_store_ref: Dict[str, ray.ObjectRef], fold_generator: StratifiedKFold,
              metric: str) -> None:
        self.estimator: BaseEstimator = ray.get(object_store_ref['estimator'])
        self.X, self.y = ray.get(object_store_ref['data'])

        self.metric = metric
        self.scorer = get_scorer(self.metric)
        self.cv_indices = fold_generator.split(self.X, self.y)

        self.mean = 0.0
        self.iter = 1

        self._build(config)

    def _build(self, config: Dict[str, Any]) -> None:
        self._estimator_params = {param: config[param] for param in config}
        self._global_best_model = None
        self._global_best_score = 0.0

    def step(self) -> Dict[str, Union[float, bool]]:
        self.estimator.set_params(**self._estimator_params)

        with PerfTimer() as train_timer:
            trained_model = self.estimator.fit(self.X, self.y)
        training_time = train_timer.duration

        with PerfTimer() as inference_timer:
            test_score = self.scorer(trained_model, self.X, self.y)
        infer_time = inference_timer.duration

        self.mean += (test_score - self.mean) / self.iter
        self.iter += 1

        if self.mean > self._global_best_score:
            self._global_best_score = test_score
            self._global_best_model = trained_model

        return {
            str(self.metric): self.mean,
            "train_time": round(training_time, 4),
            "infer_time": round(infer_time, 4),
            "is_bad": not math.isfinite(self.mean),
            "should_checkpoint": False,
        }

    def _save(self, checkpoint: Dict[str, Any]) -> None:
        self._global_best_score = checkpoint[self.metric]

    def _restore(self, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        return {"test_score": self._global_best_score}

    def save_checkpoint(self, checkpoint_dir: str) -> None:
        pass

    def reset_config(self, new_config: Dict[str, Any]) -> bool:
        del self.estimator
        self._build(new_config)
        self.config = new_config
        return True


class DecoderOptimization:
    """
    Class for managing the hyperparameter optimization process using Ray Tune.

    Attributes:
        estimator (Union[BaseEstimator, Pipeline]): The estimator or pipeline to be optimized.
        param_dist (Dict[str, Any]): The hyperparameter distribution.
        out_path (str): The output path for saving results.
        exp_name (str): The experiment name.
        num_samples (int): The number of samples to run.
        metric (str): The evaluation metric.
        fold_generator (Any): The cross-validation fold generator.
        max_concurrent (int): The maximum number of concurrent trials.
        search_scheduler (str): The search scheduler to use.
        search_optimizer (str): The search optimizer to use.
        num_cv (int): The number of cross-validation splits.
        diagnostics (Any): Diagnostics information from the best trial.
    """

    def __init__(
            self,
            estimator: Union[BaseEstimator, Pipeline],
            param_dist: Dict[str, Any],
            out_path: str,
            exp_name: str,
            num_samples: int = 10,
            metric: str = 'f1_weighted',
            fold_generator: Any = StratifiedKFold(n_splits=10),
            max_concurrent: int = 10,
            search_scheduler: str = 'AsyncHyperBand',
            search_optimizer: str = 'HEBO',
            device = 'cpu',
    ) -> None:
        self.estimator = estimator
        self.param_dist = param_dist
        self.out_path = out_path

        self.fold_generator = fold_generator
        self.max_concurrent = max_concurrent
        self.num_samples = num_samples
        self.metric = metric

        self.search_scheduler = search_scheduler
        self.search_optimizer = search_optimizer
        self.device = device

        self.exp_name = exp_name
        self.name = estimator[-1].__class__.__name__ if isinstance(estimator,
                                                                   Pipeline) else estimator.__class__.__name__
        self.num_cv = fold_generator.n_splits

        self.diagnostics = None

    def optimize(self, X: Any, y: Any) -> None:
        """
        Perform hyperparameter tuning with cross-validation.

        Args:
            X (Any): The input features.
            y (Any): The target labels.
        """
        y = y.astype("int32")

        ray.init()

        refs = {'estimator': ray.put(self.estimator), 'data': ray.put(tuple([X, y]))}

        out_path_model = os.path.join(self.out_path, self.name)
        os.makedirs(out_path_model, exist_ok=True)

        dir_results = os.path.abspath(os.path.join(out_path_model, 'ray_results'))
        os.makedirs(dir_results, exist_ok=True)

        print(f'\nStart hyperparameter search of {self.num_samples} candidates for {self.name} '
              f'\n-----------------------------------------------------------')

        cpu_per_sample = int(multiprocessing.cpu_count() / self.max_concurrent)

        print(f'\nInitialize Search Optimization Schedule...')
        opt = self.build_search_alg()
        sched = self.select_sched_alg()

        analysis = tune.run(
            tune.with_parameters(
                TrainTransformer,
                object_store_ref=refs,
                fold_generator=self.fold_generator,
                metric=self.metric
            ),
            name=self.exp_name,
            scheduler=sched,
            search_alg=opt,
            stop={"training_iteration": self.num_cv, "is_bad": True},
            config=self.param_dist,
            resources_per_trial={"cpu": cpu_per_sample},
            num_samples=self.num_samples,
            checkpoint_at_end=True,
            local_dir=dir_results,
            trial_name_creator=get_trial_name,
            checkpoint_score_attr=self.metric,
            verbose=1,
        )

        best_trial = analysis.get_best_trial(metric=self.metric, mode="max", scope="all")
        self.diagnostics = best_trial.last_result["result_grid_"]

        ray.shutdown()

    def build_search_alg(self) -> tune.search.SearchAlgorithm:
        """
        Initialize a search algorithm based on the selected optimizer.

        Returns:
            tune.suggest.SearchAlgorithm: The initialized search algorithm.
        """
        if self.search_optimizer == "BO":
            return SkOptSearch(metric=self.metric, mode='max')
        elif self.search_optimizer == "HEBO":
            return HEBOSearch(metric=self.metric, mode='max')
        elif self.search_optimizer == "AX":
            return AxSearch(metric=self.metric, mode='max')
        else:
            raise ValueError("Unknown search optimizer. Select 'BO', 'HEBO', or 'AX'.")

    def select_sched_alg(self) -> tune.schedulers.TrialScheduler:
        """
        Initialize a scheduling algorithm based on the selected scheduler.

        Returns:
            tune.schedulers.TrialScheduler: The initialized scheduling algorithm.
        """
        if self.search_scheduler == 'AsyncHyperBand':
            return AsyncHyperBandScheduler(
                time_attr='training_iteration',
                metric=self.metric,
                mode='max',
                max_t=self.num_cv,
                grace_period=1,
                reduction_factor=3,
                brackets=3,
            )
        elif self.search_scheduler == 'MedianStop':
            return MedianStoppingRule(
                time_attr='training_iteration',
                metric=self.metric,
                mode='max',
                grace_period=1,
                min_samples_required=3,
            )
        else:
            raise ValueError('Unknown search scheduler. Select "MedianStop" or "AsyncHyperBand".')

    def file_name(self, exp_name: str) -> str:
        """
        Generate a file name for the experiment results.

        Args:
            exp_name (str): The base experiment name.

        Returns:
            str: The full file name with additional details.
        """
        exp_name += f'_RF_{self.device}_CV-{self.num_cv}_M_SAMP-{self.num_samples}'
        exp_name += f'_{"Random" if self.search_optimizer is None else self.search_optimizer}'
        if self.search_scheduler:
            exp_name += f'_{self.search_scheduler}'
        return exp_name
