# -------------------------------------------------------------
# HandDecoding
# Copyright (c) 2023
#       Dirk Keller,
#       Nick Ramsey's Lab, University Medical Center Utrecht, University Utrecht
# Licensed under the MIT License [see LICENSE for detail]
# -------------------------------------------------------------
import math
import os
import warnings
from collections import OrderedDict
from typing import Dict, Any, Union, List, Tuple

import numpy
import ray
from cloudpickle import cloudpickle
from ray import tune, train
from ray.tune.schedulers import AsyncHyperBandScheduler, MedianStoppingRule
from ray.tune.search.ax import AxSearch
from ray.tune.search.hebo import HEBOSearch
from ray.tune.search.skopt import SkOptSearch
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

os.environ["RAY_AIR_NEW_OUTPUT"] = "1"

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


# TODO fix HEBO
# TODO include weights and biases for logging
# TODO handle output /mask -> hall_of_fame


def ray_callback(iteration, n_individuals, context):
    # Retrieve the batch
    context_data = ray.get(context.get.remote())
    start_idx = iteration * n_individuals
    end_idx = start_idx + n_individuals
    batch = context_data[start_idx:end_idx]

    # Calculate the average in a single pass using a generator
    av = sum(individual["Mean"][0] for individual in batch) / len(batch)

    # Prepare and report in one loop
    for individual in batch:
        transformed_individual = {
            key.lower().replace(" ", "_"): value[0] for key, value in individual.items()
        }

        # Build the report
        report = {
            **transformed_individual,
            transformed_individual["metric"]: transformed_individual["mean"],
            "batch_metric": av,
            "is_bad": not math.isfinite(transformed_individual["mean"]),
            "should_checkpoint": False,
        }

        # Report via ray
        ray.train.report(report)


def train_optimizer(
    config: Dict[str, Any],
    estimator: Any,
    data: Tuple[numpy.ndarray, numpy.ndarray],
    warm_restarts: bool,
    temp_path: str,
):
    X, y = data
    # hall_of_fame = ray.get(object_store_ref["hall_of_fame"])

    estimator_params = {param: config[param] for param in config}
    file_name = "_".join(f"{k}-{v}" for k, v in estimator_params.items()).rstrip("-")

    estimator_params["callback"] = ray_callback

    # hall_of_fame = HallOfFame(size=1)

    # # if warm_restarts and hall_of_fame:
    # if train.get_checkpoint():
    #     with train.get_checkpoint().as_directory() as checkpoint_dir:
    #         with open(os.path.join(checkpoint_dir, "hall_of_fame.ckpt"), "rb") as fp:
    #             hall_of_fame = cloudpickle.load(fp)
    #             best_mask = hall_of_fame[max(hall_of_fame.keys())]
    #
    #     if info['warm_restarts']:
    #         estimator_params["prior"] = best_mask

    estimator = estimator.set_params(**estimator_params)
    estimator.fit(X, y)

    file_path = os.path.abspath(os.path.join(temp_path, f"{file_name}.ckpt"))
    with open(file_path, "wb") as fp:
        cloudpickle.dump(
            estimator.hall_of_fame_.hall_of_fame.items(),
            fp,
        )


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
        prior_dist: List[Dict[str, Any]],
        max_iter: int,
        batch_size: int,
        out_path: str,
        exp_name: str,
        num_samples: int = 25,
        metric: str = "f1_weighted",
        max_concurrent: int = 1,
        search_scheduler: str = "AsyncHyperBand",
        search_optimizer: str = "HEBO",
        warm_restarts: bool = False,
        n_jobs: int = -1,
    ) -> None:
        self.estimator = estimator
        self.param_dist = param_dist
        self.prior_dist = prior_dist
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.out_path = out_path

        self.max_concurrent = max_concurrent
        self.num_samples = num_samples
        self.metric = metric

        self.search_scheduler = search_scheduler
        self.search_optimizer = search_optimizer
        self.warm_restarts = warm_restarts
        self.n_jobs = n_jobs

        self.exp_name = exp_name
        self.name = (
            estimator[-1].__class__.__name__
            if isinstance(estimator, Pipeline)
            else estimator.__class__.__name__
        )

    def optimize(self, X: Any, y: Any) -> None:
        """
        Perform hyperparameter tuning with cross-validation.

        Args:
            X (Any): The input features.
            y (Any): The target labels.
        """
        y = y.astype("int32")

        ray.init(num_cpus=self.n_jobs)

        out_path_model = os.path.abspath(os.path.join(self.out_path, self.name))
        os.makedirs(out_path_model, exist_ok=True)

        temp_path = os.path.abspath(os.path.join(out_path_model, "hall_of_fame"))
        dir_results = os.path.abspath(os.path.join(out_path_model, "ray_results"))
        os.makedirs(temp_path, exist_ok=True)
        os.makedirs(dir_results, exist_ok=True)

        print(
            f"\nStart hyperparameter search of {self.num_samples} candidates for {self.name} "
            f"\n-----------------------------------------------------------"
        )

        cpu_per_sample = self.n_jobs / self.max_concurrent

        print(f"\nInitialize Search Optimization Schedule...")

        opt = self.build_search_alg()
        sched = self.select_sched_alg()

        analysis = tune.run(
            tune.with_parameters(
                train_optimizer,
                estimator=self.estimator,
                data=(X, y),
                warm_restarts=self.warm_restarts,
                temp_path=temp_path,
            ),
            name=self.exp_name,
            scheduler=sched,
            search_alg=opt,
            stop={
                "training_iteration": self.max_iter * self.batch_size,
                "is_bad": True,
            },
            config=self.param_dist,
            resources_per_trial=tune.PlacementGroupFactory(
                [
                    {"CPU": 0.1 * cpu_per_sample},
                    {"CPU": 0.9 * cpu_per_sample},
                    # {"CPU": (0.9 * cpu_per_sample) ** 0.5},
                ]
                * self.max_concurrent
            ),
            max_concurrent_trials=self.max_concurrent,  # TODO adjust
            # {"cpu": cpu_per_sample},
            num_samples=self.num_samples,
            # callbacks=[
            #     WandbLoggerCallback(
            #         project=self.name,
            #         tags=f"{self.__class__.__name__}_{datetime.now().strftime('%Y-%m_%d_%H:%M')}",
            #     )
            # ],
            checkpoint_at_end=False,
            local_dir=dir_results,
            trial_name_creator=get_trial_name,
            checkpoint_score_attr=self.metric,
            verbose=1,
        )
        self.result_grid_ = analysis.dataframe()
        self.hall_of_fame_ = self.load_hall_of_fame(temp_path)

        self.mask_ = self.hall_of_fame_[max(self.hall_of_fame_.keys())]

        ray.shutdown()

    def build_search_alg(self) -> Union[SkOptSearch, HEBOSearch, AxSearch]:
        """
        Initialize a search algorithm based on the selected optimizer.

        Returns:
            tune.suggest.SearchAlgorithm: The initialized search algorithm.
        """
        if self.search_optimizer == "BO":
            return SkOptSearch(
                metric=self.metric, points_to_evaluate=self.prior_dist, mode="max"
            )
        elif self.search_optimizer == "HEBO":
            return HEBOSearch(
                metric=self.metric, points_to_evaluate=self.prior_dist, mode="max"
            )
        elif self.search_optimizer == "AX":
            return AxSearch(
                metric=self.metric, points_to_evaluate=self.prior_dist, mode="max"
            )
        else:
            raise ValueError("Unknown search optimizer. Select 'BO', 'HEBO', or 'AX'.")

    def select_sched_alg(self) -> tune.schedulers.TrialScheduler:
        """
        Initialize a scheduling algorithm based on the selected scheduler.

        Returns:
            tune.schedulers.TrialScheduler: The initialized scheduling algorithm.
        """
        if self.search_scheduler == "AsyncHyperBand":
            return AsyncHyperBandScheduler(
                time_attr="training_iteration",
                metric="batch_metric",  # self.metric,
                mode="max",
                max_t=self.max_iter * self.batch_size,
                grace_period=math.ceil(0.05 * self.max_iter) * self.batch_size,
                reduction_factor=4,
                brackets=4,
            )
        elif self.search_scheduler == "MS":
            return MedianStoppingRule(
                time_attr="training_iteration",
                metric=self.metric,
                mode="max",
                grace_period=math.ceil(0.05 * self.max_iter) * self.batch_size,
                min_samples_required=3 * self.batch_size,
            )
        else:
            raise ValueError(
                'Unknown search scheduler. Select "MedianStop" or "AsyncHyperBand".'
            )

    def file_name(self, exp_name: str) -> str:
        """
        Generate a file name for the experiment results.

        Args:
            exp_name (str): The base experiment name.

        Returns:
            str: The full file name with additional details.
        """
        exp_name += f"_RF_{self.device}_CV-{self.num_cv}_M_SAMP-{self.num_samples}"
        exp_name += (
            f'_{"Random" if self.search_optimizer is None else self.search_optimizer}'
        )
        if self.search_scheduler:
            exp_name += f"_{self.search_scheduler}"
        return exp_name

    @staticmethod
    def load_hall_of_fame(temp_path):
        concatenated_dict = OrderedDict()

        # Iterate over all files in the temp_path directory
        for file_name in os.listdir(temp_path):
            if file_name.endswith(".ckpt"):  # Check if file is a model checkpoint file
                model_path = os.path.join(temp_path, file_name)

                try:
                    # Load the model
                    with open(model_path, "rb") as fp:
                        model_dict = cloudpickle.load(fp)

                        # Check if the model is an OrderedDict, and concatenate
                        if isinstance(model_dict, OrderedDict):
                            concatenated_dict.update(model_dict)
                        else:
                            print(
                                f"Warning: {file_name} is not an OrderedDict, skipping."
                            )
                except Exception as e:
                    print(f"Error loading {file_name}: {e}")

        return concatenated_dict
