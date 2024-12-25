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
from typing import Dict, Any, Union, List

import ray
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
        #warm_restarts: bool = False,
        n_jobs: int = -1,
    ) -> None:
        self.info = {key: value for key, value in locals().items() if key != "self"}

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
        #self.warm_restarts = warm_restarts
        self.n_jobs = n_jobs

        self.exp_name = exp_name
        self.name = (
            estimator[-1].__class__.__name__
            if isinstance(estimator, Pipeline)
            else estimator.__class__.__name__
        )

        # filename=f"/{estimator.__class__.__name__}_hall_of_fame.pkl",

    @staticmethod
    def train(
        config: Dict[str, Any],
        info: Dict[str, Any],
        # warm_restarts: bool,
        object_store_ref: Dict[str, ray.ObjectRef],
    ):
        # warm_restarts = info["warm_restarts"]

        fs_transformer = ray.get(object_store_ref["estimator"])
        X, y = ray.get(object_store_ref["data"])
        # hall_of_fame = ray.get(object_store_ref["hall_of_fame"])

        estimator_params = {param: config[param] for param in config}
        estimator_params["callback"] = ray_callback

        #hall_of_fame = HallOfFame(size=1)

        # # if warm_restarts and hall_of_fame:
        # if train.get_checkpoint():
        #     with train.get_checkpoint().as_directory() as checkpoint_dir:
        #         with open(os.path.join(checkpoint_dir, "hall_of_fame.ckpt"), "rb") as fp:
        #             hall_of_fame = cloudpickle.load(fp)
        #             best_mask = hall_of_fame[max(hall_of_fame.keys())]
        #
        #     if info['warm_restarts']:
        #         estimator_params["prior"] = best_mask

        fs_transformer = fs_transformer.set_params(**estimator_params)
        fs_transformer.fit(X, y)


        # results = fs_transformer.result_grid_
        # max_score_index = results["Score"].idxmax()
        # hall_of_fame[results.loc[max_score_index, "Score"]] = results.loc[
        #     max_score_index, "Mask"
        # ]
        # hall_of_fame[numpy.max(fs_transformer.score_)] = fs_transformer.mask_

    def optimize(self, X: Any, y: Any) -> None:
        """
        Perform hyperparameter tuning with cross-validation.

        Args:
            X (Any): The input features.
            y (Any): The target labels.
        """
        y = y.astype("int32")

        ray.init(num_cpus=self.n_jobs)

        refs = {
            "estimator": ray.put(self.estimator),
            "data": ray.put(tuple([X, y])),
        }

        out_path_model = os.path.join(self.out_path, self.name)
        os.makedirs(out_path_model, exist_ok=True)

        dir_results = os.path.abspath(os.path.join(out_path_model, "ray_results"))
        os.makedirs(dir_results, exist_ok=True)

        print(
            f"\nStart hyperparameter search of {self.num_samples} candidates for {self.name} "
            f"\n-----------------------------------------------------------"
        )

        cpu_per_sample = (self.n_jobs) / self.max_concurrent

        print(f"\nInitialize Search Optimization Schedule...")

        opt = self.build_search_alg()
        sched = self.select_sched_alg()

        analysis = tune.run(
            tune.with_parameters(
                self.train,
                info=self.info,
                # warm_restarts=self.warm_restarts,
                object_store_ref=refs,
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
                [{"CPU": 0.1 * cpu_per_sample}]
                + [{"CPU": 0.9 * cpu_per_sample}] * self.max_concurrent
            ),
            max_concurrent_trials=self.max_concurrent,  # TODO adjust
            # {"cpu": cpu_per_sample},
            num_samples=self.num_samples,
            checkpoint_at_end=False,
            local_dir=dir_results,
            trial_name_creator=get_trial_name,
            checkpoint_score_attr=self.metric,
            verbose=1,
        )

        # self.hall_of_fame_ = ray.get(hall_of_fame).get.remote()
        # print(self.hall_of_fame_)
        # self.score_ = max(self.hall_of_fame_.keys())
        # self.mask_ = self.hall_of_fame_[self.score_]

        self.result_grid_ = analysis.dataframe()

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
        elif self.search_scheduler == "MedianStop":
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
