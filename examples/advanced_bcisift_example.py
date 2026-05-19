# -------------------------------------------------------------
# BCI-FeaST
# Copyright (c) 2025
#       Dirk Keller,
#       Elena Offenberg,
#       Nick Ramsey's Lab, University Medical Center Utrecht, University Utrecht
# Licensed under the MIT License [see LICENSE for detail]
# -------------------------------------------------------------

'''
This is an extensive example of using the BCI-sift toolbox, including loading the data, using nested cross-validation, running different optimization methods, 
and plotting and saving the results. The optimizers are run in a nested cross-validation scheme, where the inner loop is used for optimization and the outer loop 
is used for evaluating the performance of the optimized model on unseen data. The results are saved in a csv file for each subject, and include the best score, 
the best mask, and the number of selected channels for each method. Additionally, the importance and elimination plots are saved for each method.
How to run:
    python /home/BCI-sift/examples/optimize.py \
        -c config.yml
'''

from pathlib import Path
import argparse
import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import (
    train_test_split,
    cross_validate,
    StratifiedKFold
)
from sklearn import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.metrics import get_scorer

import sys
sys.path.insert(0, '.')
from optimizer import RecursiveFeatureElimination, RandomSearch, ParticleSwarmOptimization, SimulatedAnnealing, EvolutionaryAlgorithms
from optimizer.backend._backend import FlattenTransformer
from dataset.utils import load_config, load_by_id
from dataset.Dataset import Dataset
from dataset.Events import Events
from dataset.Epochs import Epochs
from utils.hp_tune import PerfTimer



def main(config_path):
    """
    Run all optimizers on a BCI dataset.
    """
    config = load_config(Path(config_path))
    params = config['optimization']
    task_name = config['task']
    metric = params['metric'] #performance metric to be used 
    optimization_methods = params['optimization_methods']
    dims = tuple(params['dims']) #dimensions to optimize over, for example here (1,) for timepoints only, (1,2) for timepoints and frequencies 
    seed = 120
    outer_cv = params['outer_cv']
    inner_cv = params['inner_cv']

    output_path = Path(config['save_path'] + task_name)
    output_path.mkdir(parents=True, exist_ok=True)
    #for each subject, save the feature masks, training results, and testing results in separate folders
    mask_output_path = output_path / "feature_masks"
    mask_output_path.mkdir(parents=True, exist_ok=True)
    training_output_path = output_path / "training_results"
    training_output_path.mkdir(parents=True, exist_ok=True)
    testing_output_path = output_path / "testing_results"
    testing_output_path.mkdir(parents=True, exist_ok=True)

    n_jobs = os.cpu_count() - 2 
    print(f"\nNumber of available cpu cores: {n_jobs}")

    for id in config['subject']:
        subject = load_by_id(id, data_path=config['data_path'])
        print(subject['name'])
        sr = config['sampling_rate']
        tmin = config['epochs']['tmin']
        tmax = config['epochs']['tmax']

        # Load the raw data and events for the subject, and create epochs -> can be replaced with loading pre-processed epochs if available
        raw = Dataset(id=subject['name'],
                      input_paths={k: v for k, v in subject['feature_paths'][sr].items() if k in config['features']},
                      channel_paths={k: v for k, v in subject['channel_paths'].items() if k in config['features']},
                      sampling_rate=sr)

        events = Events(id=subject['name'],
                        events_path=subject['events_path'],
                        units='seconds')
        epochs = Epochs(raw, events,
                        tmin=tmin,
                        tmax=tmax)
        epochs.data2array()

        X = epochs.data # shape in this example: (n_trials, n_times, n_frequencies, n_channels)
        y = events.data # shape in this example: (n_trials,) with class labels for each trial

        # define estimator to use on data
        estimator = Pipeline(
            [
                ("scaler", MinMaxScaler()),
                ("classifier", SVC(kernel="linear", C=1e5)),
            ]
        )
        estimator_name = (
            estimator[-1].__class__.__name__
            if isinstance(estimator, Pipeline)
            else estimator[-1].__class__.__name__
        )
        #make dummy classifier to get baseline performance on test set
        dummy = Pipeline([("scaler", MinMaxScaler()), ("classifier", DummyClassifier(strategy="most_frequent"))])

        #do nested cross-validation
        if  outer_cv > 1:
            outer_cv = StratifiedKFold(n_splits=outer_cv, shuffle=True, random_state=seed)
            for idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
                #define dictionary of possible optimizers to run
                optimizer_dict = {
                    "RFE": RecursiveFeatureElimination(
                        dims, 'tabular', estimator, scoring=metric, n_features_to_select=params['RFE_params']['n_features_to_select'],  
                        cv=inner_cv, importance_getter = params['RFE_params']['importance_getter'],
                        step=params['RFE_params']['step'], random_state=seed, n_jobs=n_jobs, verbose=True,
                    ),
                    "PSO": ParticleSwarmOptimization(
                        dims, 'tabular', estimator, scoring=metric, n_iter=params['PSO_params']['n_iter'], cv=inner_cv, 
                        random_state=seed, n_jobs=n_jobs, verbose=True, strategy="joint",
                    ),
                    "SA": SimulatedAnnealing(
                        dims, 'tabular', estimator, scoring=metric, n_iter=params['SA_params']['n_iter'], cv=inner_cv,
                        random_state=seed, n_jobs=n_jobs, verbose=True,
                    ),
                    "EA": EvolutionaryAlgorithms(
                        dims, 'tabular', estimator, scoring=metric, n_gen=params['EA_params']['n_gen'], cv=inner_cv, 
                        random_state=seed, n_jobs=n_jobs, verbose=True,
                    ),
                    "RS": RandomSearch(
                        dims, 'tabular', estimator, scoring=metric, n_iter=params['RS_params']['n_iter'], cv=inner_cv,
                        random_state=seed, n_jobs=n_jobs, verbose=True
                    ),
                }

                print(f"\nCommence Experiment on fold {idx}")
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # run optimizers and save results
                result_grids = []
                results = []
                models = []
                for method, clf in ([("Majority", dummy), (estimator_name, estimator)]
                    + [(name, optimizer) for name, optimizer in optimizer_dict.items()]):
                    if method in ["Majority", estimator_name]: #run baseline model 
                        if inner_cv > 1:
                            # Cross-validated scores for baselines
                            result = cross_validate(clf, X.reshape((X.shape[0], np.prod(X.shape[1:]))), y,
                                                    scoring=metric, cv=inner_cv, n_jobs=n_jobs) 
                            scores, train_time, test_time = result["test_score"] * 100, result["fit_time"], result["score_time"]
                            ci = np.round(stats.t.interval(0.95, len(scores) - 1, np.mean(scores),
                                                        np.std(scores) / np.sqrt(len(scores))), 3)
                            report_base = {"Method": clf.__class__.__name__, "Metric:": metric, "Task": task_name,
                                        "Mean": np.round(np.mean(scores), 3), "Median": np.round(np.median(scores), 3), "SD": np.round(np.std(scores), 3),
                                        "Train Time": np.round(np.mean(train_time), 6), "Infer Time": np.round(np.mean(test_time), 6)}
                            report_cv = {"95-CI Lower": ci[0], "95-CI Upper": ci[1], "CV Scores": list(np.round(scores, 3))}
                        else:
                            # Train-test split for baselines
                            X_train, X_test, y_train, y_test = train_test_split(X.reshape((X.shape[0], np.prod(X.shape[1:]))), y, test_size=inner_cv, random_state=seed)
                            with PerfTimer() as train_timer:
                                clf.fit(X_train, y_train)
                            with PerfTimer() as inference_timer:
                                score = get_scorer(metric)(clf, X_test, y_test) * 100
                            report_base = {"Method": clf.__class__.__name__, "Metric:": metric, "Task": task_name,
                                        "Mean": score,  "Median": score, "SS": 0.0,
                                        "Train Time": np.round(train_timer.duration, 6),
                                        "Infer Time": np.round(inference_timer.duration, 6)}
                            report_cv = {}
                    else: #run optimizers
                        if method not in optimization_methods: #only run optimizers specified in config
                            continue
                        # Fit optimizers
                        clf.fit(X_train, y_train)
                        best = clf.result_grid_[clf.result_grid_["Mean"] == clf.result_grid_["Mean"].max()].reset_index()
                        report_base = {
                            "Method": best.loc[0, "Method"], "Metric:": metric, "Task": task_name,
                            "Mean": best.loc[0, "Mean"] * 100, "Median": best.loc[0, "Median"] * 100, "SD": best.loc[0, "SD"] * 100,
                            "Train Time": best.loc[0, "Train Time"], "Infer Time": best.loc[0, "Infer Time"]
                        }
                        report_cv = {"95-CI Lower": best.loc[0, "95-CI Lower"] * 100, "95-CI Upper": best.loc[0, "95-CI Upper"] * 100,
                                    "CV Scores": list(np.round(np.array(best.filter(like="Fold", axis=1).loc[0, :]) * 100, 3))
                                    } if inner_cv > 1 else {}

                        result_grids.append(clf.result_grid_)
                        models.append(clf)
                # Common report fields for both baselines and optimizers
                report = {"Method": method, "Dimensions": str(dims), "Subject": subject["name"], "Task": task_name, "Metric": metric}
                results.append({**report, **report_base, **report_cv})
                # Save results  to a csv
                results_df = pd.DataFrame(results)
                dimensions_file = "_".join(str(d) for d in dims)
                results_df[results_df["Task"] == task_name].to_csv(
                    f"{training_output_path}/results_training_fold_{idx}_dim_{dimensions_file}.csv",
                    index=False,
                )
                result_grids = pd.concat(result_grids, axis=0)
                pd.DataFrame(result_grids).to_csv(
                    f'{training_output_path}/full_result_table_{subject["name"]}_trainings_fold_{idx}_dim_{dimensions_file}.csv',
                    index=False,
                )
                print(f"\nEvaluate performance on train and test fold {idx}")
                # Testing phase
                results = []
                for clf in models:
                    if hasattr(clf, "mask_"):
                        X_test_transformed = clf.transform(X)[test_idx]
                        X_train_transformed = clf.transform(X)[train_idx]
                        name = clf.__class__.__name__
                        est = clone(estimator)
                        mask = clf.mask_
                    else:
                        X_test_transformed = X_test
                        X_train_transformed = X_train
                        name = clf.named_steps["classifier"].__class__.__name__
                        est = clone(clf)
                        mask = np.ones(X.shape)

                    est.steps.insert(0, ("flatten", FlattenTransformer()))
                    est.fit(X_train_transformed, y_train)

                    with PerfTimer() as trainings_timer:
                        train_score = get_scorer(metric)(est, X_train_transformed, y_train)
                    with PerfTimer() as inference_timer:
                        test_score = get_scorer(metric)(est, X_test_transformed, y_test)
                    print(f"{name}. Train score: {np.round(train_score, 6)}. Test score: {np.round(test_score, 6)}.")

                    base = {"Method": name, "Metric": metric,"Task": task_name,
                          "Training Score": np.round(train_score, 6), "Training Time": np.round(trainings_timer.duration, 6),
                          "Test Score": np.round(test_score, 6), "Test Time": np.round(inference_timer.duration, 6)}
                    results.append(base)
                    np.save(f"{mask_output_path}/mask_{name}_fold_{idx}_dim_{dimensions_file}.npy", mask)
                results = pd.DataFrame(results)
                results.to_csv(
                    f"{testing_output_path}/results_test_fold_{idx}_dim_{dimensions_file}.csv",
                    index=False,
                )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters for sound generation')
    parser.add_argument('--config_path', '-c', type=str, help='Input config file', default='')
    args = parser.parse_args()

    main(args.config_path)