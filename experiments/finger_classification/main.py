# -------------------------------------------------------------
# HandDecoding
# Copyright (c) 2023
#       Dirk Keller,
#       Nick Ramsey's Lab, University Medical Center Utrecht, University Utrecht
# Licensed under the MIT License [see LICENSE for detail]
# -------------------------------------------------------------

import argparse
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from ray import tune
from scipy import stats
from sklearn.dummy import DummyClassifier
from sklearn.metrics import get_scorer
from sklearn.model_selection import (
    train_test_split,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from aux import load_raw_bids, load_info, load_markers, load_data_glove
from config import update_config_from_args
from optimizer import (
    EvolutionaryAlgorithms,
    ParticleSwarmOptimization,
    RecursiveFeatureElimination,
    RandomSearch,
    SimulatedAnnealing,
)
from preprocessing import GestureFingerDataProcessor, ECOGPreprocessor
from utils.hp_tune import DecoderOptimization, PerfTimer

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def parse_option():
    parser = argparse.ArgumentParser(
        "ECoG motor decoding: Preprocessing, training and evaluation script",
        add_help=False,
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="clm_gestures_spatial.yaml",  # required
        metavar="FILE",
        help="configs file name",
    )

    # # convenient configs modifications
    # parser.add_argument(
    #     '--experiment',
    #     type=str,
    #     default='clm_spatial_no_pca',
    #     help='model name'
    # )

    args, unparsed = parser.parse_known_args()
    configs = update_config_from_args(args)
    return args, configs


def make_groups(y):
    s = [np.where(y == i)[0] for i in np.unique(y)]
    lens = [len(i) for i in s]
    sp = [
        np.pad(sub, (0, max(lens) - l), "constant", constant_values=-1)
        for sub, l in zip(s, lens)
    ]
    spx = [[sub[i] for sub in sp] for i in range(max(lens))]
    [sub.remove(-1) for sub in spx if -1 in sub]
    groups = np.zeros(len(y)).astype(int)
    for i, sub in enumerate(spx):
        groups[sub] = i
    return groups


# fmt: off


def channel_combination_serarch(
        X,y,dims,estimator,with_hp,condition,metric,info,cv,seed,n_jobs,output_path,
):
    grid, bads = info["grid"], info["bads"]
    ed, ied = info["ed"], info["ied"]
    name = info["subject_info"]["his_id"]

    # groups = make_groups(y)
    X = X.reshape(X.shape[0], X.shape[1] * X.shape[2], X.shape[3], X.shape[4])
    # X = np.mean(X, axis=(2, 3))

    n_cv = cv if isinstance(cv, (float, int)) else cv.get_n_splits()  # groups=groups)
    estimator_name = (
        estimator[-1].__class__.__name__
        if isinstance(estimator, Pipeline)
        else estimator[-1].__class__.__name__
    )

    opt_lib = {
        # "SES": SpatialExhaustiveSearch(
        #     dims, estimator, scoring=metric, cv=cv, strategy="joint", n_jobs=n_jobs, verbose=False if with_hp else True
        # ),
        # "SSHC": SpatialStochasticHillClimbing(
        #     dims, estimator, scoring=metric, n_iter=int(len(grid.reshape(-1)) * config.SUBGRID.SSHC_FACTOR),
        #     cv=cv, random_state=seed, n_jobs=n_jobs, verbose=False if with_hp else True,
        # ),
        "RS": RandomSearch(
            dims, 'tabular', estimator, scoring=metric, n_iter=config.SUBGRID.RS_ITER, cv=cv,
            strategy="joint", random_state=seed, n_jobs=n_jobs, verbose=False if with_hp else True,
        ),
        "RFE": RecursiveFeatureElimination(
            dims, 'tabular', estimator, scoring=metric, n_features_to_select=config.SUBGRID.RFE_RATIO, cv=cv,
            strategy="joint", step=config.SUBGRID.RFE_STEP, random_state=seed, n_jobs=n_jobs, verbose=False if with_hp else True,
        ),
        "PSO": ParticleSwarmOptimization(
            dims, 'tabular', estimator, scoring=metric, n_iter=config.SUBGRID.PSO_ITER, cv=cv,
            random_state=seed, n_jobs=n_jobs, verbose=False if with_hp else True,
        ),
        "SA": SimulatedAnnealing(
            dims, 'tabular', estimator, scoring=metric, n_iter=config.SUBGRID.SA_ITER, cv=cv,
            random_state=seed, n_jobs=n_jobs, verbose=False if with_hp else True,
        ),
        "EA": EvolutionaryAlgorithms(
            dims, 'tabular', estimator, scoring=metric, n_gen=config.SUBGRID.EA_ITER, cv=cv,
            random_state=seed, n_jobs=n_jobs, verbose=False if with_hp else True,
        ),
    }

    result_grids = []
    results = []

    for method, clf in ([("Majority", DummyClassifier(strategy="most_frequent")), (estimator_name, estimator)]
                        + [(name, optimizer) for name, optimizer in opt_lib.items()]):
        if method in ["Majority", estimator_name]:
            if n_cv > 1:
                # Cross-validated scores for baselines
                result = cross_validate(clf, X.reshape((X.shape[0], np.prod(X.shape[1:]))), y,
                                         scoring=metric, cv=cv, n_jobs=n_jobs) #groups=groups,
                scores, train_time, test_time = result["test_score"] * 100, result["fit_time"], result["score_time"]
                ci = np.round(stats.t.interval(0.95, len(scores) - 1, np.mean(scores),
                                               np.std(scores) / np.sqrt(len(scores))), 3)
                report_base = {"Mean": np.round(np.mean(scores), 3), "Median": np.round(np.median(scores), 3),
                               "SD": np.round(np.std(scores), 3), "Train Time": np.round(np.mean(train_time), 6),
                               "Infer Time": np.round(np.mean(test_time), 6)}
                report_cv = {"95-CI Lower": ci[0], "95-CI Upper": ci[1], "CV Scores": list(np.round(scores, 3))}
            else:
                # Train-test split for baselines
                X_train, X_test, y_train, y_test = train_test_split(X.reshape((X.shape[0], np.prod(X.shape[1:]))), y, test_size=n_cv, random_state=seed)
                with PerfTimer() as train_timer:
                    clf.fit(X_train, y_train)
                with PerfTimer() as inference_timer:
                    score = get_scorer(metric)(clf, X_test, y_test) * 100
                report_base = {"Mean": score, "Median": score, "SS": 0.0,
                               "Train Time": np.round(train_timer.duration, 6),
                               "Infer Time": np.round(inference_timer.duration, 6)}
                report_cv = {}
        else:
            if method not in config.SUBGRID.OPTIMIZERS:
                continue
            if with_hp:
                # Hyperparameter optimization for optimizers
                param_space = {"estimator_params": {"C": tune.loguniform(0.001, 10)}}
                dc = DecoderOptimization(estimator=clf, param_dist=param_space,
                                         max_iter=clf.n_gen if method == "EA" else clf.n_iter,
                                         num_samples=25, exp_name=method, metric=config.SUBGRID.METRIC,
                                         search_optimizer="HEBO", out_path=output_path, max_concurrent=1)
                dc.optimize(X=X, y=y)
            else:
                # Fit optimizers without hyperparameter search
                clf.fit(X, y)
                best = clf.result_grid_[clf.result_grid_["Mean"] == clf.result_grid_["Mean"].max()].reset_index()
                report_base = {
                    "Method": best.loc[0, "Method"], "Mean": best.loc[0, "Mean"] * 100,
                    "Median": best.loc[0, "Median"] * 100, "SD": best.loc[0, "SD"] * 100,
                    "Train Time": best.loc[0, "Train Time"], "Infer Time": best.loc[0, "Infer Time"]
                }
                report_cv = {"95-CI Lower": best.loc[0, "95-CI Lower"] * 100, "95-CI Upper": best.loc[0, "95-CI Upper"] * 100,
                             "CV Scores": list(np.round(np.array(best.filter(like="Fold", axis=1).loc[0, :]) * 100, 3))
                             } if n_cv > 1 else {}

            result_grids.append(clf.result_grid_)

            # # Generate and save plots
            # if len(dims) == 2:
            #     plot_params = {
            #         "grid": grid,
            #         "result_grid": clf.result_grid_,
            #         "output_path": f"{output_path}/{method}",
            #         "identifier": f"{condition}_{name}",
            #         "bads": bads,
            #     }
            #     # Importance plots
            #     for viz, extra_args in [("heat_map", {}), ("grid_overlay", {"top_k": 5}), ("distribution", {})]:
            #         importance_plot(viz=viz, **plot_params, **extra_args)
            #
            #     # Elimination plots
            #     elim_params = {**plot_params, "seed": seed}
            #     elimination_plot(ied_ed=(ied, ed), **elim_params)
            #     elimination_plot(**elim_params)

        # Common report fields for both baselines and optimizers
        report = {"Method": method, "Dimensions": str(dims), "Subject": name, "Condition": condition, "Metric": metric}
        results.append({**report, **report_base, **report_cv})

    # Concatenate all results
    result_grids = pd.concat(result_grids, axis=0)
    # fmt: on
    return results, result_grids
    # fmt: on


def experiment_four_DoF(ECOG_processor, configs, estimator, with_hp, fold_gen, ch_info):
    # Channel combinator results for each subject
    comb_results_multi = []

    """For multi class or binary (all versus rest)"""
    clf_classes = ["4-DoF", "5-DoF"]
    for clf_class in clf_classes:
        print(
            f"\n---------------------------------------------------------------------------------",
            f'\n   Commence Experiment on {clf_class} for {configs.DATA.PATH.split("/")[-1].split("_")[0]}.',
            f"\n---------------------------------------------------------------------------------",
        )

        # Make Directory for the classification-type-specific output for three classes.
        out_path_clf = (
            f'./output/{configs.DATA.PATH.split("/")[-1]}/out_model_{clf_class}'
        )
        conditions = [item for item in ECOG_processor[0].trial_types_]

        # Remove least decodable Gesture (Gesture D) to make Fingers and Gestures comparable
        if (
            configs.DATA.PATH.split("/")[-1].split("_")[0] == "Gestures"
            and clf_class == "4-DoF"
        ):
            conditions = [
                item for item in ECOG_processor[0].trial_types_ if item != "gesture D"
            ]
        elif (
            configs.DATA.PATH.split("/")[-1].split("_")[0] == "BoldFingers"
            and clf_class == "5-DoF"
        ):
            print(f"Skipped ...")
            continue

        os.makedirs(out_path_clf, exist_ok=True)
        data = []
        for processor in ECOG_processor:
            X, y, info = processor.get_mne_data(
                conditions=conditions, get_raw=False, to_grid=True
            )
            data.append(X)
        X = np.stack(data, axis=3)
        print(
            f'\n Starting Experiment 2. Subgrid Search of {info["subject_info"]["his_id"]} for {clf_class} in {configs.DATA.PATH.split("/")[-1].split("_")[0]} ...'
        )

        # info['model'] = name[0]
        info["ied"] = ch_info.loc[info["subject_info"]["his_id"], "IED"]
        info["ed"] = ch_info.loc[info["subject_info"]["his_id"], "ED"]

        ch_comb_results, result_grid = channel_combination_serarch(
            X=X,
            y=y,
            dims=config.SUBGRID.DIMS,
            estimator=estimator,
            with_hp=with_hp,
            condition=clf_class,
            metric=config.SUBGRID.METRIC,
            info=info,
            cv=fold_gen,
            seed=config.SEED,
            n_jobs=num_cpu,
            output_path=out_path_clf,
        )
        comb_results_multi += ch_comb_results

        # Save results to a Channel Combination analytics to a csv
        ch_combo_df = pd.DataFrame(comb_results_multi)
        dimensions_file = "_".join(str(d) for d in config.SUBGRID.DIMS)
        ch_combo_df[ch_combo_df["Condition"] == clf_class].to_csv(
            f"{out_path_clf}/_global_channel_optimization_results_dim_{dimensions_file}.csv",
            index=False,
        )
        pd.DataFrame(result_grid).to_csv(
            f'{out_path_clf}/{info["subject_info"]["his_id"]}_dim_{dimensions_file}_full_result_table.csv',
            index=False,
        )


def experiment_eight_DoF(configs, estimator, with_hp, fold_gen, ch_info):
    # Channel combination results for each subject
    comb_results_multi = []

    clf_class = "8-DoF"
    print(
        f"\n---------------------------------------------------------------------------------",
        f'\n   Commence Experiment on {clf_class} for {configs.DATA.PATH.split("/")[-1].split("_")[0]}.',
        f"\n---------------------------------------------------------------------------------",
    )

    # Make Directory for the classification-type-specific output for three classes.
    data_paths = [f"./output/Gestures_20230717/", f"./output/BoldFingers_20230717/"]
    out_path_clf = f'./output/{configs.DATA.PATH.split("/")[-1]}/out_model_{clf_class}'
    os.makedirs(out_path_clf, exist_ok=True)

    try:
        multi_8 = {}
        for data, subs, type in zip(
            data_paths, [["01", "04", "05"], ["01", "06", "08"]], ["gesture", "finger"]
        ):
            # Filter files that are pickled and load the corresponding subjects
            filtered_files = sorted(
                [
                    file
                    for file in os.listdir(data)
                    if file.endswith("_2.0.pkl")
                    and (subs[0] in file or subs[1] in file or subs[2] in file)
                ]
            )

            # Load pickle files into memory
            for idx, file in enumerate(filtered_files):
                file_path = os.path.join(data, file)
                with open(file_path, "rb") as f:
                    multi_8[type + "_" + subs[idx]] = pickle.load(f)
        mels_rooi_habe = (
            [multi_8["gesture_01"], multi_8["finger_06"]],
            [multi_8["gesture_04"], multi_8["finger_01"]],
            [multi_8["gesture_05"], multi_8["finger_08"]],
        )
    except:
        raise RuntimeError(
            f"8-DoF setting was selected but not all "
            f"relevant subjects of the two data sets were preprocessed!"
        )

    for idx, sub in enumerate(mels_rooi_habe):
        # if idx != 3:
        #     continue
        X_G, y_G, info_G = sub[0].get_mne_data(
            conditions=[item for item in sub[0].trial_types_],
            get_raw=False,
            to_grid=True,
        )
        X_F, y_F, info_F = sub[1].get_mne_data(
            conditions=[item for item in sub[1].trial_types_],
            get_raw=False,
            to_grid=True,
        )
        y_G += 3
        y_F[y_F == 2], y_F[y_F == 3] = 7, 2

        info = info_F
        if configs.DATA.PATH.split("/")[-1].split("_")[0] == "Gestures":
            info = info_G

        X, y = np.concatenate((X_F, X_G), axis=0), np.concatenate((y_F, y_G), axis=0)
        info["labels"] = [
            "Index",
            "Little",
            "Thumb",
            "Gesture D",
            "Gesture F",
            "Gesture V",
            "Gesture Y",
            "Rest",
        ]

        print(
            f'\n Starting Experiment 2. Subgrid Search of {info["subject_info"]["his_id"]} for {clf_class} in {configs.DATA.PATH.split("/")[-1].split("_")[0]} ...'
        )

        info["ied"] = ch_info.loc[info["subject_info"]["his_id"], "IED"]
        info["ed"] = ch_info.loc[info["subject_info"]["his_id"], "ED"]

        ch_comb_results, result_grid = channel_combination_serarch(
            X=X,
            y=y,
            dims=config.SUBGRID.DIMS,
            estimator=estimator,
            with_hp=with_hp,
            condition=clf_class,
            metric=config.SUBGRID.METRIC,
            info=info,
            cv=fold_gen,
            seed=config.SEED,
            n_jobs=num_cpu,
            output_path=out_path_clf,
        )
        comb_results_multi += ch_comb_results

        # Save results to a Channel Combination analytics to a csv
        ch_combo_df = pd.DataFrame(comb_results_multi)
        ch_combo_df[ch_combo_df["Condition"] == clf_class].to_csv(
            f"{out_path_clf}/Subgrid/global_channel_optimization_results_.csv",
            index=False,
        )

        with open(
            f'{out_path_clf}/Subgrid/{info["subject_info"]["his_id"]}_spatial_channel_optimization_results.sav',
            "wb",
        ) as f:
            pickle.dump(pd.DataFrame(result_grid), f)


def main(configs):
    """"""
    np.random.seed(configs.SEED)

    global num_cpu
    num_cpu = os.cpu_count() - 2
    print(f"\nNumber of available cpu cores: {num_cpu}")

    estimator = Pipeline([("scaler", MinMaxScaler()), ("svc", SVC(kernel="linear"))])

    """Acquire data set and supplementary material"""
    print(f"\nLoad raw BIDS IEEG data and supplementary material into memory...")
    # extract parameters from BIDS directory structure to read in BIDS ieeg data
    raw = load_raw_bids(bids_root=configs.DATA.PATH)

    ch_info = load_info(path=f"{configs.DATA.PATH}/{configs.DATA.CH_FILENAME}")
    notch_bands = load_info(path=f"{configs.DATA.PATH}/{configs.DATA.NOTCH_FILENAME}")
    data_glove = load_data_glove(path=configs.DATA.PATH)
    mom = load_markers(
        marker_path=configs.DATA.EVENT_MOM_PATH, data_path=configs.DATA.PATH
    )
    if config.EXPERIMENT.EIGHT_DOF:
        tmin, tmax = -0.5, 2.0
    elif configs.DATA.PATH.split("/")[-1].split("_")[0] == "Gestures":
        tmin, tmax = -0.5, 2.5
    elif configs.DATA.PATH.split("/")[-1].split("_")[0] == "BoldFingers":
        tmin, tmax = -0.5, 1.5

    # Iterate through your bids_ieeg_data dictionary
    for subject, session in raw.items():
        if subject not in config.EXPERIMENT.SUBJECTS:
            continue

        """Execute Data-specific Preprocessing"""
        print(f"\nExecute Data-specific Preprocessing...")
        ECOG_processor = []
        pickled = False

        for band, fmin, fmax in config.PREP.BANDS:
            preprocessor_file_path = f'./output/{configs.DATA.PATH.split("/")[-1]}/{band}-ECOG_processor_{subject}_{tmax}.pkl'
            if os.path.exists(preprocessor_file_path):
                pickled = True
                with open(preprocessor_file_path, "rb") as file:
                    ECOG_processor.append(pickle.load(file))
            else:
                pickled = False

            if not pickled:
                print(f"\nExecute Signal Preprocessing for patient '{subject}'\n")
                data_set_processor = GestureFingerDataProcessor(
                    tmin=tmin,
                    tmax=tmax,
                    ch_info=ch_info,
                    data_glove=data_glove,
                    mom=mom,
                    gms=None,
                    multi_clf="multi",
                    remove_bads=configs.DATA.REMOVE_BADS,
                    baseline_name=configs.DATA.BASELINE_NAME,
                    data_set_name=configs.DATA.PATH.split("/")[-1],
                    n_jobs=num_cpu,
                    verbose=False,
                )
                print(f"Prepare Data sets for patient {subject} ...")
                session = data_set_processor.fit_transform(session)

                data_set = configs.DATA.PATH.split("/")[-1]
                file_name = next(
                    file_name
                    for file_name in [
                        os.path.join(f"./data/grid_forms/{data_set}", f)
                        for f in os.listdir(f"./data/grid_forms/{data_set}/")
                        if os.path.isfile(
                            os.path.join(f"./data/grid_forms/{data_set}", f)
                        )
                    ]
                    if session.info["subject_info"]["his_id"] in file_name
                )
                for band, fmin, fmax in config.PREP.BANDS:
                    preprocessor_file_path = f'./output/{configs.DATA.PATH.split("/")[-1]}/{band}-ECOG_processor_{subject}_{tmax}.pkl'
                    processor = ECOGPreprocessor(
                        sfreq=configs.PREP.SAMPLING_RATE,
                        tmin=tmin,
                        tmax=tmax,
                        fmin=fmin,
                        fmax=fmax,
                        filter_method=configs.PREP.FILTER,
                        tfr_method=configs.PREP.TFR,
                        tfr_width=configs.PREP.TFR_WIDTH,
                        n_cycles=configs.PREP.TFR_CYCLES,
                        notch_info=notch_bands,
                        decim=configs.PREP.DECIM,  # 5
                        grid_files=file_name,
                        output_path=f"./output/{data_set}/out_prep/{band}",
                        n_jobs=num_cpu,
                        verbose=True,
                        plots=2,
                    )
                    processor.fit_transform(session)

                    # Pickle the preprocessing to avoid recomputation
                    with open(preprocessor_file_path, "wb") as file:
                        pickle.dump(processor, file)

        # Run experiment on 4-DoF
        if config.EXPERIMENT.FOUR_DOF:
            # print(f'Commence Experiment on 4-DoF for {configs.DATA.PATH.split("/")[-1].split("_")[0]}.')
            experiment_four_DoF(
                ECOG_processor, config, estimator, config.SUBGRID.HP, config.SUBGRID.CV, ch_info
            )
        else:
            print(
                f'Experiment on 4-DoF for {configs.DATA.PATH.split("/")[-1].split("_")[0]} skipped...'
            )

        # Run experiment on 8-DoF
        if config.EXPERIMENT.EIGHT_DOF:
            # print(f'Commence Experiment on 8-DoF for all Hand Movements.')
            experiment_eight_DoF(
                config, estimator, config.SUBGRID.HP, config.SUBGRID.CV, ch_info
            )
        else:
            print(f"Experiment on 8-DoF for all Hand Movements skipped...")
        print(f"Done with {subject}")


if __name__ == "__main__":
    _, config = parse_option()
    main(config)
