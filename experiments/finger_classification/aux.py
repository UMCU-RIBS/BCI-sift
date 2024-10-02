# -------------------------------------------------------------
# HandDecoding
# Copyright (c) 2023
#       Dirk Keller,
#       Nick Ramsey's Lab, University Medical Center Utrecht, University Utrecht
# Licensed under the MIT License [see LICENSE for detail]
# -------------------------------------------------------------


import concurrent.futures
import json
import os
import re
from collections import defaultdict
from typing import List, Dict

import mne_bids
import pandas as pd
import scipy.io


def convert_to_sub2run_dict(raws: list) -> Dict[str, List[any]]:
    """

    :param raws: (list) A list of RawBrainVision vision objects containing all runs for each subject.
    :return: (Dict[str: list]) Returns a Dictionary containing a list of runs (values) per subject (keys)
    """

    subject_runs_raw = defaultdict(
        list
    )  # Use defaultdict to store the subjects and their runs

    for raw_object in raws:
        subject_match = re.search(r"sub-(\d+)", raw_object.filenames[0])
        run_match = re.search(r"run-(\d+)", raw_object.filenames[0])

        if subject_match and run_match:
            subject_runs_raw["".join(["sub-", subject_match.group(1)])].append(
                raw_object
            )
    return dict(subject_runs_raw)


def load_raw_bids(bids_root: str) -> Dict[str, List[any]]:
    """

    :param bids_root: (str) The path of the bids root.
    :return: (Dict[str: list]) Returns a Dictionary containing a list of runs (values) per subject (keys)
    """

    bids_paths = mne_bids.find_matching_paths(
        bids_root,
        datatypes="ieeg",
        sessions=mne_bids.get_entity_vals(
            bids_root, entity_key="session", ignore_sessions="on"
        ),
        extensions=".vhdr",
    )
    return convert_to_sub2run_dict(
        [
            mne_bids.read_raw_bids(bids_path=bids_path, verbose=False)
            for _, bids_path in enumerate(bids_paths)
        ]
    )


##########################################
# Read Supplementary Material
##########################################


def load_info(path: str) -> pd.DataFrame:
    """
    Read supplementary information from a CSV or MATLAB file and return it as a Pandas DataFrame.

    :param path: (str) The path to the file.

    :return: A Pandas DataFrame containing the supplementary data.
    """
    if path.lower().endswith(".csv"):
        return pd.read_csv(path, index_col=0)
    elif path.lower().endswith(".mat"):
        return pd.DataFrame(scipy.io.loadmat(path))
    else:
        raise ValueError(
            "Unsupported file format. Supported formats are .csv and .mat."
        )


# f'{output_path}/{condition}_{subject}_channel_importance_{viz}_plot.png',
def load_markers(marker_path: str, data_path: str) -> Dict[str, Dict[str, List[float]]]:
    """
    Load onset markers from multiple .mat files in parallel.

    :param marker_path: (str) The directory where .mat files are located.
    :param data_path: (str) The directory tyo the data set.

    :return: A dictionary containing subject names as keys and a dictionary of onset markers as values.
             Each onset marker dictionary contains gesture names as keys and a list of onset marker values as values.
    """
    data_set = data_path.split("/")[-1].split("_")[0]  # TODO use find
    marker = marker_path.split("/")[-1]

    if data_set == "Gestures":  # TODO move to csv file
        data_num = 1
        motor_strategy = {
            "0": "gesture D",
            "1": "gesture F",
            "2": "gesture V",
            "3": "gesture Y",
        }  # same names as in BIDS, otherwise broadcasting error
        sub_code = {
            "mels": "sub-01_run-01",
            "terp_run1": "sub-02_run-01",
            "terp_run2": "sub-02_run-02",
            "brem": "sub-03_run-01",
            "rooi": "sub-04_run-01",
            "habe": "sub-05_run-01",
        }
    elif data_set == "BoldFingers":
        data_num = 7
        motor_strategy = {
            "0": "little move",
            "1": "index move",
            "2": "thumb move",
        }  # same names as in BIDS, otherwise broadcasting error
        sub_code = {
            "rooi": "sub-01_run-01",
            "heek": "sub-02_run-01",
            "franeker": "sub-03_run-01",
            "ommen_run1": "sub-04_run-01",
            "ommen_run2": "sub-04_run-02",
            "vledder": "sub-05_run-01",
            "mels": "sub-06_run-01",
            "duiven": "sub-07_run-01",
            "habe": "sub-08_run-01",
        }
    else:
        ValueError(
            f'Accepts only onset markers for "Gesture" or "BoldFingers" data set. Got {data_set}'
        )

    if marker == "gammaSlopemarkers":
        id = "allbvalues"
    elif marker == "movementOnsetMarkers":
        id = "MOmarkers"
    else:
        ValueError(
            f'Accepts only a path to the marker with a file name "movementOnsetMarkers" '
            f'or "gammaSlopemarkers" (only for Gesture data set). Got {marker}'
        )

    def load_mat_file(
        file_path: str,
        motor_strategy: Dict[str, str],
        sub_code: Dict[str, str],
        marker_id: str,
    ) -> tuple[str, Dict[str, List[float]]]:
        """
        Load onset markers from a single .mat file.

        :param file_path: (str) The path to the .mat file.
        :param motor_strategy: (Dict[str, str]) Dictionary mapping gesture code keys to gesture names.
        :param marker_id: (str) Matlab-specific variable name.

        :return: A tuple containing the subject name (str) and a dictionary of onset markers.
                 The onset markers dictionary contains gesture names as keys and a list of onset marker values as values.
        """
        data_dict = {}
        # Extract the subject name from the filename and load the data
        sub = [x for x in sub_code.keys() if x in file_path]
        assert len(sub) == 1, RuntimeWarning(
            f"No code or multiple codes matches the file: {file_path}"
        )

        # Load .mat file
        raws = scipy.io.loadmat(file_path)
        assert len(motor_strategy) == raws[marker_id].shape[1], (
            f"The motor strategy and .mat file for subject {sub} "
            f'do not have the same length. Got {len(motor_strategy)} and {raws["MOmarkers"].shape[1]}'
        )
        for i, raw in enumerate(raws[marker_id][0].tolist()):
            # Access the corresponding gesture code and use it to get the gesture nam
            strat = motor_strategy[str(i)]  # Use the code to get the gesture
            data_dict[strat] = raw
        return sub_code[sub[0]], data_dict

    mat_files = [
        os.path.join(marker_path, filename)
        for filename in os.listdir(marker_path)
        if filename.startswith(f"{marker[:-1]}_{data_num}")
        and (
            re.match(
                "^((movementOnsetMarker)|(gammaSlopemarker))_\d+_[a-zA-Z]+(?:_run\d+)?\.mat$",
                filename,
            )
            or re.match(
                "^((movementOnsetMarker)|(gammaSlopemarker))_\d+_[a-zA-Z]+\.mat$",
                filename,
            )
        )
    ]

    # Create a ThreadPoolExecutor with as many workers as needed
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Get a list of all .mat files in the directory
        # (re.match(r'^movementOnsetMarker_\d+_[a-zA-Z]+(?:_run\d+)?\.mat$', filename) or
        #  re.match(r'^movementOnsetMarker_\d+_[a-zA-Z]+\.mat$', filename))] #TODO match instead of strting seperastor
        # Use concurrent.futures to load files in parallel
        results = list(
            executor.map(
                load_mat_file,
                mat_files,
                [motor_strategy] * len(mat_files),
                [sub_code] * len(mat_files),
                [id] * len(mat_files),
            )
        )

    return {sub_name: data for sub_name, data in results}


def load_data_glove(path):
    results = {}
    data_name = path.split("/")[-1].split("_")[0].lower()
    data_name = data_name[:-1] if data_name == "boldfingers" else data_name
    for p in os.listdir(path):
        if not p.startswith("sub") or not os.path.isdir(os.path.join(path, p)):
            continue
        dir_path = f"{path}/{p}/ses-iemu/ieeg"
        files = []
        run = 1
        for f in os.listdir(dir_path):
            if not f.endswith("dataglove_physio.tsv"):
                continue

            file_path = f"{dir_path}/{p}_ses-iemu_task-{data_name}_run-{run if run > 10 else f'0{run}'}_recording-dataglove_physio"

            data = pd.read_csv(file_path + ".tsv", sep="\t")
            with open(file_path + ".json") as f:
                metadata = json.load(f)
            data.columns = metadata["Columns"][1:]
            files.append(data)
            run += 1
        results[p] = files
    return results
