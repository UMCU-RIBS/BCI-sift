# -------------------------------------------------------------
# HandDecoding
# Copyright (c) 2023
#       Dirk Keller,
#       Nick Ramsey's Lab, University Medical Center Utrecht, University Utrecht
# Licensed under the MIT License [see LICENSE for detail]
# -------------------------------------------------------------

import os

import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = [""]

# -----------------------------------------------------------------------------
# Data Settings
# -----------------------------------------------------------------------------
_C.DATA = CN()

# Directory of the data. Valid options are: 'BoldFingers_20230717' and 'Gestures_20230717'
_C.DATA.PATH = "./data/BoldFingers_20230717"
# Remove prior identified bad trials (Gestures and BoldFingers)
_C.DATA.REMOVE_BADS = True
# Unitary Baseline Name for both Data Sets (Gestures and BoldFingers)
_C.DATA.BASELINE_NAME = "rest"
# Directory of the supplementary info
_C.DATA.CH_FILENAME = "sub-ch_and_bads.csv"
_C.DATA.NOTCH_FILENAME = "sub-notch_bands.csv"
# Directory of the Onset Markers
_C.DATA.EVENT_MOM_PATH = "./data/movementOnsetMarkers"
_C.DATA.EVENT_GMS_PATH = "./data/gammaSlopemarkers"

# -----------------------------------------------------------------------------
# Preprocessing Settings
# -----------------------------------------------------------------------------
_C.PREP = CN()
# Sampling rate of signal
_C.PREP.SAMPLING_RATE = 500
# Filter Method used in during Signal Preprocessing
# Chose from: 'fir', 'iir', 'spectrum'
_C.PREP.FILTER = "fir"
# TFR Method used in during Signal Preprocessing
# Chose from: 'fft', 'morlet', 'stockwell'
_C.PREP.TFR = "morlet"
# Lower Bound for Filter Window and Time-Frequency Transformation
_C.PREP.BANDS = [
    ("delta", 1, 3),
    ("theta", 4, 7),
    ("beta", 8, 12),
    ("alpha", 13, 30),
    ("gamma", 70, 130),
]
# Lower Bound for Filter Window and Time-Frequency Transformation
_C.PREP.FMIN = 1
# Upper Bound for Filter Window and Time-Frequency Transformation
_C.PREP.FMAX = 130
# Time Bandwith (fft) or Gaussian Width (stockwell)
# (fft) Product of temporal window (s) and frequency bandwidth (Hz). If < freqs, more time smoothing, less variance
# (stockwell) Gaussian window. If < 1, increased temporal resolution; If > 1, increased frequency resolution
_C.PREP.TFR_WIDTH = 2
# Number of cycles in the wavelet (fft only)
# Chose from 'auto' or a number
_C.PREP.TFR_CYCLES = "auto"
# Decimation rate
_C.PREP.DECIM = 10
# -----------------------------------------------------------------------------
# Experiment Settings
# -----------------------------------------------------------------------------
_C.EXPERIMENT = CN()
# Experiment 2: Multiclass with 4 Degrees of Freedom (DOI).
_C.EXPERIMENT.FOUR_DOF = True
# Experiment 2: Subjects to decode. Valid options:
# 'sub-01','sub-02', 'sub-03','sub-05','sub-06','sub-07','sub-08',
_C.EXPERIMENT.SUBJECTS = [
    # "sub-01",
    # "sub-02",
    # "sub-03",
    # "sub-05",
    "sub-06",
    # "sub-07",
    # "sub-08",
]
# Experiment 2: Multiclass with 8 Degrees of Freedom (DOI).  Cross Gesture and Finger Multi.
_C.EXPERIMENT.EIGHT_DOF = False

# -----------------------------------------------------------------------------
# Subgrid Search Settings
# -----------------------------------------------------------------------------
_C.SUBGRID = CN()
# Define which classifiers to experiment with. Possible options:
_C.SUBGRID.DIMS = (2, 1, 3)  #
# Early stopping criteria for the dimensions
_C.SUBGRID.PATIENCE = (20, 75, 75)  # (20, 75)
# _C.SUBGRID.CLASSIFIERS = ['base', 'lr', 'lda', 'xgb', 'svm']
# Optimizer options: 'PS', 'SES', 'SSHC', 'RFE', 'EA', 'SA', 'PSO'
_C.SUBGRID.OPTIMIZERS = ["PSO"]
# With or without hyperparameter tuning
_C.SUBGRID.HP = False
# Number of Crossvalidation Folds
_C.SUBGRID.CV = 0.8  # 10
# Define which metric top use to evaluate the classifier.
_C.SUBGRID.METRIC = "accuracy"
# Number of random mask generations
_C.SUBGRID.RS_ITER = 2  # 150  # 150
# Iteration factor that is multiplied with the total number of electrodes in the grid.
# Determines the number of reinitialization (random starting positions) of the algorithm.
_C.SUBGRID.SSHC_FACTOR = 0.1  # 1.5  # 150
# Percentage of Retained Features of the Recursive Feature Elimination Algorithm.
_C.SUBGRID.RFE_RATIO = 1
_C.SUBGRID.RFE_STEP = 1  # 10
# Number of Generations of the Evolutionary Algorithm.
_C.SUBGRID.EA_ITER = 2  # 150  # 250  # 250
# Number of Iterations of the Dual Simulated Annealing Algorithm.
# Iterations are somewhat lower due to longer convergence time.
_C.SUBGRID.SA_ITER = 2  # 150  # 150  # 150
# Number of Iterations of the Particle Swarm Optimization Algorithm.
_C.SUBGRID.PSO_ITER = 10  # 150  # 250  # 250

# -----------------------------------------------------------------------------
# Output Paths
# -----------------------------------------------------------------------------

_C.OUTPUT = CN()
# Path to output folder, overwritten by command line argument
_C.OUTPUT.CONFIG = "./config"
# # Path to output folder

# -----------------------------------------------------------------------------
# MISC
# -----------------------------------------------------------------------------
# Fixed random seed
_C.SEED = 42


def get_default_config():
    """
    Get a yacs CfgNode object with default values
    """
    # Returns a clone so that the defaults will not be altered
    # It will be subsequently overwritten with local YAML.
    return _C.clone()


def load_config(file_path):
    with open(file_path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def save_config(config, file_path):
    with open(file_path, "w") as f:
        f.write(config.dump())


def _update_config_from_file(config, cfg_file):
    config.defrost()

    yamal_cfg = load_config(cfg_file)

    for cfg in yamal_cfg.setdefault("BASE", [""]):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print(f"=> merge config from {cfg_file}")
    config.merge_from_file(cfg_file)

    config.freeze()


def update_config_from_args(args):
    config = get_default_config()

    # if os.path.isfile(args.cfg):
    #     config_path = os.join.path(config.OUTPUT.CONFIG, args.cfg)
    #     _update_config_from_file(config, config_path)

    config.defrost()

    # # merge from specific arguments
    # if hasattr(args, '--experiment') and args.experiment:
    #     config.MODEL.NAME = args.experiment

    # Save to output folder
    if not os.path.isdir(config.OUTPUT.CONFIG):
        os.makedirs(config.OUTPUT.CONFIG, exist_ok=True)

    name = args.cfg.split("/")[-1].replace(".yaml", "")
    config.OUTPUT.CONFIG = os.path.join(config.OUTPUT.CONFIG, name)
    save_config(config, f"{config.OUTPUT.CONFIG}.yaml")

    config.freeze()
    return config


# TODO update config, sub from_args (save=True), sub from_args (save=True); call config_attributes  (merge)
