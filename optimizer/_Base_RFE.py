# -------------------------------------------------------------
# Channel Elimination
# Copyright (c) 2024
#       Dirk Keller,
#       Nick Ramsey's Lab, University Medical Center Utrecht, University Utrecht
# Licensed under the MIT License [see LICENSE for detail]
# -------------------------------------------------------------

import os
import random
import re
from abc import ABC, abstractmethod
from copy import copy
from typing import Tuple, List, Union, Dict, Any, Optional, Type

import matplotlib
import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas
import pandas as pd
import seaborn as sns
from deap import base, creator, tools, algorithms
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from pyswarms.single import GlobalBestPSO, LocalBestPSO
from scipy import stats
from scipy.io import savemat
from scipy.optimize import dual_annealing
from sklearn.base import TransformerMixin, MetaEstimatorMixin
from sklearn.metrics import get_scorer
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
# from sklearn.utils._metadata_requests import _RoutingNotSupportedMixin
from sklearn.utils.validation import check_is_fitted as sklearn_is_fitted
from operator import attrgetter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

from utils import (
    SimulatedAnnealingReporter, to_dict_keys, channel_id_to_int,
    grid_to_channel_id, compute_subgrid_dimensions
)