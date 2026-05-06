# -------------------------------------------------------------
# BCI-sift
# Copyright (c) 2025
#       Dirk Keller,
#       Elena Offenberg,
#       Nick Ramsey's Lab, University Medical Center Utrecht, University Utrecht
# Licensed under the MIT License [see LICENSE for detail]
# -------------------------------------------------------------
import typing
import numpy as np
import pandas as pd
from collections import OrderedDict
pd.set_option('display.max_rows', 500)

class Dataset:
    """
    Dataset class to load and store data and channel information for a given dataset. 
    The class also has a method to convert the data into an array format that can be used 
    for training machine learning models.
    Parameters:
    -----------
    :param id: str
        Name of subject or alias
    :param input_paths: dict
        A dictionary with keys corresponding to the feature names and values corresponding to the paths of the input
    :param channel_paths: dict
        A dictionary with keys corresponding to the feature names and values corresponding to the paths of the channel information
    :param sampling_rate: float
        The sampling rate of the data, to convert the time information in the channel information to sample indices.
        Sampling rate is not needed if the time information in the channel information is already in sample indices. 

    Methods:
    --------
    - data2array: Convert the data into an array format that can be used for training machine learning models. The data is stored
    in a dictionary with keys corresponding to the feature names.

    Returns:
    --------
    :return: None
    """
    def __init__(self, id: str,
                        input_paths: typing.Dict = None,
                        channel_paths: typing.Dict = None,
                        sampling_rate: float = None) -> None:
        

        self.id = id
        if input_paths is not None:
            self.input_paths = input_paths
            self.data = OrderedDict({k:[] for k in self.input_paths.keys()})
            for k, v in input_paths.items():
                self.data[k] = np.load(v)
        if channel_paths is not None:
            self.channel_paths = channel_paths
            self.channels = OrderedDict({k: [] for k in self.channel_paths.keys()})
            for k, v in channel_paths.items():
                self.channels[k] = pd.read_csv(v)
        assert input_paths.keys() == channel_paths.keys(), 'Inputs and channels do not correspond to the same features'
        if sampling_rate is not None:
            self.sampling_rate = sampling_rate

    def data2array(self):
        """Convert the data into an array format that can be used for training machine learning models.
        The data is stored in a dictionary with keys corresponding to the feature names."""
        assert all([isinstance(d, np.ndarray) for d in self.data.values()]), \
                                    'All data needs to be in arrays'
        x, names = [], []
        for k, v in self.data.items():
            names.append(k)
            x.append(v)

        x = np.array(x)[None,:]
        self.data_array = x
        self.feature_names = names
