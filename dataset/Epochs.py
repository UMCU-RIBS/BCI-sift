# -------------------------------------------------------------
# BCI-sift
# Copyright (c) 2025
#       Dirk Keller,
#       Elena Offenberg,
#       Nick Ramsey's Lab, University Medical Center Utrecht, University Utrecht
# Licensed under the MIT License [see LICENSE for detail]
# -------------------------------------------------------------
import numpy as np
from .Dataset import Dataset
from .Events import Events
from .utils import sec2ind
from collections import OrderedDict
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class Epochs:
    """
    Epochs class to create epochs from a given dataset and events. The class also has a method to convert the 
    data into an array format that can be used for training machine learning models.
    Parameters:
    -----------
    :param raw: Dataset
        The raw dataset to create epochs from.
    :param events: Events
        The behavioral events to create epochs from.
    :param tmin: float
        Onset in time to start each epoch relative to each event's starting time point.
    :param tmax: float or str
        Offset in time to end each epoch relative to each event's starting time point. If tmax is a string, it can be either 
        'maxlen' or 'unequal'. If tmax is 'maxlen', the length of each epoch is determined by the longest event in the dataset. 
        If tmax is 'unequal', the length of each epoch is determined by the duration of each event in the dataset, which needs 
        to be provided in the events dataframe as 'xmax'.

    Methods:
    --------
    - data2array: Convert the data into an array format that can be used for training machine learning models. The data is stored 
    in a dictionary with keys corresponding to the feature names.

    Returns:
    --------
    :return: None
    """
    def __init__(self, raw: Dataset, events: Events, tmin: float, tmax: object) -> None:

        self.data_ = OrderedDict({k:[] for k in raw.data.keys()})
        self.onsets = events.dataframe['xmin'] + tmin
        self.offsets = None

        if type(tmax) is str:
            if tmax == 'maxlen':
                tmax = events.dataframe['duration'].max()
                fixed_duration = tmax - tmin
            elif tmax == 'unequal':
                assert 'xmax' in events.dataframe, 'If tmax is uniqual, xmax needs to be provided per event'
                fixed_duration = None
                self.offsets = events.dataframe['xmax']
            else:
                raise NotImplementedError
        elif type(tmax) == int or type(tmax) == float:
            tmax = float(tmax)
            fixed_duration = tmax - tmin
        else:
            raise NotImplementedError

        self.fixed_duration = fixed_duration

        sr = raw.sampling_rate
        for k in self.data_.keys():
            if fixed_duration is not None:
                for ons in self.onsets:
                    if sec2ind(ons, sr) < 0 or sec2ind(ons, sr) + sec2ind(fixed_duration, sr) > len(raw.data[k]):
                        warnings.warn('Index out of range, wrapping the array')
                    self.data_[k].append(raw.data[k].take(range(sec2ind(ons, sr),
                                                                sec2ind(ons, sr) + sec2ind(fixed_duration, sr)),
                                                          axis=0, mode='wrap'))
                self.data_[k] = np.array((self.data_[k]))
            else:
                for ons, off in zip(self.onsets, self.offsets):
                    self.data_[k].append(raw.data[k][sec2ind(ons, sr):sec2ind(off, sr)])


    def data2array(self):
        assert all([isinstance(d, np.ndarray) for d in self.data_.values()]), \
                                    'Cannot concatenate all data if epochs are not equal size'
        x, names = [], []
        for k, v in self.data_.items():
            names.append(k)
            x.append(v)

        x = np.array(x)
        xx = x.transpose((1, 2, 0, 3)) # epochs x timepoints x feature sets x channels
        self.data = xx
        self.feature_names = names


