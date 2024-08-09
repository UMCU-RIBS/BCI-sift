'''

What is changed:
    - added events_df and select_df as possible inputs to Events, not only paths
    - changed channel_path str to channel_paths dict: allowing different featuers have different channels
'''

import typing
import numpy as np
import pandas as pd
from collections import OrderedDict
pd.set_option('display.max_rows', 500)

class Dataset:
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
        assert all([isinstance(d, np.ndarray) for d in self.data.values()]), \
                                    'All data needs to be in arrays'
        x, names = [], []
        for k, v in self.data.items():
            names.append(k)
            x.append(v)

        x = np.array(x)[None,:]
        self.data_array = x
        self.feature_names = names
