'''

What is changed:
    - changed sec2ind from local to general (July 20, 2023)
    - added possibility of unequal epochs: epochs.data_ is a list (July 20, 2023)
    - added wrapping around the array if tmin or tmax indices are out of range: use np.take (July 25, 2023)
'''
import numpy as np
from .Dataset import Dataset
from .Events import Events
from .utils import sec2ind
from collections import OrderedDict
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class Epochs:
    def __init__(self, raw: Dataset, events: Events, tmin: float, tmax: object) -> None:

        self.data_ = OrderedDict({k:[] for k in raw.data.keys()})
        self.onsets = events.dataframe['xmin'] + tmin
        self.offsets = None

        if type(tmax) is str:
            if tmax == 'maxlen':
                tmax = events.dataframe['duration'].max()
                #offsets = events.dataframe['xmin'] + tmax
                fixed_duration = tmax - tmin
            elif tmax == 'unequal':
                assert 'xmax' in events.dataframe, 'If tmax is uniqual, xmax needs to be provided per event'
                fixed_duration = None
                self.offsets = events.dataframe['xmax']
            else:
                raise NotImplementedError
        elif type(tmax) == int or type(tmax) == float:
            tmax = float(tmax)
            #offsets = events.dataframe['xman'] + tmax
            fixed_duration = tmax - tmin
        else:
            raise NotImplementedError

        self.fixed_duration = fixed_duration

        #sec2ind = lambda s: int(round(s * raw.sampling_rate))
        sr = raw.sampling_rate
        for k in self.data_.keys():
            if fixed_duration is not None:
                for ons in self.onsets:
                    if sec2ind(ons, sr) < 0 or sec2ind(ons, sr) + sec2ind(fixed_duration, sr) > len(raw.data[k]):
                        warnings.warn('Index out of range, wrapping the array')
                    self.data_[k].append(raw.data[k].take(range(sec2ind(ons, sr),
                                                                sec2ind(ons, sr) + sec2ind(fixed_duration, sr)),
                                                          axis=0, mode='wrap'))
                    # self.data_[k].append(raw.data[k][max(sec2ind(ons, sr), 0):sec2ind(ons, sr) + sec2ind(fixed_duration, sr)])
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
        # xxx = xx.reshape((xx.shape[0], xx.shape[1], -1)) # stack channels and feature sets: Lennart has default order C
        # xxxx = xxx.reshape((xx.shape[0], -1), order='F') # stack over timestamps: Lennart has order F

        #self.data =  xxxx
        self.data = xx
        self.feature_names = names


