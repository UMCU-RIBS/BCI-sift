# -------------------------------------------------------------
# BCI-sift
# Copyright (c) 2025
#       Dirk Keller,
#       Elena Offenberg,
#       Nick Ramsey's Lab, University Medical Center Utrecht, University Utrecht
# Licensed under the MIT License [see LICENSE for detail]
# -------------------------------------------------------------

'''
Load and save configuration files, and other utility functions for the dataset.
'''
import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP

def load_config(config_path):
    '''Load a configuration file from a given path. The configuration file needs to be in yaml format.'''
    assert Path.exists(config_path), f'{config_path} does not exist'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def save_config(config, save_path):
    '''Save a configuration file to a given path. The configuration file is saved in yaml format.'''
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

def load_by_id(id, is_alias=False, data_path=None):
    '''Load a configuration file for a given subject id or alias. The configuration file needs to be in yaml format and located in the data path. 
    The name of the configuration file needs to be the same as the subject id or alias, with the extension .yml. If is_alias is True, the id is 
    treated as an alias and the corresponding subject id is looked up in the subjects.json file located in the data path.'''
    data_path = Path(data_path)
    if is_alias:
        subject = get_subject_by_alias(id)
    else:
        subject = id
    config_path = data_path / Path(subject + '.yml')
    return load_config(config_path)

def sec2ind(s, sr):
    '''Convert seconds to sample indices, given a sampling rate.'''
    return int(Decimal(s * sr).quantize(0, ROUND_HALF_UP))

def write_config(save_path, config):
    '''Write a configuration file to a given path. The configuration file is saved in yaml format.'''
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

def get_subject_by_alias(alias, data_path=None):
    '''Get the subject id corresponding to a given alias. The mapping between aliases and subject ids is stored in the subjects.json file located in the data path.'''
    subject_file = Path(f'{data_path}subjects.json')
    subjects = json.load(open(subject_file, 'r'))
    return subjects[alias]

def get_subjects(data_path=None):
    '''Get a list of all subject ids. The subject ids are stored in the subjects.json file located in the data path.'''
    subject_file = Path(f'{data_path}subjects.json')
    return json.load(open(subject_file, 'r'))

def get_channels_numeric(labels):
    '''Get the numeric channel indices from a list of channel labels. The channel labels need to be in the format 'chanX' or 'chX', where X is the channel index. 
    The function returns a list of numeric channel indices, where the channel index is 0-based (i.e., chan1 corresponds to index 0).'''
    try:
        channels = [int(s.split('chan')[1]) - 1 for s in labels]
    except:
        channels = [int(s.split('ch')[1]) - 1 for s in labels]
    return channels

def get_bad_on_grid(grid, labels):
    '''Get the channel indices that are present in the grid but not in the labels. The grid is a 2D array where each element is either a channel index or -1 
    (indicating no channel).'''
    grid_flat = np.delete(grid.flatten(), np.where(grid.flatten() == -1))
    channels = get_channels_numeric(labels)
    bad = grid_flat[np.where(np.in1d(grid_flat, channels) == False)[0]]
    return bad

def map_values_to_grid(values, grid, labels):
    '''Map a list of values to a grid based on the channel labels. The grid is a 2D array where each element is either a channel index or -1 (indicating no channel). 
    The labels are a list of channel labels in the format 'chanX' or 'chX', where X is the channel index. The function returns a 1D array of values, where the value 
    at each index corresponds to the channel index in the grid. If a channel index in the grid is not present in the labels, the corresponding value is set to NaN.'''
    grid_flat = np.delete(grid.flatten(), np.where(grid.flatten() == -1))
    values_ = np.zeros(len(grid_flat))
    bad = get_bad_on_grid(grid, labels)
    values_[bad] = np.nan
    values_[np.setdiff1d(range(values_.shape[0]), bad)] = values
    return values_


def data2csv(save_path, save_name, **kwargs):
    '''Save a dictionary of data to a csv file. The keys of the dictionary correspond to the column names, and the values correspond to the column values. 
    The csv file is saved to the specified path with the specified name.'''
    save_path.mkdir(parents=True, exist_ok=True)
    out2 = pd.DataFrame()
    for k, v in kwargs.items():
        out2[k] = v
    out2.to_csv(save_path / f'{save_name}.csv')


def dict2json(save_path, save_name, d):
    '''Save a dictionary to a json file. The json file is saved to the specified path with the specified name.'''
    save_path.mkdir(parents=True, exist_ok=True)
    if type(d) == dict:
        d_ = d
    elif hasattr(d, '__dict__'):
        d_ = d.__dict__
    else:
        raise NotImplementedError
    with open(save_path / f'{save_name}.json', 'w') as fp:
        json.dump(d_, fp, indent=4)



