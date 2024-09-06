'''
Load private info from files
'''
import json
import yaml
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP

def load_config(config_path):
    assert Path.exists(config_path), f'{config_path} does not exist'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_by_id(id, is_alias=False, data_path=None):
    data_path = Path(data_path)
    if is_alias:
        subject = get_subject_by_alias(id)
    else:
        subject = id
    config_path = data_path / Path(subject + '.yml')
    return load_config(config_path)

def sec2ind(s, sr):
    return int(Decimal(s * sr).quantize(0, ROUND_HALF_UP))

def write_config(save_path, config):
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

def get_subject_by_alias(alias, data_path=None):
    subject_file = Path(f'{data_path}subjects.json')
    subjects = json.load(open(subject_file, 'r'))
    return subjects[alias]

def get_subjects(data_path=None):
    subject_file = Path(f'{data_path}subjects.json')
    return json.load(open(subject_file, 'r'))

def get_channels_numeric(labels):
    try:
        channels = [int(s.split('chan')[1]) - 1 for s in labels]
    except:
        channels = [int(s.split('ch')[1]) - 1 for s in labels]
    return channels

def get_bad_on_grid(grid, labels):
    grid_flat = np.delete(grid.flatten(), np.where(grid.flatten() == -1))
    channels = get_channels_numeric(labels)
    bad = grid_flat[np.where(np.in1d(grid_flat, channels) == False)[0]]
    return bad

def map_values_to_grid(values, grid, labels):
    grid_flat = np.delete(grid.flatten(), np.where(grid.flatten() == -1))
    values_ = np.zeros(len(grid_flat))
    bad = get_bad_on_grid(grid, labels)
    values_[bad] = np.nan
    values_[np.setdiff1d(range(values_.shape[0]), bad)] = values
    return values_


def data2csv(save_path, save_name, **kwargs):
    save_path.mkdir(parents=True, exist_ok=True)
    out2 = pd.DataFrame()
    for k, v in kwargs.items():
        out2[k] = v
    out2.to_csv(save_path / f'{save_name}.csv')


def dict2json(save_path, save_name, d):
    save_path.mkdir(parents=True, exist_ok=True)
    if type(d) == dict:
        d_ = d
    elif hasattr(d, '__dict__'):
        d_ = d.__dict__
    else:
        raise NotImplementedError
    with open(save_path / f'{save_name}.json', 'w') as fp:
        json.dump(d_, fp, indent=4)



