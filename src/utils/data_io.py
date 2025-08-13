import json
import numpy as np
import pickle


def load_json_config(path_to_file):
    with open(path_to_file, "r") as f:
        data = json.load(f)
    return data


def load_npy(path_to_file):
    with open(path_to_file, 'rb') as f:
        data = np.load(f)
    return data


def load_pickle_dict(path_to_file):
    with open(path_to_file, 'rb') as f:
        data = pickle.load(f)
    return data
