import pickle
import os
import numpy as np


def get_nx_test_dataset():
    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, 'uNatQ_nx_graphs_subsample.pkl')
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def get_y():
    graphs = get_nx_test_dataset()
    y = np.array([graph.graph['target_tzvp_homo_lumo_gap'] for graph in graphs])
    return y