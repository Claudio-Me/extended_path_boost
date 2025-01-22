import pickle
import os
import numpy as np


def get_full_nx_dataset():
    file_path = os.path.dirname(__file__)
    file_path = os.path.join(file_path, 'examples_data')
    file_path = os.path.join(file_path, 'uNatQ_nx_graphs.pkl')
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def get_y(graphs=None):
    if graphs is None:
        graphs = get_full_nx_dataset()
    y = np.array([graph.graph['target_tzvp_homo_lumo_gap'] for graph in graphs])
    return y


def get_full_nx_dataset_with_y():
    graphs = get_full_nx_dataset()
    y = get_y(graphs=graphs)
    return graphs, y
