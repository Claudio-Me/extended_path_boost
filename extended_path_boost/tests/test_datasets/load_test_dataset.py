import pickle
import os

def get_nx_test_dataset():
    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, 'uNatQ_nx_graphs_subsample.pkl')
    with open(file_path, 'rb') as f:
        return pickle.load(f)