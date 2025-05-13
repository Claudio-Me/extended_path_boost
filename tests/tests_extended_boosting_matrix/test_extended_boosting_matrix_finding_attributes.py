import networkx as nx
from extended_path_boost.utils.classes.extended_boosting_matrix import ExtendedBoostingMatrix
from tests.datasets_used_for_tests import load_test_dataset


def test_on_molecules_dataset():
    dataset = load_test_dataset.get_nx_test_dataset()
    graph = dataset[0]
    path = ('0', '2')
    result = ExtendedBoostingMatrix.get_attributes_of_last_part_of_the_path(graph, path)
    assert result == {'feature_antibond_energy_min_max_difference': 0.0,
                      'feature_antibond_min_d_occupation': 0.018285833333333334, 'feature_antibond_min_energy': 0.0,
                      'feature_antibond_min_occupation': 0.0, 'feature_antibond_min_p_occupation': 0.6547841666666667,
                      'feature_antibond_min_s_occupation': 0.3266175, 'feature_atomic_number': 16,
                      'feature_bond_distance': 2.4588081030430167, 'feature_bond_energy_min_max_difference': 0.0,
                      'feature_bond_max_d_occupation': 0.018285833333333334, 'feature_bond_max_energy': 0.0,
                      'feature_bond_max_occupation': 0.0, 'feature_bond_max_p_occupation': 0.6547841666666667,
                      'feature_bond_max_s_occupation': 0.3266175, 'feature_hydrogen_count': 0,
                      'feature_lone_pair_energy_min_max_difference': 0.27125000000000005,
                      'feature_lone_pair_max_d_occupation': 0.0028000000000000004,
                      'feature_lone_pair_max_energy': -0.2434, 'feature_lone_pair_max_occupation': 1.84104,
                      'feature_lone_pair_max_p_occupation': 0.9959, 'feature_lone_pair_max_s_occupation': 0.0012,
                      'feature_lone_vacancy_energy_min_max_difference': 0.0,
                      'feature_lone_vacancy_min_d_occupation': 0.0, 'feature_lone_vacancy_min_energy': 0.0,
                      'feature_lone_vacancy_min_occupation': 0.0, 'feature_lone_vacancy_min_p_occupation': 0.0,
                      'feature_lone_vacancy_min_s_occupation': 0.0, 'feature_n_bn': 0, 'feature_n_lone_pairs': 3,
                      'feature_n_lone_vacancies': 0, 'feature_n_nbn': 0, 'feature_natural_atomic_charge': -0.38134,
                      'feature_natural_electron_configuration_d_occupation': 0.03,
                      'feature_natural_electron_configuration_p_occupation': 4.57,
                      'feature_natural_electron_configuration_s_occupation': 1.78,
                      'feature_natural_electron_population_valence': 6.34742, 'feature_wiberg_bond_order': 0.3398}


def test_get_path_attributes_with_node_and_edge_attributes():
    graph = nx.Graph()
    graph.add_nodes_from([
        (1, {"attr1": 1}),
        (2, {"attr2": 2})
    ])
    graph.add_edge(1, 2, attr3=3)
    path = [1, 2]
    result = ExtendedBoostingMatrix.get_attributes_of_last_part_of_the_path(graph, path)

    assert result == {"attr2": 2, "attr3": 3}


def test_get_path_attributes_with_no_attributes():
    graph = nx.Graph()
    graph.add_nodes_from([1, 2])
    graph.add_edge(1, 2)
    path = [1, 2]
    result = ExtendedBoostingMatrix.get_attributes_of_last_part_of_the_path(graph, path)
    assert result == {}


def test_get_path_attributes_with_single_node():
    graph = nx.Graph()
    graph.add_node(1, attr1=1)
    graph.add_node(2, attr2=2)
    graph.add_edge(1, 2, attr3=3)
    path = [1]
    result = ExtendedBoostingMatrix.get_attributes_of_last_part_of_the_path(graph, path)
    assert result == {"attr1": 1}


def test_get_path_same_attributes_name():
    # this is a borderline situation, the behaviour might change in the future
    graph = nx.Graph()
    graph.add_node(1, attr1=1)
    graph.add_node(2, attr1=2)
    graph.add_edge(1, 2, attr1=3)
    path = [1, 2]
    result = ExtendedBoostingMatrix.get_attributes_of_last_part_of_the_path(graph, path)
    assert result == {"attr1": 3}


def test_get_path_attributes_complex_graph():
    graph = nx.Graph()
    graph.add_nodes_from([
        (1, {"attr1": 1}),
        (2, {"attr2": 2}),
        (3, {"attr3": 3}),
        (4, {"attr4": 4}),
        (5, {"attr5": 5})
    ])
    graph.add_edges_from([
        (1, 2, {"attr6": 6}),
        (2, 3, {"attr7": 7}),
        (3, 4, {"attr8": 8}),
        (4, 5, {"attr9": 9}),
        (1, 5, {"attr10": 10})
    ])
    path = [1, 2, 3, 4, 5]
    result = ExtendedBoostingMatrix.get_attributes_of_last_part_of_the_path(graph, path)
    assert result == {"attr5": 5, "attr9": 9}
