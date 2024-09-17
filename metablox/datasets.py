"""Example datasets"""

import pickle
from os.path import dirname, join
from graph_tool.all import load_graph


def _get_dataset_file_path(file_name: str) -> str:
    dataset_dir = join(dirname(__file__), '..', 'datasets')
    file_path = join(dataset_dir, file_name)
    return file_path


def _load_pickle(filename):
    with open(filename, 'rb') as f:
        pickledfile = pickle.load(f)
    return pickledfile


def load_scbm_graph():
    """
    Load and return the scbm graph used in the metablox paper, generated using the scbm model in
    Mangold, Lena, and Camille Roth. "Generative models for two-ground-truth partitions in networks." Physical Review E
    108.5 (2023): 054308.

    :return: graphml object from graph tool library
    """

    file_path = _get_dataset_file_path('g_scbm.graphml')
    return load_graph(file_path)


def load_kc_graph():
    """
    Load and return the karate club graph.

    :return: graphml object from graph tool library
    """

    file_path = _get_dataset_file_path('g_kc.graphml')
    return load_graph(file_path)


def load_football_graph():
    """
    Load and return the football graph.

    :return: graphml object from graph tool library
    """

    file_path = _get_dataset_file_path('g_football.graphml')
    return load_graph(file_path)


def load_kc_state_sbm():
    """
    Load and return an sbm block state of the karate club.

    :return: blockstate object from graph tool library
    """

    file_path = _get_dataset_file_path('state_kc_sbm.pickle')
    return _load_pickle(file_path)


def load_kc_state_dcsbm():
    """
    Load and return a dcsbm block state of the karate club.

    :return: blockstate object from graph tool library
    """

    file_path = _get_dataset_file_path('state_kc_dcsbm.pickle')
    return _load_pickle(file_path)


def load_football_state_pp():
    """
    Load and return an pp block state of the football network.

    :return: blockstate object from graph tool library
    """

    file_path = _get_dataset_file_path('state_football_pp.pickle')
    return _load_pickle(file_path)
