"""Prepare fixtures for testing"""

import pytest
import numpy as np
import graph_tool.all as gt

from metablox.datasets import (
    load_kc_graph,
    load_kc_state_sbm,
    load_kc_state_dcsbm,
    load_kc_state_dcsbm_single_node,
    load_football_graph,
    load_football_state_pp,
)


@pytest.fixture()
def kc_graph() -> gt.Graph:
    return load_kc_graph()


@pytest.fixture()
def kc_sbm_blockstate() -> gt.BlockState:
    return load_kc_state_sbm()


@pytest.fixture()
def kc_dcsbm_blockstate() -> gt.BlockState:
    return load_kc_state_dcsbm()


@pytest.fixture()
def kc_dcsbm_blockstate_single_node() -> gt.BlockState:
    return load_kc_state_dcsbm_single_node()


@pytest.fixture()
def kc_sbm_partition() -> np.array:
    state = load_kc_state_sbm()
    return np.array(state.get_blocks().a)


@pytest.fixture()
def kc_dcsbm_partition() -> np.array:
    state = load_kc_state_dcsbm()
    return np.array(state.get_blocks().a)


@pytest.fixture()
def kc_dcsbm_partition_single_node() -> np.array:
    state = kc_dcsbm_blockstate_single_node()
    return np.array(state.get_blocks().a)


@pytest.fixture()
def football_graph() -> gt.Graph:
    return load_football_graph()


@pytest.fixture()
def football_pp_blockstate() -> gt.BlockState:
    return load_football_state_pp()


@pytest.fixture()
def football_pp_partition() -> np.array:
    state = load_football_state_pp()
    return np.array(state.get_blocks().a)
