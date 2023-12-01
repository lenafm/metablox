"""Tests for description length calculations"""

import pytest

import graph_tool.all as gt

from metablox.dl import (
    sbm_entropy,
    dl_partition,
    dl_degree_sequence,
    dl_edge_counts,
    calculate_dl,
)
from tests.fixtures import (
    kc_graph,
    kc_sbm_blockstate,
    kc_dcsbm_blockstate,
    kc_sbm_partition,
    kc_dcsbm_partition,
    football_graph,
    football_pp_blockstate,
    football_pp_partition,
)

multigraph = False
dense = False
exact = True
recs = False
recs_dl = False
Bfield = False


###############################################################################
# Tests of individual components of description length


def test_entropy_sbm(kc_graph, kc_sbm_blockstate, kc_sbm_partition):
    dc = False
    expected_entropy = kc_sbm_blockstate.entropy(adjacency=True, dl=False,
                                                 multigraph=multigraph, dense=dense, exact=exact,
                                                 deg_entropy=dc,
                                                 recs=recs, recs_dl=recs_dl, Bfield=Bfield)
    entropy = sbm_entropy(g=kc_graph, b=kc_sbm_partition, dc=dc)
    assert entropy == pytest.approx(expected_entropy)


def test_entropy_dcsbm(kc_graph, kc_dcsbm_blockstate, kc_dcsbm_partition):
    dc = True
    expected_entropy = kc_dcsbm_blockstate.entropy(adjacency=True, dl=False,
                                                   multigraph=multigraph, dense=dense, exact=exact,
                                                   deg_entropy=dc,
                                                   recs=recs, recs_dl=recs_dl, Bfield=Bfield)
    entropy = sbm_entropy(g=kc_graph, b=kc_dcsbm_partition, dc=dc)
    assert entropy == pytest.approx(expected_entropy)


def test_entropy_partition(kc_graph, kc_dcsbm_blockstate, kc_dcsbm_partition):
    expected_entropy = kc_dcsbm_blockstate.entropy(adjacency=False, dl=True,
                                                   partition_dl=True, edges_dl=False, degree_dl=False,
                                                   multigraph=multigraph, dense=dense, exact=exact,
                                                   deg_entropy=True,  # the deg_entropy parameter does not matter here
                                                   recs=recs, recs_dl=recs_dl, Bfield=Bfield)
    entropy = dl_partition(g=kc_graph, b=kc_dcsbm_partition)
    assert entropy == pytest.approx(expected_entropy)


def test_entropy_edges(kc_graph, kc_dcsbm_blockstate, kc_dcsbm_partition):
    expected_entropy = kc_dcsbm_blockstate.entropy(adjacency=False, dl=True,
                                                   partition_dl=False, edges_dl=True, degree_dl=False,
                                                   multigraph=multigraph, dense=dense, exact=exact,
                                                   deg_entropy=True,
                                                   recs=recs, recs_dl=recs_dl, Bfield=Bfield)
    entropy = dl_edge_counts(g=kc_graph, b=kc_dcsbm_partition)
    assert entropy == pytest.approx(expected_entropy)


def test_entropy_degree_uniform(kc_graph, kc_dcsbm_blockstate, kc_dcsbm_partition):
    degree_dl_kind = 'uniform'
    expected_entropy = kc_dcsbm_blockstate.entropy(adjacency=False, dl=True,
                                                   partition_dl=False, edges_dl=False, degree_dl=True,
                                                   multigraph=multigraph, dense=dense, exact=exact,
                                                   deg_entropy=True,
                                                   degree_dl_kind=degree_dl_kind,
                                                   recs=recs, recs_dl=recs_dl, Bfield=Bfield)
    entropy = dl_degree_sequence(g=kc_graph, b=kc_dcsbm_partition, degree_dl_kind=degree_dl_kind)
    assert entropy == pytest.approx(expected_entropy)


def test_entropy_degree_distributed(kc_graph, kc_dcsbm_blockstate, kc_dcsbm_partition):
    degree_dl_kind = 'distributed'
    expected_entropy = kc_dcsbm_blockstate.entropy(adjacency=False, dl=True,
                                                   partition_dl=False, edges_dl=False, degree_dl=True,
                                                   multigraph=multigraph, dense=dense, exact=exact,
                                                   deg_entropy=True,
                                                   degree_dl_kind=degree_dl_kind,
                                                   recs=recs, recs_dl=recs_dl, Bfield=Bfield)
    entropy = dl_degree_sequence(g=kc_graph, b=kc_dcsbm_partition, degree_dl_kind=degree_dl_kind)
    assert entropy == pytest.approx(expected_entropy, abs=0.6)


###############################################################################
# Tests of complete description length calculations


def test_dl_pp_uniform_distributed_degree_dl_kind(football_graph, football_pp_blockstate, football_pp_partition):
    uniform = True
    degree_dl_kind = 'distributed'
    expected_entropy = football_pp_blockstate.entropy(uniform=uniform, degree_dl_kind=degree_dl_kind)
    entropy = calculate_dl(g=football_graph, b=football_pp_partition, blockstate='PPBlockState',
                           uniform=uniform, degree_dl_kind=degree_dl_kind)
    assert entropy == pytest.approx(expected_entropy, abs=2)


def test_dl_pp_nonuniform_distributed_degree_dl_kind(football_graph, football_pp_blockstate, football_pp_partition):
    uniform = False
    degree_dl_kind = 'distributed'
    expected_entropy = football_pp_blockstate.entropy(uniform=uniform, degree_dl_kind=degree_dl_kind)
    entropy = calculate_dl(g=football_graph, b=football_pp_partition, blockstate='PPBlockState',
                           uniform=uniform, degree_dl_kind=degree_dl_kind)
    assert entropy == pytest.approx(expected_entropy, abs=2)


def test_dl_pp_uniform_uniform_degree_dl_kind(football_graph, football_pp_blockstate, football_pp_partition):
    uniform = True
    degree_dl_kind = 'uniform'
    expected_entropy = football_pp_blockstate.entropy(uniform=uniform, degree_dl_kind=degree_dl_kind)
    entropy = calculate_dl(g=football_graph, b=football_pp_partition, blockstate='PPBlockState',
                           uniform=uniform, degree_dl_kind=degree_dl_kind)
    assert entropy == pytest.approx(expected_entropy)


def test_dl_pp_nonuniform_uniform_degree_dl_kind(football_graph, football_pp_blockstate, football_pp_partition):
    uniform = False
    degree_dl_kind = 'uniform'
    expected_entropy = football_pp_blockstate.entropy(uniform=uniform, degree_dl_kind=degree_dl_kind)
    entropy = calculate_dl(g=football_graph, b=football_pp_partition, blockstate='PPBlockState',
                           uniform=uniform, degree_dl_kind=degree_dl_kind)
    assert entropy == pytest.approx(expected_entropy)


def test_dl_dcsbm_uniform_degree_dl_kind(kc_graph, kc_dcsbm_blockstate, kc_dcsbm_partition):
    degree_dl_kind = 'uniform'
    dc = True
    expected_entropy = kc_dcsbm_blockstate.entropy(adjacency=True, dl=True,
                                                   multigraph=multigraph, dense=dense, exact=exact,
                                                   degree_dl_kind=degree_dl_kind,
                                                   deg_entropy=dc,
                                                   recs=recs, recs_dl=recs_dl, Bfield=Bfield)
    entropy = calculate_dl(g=kc_graph, b=kc_dcsbm_partition,
                           dc=dc, degree_dl_kind=degree_dl_kind)
    assert entropy == pytest.approx(expected_entropy)


def test_dl_dcsbm_distributed_degree_dl_kind(kc_graph, kc_dcsbm_blockstate, kc_dcsbm_partition):
    degree_dl_kind = 'distributed'
    dc = True
    expected_entropy = kc_dcsbm_blockstate.entropy(adjacency=True, dl=True,
                                                   multigraph=multigraph, dense=dense, exact=exact,
                                                   degree_dl_kind=degree_dl_kind,
                                                   deg_entropy=dc,
                                                   recs=recs, recs_dl=recs_dl, Bfield=Bfield)
    entropy = calculate_dl(g=kc_graph, b=kc_dcsbm_partition,
                           dc=dc, degree_dl_kind=degree_dl_kind)
    assert entropy == pytest.approx(expected_entropy, abs=0.6)


def test_dl_sbm_uniform_degree_dl_kind(kc_graph, kc_sbm_blockstate, kc_sbm_partition):
    degree_dl_kind = 'uniform'
    dc = False
    expected_entropy = kc_sbm_blockstate.entropy(adjacency=True, dl=True,
                                                 multigraph=multigraph, dense=dense, exact=exact,
                                                 degree_dl_kind=degree_dl_kind,
                                                 deg_entropy=dc,
                                                 recs=recs, recs_dl=recs_dl, Bfield=Bfield)
    entropy = calculate_dl(g=kc_graph, b=kc_sbm_partition,
                           dc=dc, degree_dl_kind=degree_dl_kind)
    assert entropy == pytest.approx(expected_entropy)


def test_dl_sbm_distributed_degree_dl_kind(kc_graph, kc_sbm_blockstate, kc_sbm_partition):
    degree_dl_kind = 'distributed'
    dc = False
    expected_entropy = kc_sbm_blockstate.entropy(adjacency=True, dl=True,
                                                 multigraph=multigraph, dense=dense, exact=exact,
                                                 degree_dl_kind=degree_dl_kind,
                                                 deg_entropy=dc,
                                                 recs=recs, recs_dl=recs_dl, Bfield=Bfield)
    entropy = calculate_dl(g=kc_graph, b=kc_sbm_partition,
                           dc=dc, degree_dl_kind=degree_dl_kind)
    assert entropy == pytest.approx(expected_entropy, abs=0.5)
