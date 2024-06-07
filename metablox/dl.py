"""Description length functions, implementations from the graph tool library:
Tiago P. Peixoto, “The graph-tool python library”, figshare. (2014) DOI: 10.6084/m9.figshare.1164194
The code in the library is primarily based on the following two papers:
    1) Peixoto, Tiago P. "Nonparametric Bayesian inference of the microcanonical stochastic block model."
    Physical Review E 95.1 (2017): 012317.
    2) Zhang, Lizhi, and Tiago P. Peixoto. "Statistical inference of assortative community structures."
    Physical Review Research 2.4 (2020): 043271.
"""

import graph_tool.all as gt
import numpy as np
from math import lgamma
from collections import defaultdict
from typing import Union, Tuple, Optional, Dict, Type

from metablox.utils import lbinom, safelog, num_partitions


def calculate_dl(state: Union[gt.BlockState, gt.PPBlockState] = None,
                 g: Optional[gt.Graph] = None,
                 b: Optional[gt.VertexPropertyMap] = None,
                 dc: bool = True,
                 blockstate: str = 'BlockState',
                 uniform: bool = False,
                 degree_dl_kind: str = 'distributed') -> float:
    """
    Calculate the description length of a graph under the stochastic block model.

    Args:
        state (Union[gt.BlockState, gt.PPBlockState]): A block state representing the partition of the graph.
        g (Optional[gt.Graph]): The input graph.
        b (Optional[gt.VertexPropertyMap]): The vertex property map representing the partition.
        dc (bool): Flag indicating whether to use degree correction.
        blockstate (str): Type of block state to use. Should be one of 'BlockState' or 'PPBlockState'.
        uniform (bool): Flag indicating whether to use uniform edge count entropy.
        degree_dl_kind (str): The kind of degree distribution used in entropy estimation.

    Returns:
        float: The description length of the graph.

    Raises:
        Exception: If neither state nor both graph g and partition b are specified.

    Notes:
        - The function computes various entropy terms and combines them to calculate the description length.
    """
    if state is not None:
        block_stats = get_block_stats(state=state, g=None, b=b)
    else:
        block_stats = get_block_stats(state=state, g=g, b=b)

    ent_sbm = sbm_entropy(block_stats=block_stats, g=g, dc=dc)
    ent_partition = dl_partition(block_stats=block_stats, g=g)
    ent_edge_count = dl_edge_counts(block_stats=block_stats, g=g, blockstate=blockstate, uniform=uniform)
    ent_degree_sequence = dl_degree_sequence(block_stats=block_stats, g=g, degree_dl_kind=degree_dl_kind)

    dl = ent_sbm + ent_partition + ent_edge_count
    if dc:
        dl += ent_degree_sequence
    return dl


def sbm_entropy(block_stats: Optional[tuple] = None,
                state: Optional[gt.BlockState] = None,
                g: Optional[gt.Graph] = None,
                b: Optional[gt.VertexPropertyMap] = None,
                dc: bool = False) -> float:
    """
    Calculate the entropy of a Stochastic Block Model (SBM). This is uses the sparse variant of the SBM entropy. It
    does not use Stirling's approximation for the factorials (i.e. it is the 'exact' version of the entropy).

    Args:
        block_stats (Optional[tuple]): Tuple containing block statistics.
        state (Optional[gt.BlockState]): A block state representing the partition of the graph.
        g (Optional[gt.Graph]): The input graph.
        b (Optional[gt.VertexPropertyMap]): The vertex property map representing the partition.
        dc (bool): Flag indicating whether to use degree correction.

    Returns:
        float: The entropy of the stochastic block model.

    Raises:
        Exception: If neither block_stats nor both graph g and partition b are specified.
    """
    if block_stats is None:
        if state is None:
            if b is None or g is None:
                raise Exception("Either state or both graph g and partition b must be specified.")
        graph_copy = g.copy() if state is None else None
        block_stats = get_block_stats(state=state, g=graph_copy, b=b)
    bg, b, N, E, B, ers, nr, er, ers_mat = block_stats

    S = 0
    for e in bg.edges():
        r = e.source()
        s = e.target()
        val = lgamma(ers[e] + 1)  # log(ers!)
        if r != s:
            S += -val
        else:
            S += -val - (ers[e] * safelog(2))

    for v in bg.vertices():
        if dc:
            S += lgamma(er[v] + 1)  # log(er!)
        else:
            S += er[v] * safelog(nr[v])  # er * log(nr)

    if dc:
        for v in g.vertices():
            k = v.out_degree()
            S += -lgamma(k + 1)

    return S


def get_block_stats(state: Optional[gt.BlockState] = None,
                    g: Optional[gt.Graph] = None,
                    b: Optional[gt.VertexPropertyMap] = None,
                    verbose: bool = True) -> tuple:
    """
    Get block statistics including block graph, partition, and relevant counts.

    Args:
        state (Optional[gt.BlockState]): A block state representing the partition of the graph.
        g (Optional[gt.Graph]): The input graph.
        b (Optional[gt.VertexPropertyMap]): The vertex property map representing the partition.
        verbose (bool): Flag indicating whether to display verbose output.

    Returns:
        tuple: Tuple containing block graph, partition, and relevant counts.

    Raises:
        Exception: If neither state nor both graph g and partition b are specified.
    """
    if state is None:
        if verbose:
            if g is None or b is None:
                raise Exception("Either state or both graph g and partition b must be specified.")
        if not isinstance(b, gt.VertexPropertyMap):
            if np.max(b) > len(set(b)) - 1:
                b = gt.contiguous_map(np.array(b))
            b_new = g.new_vp('int')
            b_new.a = np.array(b)
            b = b_new.copy()
        else:
            b = gt.contiguous_map(b)
        bg = get_block_graph(g, b)
        B = bg.num_vertices()
        N = g.num_vertices()
        E = g.num_edges()
        ers = bg.ep["count"]
        nr = bg.vp["count"]
        er = bg.degree_property_map("out", weight=ers)
        ers_mat = gt.adjacency(bg, ers)
    else:
        if verbose:
            if g is not None or b is not None:
                print('Graph g and or partition b was specified although state was specified - state is being used.')
        b = gt.contiguous_map(state.get_blocks())
        state = state.copy(b=b)
        bg = state.get_bg()
        B = state.get_B()
        N = state.get_N()
        ers = state.mrs
        nr = state.wr
        er = state.mrp
        E = sum(er.a) / 2
        ers_mat = state.get_matrix().todense()
    return bg, b, N, E, B, ers, nr, er, ers_mat


def get_block_graph(g: gt.Graph, b: gt.VertexPropertyMap) -> gt.Graph:
    """
    Get the block graph induced by a partition.

    Args:
        g (gt.Graph): The input graph.
        b (gt.VertexPropertyMap): The vertex property map representing the partition.

    Returns:
        gt.Graph: The block graph induced by the partition.
    """
    B = len(set(b))
    cg, br, vc, ec, av, ae = gt.condensation_graph(g, b,
                                                   self_loops=True)
    cg.vp.count = vc
    cg.ep.count = ec
    rs = np.setdiff1d(np.arange(B, dtype="int"), br.fa,
                      assume_unique=True)
    if len(rs) > 0:
        cg.add_vertex(len(rs))
        br.fa[-len(rs):] = rs

    cg = gt.Graph(cg, vorder=br)
    return cg


def dl_partition(block_stats: Optional[tuple] = None,
                 state: Optional[gt.BlockState] = None,
                 g: Optional[gt.Graph] = None,
                 b: Optional[gt.VertexPropertyMap] = None) -> float:
    """
    Calculate the entropy contribution from the partition in the description length.

    Args:
        block_stats (Optional[tuple]): Tuple containing block statistics.
        state (Optional[gt.BlockState]): A block state representing the partition of the graph.
        g (Optional[gt.Graph]): The input graph.
        b (Optional[gt.VertexPropertyMap]): The vertex property map representing the partition.

    Returns:
        float: The entropy contribution from the partition.

    Raises:
        Exception: If neither block_stats nor both graph g and partition b are specified.
    """
    if block_stats is None:
        graph_copy = g.copy() if state is None else None
        block_stats = get_block_stats(state=state, g=graph_copy, b=b)
    bg, b, N, E, B, ers, nr, er, ers_mat = block_stats
    S = 0
    S += lbinom(N - 1, B - 1)  # log(binom(N-1, B-1))
    S += lgamma(N + 1)  # log(N!)
    for n in nr.a:
        S -= lgamma(n + 1)  # -sum_r (log(nr!))
    S += safelog(N)
    return S


def dl_degree_sequence(block_stats: Optional[tuple] = None,
                       state: Optional[gt.BlockState] = None,
                       g: Optional[gt.Graph] = None,
                       b: Optional[gt.VertexPropertyMap] = None,
                       degree_dl_kind: str = 'uniform') -> float:
    """
    Calculate the entropy contribution from the degree sequence in the description length.

    Args:
        block_stats (Optional[tuple]): Tuple containing block statistics.
        state (Optional[gt.BlockState]): A block state representing the partition of the graph.
        g (Optional[gt.Graph]): The input graph.
        b (Optional[gt.VertexPropertyMap]): The vertex property map representing the partition.
        degree_dl_kind (str): The kind of degree distribution used in entropy estimation.

    Returns:
        float: The entropy contribution from the degree sequence.

    Raises:
        Exception: If neither block_stats nor both graph g and partition b are specified.
                  If degree_dl_kind is not one of 'uniform' or 'distributed'.
    """
    if block_stats is None:
        graph_copy = g.copy() if state is None else None
        block_stats = get_block_stats(state=state, g=graph_copy, b=b)
    bg, b, N, E, B, ers, nr, er, ers_mat = block_stats
    S = 0
    if degree_dl_kind == 'uniform':
        for r in bg.vertices():
            S += lbinom(nr[r] + er[r] - 1, er[r])
    elif degree_dl_kind == 'distributed':
        deg_hists = get_degree_histograms(state=state, g=g, b=b)
        for r in bg.vertices():
            hist = deg_hists[int(r)]
            for k in hist:
                S -= lgamma(k[1] + 1)
            q = num_partitions(er[r], nr[r])
            S += safelog(float(q))
            S += lgamma(nr[r] + 1)
    else:
        raise Exception("Degree dl kind must be one of 'uniform' or 'distributed'.")
    return S


def dl_edge_counts(block_stats: Optional[tuple] = None,
                   state: Optional[gt.BlockState] = None,
                   g: Optional[gt.Graph] = None,
                   b: Optional[Union[np.array, gt.VertexPropertyMap]] = None,
                   blockstate: str = 'BlockState',
                   uniform: bool = True) -> float:
    """
    Calculate the description length of edge counts in a Stochastic Blockmodel.

    Args:
        block_stats (Optional[tuple]): Tuple containing block statistics.
        state (Optional[gt.BlockState]): The block state of the graph.
        g (Optional[gt.Graph]): The graph.
        b (Optional[Union[np.array, gt.VertexPropertyMap]]): The vertex property map representing the partition of the graph.
        blockstate (str): Type of block state to use. Should be one of 'BlockState' or 'PPBlockState'.
        uniform (bool): Flag indicating whether to use uniform entropy estimation (default: True).

    Returns:
        float: The description length of edge counts in the Stochastic Blockmodel.

    Raises:
        Exception: If both state and graph/partition are not specified.

    """
    if block_stats is None:
        graph_copy = g.copy() if state is None else None
        block_stats = get_block_stats(state=state, g=graph_copy, b=b)
    bg, b, N, E, B, ers, nr, er, ers_mat = block_stats
    if blockstate == 'BlockState':
        NB = (B * (B + 1)) / 2
        S = lbinom(NB + E - 1, E)
    elif blockstate == 'PPBlockState':
        diag = np.diag_indices(ers_mat.shape[0])
        sum_diag = np.sum(ers_mat[diag])
        sum_off_diag = np.sum(ers_mat) - sum_diag
        e_in = sum_diag / 2
        e_out = sum_off_diag / 2
        S = 0
        if uniform:
            S -= lgamma(e_in + 1)
            S -= lgamma(e_out + 1)
            S += e_in * safelog(B)
            S += e_out * lbinom(B, 2)
            if B > 1:
                S += safelog(E + 1)
            for e in bg.edges():
                S += lgamma(ers[e] + 1)
        else:
            S -= lgamma(e_out + 1)
            S += e_out * lbinom(B, 2)
            if B > 1:
                S += safelog(E + 1)
            S += lbinom(B + e_in - 1, e_in)
            for e in bg.edges():
                r = e.source()
                s = e.target()
                if r != s:
                    S += lgamma(ers[e] + 1)
    else:
        raise Exception('Argument blockstate needs to be one of BlockState or PPBlockState.')
    return S


def get_degree_histograms(state: gt.BlockState = None, g: gt.Graph = None, b: Union[np.array, gt.VertexPropertyMap] = None) -> Dict:
    """
    Get the degree histograms for each block in the Stochastic Blockmodel.

    Args:
        state (gt.BlockState): The block state of the graph.
        g (gt.Graph): The graph.
        b (Union[np.array, gt.VertexPropertyMap]): The vertex property map representing the partition of the graph.

    Returns:
        Dict: Dictionary containing degree histograms for each block.

    Raises:
        Exception: If graph g is not specified.

    """
    if g is None:
        raise Exception("Need to specify graph g.")
    if state is None and b is None:
        raise Exception("Need to specify either state or partition b.")
    if b is None:
        b = state.b
    hist = defaultdict(dict)
    degrees = g.get_out_degrees(g.get_vertices())
    for v in g.vertices():
        r = b[v]
        if r not in hist: hist[r] = {}
        deg = degrees[int(v)]
        if deg not in hist[r]: hist[r][deg] = 0
        hist[r][deg] += 1
    hist = {k: [(key, val) for key, val in v.items()] for k, v in hist.items()}
    return hist
