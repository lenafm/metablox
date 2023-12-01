"""Utility functions for description length and gamma calculations"""

import graph_tool.all as gt
import numpy as np
import scipy.special as sp
from typing import Union


def lbinom(n: int, k: int) -> Union[float, int]:
    """
    Calculate the logarithm of the binomial coefficient (n choose k).

    Args:
        n (int): The total number of items.
        k (int): The number of items to choose.

    Returns:
        Union[float, int]: The logarithm of the binomial coefficient.

    Notes:
        - Uses the safelog function to safely compute logarithms.
    """
    return safelog(sp.binom(n, k))


def safelog(x: Union[float, np.ndarray], base: Union[float, int] = np.e) -> Union[float, np.ndarray]:
    """
    Safely compute the logarithm, handling cases where the input is zero.

    Args:
        x (Union[float, np.ndarray]): The input value(s).
        base (Union[float, int]): The base of the logarithm (default: np.e).

    Returns:
        Union[float, np.ndarray]: The logarithm of the input value(s).

    Notes:
        - Returns 0 if the input is zero to avoid undefined results.
    """
    if x == 0:
        return 0
    if base == 2:
        return np.log2(x)
    elif base == np.e:
        return np.log(x)
    elif base == 10:
        return np.log10(x)


def xlogx(x: Union[float, np.ndarray], base: float = np.e) -> Union[float, np.ndarray]:
    """
    Calculate x * log(x) for a given base.

    Args:
        x (Union[float, np.ndarray]): The input value(s).
        base (float): The base of the logarithm (default: np.e).

    Returns:
        Union[float, np.ndarray]: The result of x * log(x).

    Notes:
        - Returns 0 if the input is zero to avoid undefined results.
    """
    return x * safelog(x, base=base)


def num_partitions(n: int, k: int) -> int:
    """
    Returns the number of partitions of integer n into at most k parts.

    Args:
        n (int): The integer to be partitioned.
        k (int): The maximum number of parts in the partition.

    Returns:
        int: The number of partitions of n into at most k parts.

    Notes:
        - Uses dynamic programming to calculate the number of partitions.
    """
    # Initialize table with base cases
    table = [[1] * (n + 1) for _ in range(k + 1)]

    # Fill in table using dynamic programming
    for i in range(2, k + 1):
        for j in range(1, n + 1):
            if j >= i:
                table[i][j] = table[i - 1][j] + table[i][j - i]
            else:
                table[i][j] = table[i - 1][j]

    return table[k][n]


def str_to_int(strings):
    """
    Map strings to integers based on their order of appearance.

    Args:
        strings: A list or array of strings.

    Returns:
        A dictionary mapping each unique string to its corresponding integer value.
        The integers are assigned based on the order of appearance in the input array.
    """

    mapping = str_to_int_mapping(strings)
    mapped_values = [mapping[string] for string in strings]
    return mapped_values


def str_to_int_mapping(strings):
    unique_strings = list(set(strings))
    return {string: index for index, string in enumerate(unique_strings)}


def make_list(item):
    """
    Checks if an item is a list and converts it to a list if it is not.

    Args:
        item: The item to check.

    Returns:
        A list containing the item, either as a single element or as the original list.

    """
    if isinstance(item, list):
        return item
    else:
        return [item]


def flatten_list(ls):
    return [item for row in ls for item in row]


def is_multigraph(g):
    """
    Check whether a network is a multigraph or not.

    Args:
        graph: A graph_tool.Graph object.

    Returns:
        True if the graph is a multigraph, False otherwise.
    """
    edge_counts = {}

    for edge in g.edges():
        source = int(edge.source())
        target = int(edge.target())

        if (source, target) in edge_counts:
            edge_counts[(source, target)] += 1
        else:
            edge_counts[(source, target)] = 1

    return any(count > 1 for count in edge_counts.values())


def simplify_multigraph(multigraph):
    """
    Convert an undirected multigraph into a simple graph by removing duplicate edges.
    Preserve vertex properties from the input graph.

    Args:
        multigraph: A graph_tool.Graph object representing the undirected multigraph.

    Returns:
        A new graph_tool.Graph object representing the simplified graph with preserved vertex properties.
    """
    simple_graph = gt.Graph(directed=False)
    node_dict = {}  # Mapping of original node ids to new node ids

    # Copy vertex properties from the input multigraph to the simplified graph
    for prop_name, prop_value in multigraph.vp.items():
        simple_graph.vp[prop_name] = simple_graph.new_vertex_property(prop_value.value_type())

    for v in multigraph.vertices():
        new_v = simple_graph.add_vertex()

        for prop_name, prop_value in multigraph.vp.items():
            simple_graph.vp[prop_name][new_v] = prop_value[v]

        node_dict[int(v)] = new_v

    for edge in multigraph.edges():
        source = int(edge.source())
        target = int(edge.target())
        new_source = node_dict[source]
        new_target = node_dict[target]

        # Add the edge to the new graph if it doesn't exist already
        if not simple_graph.edge(new_source, new_target):
            simple_graph.add_edge(new_source, new_target)

    return simple_graph

