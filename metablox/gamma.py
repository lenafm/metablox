"""Metablox calculations"""

import copy
import numpy as np
import graph_tool.all as gt
from tqdm import tqdm
from metablox.dl import calculate_dl
from metablox.utils import is_multigraph, simplify_multigraph, make_list, str_to_int_mapping


def calculate_metadata_relevance(g, metadata, use_gt=True, allow_multigraphs=True, variants='all', iters_rand=100,
                                 new_metadata_names=None, uniform=False, degree_dl_kind='distributed',
                                 variants_infer='all', refine_states=False, iters_refine=1000, return_dls=False,
                                 return_states=False, output_format='dict', verbose=True):
    """
    Calculates the gamma values for each metadata attribute based on the graph and metadata.

    Parameters:
    ----------
    g : graph_tool.Graph
        A graph_tool.Graph object representing the graph.

    metadata : list
        The metadata input, which can be a list of strings or a list of NumPy arrays.

    use_gt : bool, optional
        Flag indicating if the description length calculation from the graph-tool library should be used
        (default: True).

    allow_multigraphs : bool, optional
        Flag indicating if multigraphs should be allowed (can only be True if `use_gt=True`; default: True).

    variants : str or list, optional
        A string 'all' or a list of SBM variants ('dc', 'ndc', 'pp') to be included in the calculation of gamma
        (default: 'all').

    iters_rand : int, optional
        The number of random iterations for computing statistical significance (default: 100).

    new_metadata_names : list, optional
        Labels for metadata if metadata is a list of arrays (default: None).

    uniform : bool, optional
        Flag indicating whether to use uniform entropy estimation for the pp SBM (default: False).

    degree_dl_kind : str, optional
        The kind of degree distribution used in entropy estimation (default: 'distributed').

    variants_infer : str or list, optional
        A string 'all' or a list of SBM variants ('dc', 'ndc', 'pp') for inference of the optimal partition
        (default: 'all').

    refine_states : bool, optional
        Flag indicating whether to refine minimization of block states by running 10 * `iters_refine` sweeps of the
        MCMC with zero temperature (default: False).

    iters_refine : int, optional
        If `refine_states=True`, indicates the number of times the multiflip MCMC algorithm performs 10 sweeps.
        The total number of performed sweeps will be 10 * `iters_refine` (default: 1000).

    return_dls : bool, optional
        Flag indicating whether to return the description lengths of the metadata, random metadata, optimal partition,
        and blocklabel partition (if synthetic=True) (default: False).

    return_states : bool, optional
        Flag indicating whether to return the blockmodel states of the fitted SBMs (default: False).

    output_format : str, optional
        String indicating whether the output format should be dictionaries ('dict') or lists ('list'), (default: 'dict').

    verbose : bool, optional
        Flag indicating whether to display detailed output (default: True).

    Returns:
    -------
    dict
        A dictionary containing tuples of gamma values and edge compression for each metadata attribute and each
        variant, e.g.
        {
        'attr1': {'dc': gamma_value_dc_attr1, 'pp': gamma_value_pp_attr1},
        'attr2': {'dc': gamma_value_dc_attr2, 'pp': gamma_value_pp_attr2}
        }

    If `return_dls` and/or `return_states` are set to True, the function returns a tuple with the dictionary as the
    first element, and additional description lengths or states as the second element.
    """
    if not use_gt:
        if allow_multigraphs:
            raise ValueError('Can only allow for multigraphs if gt = True.')
    variants = check_variants(variants=variants)
    variants_infer = check_variants(variants=variants_infer)
    g, metadata = check_input(graph=g, metadata=metadata, allow_multigraphs=allow_multigraphs,
                              new_metadata_names=new_metadata_names, verbose=verbose)
    if verbose:
        tqdm.write("Minimizing block models and nested block models.")
    states = optimise_block_states(g=g, variants=variants_infer, refine=refine_states, iters_refine=iters_refine)
    states_dls = get_dls_states(states=states, allow_multigraphs=allow_multigraphs)
    if verbose:
        tqdm.write("Calculating description lengths for metadata.")
    meta_dls = calculate_meta_dls(g=g, metadata=metadata, variants=variants, uniform=uniform,
                                  degree_dl_kind=degree_dl_kind, use_gt=use_gt, allow_multigraphs=allow_multigraphs)
    if verbose:
        tqdm.write("Calculating description lengths for randomised metadata.")
    disable_progress_bar = not verbose
    meta_dls_randomised = calculate_meta_dls_randomised(g=g, metadata=metadata, iters_rand=iters_rand,
                                                        variants=variants, uniform=uniform,
                                                        degree_dl_kind=degree_dl_kind, use_gt=use_gt,
                                                        allow_multigraphs=allow_multigraphs,
                                                        disable_progress_bar=disable_progress_bar)

    if verbose:
        tqdm.write("Calculating gamma values.")
    gamma = compute_gamma_values(meta_dls=meta_dls, meta_dls_randomised=meta_dls_randomised, optimal_dl=states_dls,
                                 metadata=metadata, variants=variants)

    second_dim = compute_edge_compression(optimal_dl=states_dls, variants=variants, num_edges=g.num_edges())

    if output_format == 'list':
        gamma = {variant: [val[variant] for key, val in gamma.items()] for variant in variants}

    if return_dls:
        extra_output = {'meta_dls': meta_dls,
                        'meta_dls_randomised': meta_dls_randomised}
        if return_states:
            extra_output['states'] = states
        return gamma, second_dim, extra_output
    elif return_states:
        return gamma, second_dim, {'states': states}
    else:
        return gamma, second_dim


def compute_edge_compression(optimal_dl, variants, num_edges):
    """
    Computes the compression per edge for each variant by dividing the optimal description length (DL) by the total
    number of edges in the graph.

    Parameters:
    ----------
    optimal_dl : dict
        A dictionary where keys are variant names and values are their optimal description lengths (DL).

    variants : list
        A list of variant names for which to compute the edge compression.

    num_edges : int
        The total number of edges in the graph.

    Returns:
    -------
    dict
        A dictionary mapping each variant to its corresponding edge compression ratio (optimal DL / num_edges).
    """
    vals = {}
    for variant in variants:
        vals[variant] = optimal_dl[variant] / num_edges
    return vals


def check_variants(variants):
    """
    Checks and validates Stochastic Blockmodel (SBM) variants.

    Parameters:
    ----------
    variants : str or list
        A string 'all' or a list of SBM variants to check.

    Returns:
    -------
    list
        A list of validated SBM variants.

    Raises:
    ------
    ValueError
        If any variant in the input list is invalid.
    """
    valid_variants = ['dc', 'ndc', 'pp']

    if variants == 'all':
        return valid_variants
    else:
        return validate_variants(input_variants=variants, valid_variants=valid_variants)


def validate_variants(input_variants, valid_variants):
    """
    Checks the validity of Stochastic Blockmodel (SBM) variants.

    Parameters:
    ----------
    input_variants : list
        A list of SBM variants to check.

    valid_variants : list
        A list of valid SBM variants.

    Returns:
    -------
    list
        A list of valid variants.

    Raises:
    ------
    ValueError
        If any variant in the input list is invalid.
    """
    if not np.all([variant in valid_variants for variant in input_variants]):
        raise ValueError("Invalid input variant.")

    return input_variants


def calculate_blocklabel_dls(g, variants, uniform, degree_dl_kind, use_gt, allow_multigraphs):
    """
    Calculates description lengths based on block labels within a graph.

    Parameters:
    ----------
    g : graph_tool.Graph
        The input graph.

    variants : list
        A list of SBM variants to consider.

    uniform : bool
        Flag indicating whether to use uniform entropy estimation.

    degree_dl_kind : str
        The kind of degree distribution used in entropy estimation.

    use_gt : bool
        Flag indicating if the description length calculation from the graph tool library should be used.

    allow_multigraphs : bool
        Flag indicating if multigraphs should be allowed (only possible if `use_gt=True`).

    Returns:
    -------
    dict
        A dictionary containing calculated description lengths for each SBM variant.
    """
    blocklabel_partition = np.array(g.vp.blocklabel.a, dtype=int)
    return calculate_dls(g=g, meta_partition=blocklabel_partition, variants=variants, uniform=uniform,
                         degree_dl_kind=degree_dl_kind, use_gt=use_gt, allow_multigraphs=allow_multigraphs)


def calculate_meta_dls(g, metadata, variants, uniform, degree_dl_kind, use_gt, allow_multigraphs):
    """
    Calculates the description lengths of each metadata attribute under the DCSBM and PP-DCSBM.

    Parameters:
    ----------
    g : graph_tool.Graph
        A graph_tool.Graph object representing the graph.

    metadata : list
        A list of strings representing the metadata attributes.

    variants : list
        A list of strings indicating the variants for which the description length should be calculated.

    uniform : bool
        Flag indicating whether to use uniform entropy estimation.

    degree_dl_kind : str
        The kind of degree distribution used in entropy estimation.

    use_gt : bool
        Flag indicating if the description length calculation from the graph tool library should be used.

    allow_multigraphs : bool
        Flag indicating if multigraphs should be allowed (only possible if `use_gt=True`).

    Returns:
    -------
    dict
        A dictionary containing the description lengths for each metadata attribute.
    """
    meta_dls = {}
    for meta in metadata:
        meta_partition = np.array(g.vp[meta].a, dtype=int)
        meta_dls[meta] = calculate_dls(g=g, meta_partition=meta_partition, variants=variants, uniform=uniform,
                                       degree_dl_kind=degree_dl_kind,
                                       use_gt=use_gt, allow_multigraphs=allow_multigraphs)
    return meta_dls


def calculate_dls(g, meta_partition, variants, uniform, degree_dl_kind, use_gt, allow_multigraphs):
    """
    Calculates description lengths for multiple SBM variants.

    Parameters:
    ----------
    g : graph_tool.Graph
        The input graph.

    meta_partition : array-like
        The metadata partition to be used for description length calculation.

    variants : list
        A list of SBM variants for which description lengths should be calculated.

    uniform : bool
        Flag indicating whether to use uniform entropy estimation in the PPSBM case.

    degree_dl_kind : str
        The kind of degree distribution used in entropy estimation.

    use_gt : bool
        Flag indicating if the description length calculation from the graph tool library should be used.

    allow_multigraphs : bool
        Flag indicating if multigraphs should be allowed (only possible if `use_gt=True`).

    Returns:
    -------
    dict
        A dictionary containing calculated description lengths for each SBM variant.
    """
    return {variant: calculate_dl_variant(g=g, meta_partition=meta_partition, variant=variant, uniform=uniform,
                                          degree_dl_kind=degree_dl_kind,
                                          use_gt=use_gt, allow_multigraphs=allow_multigraphs)
            for variant in variants}


def calculate_dl_variant(g, meta_partition, variant, uniform, degree_dl_kind, use_gt, allow_multigraphs):
    """
    Calculates the description length for a specific SBM variant.

    Parameters:
    ----------
    g : graph_tool.Graph
        The input graph.

    meta_partition : array-like
        The metadata partition to be used for description length calculation.

    variant : str
        The SBM variant ('dc', 'ndc', or 'pp').

    uniform : bool
        Flag indicating whether to use uniform entropy estimation.

    degree_dl_kind : str
        The kind of degree distribution used in entropy estimation.

    use_gt : bool
        Flag indicating if the description length calculation from the graph tool library should be used.

    allow_multigraphs : bool
        Flag indicating if multigraphs should be allowed (only possible if `use_gt=True`).

    Returns:
    -------
    float
        The calculated description length for the specified SBM variant.
    """
    dc = False
    if variant in ['dc', 'pp']:
        dc = True
    if use_gt:
        if variant == 'pp':
            return gt.PPBlockState(g=g, b=meta_partition).entropy(uniform=uniform, degree_dl_kind=degree_dl_kind)
        return gt.BlockState(g=g, b=meta_partition, deg_corr=dc).entropy(multigraph=allow_multigraphs)
    else:
        blockstate = 'BlockState'
        if variant == 'pp':
            blockstate = 'PPBlockState'
        return calculate_dl(g=g, b=meta_partition, dc=dc,
                            blockstate=blockstate, uniform=uniform, degree_dl_kind=degree_dl_kind)


def calculate_meta_dls_randomised(g, metadata, iters_rand, variants, uniform, degree_dl_kind, use_gt,
                                  allow_multigraphs, disable_progress_bar=False):
    """
    Calculates the description lengths of randomised metadata attributes for various SBM variants.

    Parameters:
    ----------
    g : graph_tool.Graph
        A graph_tool.Graph object representing the graph.

    metadata : list
        A list of strings representing the metadata attributes.

    iters_rand : int
        The number of random iterations to compute statistical significance.

    variants : list
        A list of SBM variants for which to calculate the description lengths.

    uniform : bool
        Flag indicating whether to use uniform entropy estimation.

    degree_dl_kind : str
        The kind of degree distribution used in entropy estimation.

    use_gt : bool
        Flag indicating if the description length calculation from the graph-tool library should be used.

    allow_multigraphs : bool
        Flag indicating if multigraphs should be allowed (only possible if `use_gt=True`).

    disable_progress_bar : bool, optional
        Flag indicating whether to disable the progress bar (default: False).

    Returns:
    -------
    dict
        A dictionary containing the description lengths for randomised metadata attributes.
    """
    meta_dls_randomised = {}
    for meta in metadata:
        meta_partition = np.array(g.vp[meta].a, dtype=int)
        dls_rand = {variant: np.zeros(iters_rand) for variant in variants}

        with tqdm(total=iters_rand, desc=f'Random Iterations for {meta}', unit='iter', ncols=80,
                  leave=False, disable=disable_progress_bar) as pbar:
            for j in range(iters_rand):
                meta_partition_randomised = np.random.permutation(meta_partition)
                for variant in variants:
                    dls_rand[variant][j] = calculate_dl_variant(g=g, meta_partition=meta_partition_randomised,
                                                                variant=variant, uniform=uniform,
                                                                degree_dl_kind=degree_dl_kind, use_gt=use_gt,
                                                                allow_multigraphs=allow_multigraphs)
                pbar.update(1)

        meta_dls_randomised[meta] = dls_rand
    return meta_dls_randomised


def get_dls_states(states, allow_multigraphs):
    """
    Calculates the description lengths of the provided block states.

    Parameters:
    ----------
    states : dict
        A dictionary of block states to compute description lengths for.

    allow_multigraphs : bool
        Flag indicating if multigraphs should be allowed.

    Returns:
    -------
    dict
        A dictionary where keys are SBM variants and values are their corresponding description lengths.
    """
    return {variant: s.entropy() if variant == 'pp' else s.entropy(multigraph=allow_multigraphs)
            for variant, s in states.items()}


def calculate_uncompressed_dls(g, variants, allow_multigraphs):
    """
    Calculates the uncompressed description lengths (with B=1).

    Parameters:
    ----------
    g : graph_tool.Graph
        A graph_tool.Graph object representing the graph.

    variants : list
        A list of SBM variants for which to calculate the description lengths.

    allow_multigraphs : bool
        Flag indicating if multigraphs should be allowed.

    Returns:
    -------
    dict
        A dictionary containing the uncompressed description lengths for each SBM variant.
    """
    deg_corr = {'dc': True,
                'ndc': False}
    b = [0] * g.num_vertices()
    return {variant: gt.PPBlockState(g=g, b=b).entropy()
            if variant == 'pp'
            else gt.BlockState(g=g, B=1, deg_corr=deg_corr[variant]).entropy(multigraph=allow_multigraphs)
            for variant in variants}


def compute_gamma_values(meta_dls, meta_dls_randomised, variants, optimal_dl, metadata, percentile=1):
    """
    Computes the gamma values for each metadata attribute based on description lengths and statistical significance.

    Parameters:
    ----------
    meta_dls : dict
        A dictionary containing the description lengths for each metadata attribute.

    meta_dls_randomised : dict
        A dictionary containing the description lengths for randomised metadata.

    variants : list
        A list of SBM variants for which to calculate the gamma values.

    optimal_dl : dict
        A dictionary containing the optimal description lengths for each variant.

    metadata : list
        A list of strings representing the metadata attributes.

    percentile : int, optional
        Percentile used to determine statistical significance (default: 1).

    Returns:
    -------
    dict
        A dictionary containing gamma values for each metadata attribute and each variant. Keys are metadata names,
        and values are dictionaries with SBM variants as keys and tuples of (gamma, edge compression) as values.
    """
    vals = {}
    for meta in metadata:
        vals[meta] = {variant: gamma(partition_dl=meta_dls[meta][variant],
                                     optimal_dl=optimal_dl[variant],
                                     random_dl=np.percentile(meta_dls_randomised[meta][variant], percentile))
                      for variant in variants}
    return vals


def gamma(partition_dl, optimal_dl, random_dl):
    """
    Calculates the gamma value for a specific partition description length.

    Parameters:
    ----------
    partition_dl : float
        The description length of the partition.

    optimal_dl : float
        The optimal description length.

    random_dl : float
        The description length of random partition(s).

    Returns:
    -------
    float
        The calculated gamma value.
    """
    return (partition_dl - optimal_dl) / (random_dl - optimal_dl)


def check_input(graph, metadata, allow_multigraphs, new_metadata_names=None, verbose=True):
    """
    Performs checks on the graph and metadata as preparation for calculating the gamma vector.

    Parameters:
    ----------
    graph : graph_tool.Graph
        A graph_tool.Graph object representing the graph.

    metadata : list
        The metadata input, which can be a list of strings or a list of NumPy arrays.

    allow_multigraphs : bool
        Flag indicating if multigraphs should be allowed.

    new_metadata_names : list, optional
        A list of strings representing new metadata names corresponding to NumPy arrays in 'metadata' (default: None).

    verbose : bool, optional
        Flag indicating whether to display detailed output (default: True).

    Returns:
    -------
    tuple
        A tuple containing the processed graph and metadata.

    Raises:
    ------
    ValueError
        If the graph is not undirected, not a simple graph, or if metadata is neither a string nor a list of strings.

    KeyError
        If the metadata property does not exist in the graph.

    TypeError
        If the metadata property values are not integers or integers written as floats.
    """
    g = copy.deepcopy(graph)
    metadata = make_list(metadata)

    # Check if g is a graph_tool.Graph object
    if not isinstance(g, gt.Graph):
        raise ValueError("The 'g' parameter should be a graph_tool.Graph object.")

    # Check if the graph is directed, if so make undirected
    if g.is_directed():
        g.set_directed(False)
        if verbose:
            print("Converted graph to undirected.")

    # Check if the graph is a simple graph
    if not allow_multigraphs:
        if is_multigraph(g=g):
            g = simplify_multigraph(multigraph=g)
            if verbose:
                print("Simplified graph.")

    # Check metadata and convert if necessary
    if isinstance(metadata, list) and isinstance(metadata[0], np.ndarray):
        if new_metadata_names is None or not isinstance(new_metadata_names, list) or len(metadata) != len(
                new_metadata_names):
            raise ValueError("When 'metadata' is a list of NumPy arrays, "
                             "'new_metadata_names' must be a list of strings with the same length.")

        for i, meta_arr in enumerate(metadata):
            if not isinstance(meta_arr, np.ndarray):
                raise ValueError("Each element in 'metadata' must be a NumPy array.")

            if meta_arr.shape[0] != g.num_vertices():
                raise ValueError(
                    "The length of each NumPy array in 'metadata' must match the number of vertices in the graph.")

            meta_name = new_metadata_names[i]
            g.vp[meta_name] = g.new_vertex_property('int')
            g.vp[meta_name].a = meta_arr
        meta_checked = new_metadata_names

    elif isinstance(metadata, list) and isinstance(metadata[0], str):
        # Check if metadata properties exist in the graph
        for prop_name in metadata:
            if prop_name not in g.vertex_properties:
                raise KeyError('Metadata property {} does not exist in the graph.'.format(prop_name))

        # Check if metadata property values are integers or integers written as floats
        for prop_name in metadata:
            prop_values = [g.vp[prop_name][v] for v in g.vertices()]
            if not all(isinstance(value, (int, float)) and int(value) == value for value in prop_values):
                if all(isinstance(value, str) for value in prop_values):
                    new_vp = g.new_vertex_property("int")
                    mapping = str_to_int_mapping(prop_values)
                    gt.map_property_values(g.vp[prop_name], new_vp, lambda x: mapping[x])
                    g.vp[prop_name] = new_vp
                    if verbose:
                        print("Converted metadata {} from type to str to int.".format(prop_name))
                else:
                    raise TypeError(f"Metadata property '{prop_name}' should have only integer or only string values.")

        meta_checked = metadata
    else:
        raise ValueError("Input 'metadata' must be either a list of strings or a list of NumPy arrays.")

    return g, meta_checked


def optimise_block_states(g, variants, refine=True, iters_refine=1000):
    """
    Minimise description length for one or multiple SBM variants for a given graph.

    Parameters:
    ----------
    g : graph_tool.Graph
        A graph_tool.Graph object representing the graph.

    variants : list
        A list of SBM variants to be included in the inference of block states (default: ['dc', 'ndc', 'pp']).

    refine : bool, optional
        Flag indicating whether to refine the block state minimization (default: True).

    iters_refine : int, optional
        If `refine=True`, indicates the number of times the multiflip MCMC algorithm performs 10 sweeps (default: 1000).

    Returns:
    -------
    dict
        A dictionary containing the block states for each variant. Keys are SBM variants and values are the corresponding states.
    """
    # Compute block models
    states = {}
    for variant in variants:
        if variant == 'dc':
            dc = True
            pp = False
        elif variant == 'ndc':
            dc = False
            pp = False
        elif variant == 'pp':
            dc = True
            pp = True
        else:
            raise ValueError("Invalid variant.")
        state = refine_minimize_blockmodel_dl(g=g, dc=dc, pp=pp, nested=False, refine=refine,
                                              iters=iters_refine, sweeps=10)
        states[variant] = state

    return states


def refine_minimize_blockmodel_dl(g, dc, pp, nested, refine=True, iters=1000, sweeps=10):
    """
    Refines and minimizes the description length of block models for the given graph.

    Parameters:
    ----------
    g : graph_tool.Graph
        A graph_tool.Graph object representing the graph.

    dc : bool
        Flag indicating whether to use degree correction.

    pp : bool
        Flag indicating whether to use the planted partition model.

    nested : bool
        Flag indicating whether to use nested block models.

    refine : bool, optional
        Flag indicating whether to refine the block state minimization (default: True).

    iters : int, optional
        Number of times the multiflip MCMC algorithm performs sweeps if `refine=True` (default: 1000).

    sweeps : int, optional
        Number of sweeps performed in each iteration if `refine=True` (default: 10).

    Returns:
    -------
    gt.BlockState
        The refined and minimized block model state.
    """
    if nested:
        if pp:
            raise ValueError('Nested version of pp variant not implemented.')
        state = gt.minimize_nested_blockmodel_dl(g, state_args=dict(deg_corr=dc))
    else:
        if pp:
            if not dc:
                raise ValueError('Non-degree-corrected version of pp variant not implemented.')
            state = gt.minimize_blockmodel_dl(g, state=gt.PPBlockState)
        else:
            state = gt.minimize_blockmodel_dl(g, state_args=dict(deg_corr=dc))
    if refine:
        for i in range(iters):
            state.multiflip_mcmc_sweep(beta=np.inf, niter=sweeps)
    return state
