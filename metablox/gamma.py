"""Gamma calculations"""

import copy
import numpy as np
import graph_tool.all as gt
from tqdm import tqdm
from metablox.dl import calculate_dl
from metablox.utils import is_multigraph, simplify_multigraph, make_list, str_to_int_mapping


def calculate_gamma(g, metadata,
                    variants='all',
                    iters_rand=100,
                    new_metadata_names=None,
                    uniform=False, degree_dl_kind='distributed',
                    variants_infer='all',
                    refine_states=False,
                    iters_refine=1000,
                    return_dls=False, return_states=False,
                    verbose=True, synthetic=False):
    """
    Calculate the gamma values for each metadata attribute based on the graph and metadata.

    Args:
        g: A graph_tool.Graph object representing the graph.
        metadata: The metadata input, which can be a list of strings or a list of NumPy arrays.
        variants (str or list): A string 'all' or a list of SBM variants (options: 'dc', 'ndc', and 'pp') to be included
        as models for the calculation of gamma (default: 'all').
        iters_rand (optional): The number of random iterations for computing statistical significance (default: 100).
        new_metadata_names (optional): Labels for metadata if metadata is a list of arrays (default: None).
        uniform (optional): Flag indicating whether to use uniform entropy estimation (default: False).
        degree_dl_kind (optional): The kind of degree distribution used in entropy estimation (default: 'distributed').
        exact (optional): Flag indicating whether to use exact entropy estimation (default: True).
        variants_infer (str or list): A string 'all' or a list of SBM variants (options: 'dc', 'ndc', and 'pp') to be
        included for the inference of the optimal partition (default: 'all').
        refine_states: Flag indicating whether to refine minimisation of block states by running 10*iters_refine sweeps
        of the MCMC with zero temperature (default: False).
        iters_refine: If refine_states=True, indicates the number of times the multiflip mcmc algorithm performs 10
        sweeps. The total number of performed sweeps will therefore be 10*iters_refine (default: 1000).
        return_dls: Flag indicating whether the description lengths of the metadata, the randomised metadata, the
        optimal partition, and the blocklabel partition (if synthetic = True) should be returned alongside gamma
        (default: False).
        return_states: Flag indicating whether to return the blockmodel states of the fitted SBMs (default: False).
        verbose: Flag indicating whether to display detailed output (default: False).
        synthetic: Flag indicating whether the network is a synthetic network with a planted partition; if True, and
        if return_dls = True, the function also returns the description length of the planted partition (default: False)

    Returns:
        A dictionary containing gamma values for each metadata attribute and for each variant:
        {
            'metadata_attribute_1': {
                'gamma_dc': gamma_value_dc,
                'gamma_pp': gamma_value_pp
            },
            'metadata_attribute_2': {
                'gamma_dc': gamma_value_dc,
                'gamma_pp': gamma_value_pp
            },
            ...
        }
    """
    variants = check_variants(variants=variants)
    variants_infer = check_variants(variants=variants_infer)
    g, metadata = check_input(graph=g, metadata=metadata, new_metadata_names=new_metadata_names, verbose=verbose)
    if verbose:
        tqdm.write("Minimizing block models and nested block models.")
    states = get_states(g=g, variants=variants_infer, refine=refine_states, iters_refine=iters_refine)
    states_dls = get_dls_states(states=states)
    if verbose:
        tqdm.write("Calculating description lengths for metadata.")
    meta_dls = calculate_meta_dls(g=g, metadata=metadata, variants=variants,
                                  uniform=uniform, degree_dl_kind=degree_dl_kind)
    if verbose:
        tqdm.write("Calculating description lengths for randomised metadata.")
    disable_progress_bar = not verbose
    meta_dls_randomised = calculate_meta_dls_randomised(g=g, metadata=metadata, variants=variants,
                                                        iters_rand=iters_rand,
                                                        uniform=uniform, degree_dl_kind=degree_dl_kind,
                                                        disable_progress_bar=disable_progress_bar)

    if verbose:
        tqdm.write("Calculating gamma values.")
    gammas = calculate_gamma_values(meta_dls=meta_dls, meta_dls_randomised=meta_dls_randomised, optimal_dl=states_dls,
                                    metadata=metadata, variants=variants)

    if return_dls:
        if synthetic:
            if verbose:
                tqdm.write("Determining description length of planted partition.")
            blocklabel_dl = calculate_blocklabel_dls(g=g, variants=variants,
                                                     uniform=uniform, degree_dl_kind=degree_dl_kind)
            extra_output = {'meta_dls': meta_dls,
                            'meta_dls_randomised': meta_dls_randomised,
                            'state_dls': states_dls,
                            'blocklabel_dl': blocklabel_dl}
        else:
            extra_output = {'meta_dls': meta_dls,
                            'meta_dls_randomised': meta_dls_randomised,
                            'state_dls': states_dls}
        if return_states:
            extra_output['states'] = states
        return gammas, extra_output
    elif return_states:
        return gammas, {'states': states}
    else:
        return gammas


def check_variants(variants):
    """
    Check and validate Stochastic Blockmodel (SBM) variants.

    Args:
        variants (str or list): A string 'all' or a list of SBM variants to check.

    Returns:
        list: A list of validated SBM variants or raises a ValueError if any variant is invalid.

    Raises:
        ValueError: If any variant in the input list is invalid.
    """
    valid_variants = ['dc', 'ndc', 'pp']

    if variants == 'all':
        return valid_variants
    else:
        return validate_variants(input_variants=variants, valid_variants=valid_variants)


def validate_variants(input_variants, valid_variants):
    """
    Check the validity of Stochastic Blockmodel (SBM) variants.

    Args:
        input_variants (list): A list of SBM variants to check.
        valid_variants (list): A list of valid SBM variants.

    Raises:
        ValueError: If any variant in the input list is invalid.

    Returns:
        List of valid variants.
    """

    if not np.all([variant in valid_variants for variant in input_variants]):
        raise ValueError("Invalid input variant.")

    return input_variants


def calculate_blocklabel_dls(g, variants, uniform, degree_dl_kind):
    """
    Calculate description lengths based on block labels within a graph.

    Args:
        g (graph_tool.Graph): The input graph.
        variants (list): A list of SBM variants to consider.
        uniform (bool): Flag indicating whether to use uniform entropy estimation.
        degree_dl_kind (str): The kind of degree distribution used in entropy estimation.

    Returns:
        list: A list of calculated description lengths for each SBM variant.

    """
    blocklabel_partition = np.array(g.vp.blocklabel.a, dtype=int)
    return calculate_dls(g=g, meta_partition=blocklabel_partition, variants=variants,
                         uniform=uniform, degree_dl_kind=degree_dl_kind)


def calculate_meta_dls(g, metadata, variants, uniform, degree_dl_kind):
    """
    Calculate the description lengths of each metadata attribute under the DCSBM and PP-DCSBM.

    Args:
        g: A graph_tool.Graph object representing the graph.
        metadata: A list of strings representing the metadata attributes.
        variants: A list of strings indicating the variants for which the description length should be calculated.
        uniform: Flag indicating whether to use uniform entropy estimation.
        degree_dl_kind: The kind of degree distribution used in entropy estimation.

    Returns:
        A dictionary containing the description lengths for each metadata attribute.
    """
    meta_dls = {}
    for meta in metadata:
        meta_partition = np.array(g.vp[meta].a, dtype=int)
        meta_dls[meta] = calculate_dls(g=g, meta_partition=meta_partition, variants=variants,
                                       uniform=uniform, degree_dl_kind=degree_dl_kind)
    return meta_dls


def calculate_dls(g, meta_partition, variants, uniform, degree_dl_kind):
    """
    Calculate description lengths for multiple SBM variants.

    Args:
        g (graph_tool.Graph): The input graph.
        meta_partition (array-like): The metadata partition to be used for description length calculation.
        variants (list): A list of SBM variants for which description lengths should be calculated.
        uniform (bool): Flag indicating whether to use uniform entropy estimation in the PPSBM case.
        degree_dl_kind (str): The kind of degree distribution used in entropy estimation.

    Returns:
        dict: A dictionary containing calculated description lengths for each SBM variant.

    """
    return {variant: calculate_dl_variant(g=g, meta_partition=meta_partition, variant=variant,
                                          uniform=uniform, degree_dl_kind=degree_dl_kind)
            for variant in variants}


def calculate_dl_variant(g, meta_partition, variant, uniform, degree_dl_kind):
    """
    Calculate the description length for a specific SBM variant.

    Args:
        g (graph_tool.Graph): The input graph.
        meta_partition (array-like): The metadata partition to be used for description length calculation.
        variant (str): The SBM variant ('dc', 'ndc', or 'pp').
        uniform (bool): Flag indicating whether to use uniform entropy estimation.
        degree_dl_kind (str): The kind of degree distribution used in entropy estimation.

    Returns:
        float: The calculated description length for the specified SBM variant.

    """
    blockstate = 'BlockState'
    if variant == 'pp':
        blockstate = 'PPBlockState'
    dc = False
    if variant in ['dc', 'pp']:
        dc = True
    return calculate_dl(g=g, b=meta_partition, dc=dc,
                        blockstate=blockstate, uniform=uniform, degree_dl_kind=degree_dl_kind)


def calculate_meta_dls_randomised(g, metadata, iters_rand, variants, uniform, degree_dl_kind,
                                  disable_progress_bar=False):
    """
    Calculate the description lengths of randomised metadata under the DCSBM and PP-DCSBM.

    Args:
        g: A graph_tool.Graph object representing the graph.
        metadata: A list of strings representing the metadata attributes.
        iters_rand: The number of random iterations for computing statistical significance.
        variants: A list of strings indicating the variants for which the description length should be calculated.
        uniform: Flag indicating whether to use uniform entropy estimation.
        degree_dl_kind: The kind of degree distribution used in entropy estimation.
        disable_progress_bar: Flag indicating whether to disable progress bar (default: False).

    Returns:
        A dictionary containing the description lengths for randomised metadata.
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
                                                                variant=variant,
                                                                uniform=uniform,
                                                                degree_dl_kind=degree_dl_kind)
                pbar.update(1)

        meta_dls_randomised[meta] = dls_rand
    return meta_dls_randomised


def get_dls_states(states):
    """
    Calculate the description lengths of the given block states.

    Args:
        states: A dictionary of block states to compute description lengths for.

    Returns:
        A dictionary of description lengths for each block state.
    """
    return {variant: s.entropy() if variant == 'pp' else s.entropy(multigraph=False) for variant, s in states.items()}


def calculate_gamma_values(meta_dls, meta_dls_randomised, variants, optimal_dl, metadata, percentile=1):
    """
    Calculate the gamma values based on the description lengths.

    Args:
        meta_dls: A dictionary containing the description lengths for each metadata attribute.
        meta_dls_randomised: A dictionary containing the description lengths for randomised metadata.
        variants: A list of strings indicating the variants for which the description length should be calculated.
        optimal_dl: The optimal description length.
        metadata: A list of strings representing the metadata attributes.
        percentile: Percentile to be used to determine statistical significance (default: 1).

    Returns:
        A dictionary containing the gamma values for each metadata attribute.
    """
    gamma_val = {}
    for meta in metadata:
        gamma_val[meta] = {variant: gamma(partition_dl=meta_dls[meta][variant],
                                          optimal_dl=optimal_dl[variant],
                                          random_dl=np.percentile(meta_dls_randomised[meta][variant], percentile))
                           for variant in variants}
    return gamma_val


def gamma(partition_dl, optimal_dl, random_dl):
    """
    Calculate the gamma value for a specific partition description length.

    Args:
        partition_dl (float): Description length of the partition.
        optimal_dl (float): Optimal description length.
        random_dl (float): Description length of random partition(s).

    Returns:
        float: The calculated gamma value.
    """
    return (partition_dl - optimal_dl) / (random_dl - optimal_dl)


def check_input(graph, metadata, new_metadata_names=None, verbose=True):
    """
    Perform checks on g and metadata as preparation for the calculation of the gamma vector.

    Args:
        graph: A graph_tool.Graph object representing the graph.
        metadata: The metadata input, which can be a list of strings or a list of NumPy arrays.
        new_metadata_names: (Optional) A list of strings representing new metadata names
                            corresponding to the NumPy arrays in 'metadata'.
        verbose: Flag indicating whether to display detailed output (default: True).

    Raises:
        ValueError: If the graph is not undirected, not a simple graph,
                    or if metadata is neither a string nor a list of strings.
        KeyError: If the metadata property does not exist in the graph.
        TypeError: If the metadata property values are not integers or integers written as floats.
    """
    g = copy.deepcopy(graph)
    metadata = make_list(metadata)

    # Check if g is a graph_tool.Graph object
    if not isinstance(g, gt.Graph):
        raise ValueError("The 'g' parameter should be a graph_tool.Graph object.")

    # Check if the graph directed, if so make undirected
    if g.is_directed():
        g.set_directed(False)
        if verbose:
            print("Converted graph to undirected.")

    # Check if the graph is a simple graph
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


def get_states(g, variants, refine=True, iters_refine=1000):
    """
    Compute various block models and nested block models of a given graph.

    Args:
        g: A graph_tool.Graph object representing the graph.
        variants (list): List of strings indicating which variants should be included in the inference of block states
        (default: ['dc', 'ndc', 'pp']).
        refine: Boolean indicating whether to refine the block state minimisation (default: True).
        iters_refine: If refine=True, indicates the number of times the multiflip mcmc algorithm performs 10
        sweeps. The total number of performed sweeps will therefore be 10*iters_refine (default: 1000).

    Returns:
        A dictionary containing the block states:
            - state_dc: SBM state obtained using degree-corrected SBM.
            - state_ndc: SBM state obtained using non-degree-corrected SBM.
            - state_pp: SBM state obtained using planted partition degree-corrected SBM.
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
