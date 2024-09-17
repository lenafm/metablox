"""Plotting functions for gamma"""

import matplotlib.pyplot as plt


def plot_metadata_relevance(data, metadata, ax, highlight_min_edge_compression=True,
                            marker_col='#2a9d8f', marker_size=10, alphadots=0.65):
    """
    Plots the relevance of metadata for various variants, highlighting the variant with the smallest edge compression.

    Parameters:
    ----------
    data : tuple
        A tuple where:
        - The first element is a dictionary with variant names as keys and lists of metric values as values.
        - The second element is a dictionary with variant names as keys and summary values as values.

    metadata : list
        A list of strings representing the metadata names to be used as x-axis ticks.

    ax : matplotlib.axes.Axes
        The Axes object on which to plot the data.

    highlight_min_edge_compression : bool, optional
        A flag indicating whether to highlight the variant with the smallest edge compression (default is True).

    marker_col : str, optional
        The color of the marker edge and fill (default is '#2a9d8f').

    marker_size : int, optional
        The size of the markers (default is 10).

    alphadots : float, optional
        The transparency level of the dots (default is 0.65).

    Returns:
    -------
    matplotlib.axes.Axes
        The Axes object with the plot.
    """
    # Extract the data
    metrics = data[0]
    summary = data[1]

    # Find the key with the smallest value in the summary dictionary
    highlight_variant = min(summary, key=summary.get)

    # Define marker styles
    markers = {'dc': 'D', 'ndc': 'o', 'pp': 's'}

    # Plot each metric with different markers
    for variant, values in metrics.items():
        x = metadata
        y = values
        marker = markers.get(variant, 'o')  # Default to 'o' if variant is not in markers
        ax.plot(x, y,
                mfc=marker_col,
                color=marker_col,
                markeredgewidth=1.5,
                marker=marker,
                markersize=marker_size,
                label=variant,
                alpha=alphadots, linestyle='None')
        if highlight_min_edge_compression:
            if variant == highlight_variant:
                ax.plot(x, y,
                        mfc=marker_col,
                        color='red',
                        markeredgewidth=1.5,
                        marker=marker,
                        markersize=marker_size,
                        zorder=5,
                        fillstyle='none', linestyle='None')

    # Add the first legend for the metrics
    legend1 = ax.legend(loc='upper left')
    ax.add_artist(legend1)

    if highlight_min_edge_compression:
        # Add a second legend for the highlighted node(s)
        highlight_marker = plt.Line2D([], [], color='red', marker='.', linestyle='None',
                                      label='Min. edge compression')
        legend2 = ax.legend(handles=[highlight_marker], loc='lower right')
        ax.add_artist(legend2)

    # Customize the plot
    ax.axhline(1., color='grey', linestyle='solid', linewidth=0.4, zorder=-1)
    ax.axhline(0., color='grey', linestyle='solid', linewidth=0.4, zorder=-1)
    ax.grid(linewidth=0.1, color='grey')
    ax.set_ylabel(r'$\gamma$')

    return ax

