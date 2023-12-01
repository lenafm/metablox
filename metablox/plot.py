"""Plotting functions for gamma"""

import os
import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text


def plot_gamma(gamma, dimx, dimy, metadata=None,
               title=None, annotations=None, lims=None,
               c=None, plotxy=True, outfile=None, file_format='png'):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    possible_dimensions = ['dc', 'ndc', 'pp']
    if (dimx not in possible_dimensions) or (dimy not in possible_dimensions):
        raise ValueError('Parameters "dimx" and "dimy" must be on of "dc", "ndc", "pp"')

    if metadata is None:
        metadata = list(gamma.keys())

    x = np.array([gamma[meta][dimx] for meta in metadata])
    y = np.array([gamma[meta][dimy] for meta in metadata])

    ax.axhline(y=1., color='black', linestyle='solid', linewidth=1.2)
    ax.axvline(x=1., color='black', linestyle='solid', linewidth=1.2)
    scatter = ax.scatter(x, y, c=c, alpha=0.6)

    if lims is None:
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]
        ax.set_xlim((-0.01, lims[1]))
        ax.set_ylim((-0.01, lims[1]))
    else:
        ax.set_xlim(lims)
        ax.set_ylim(lims)

    if plotxy:
        new_lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]
        ax.plot(new_lims, new_lims, 'k-', alpha=0.75, zorder=0, linewidth=0.8)
    ax.set_aspect('equal')

    ax.grid(linewidth=0.3)
    ax.set_ylabel(r'$\gamma^{{{}}}$'.format(dimy.upper()), fontsize=14)
    if title is not None:
        ax.set_title(title, fontsize=16)
    ax.set_xlabel(r'$\gamma^{{{}}}$'.format(dimx.upper()), fontsize=14)

    if annotations is not None:
        texts = []
        for i, meta in enumerate(metadata):
            texts += [ax.text(x[i], y[i], annotations[meta], fontsize=12)]
        adjust_text(texts,
                    ax=ax,
                    arrowprops=dict(arrowstyle='-', color='gray', alpha=.5))

    fig.tight_layout()
    if outfile is None:
        plt.show()
    else:
        if file_format == 'png':
            plt.savefig(outfile, bbox_inches='tight')
        elif file_format == 'svg':
            plt.savefig(outfile, bbox_inches='tight')
        else:
            raise ValueError('Parameter file_format needs to be "png" or "svg".')
        plt.close()
