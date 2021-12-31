import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
"""
Author: María Fernanda Morales Oreamuno 

Created: 31/05/2021
Last update: 12/07/2021

Functions to plot and save different results
"""
# Set font sizes:
SMALL_SIZE = 9
MEDIUM_SIZE = 10
BIGGER_SIZE = 11

# LaTex document properties
textwidth = 448
textheight = 635.5
#plt.rc("text", usetex='True')
#plt.rc('font', family='serif', serif='Arial')

# Set plot properties:
# plt.rc('figure', figsize=(15, 15))
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # font size of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # font size of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # font size of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # font size of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend font size
plt.rc('figure', titlesize=BIGGER_SIZE)  # font size of the figure title

plt.rc('savefig', dpi=1000)


def plot_size(fraction=1, height_fraction=0):
    """
    Set figure dimensions to avoid scaling in LaTeX.
    Source: https://jwalton.info/Embed-Publication-Matplotlib-Latex/
        Parameters
        ----------
        :param fraction: float, optional (Fraction of the width which you wish the figure to occupy)
        :param height_fraction: float, optional (Fraction of the entire page height you wish the figure to occupy)

        Returns
        -------
        fig_dim: tuple (Dimensions of figure in inches)
        """
    # Width of figure (in pts)
    fig_width_pt = textwidth * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5 ** .5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    if height_fraction == 0 :
        # Figure height in inches: golden ratio
        fig_height_in = fig_width_in * golden_ratio
    else:
        fig_height_in = textheight * inches_per_pt * height_fraction

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


def plot_likelihoods(model_list, graph_name):
    """
    Function plots the likelihoods for different parameter 1 and parameter 2 combinations (from the prior) as a scatter
    plot, with plot marker colors set by the likelihood values.

    Args:
    ------------------------------
    :param model_list: list with instances of Bayes_Inference classes
    :param name: name with which to save the plot
    """
    row = int(math.ceil(len(model_list)/2))
    col = 2
    fig, ax = plt.subplots(figsize=plot_size(height_fraction=0.8))

    for i, model in enumerate(model_list):
        # Values
        x_prior = model[:, 0]
        y_prior = model[:, 1]
        z_prior = model[:, 2]

        # Set axis
        ax = plt.subplot(row, col, i+1)
        # Set axis limits:
        vmax, vmin = np.max(model[:, 2]), np.min(model[:, 2])

        # Plot
        im = ax.scatter(x_prior, y_prior, c=z_prior, s=10, vmax=vmax, vmin=vmin, cmap='jet', marker='o')

        # Labels
        ax.set_xlabel('P1')
        ax.set_ylabel('P2')

        # Titles
        if i == row - 1:
            name = "ref"
        ax.set_title(graph_name[i], loc='left', fontweight='bold')


    # Plot config
    fig.subplots_adjust(right=0.80)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    cb=fig.colorbar(im, cax=cbar_ax)
    cb.set_label('Likelihood', labelpad=-1)
    plt.subplots_adjust(top=0.9, bottom=0.1, wspace=0.32, hspace=0.32)

    plt.show()
    x=1