
"""
Contains plotting utility functions and constants.

"""

__author__ = "Christian Palmroos"

import matplotlib.pyplot as plt
import pandas as pd

# Constants:
STANDARD_QUICKLOOK_FIGSIZE = (13,7)
STANDARD_FIGSIZE = (26,11)

STANDARD_TITLE_FONTSIZE = 32
STANDARD_LEGENDSIZE = 26
STANDARD_AXIS_LABELSIZE = 20

STANDARD_TICK_LABELSIZE = 25

STANDARD__MAJOR_TICKLEN = 11
STANDARD_MINOR_TICKLEN = 8

STANDARD_MAJOR_TICKWIDTH = 2.8
STANDARD_MINOR_TICKWIDTH = 2.1

DEFAULT_SELECTION_ALPHA = 0.12
BREAKPOINT_SHADING_ALPHA = 0.18

LATEX_PM = r"$\pm$"

def set_standard_ticks(ax, labelsize:int=None) -> None:
    """
    Handles tickmarks, their sizes etc...
    """

    if labelsize is None:
        labelsize = STANDARD_TICK_LABELSIZE

    ax.tick_params(which="major", length=STANDARD__MAJOR_TICKLEN, width=STANDARD_MAJOR_TICKWIDTH, labelsize=labelsize)
    ax.tick_params(which="minor", length=STANDARD_MINOR_TICKLEN, width=STANDARD_MINOR_TICKWIDTH, labelsize=labelsize-5)

def set_xlims(ax:plt.Axes, data:pd.DataFrame, xlim:list[str]) -> None:
    """
    Sets the x-axis boundaries for the plot

    Parameters:
    -----------
    ax : {plt.Axes} The axes of the figure.
    data : {pd.DataFrame} The data being plotted.
    xlim : {list[str]} A pair of datetime strings to set the plot boundaries.
    """

    if xlim is None:
        ax.set_xlim(data.index.values[0], data.index.values[-1])
    else:
        ax.set_xlim(pd.to_datetime(xlim[0]), pd.to_datetime(xlim[1]))