
"""
Contains the first development version for ramp regression and break regression algorithm -based
SEP event onset finding tools.
"""

__author__ = "Christian Palmroos"

import pandas as pd
import matplotlib.pyplot as plt

import piecewise_regression

from .calc_utilities import resample_df
from .plotting_utilities import set_standard_ticks, STANDARD_FIGSIZE, STANDARD_LEGENDSIZE

def break_regression(ints, indices, starting_values:list=None):

    NUM_OF_BREAKPOINTS = 1

    fit = piecewise_regression.Fit(xx=indices,
                                   yy=ints,
                                   start_values=starting_values,
                                   n_breakpoints=NUM_OF_BREAKPOINTS)

    return fit.get_results()

def quicklook(data, channel:str=None, resample:str=None, xlim:list=None):
    """
    Makes a quicklook plot of one or more channels for a given dataframe.
    
    data : dataframe
    
    channel : str, list
    
    resample : str
    
    xlim : list
    """

    #color = plt.cmap("plasma")
    if resample is not None:
        data = resample_df(df=data, avg=resample)

    # Make sure that channel is a list to iterate over
    if channel is None:
        channel = list(data.columns)
    if isinstance(channel,(str,int)):
        channel = [channel]

    fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE)
    
    ax.set_yscale("log")

    for ch in channel:

        ax.step(data.index.values, data[ch].values, where="mid", label=ch)

    # The x-axis boundaries
    if xlim is None:
        ax.set_xlim(data.index.values[0], data.index.values[-1])
    else:
        ax.set_xlim(pd.to_datetime(xlim[0]), pd.to_datetime(xlim[1]))

    ax.legend(fontsize=STANDARD_LEGENDSIZE)

    set_standard_ticks(ax=ax, labelsize=None)
    
    plt.show()
