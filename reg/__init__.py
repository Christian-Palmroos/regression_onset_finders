
"""
Contains the first development version for ramp regression and break regression algorithm -based
SEP event onset finding tools.
"""

__author__ = "Christian Palmroos"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import piecewise_regression

# Relative imports cannot be used with "import .a" form; use "from . import a" insteadPylance
from . import calc_utilities as calc
from .plotting_utilities import set_standard_ticks, set_xlims, STANDARD_FIGSIZE, STANDARD_LEGENDSIZE


def workflow(data, channel:str, resample:str=None, xlim:list=None, fill_style:str="bfill",
            window:int=None ,threshold:float=None, plot:bool=True):
    """

    Parameters:
    -----------
    data : {pd.DataFrame}
    channel : {str}
    resample : {str}
    xlim : {list}
    fill_style : {str}
    window : {str}
    threshold : {float}
    plot : {bool}
    """

    # Choose resampling:
    if isinstance(resample, str):
        data = calc.resample_df(df=data, avg=resample)
    # If no resampling, just take a copy of the original data to avert 
    # modifying it by accident
    else:
        data = data.copy(deep=True)

    # Choose channel
    series = data[channel]

    # Convert to log
    series = calc.ints2log10(intensity=series, fill_style=fill_style)

    # Get the numerical index of the first peak to choose the selection from 
    # background to first peak. Also generate numerical index to run from 0 to max_idx
    max_val, max_idx = calc.search_first_peak(ints=series, window=window, threshold=threshold)

    series = series[:max_idx]
    numerical_indices = np.linspace(start=0, stop=max_idx, num=max_idx)

    # Get the fit results
    fit_results = break_regression(ints=series.values, indices=numerical_indices)

    # The results are a dictionary, extract values here
    estimates = fit_results["estimates"]

    const = estimates["const"]["estimate"]
    alpha1 = estimates["alpha1"]["estimate"]
    alpha2 = estimates["alpha2"]["estimate"]
    
    break_point = estimates["breakpoint1"]["estimate"]
    break_point_errs = estimates["breakpoint1"]["confidence_interval"]

    if plot:
        # Generate the fit lines to display on the plot
        line1, line2 = calc.generate_fit_lines(indices=numerical_indices, const=const,
                                               alpha1=alpha1, alpha2=alpha2, break_point=break_point)

        # Init figure
        fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE)

        # Plot the intensities
        ax.step(numerical_indices, series.values, label=channel)

        # Plot the fit results on the real data
        ax.plot(line1.index, line1.values, ls="--", c="navy")
        ax.plot(line2.index, line2.values, ls="--", c="navy")
        
        ax.axvspan(xmin=break_point_errs[0], xmax=break_point_errs[1], alpha=0.25, color="red")
        ax.axvline(x=break_point, c="red")

        ax.legend(fontsize=STANDARD_LEGENDSIZE)

        set_xlims(ax=ax, data=series, xlim=xlim)
        set_standard_ticks(ax=ax)

        plt.show()

    results_dict = {"const": const,
                    "slope1": alpha1,
                    "slope2": alpha2,
                    "break_point": break_point,
                    "break_errors": break_point_errs}

    return [const, alpha1, alpha2, break_point, break_point_errs]


def break_regression(ints, indices, starting_values:list=None) -> dict:

    NUM_OF_BREAKPOINTS = 1

    fit = piecewise_regression.Fit(xx=indices,
                                   yy=ints,
                                   start_values=starting_values,
                                   n_breakpoints=NUM_OF_BREAKPOINTS)

    return fit.get_results()

def quicklook(data:pd.DataFrame, channel:str=None, resample:str=None, xlim:list=None):
    """
    Makes a quicklook plot of one or more channels for a given dataframe.
    
    data : dataframe
    
    channel : str, list
    
    resample : str
    
    xlim : list
    """

    #color = plt.cmap("plasma")
    if resample is not None:
        data = calc.resample_df(df=data, avg=resample)

    # Make sure that channel is a list to iterate over
    if channel is None:
        channel = list(data.columns)
    if isinstance(channel,(str,int)):
        channel = [channel]

    fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE)
    
    ax.set_yscale("log")

    for ch in channel:
        ax.step(data.index.values, data[ch].values, where="mid", label=ch)

    ax.legend(fontsize=STANDARD_LEGENDSIZE)

    set_xlims(ax=ax, data=data, xlim=xlim)
    set_standard_ticks(ax=ax, labelsize=None)
    
    plt.show()


