
"""
Contains the first development version for ramp regression and break regression algorithm -based
SEP event onset finding tools.
"""

__author__ = "Christian Palmroos"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.dates import DateFormatter

import piecewise_regression

# Relative imports cannot be used with "import .a" form; use "from . import a" instead. -Pylance
from . import calc_utilities as calc
from .plotting_utilities import set_standard_ticks, set_xlims, STANDARD_FIGSIZE, STANDARD_LEGENDSIZE

INDEX_NUMBER_COL_NAME = "time_s"
COUNTING_NUMBERS_COL_NAME = "counting_numbers"

def select_channel_nonzero_ints(df:pd.DataFrame, channel:str):

    counting_numbers =  np.linspace(start=0, stop=len(df)-1, num=len(df))
    df[COUNTING_NUMBERS_COL_NAME] = counting_numbers.astype(int)

    selection = df[[channel, INDEX_NUMBER_COL_NAME, COUNTING_NUMBERS_COL_NAME]]
    selection = selection.loc[selection[channel]!=0]
    return selection


def produce_index_numbers(df:pd.DataFrame):
    index_numbers = df.index.strftime("%s")
    df[INDEX_NUMBER_COL_NAME] = index_numbers.astype(int)
    return df


def workflow(data, channel:str, resample:str=None, xlim:list=None, fill_style:str="bfill",
            window:int=None, threshold:float=None, plot:bool=True, diagnostics=False,
            index_choice="time_s"):
    """
    Seeks for the first peak in the given data. Cuts the data and only considers that part which comes
    before the first peak. In this chosen part, seek a break in the linear trend that is the background
    of the event. The break corresponds to the start of the event, and the second linear fit corresponds
    to the slope of the rising phase of the event (the linear slope of the 10-based logarithm).

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
    diagnostics : {bool}

    Returns:
    ----------
    results_dict : {dict} A dictionary of results that contains 'const', 'slope1', 'slope2', 
                          'break_point' and 'break_errors'.
    """

    # Choose resampling:
    if isinstance(resample, str):
        data = calc.resample_df(df=data, avg=resample)
    # If no resampling, just take a copy of the original data to avert 
    # modifying it by accident
    else:
        data = data.copy(deep=True)

    # Select the channel and produce indices for them. The indices are stored in the 
    # column "time_s", for they read seconds since the Epoch (1970-01-01 00:00).
    # The index numbers are used for the regression algorithm instead of datetime values. 
    data = produce_index_numbers(df=data)
    data = select_channel_nonzero_ints(df=data, channel=channel)

    # Convert to log
    series = calc.ints2log10(intensity=data[channel])
    # plot_series = calc.ints2log10(intensity=data[channel])
    plot_series = series.copy(deep=True)

    # Get the numerical index of the first peak to choose the selection from 
    # background to first peak. Also generate numerical index to run from 0 to max_idx
    max_val, max_idx = calc.search_first_peak(ints=series, window=window, threshold=threshold)

    # Apply a slice/selection to the data series and the numerical indices (seconds since Epoch)
    # according to the first peak found
    series = series[:max_idx]

    numerical_indices = data[index_choice].values[:max_idx]

    # Get the fit results
    fit_results = break_regression(ints=series.values, indices=numerical_indices)

    # The results are a dictionary, extract values here
    estimates = fit_results["estimates"]

    const = estimates["const"]["estimate"]
    alpha1 = estimates["alpha1"]["estimate"]
    alpha2 = estimates["alpha2"]["estimate"]

    break_point = estimates["breakpoint1"]["estimate"]
    break_point_errs = estimates["breakpoint1"]["confidence_interval"]

    # return estimates, data, series

    # Finds corresponding timestamps to the numerical indices
    if index_choice == "counting_numbers":
        # Choose the LAST entry of a linear space of integers that map to numerical_indices smaller than
        # the break_point. This is how manieth data point break_point is in series.
        lin_idx = np.linspace(start=0, stop=len(series)-1, num=len(series))
        break_point_idx = lin_idx[numerical_indices<break_point][-1]
        break_point_err_minus_idx = lin_idx[numerical_indices<break_point_errs[0]][-1]
        break_point_err_plus_idx = lin_idx[numerical_indices<break_point_errs[1]][-1]
        # onset_time = calc.get_interpolated_timestamp(datetimes=series.index, break_point=break_point)
        # onset_time_minus_err = calc.get_interpolated_timestamp(datetimes=series.index, break_point=break_point_errs[0])
        # onset_time_plus_err = calc.get_interpolated_timestamp(datetimes=series.index, break_point=break_point_errs[1])
        onset_time = calc.get_interpolated_timestamp(datetimes=series.index, break_point=break_point_idx)
        onset_time_minus_err = calc.get_interpolated_timestamp(datetimes=series.index, break_point=break_point_err_minus_idx)
        onset_time_plus_err = calc.get_interpolated_timestamp(datetimes=series.index, break_point=break_point_err_plus_idx)
    else:
        onset_time = pd.to_datetime(break_point, unit='s')
        onset_time_minus_err = pd.to_datetime(break_point_errs[0], unit='s')
        onset_time_plus_err = pd.to_datetime(break_point_errs[1], unit='s')

    results_dict = {"const": const,
                    "slope1": alpha1,
                    "slope2": alpha2,
                    "onset_time": onset_time,
                    "onset_time_error_minus": onset_time_minus_err,
                    "onset_time_error_plus": onset_time_plus_err}

    if plot:

        # Init figure
        fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE)

        if diagnostics:
            # Generate the fit lines to display on the plot
            line1, line2 = calc.generate_fit_lines(indices=numerical_indices, const=const,
                                                alpha1=alpha1, alpha2=alpha2, break_point=break_point)

            # Plot the fit results on the real data
            ax.plot(series.index[:len(line1)], line1.values, lw=2.8, ls="--", c="maroon", zorder=3)
            ax.plot(series.index[-len(line2):], line2.values, lw=2.8, ls="-.", c="maroon", zorder=3)

            # Apply a span over xmin=start and xmax=max_idx to display the are considered for the fit
            ax.axvspan(xmin=series.index[0], xmax=series.index[-1], facecolor="green", alpha=0.1)

        ax.set_ylabel("Log(intensity)", fontsize=STANDARD_LEGENDSIZE)

        # Plot the intensities
        # ax.step(plot_series.index, plot_series.values, label=channel, zorder=1)
        ax.scatter(plot_series.index, plot_series.values, label=channel, zorder=1)

        ax.axvspan(xmin=onset_time_minus_err, xmax=onset_time_plus_err, alpha=0.20, color="red")
        ax.axvline(x=onset_time, c="red", lw=1.8, label=f"onset time: {onset_time.strftime('%H:%M:%S')}")

        ax.legend(fontsize=STANDARD_LEGENDSIZE)

        set_xlims(ax=ax, data=data, xlim=xlim)
        set_standard_ticks(ax=ax)

        # Format the x-axis
        ax.xaxis.set_major_formatter(DateFormatter("%H:%M\n%d"))
        ax.set_xlabel(f"date of {onset_time.strftime('%b %Y')}", fontsize=STANDARD_LEGENDSIZE)

        plt.show()

    # When diagnostics is enabled, return additional info about the run
    if diagnostics:
        results_dict["line1"] = line1
        results_dict["line2"] = line2,
        results_dict["series"] = series,
        results_dict["indices"] = numerical_indices

    return results_dict


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


