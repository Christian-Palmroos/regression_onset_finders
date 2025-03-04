
"""
Contains the first development version for ramp regression and break regression algorithm -based
SEP event onset finding tools.
"""

__author__ = "Christian Palmroos"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import ipympl

from matplotlib.dates import DateFormatter

import piecewise_regression

# Relative imports cannot be used with "import .a" form; use "from . import a" instead. -Pylance
from . import calc_utilities as calc
from .plotting_utilities import set_standard_ticks, set_xlims, STANDARD_QUICKLOOK_FIGSIZE, \
                                STANDARD_FIGSIZE, STANDARD_LEGENDSIZE

from .validate import _validate_index_choice, _validate_plot_style, _validate_fit_convergence

DEFAULT_NUM_OF_BREAKPOINTS = 1
DEFAULT_SELECTION_ALPHA = 0.12

BREAKPOINT_SHADING_ALPHA = 0.18


class Reg:

    def __init__(self, data:pd.DataFrame):
        self.data = data
        self.selection_max_x = pd.NaT
        self.selection_max_y = np.nan


    def set_selection_max(self, x, y):
        """
        Sets the parameters by which data selection will be applied when running
        regression analysis.
        """
        self.selection_max_x = x
        self.selection_max_y = y


    def onclick(self, event):
        """
        Store coordinates to class attributes when clicking the interactive plot.
        Also draws a vertical line marking the end of the selection criterion.
        """
        if event.xdata is not None and event.ydata is not None:
            self.set_selection_max(x=event.xdata, y=event.ydata)

        self.ax.axvline(x=self.clicked_coords[-1][0])


    def draw_selection_line_marker(self, x):
        self.quicklook_ax.axvline(x=x)


    def quicklook(self, channel:str=None, resample:str=None, xlim:list=None) -> None:
        """
        Makes a quicklook plot of one or more channels for a given dataframe.
        Meant to be used in interactive mode, so that the user can apply data selection
        by clicking.

        Comprehensive example of ipympl: https://matplotlib.org/ipympl/examples/full-example.html

        Parameters:
        --------------
        channel : str, list
        resample : str
        xlim : list
        """

        # Apply resampling if asked to
        if isinstance(resample,str):
            data = calc.resample_df(df=self.data, avg=resample)
        else:
            data = self.data.copy(deep=True)

        # Make sure that channel is a list to iterate over
        if channel is None:
            channel = list(data.columns)
        if isinstance(channel,(str,int)):
            channel = [channel]

        # Attach the fig and axes to class attributes
        self.quicklook_fig, self.quicklook_ax = plt.subplots(figsize=STANDARD_QUICKLOOK_FIGSIZE)

        # Attach the onclick() -method to a mouse button press event for the interactive plot
        self.quicklook_fig.canvas.mpl_connect('button_press_event', self.onclick)

        # Set the axis settings
        self.quicklook_ax.set_yscale("log")
        set_xlims(ax=self.quicklook_ax, data=data, xlim=xlim)
        set_standard_ticks(ax=self.quicklook_ax, labelsize=None)

        # Plot the curves
        for ch in channel:
            self.quicklook_ax.step(data.index.values, data[ch].values, where="mid", label=ch)

        # Add the legend and show the figure
        self.quicklook_ax.legend(fontsize=STANDARD_LEGENDSIZE)
        plt.show()


def workflow(data, channel:str, resample:str=None, xlim:list=None,
            window:int=None, threshold:float=None, plot:bool=True, diagnostics=False,
            index_choice="time_s", plot_style="step", breaks=1):
    """
    Seeks for the first peak in the given data. Cuts the data and only considers that part which comes
    before the first peak. In this chosen part, seek (a) break/s in the linear trend that is the background
    of the event. The break corresponds to the start of the event, and the second linear fit corresponds
    to the slope of the rising phase of the event (the linear slope of the 10-based logarithm).

    Parameters:
    -----------
    data : {pd.DataFrame}
    channel : {str}
    resample : {str}
    xlim : {list}
    window : {str}
    threshold : {float}
    plot : {bool}
    diagnostics : {bool}
    index_choice : {str}
    plot_style : {str} Either 'step' or 'scatter'
    breaks : {int} Number of breaks

    Returns:
    ----------
    results_dict : {dict} A dictionary of results that contains 'const', 'slope1', 'slope2', 
                          'break_point' and 'break_errors'.
    """

    # Run checks
    _validate_index_choice(index_choice=index_choice)
    _validate_plot_style(plot_style=plot_style)

    # Choose resampling:
    if isinstance(resample, str):
        data = calc.resample_df(df=data, avg=resample)
    # If no resampling, just take a copy of the original data to avert 
    # modifying it by accident
    else:
        data = data.copy(deep=True)

    # Select the channel and produce indices for them. The indices are stored in the 
    # column "time_s", for they read seconds since the Epoch (1970-01-01 00:00).
    # The index numbers can be used for the regression algorithm instead of datetime values. 
    data = calc.produce_index_numbers(df=data)
    data = calc.select_channel_nonzero_ints(df=data, channel=channel)

    # Convert to log
    series = calc.ints2log10(intensity=data[channel])
    # This is what's getting plotted
    plot_series = series.copy(deep=True)

    # Get the numerical index of the first peak to choose the selection from 
    # background to first peak. Also generate numerical index to run from 0 to max_idx
    max_val, max_idx = calc.search_first_peak(ints=series, window=window, threshold=threshold)

    # Apply a slice/selection to the data series and the numerical indices (seconds since Epoch)
    # according to the first peak found
    series = series[:max_idx]
    numerical_indices = data[index_choice].values[:max_idx]

    # Get the fit results
    fit_results = break_regression(ints=series.values, indices=numerical_indices, num_of_breaks=breaks)

    # The results are a dictionary, extract values here. Also check that the result converged.
    estimates = fit_results["estimates"]
    regression_converged = fit_results["converged"]

    _validate_fit_convergence(regression_converged=regression_converged)

    const, list_of_alphas, list_of_breakpoints, list_of_breakpoint_errs = unpack_fit_results(fit_results=estimates,
                                                                                             num_of_breaks=breaks)

    # Finds corresponding timestamps to the numerical indices
    list_of_dt_breakpoints, list_of_dt_breakpoint_errs = breakpoints_to_datetime(series=series, numerical_indices=numerical_indices,
                                                                                 list_of_breakpoints=list_of_breakpoints,
                                                                                 list_of_breakpoint_errs=list_of_breakpoint_errs,
                                                                                 index_choice=index_choice)

    # Compile a results dictionary to eventually return
    results_dict = {"const": const}
    for i, alpha in enumerate(list_of_alphas):
        results_dict[f"alpha{i}"] = alpha
    for i, bp in enumerate(list_of_dt_breakpoints):
        results_dict[f"breakpoint{i}"] = bp
    for i, bp_errs in enumerate(list_of_dt_breakpoint_errs):
        results_dict[f"breakpoint{i}_errors"] = bp_errs

    if plot:

        # Init figure
        fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE)

        if diagnostics:
            print(f"Data selection: {series.index[0]}, {series.index[-1]}")
            print(f"Regression converged: {regression_converged}")
            # Generate the fit lines to display on the plot
            list_of_fit_series = calc.generate_fit_lines(data_df=data, indices=numerical_indices, const=const,
                                                        list_of_alphas=list_of_alphas, 
                                                        list_of_breakpoints=list_of_breakpoints, index_choice=index_choice)

            # Plot the fit results on the real data
            for line in list_of_fit_series:
                ax.plot(line.index, line.values, lw=2.8, ls="--", c="maroon", zorder=3)

            # Apply a span over xmin=start and xmax=max_idx to display the are considered for the fit
            ax.axvspan(xmin=series.index[0], xmax=series.index[-1], facecolor="green", alpha=DEFAULT_SELECTION_ALPHA)

        ax.set_ylabel("Log(intensity)", fontsize=STANDARD_LEGENDSIZE)

        # Plot the intensities
        if plot_style=="step":
            ax.step(plot_series.index, plot_series.values, label=channel, zorder=1, where="mid")
        if plot_style=="scatter":
            ax.scatter(plot_series.index, plot_series.values, label=channel, zorder=1)

        for i, breakpoint_dt in enumerate(list_of_dt_breakpoints):

            ax.axvspan(xmin=list_of_dt_breakpoint_errs[i][0], xmax=list_of_dt_breakpoint_errs[i][1], alpha=BREAKPOINT_SHADING_ALPHA, color="red")
            ax.axvline(x=breakpoint_dt, c="red", lw=1.8, label=f"breakpoint{i}: {breakpoint_dt.strftime('%H:%M:%S')}")

        ax.legend(fontsize=STANDARD_LEGENDSIZE)

        set_xlims(ax=ax, data=data, xlim=xlim)
        set_standard_ticks(ax=ax)

        # Format the x-axis
        ax.xaxis.set_major_formatter(DateFormatter("%H:%M\n%d"))
        ax.set_xlabel(f"date of {breakpoint_dt.strftime('%b %Y')}", fontsize=STANDARD_LEGENDSIZE)

        plt.show()

    # When diagnostics is enabled, return additional info about the run
    if diagnostics:
        results_dict["series"] = series
        results_dict["indices"] = numerical_indices
        results_dict["data_df"] = data
        for i, line in enumerate(list_of_fit_series):
            results_dict[f"line{i}"] = line

    return results_dict


def break_regression(ints, indices, starting_values:list=None, num_of_breaks:int=None) -> dict:
    """
    Initializes the Fit of piecewise_regression package, effectively running the algorithm for
    given data.

    Parameters:
    -----------
    ints : {array-like} The intensity (logarithms)
    indices : {array-like} The x-axis values (ordinal numbers or such)
    starting_values : {list}
    num_of_breaks : {int} Number of expected breakpoints.

    Returns:
    --------
    fit_results : {dict} A dictionary that contains the results of analysis.
    """

    if num_of_breaks is None:
        num_of_breaks = DEFAULT_NUM_OF_BREAKPOINTS

    fit = piecewise_regression.Fit(xx=indices,
                                   yy=ints,
                                   start_values=starting_values,
                                   n_breakpoints=num_of_breaks)

    return fit.get_results()


def unpack_fit_results(fit_results:dict, num_of_breaks:int) -> tuple:
    """

    Parameters:
    -----------
    fit_results : {dict}

    Returns:
    -----------
    const : {float} The constant of the first fit
    list_of_alphas : {list[float]} A list of slopes for the polynomial fits.
    list_of_breakpoints : {list[float]} A list of breakpoints for the fits.
    """

    # The constant and slope of the first fit are always there.
    const = fit_results["const"]["estimate"]
    alpha = fit_results["alpha1"]["estimate"]

    # Initialize lists to collect values into
    list_of_alphas = [alpha]
    list_of_breakpoints = []
    list_of_breakpoint_errs = []

    # For a single break, there will be one iteration in the loop -> one additional slope. 
    for i in range(num_of_breaks):

        # Access the i+2th index in fit_results, because that package's indexing somehow starts from 1, not 0.
        alpha = fit_results[f"alpha{i+2}"]["estimate"]
        break_point = fit_results[f"breakpoint{i+1}"]["estimate"]
        break_point_errs = fit_results[f"breakpoint{i+1}"]["confidence_interval"]

        list_of_alphas.append(alpha)
        list_of_breakpoints.append(break_point)
        list_of_breakpoint_errs.append(break_point_errs)

    return const, list_of_alphas, list_of_breakpoints, list_of_breakpoint_errs


def breakpoints_to_datetime(series:pd.Series, numerical_indices:np.ndarray, list_of_breakpoints:list, 
                                list_of_breakpoint_errs:list, index_choice:str):
    """
    Converts breakpoints (along with their errors) that are floats to datetimes.

    Parameters:
    -----------
    series : {pd.Series} The data series indexed by time.
    numerical_indices : {np.ndarray} The numerical indices of data, either ordinal numbers or seconds.

    list_of_breakpoints : {list[float]}
    list_of_breakpoint_errs : {list[tuple]}
    index_choice : {str} Either 'counting_numbers' or 'time_s'

    Returns:
    -----------
    list_of_dt_breakpoints : {list[datetime]}
    list_of_dt_breakpoint_errs : {list[tuple]}
    """

    list_of_dt_breakpoints = []
    list_of_dt_breakpoint_errs = []

    if index_choice == "counting_numbers":
        # Choose the LAST entry of a linear space of integers that map to numerical_indices smaller than
        # the break_point. This is "how manieth" data point break_point is in series.
        lin_idx = np.linspace(start=0, stop=len(series)-1, num=len(series))
        for i, break_point in enumerate(list_of_breakpoints):
            break_point_idx = lin_idx[numerical_indices<break_point][-1]
            break_point_err_minus_idx = lin_idx[numerical_indices<list_of_breakpoint_errs[i][0]][-1]
            break_point_err_plus_idx = lin_idx[numerical_indices<list_of_breakpoint_errs[i][1]][-1]
            breakpoint_dt = calc.get_interpolated_timestamp(datetimes=series.index, break_point=break_point_idx)
            breakpoint_dt_minus_err = calc.get_interpolated_timestamp(datetimes=series.index, break_point=break_point_err_minus_idx)
            breakpoint_dt_plus_err = calc.get_interpolated_timestamp(datetimes=series.index, break_point=break_point_err_plus_idx)
            list_of_dt_breakpoints.append(breakpoint_dt)
            list_of_dt_breakpoint_errs.append((breakpoint_dt_minus_err, breakpoint_dt_plus_err))
    else:
        for i, break_point in enumerate(list_of_breakpoints):
            breakpoint_dt = pd.to_datetime(break_point, unit='s')
            breakpoint_dt_minus_err = pd.to_datetime(list_of_breakpoint_errs[i][0], unit='s')
            breakpoint_dt_plus_err = pd.to_datetime(list_of_breakpoint_errs[i][1], unit='s')
            list_of_dt_breakpoints.append(breakpoint_dt)
            list_of_dt_breakpoint_errs.append((breakpoint_dt_minus_err, breakpoint_dt_plus_err))

    return list_of_dt_breakpoints, list_of_dt_breakpoint_errs





