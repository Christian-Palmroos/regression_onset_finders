
"""
Contains calculation utility functions and constants for linear regression model -based SEP event onset analysis
python package.
"""

__author__ = "Christian Palmroos"

import numpy as np
import pandas as pd


# Global constants
INDEX_NUMBER_COL_NAME = "time_s"
COUNTING_NUMBERS_COL_NAME = "counting_numbers"


def select_channel_nonzero_ints(df:pd.DataFrame, channel:str, dropnan:bool=True):
    """
    Selects the intensities (values) from the dataframe[channel] selection such that
    no zeroes are left in the dataframe. Also drops nans if dropnan (default True) is 
    enabled.
    """

    # Work on a copy to not alter the original one
    df = df.copy(deep=True)

    counting_numbers =  np.linspace(start=0, stop=len(df)-1, num=len(df))
    df[COUNTING_NUMBERS_COL_NAME] = counting_numbers.astype(int)

    selection = df[[channel, INDEX_NUMBER_COL_NAME, COUNTING_NUMBERS_COL_NAME]]
    selection = selection.loc[selection[channel]!=0]

    if dropnan:
        # Selects the entries for which "channel" column has no nans
        selection = selection[~selection[channel].isnull()]

    return selection


def produce_index_numbers(df:pd.DataFrame):
    # Work on a copy to not alter the original one
    df = df.copy(deep=True)
    index_numbers = df.index.strftime("%s")
    df[INDEX_NUMBER_COL_NAME] = index_numbers.astype(int)
    return df


def resample_df(df:pd.DataFrame, avg:str) -> pd.DataFrame:
    """
    Resamples a dataframe such that care is taken on the offset and origin of the data index.

    Parameters:
    ----------
    df : {pd.DataFrame}
    avg : {str} Resampling string.
    """

    if isinstance(avg,str):
        avg = pd.Timedelta(avg)

    copy_df = df.resample(rule=avg, origin="start", label="left").mean()

    # After resampling, a time offset must be added to preserve the correct alignment of the time bins
    copy_df.index = copy_df.index + pd.tseries.frequencies.to_offset(avg/2)

    return copy_df


def ints2log10(intensity) -> pd.Series:
    """
    Converts intensities to log(intensity).

    Parameters:
    -----------
    intensity : {pd.Series}

    Returns:
    ----------
    logints : {pd.Series}
    """

    # Takes the logarithm of the ints
    logints = np.log10(intensity)

    # There may be zeroes in the intensities, which get converted to -inf
    # Convert -infs to nan
    logints.replace([np.inf, -np.inf], np.nan, inplace=True)

    return logints


def generate_fit_lines(data_df:pd.DataFrame, indices:np.ndarray, const:float, list_of_alphas:list[float], 
                       list_of_breakpoints:list[float], index_choice:str) -> list[pd.Series]:
    """
    Generates a list of first order polynomials as pandas Series from given fit parameters.

    Parameters:
    ----------
    data_df : {pd.DataFrame} The intensity dataframe, indexed by time.
    indices : {array-like} The numerical indices of the data, the x-axis. They are either ordinal numbers or seconds.
    const : {float} The constant of the first linear fit.
    list_of_alphas : {list[float]} The slopes of the fits. Is always one longer than list_of_breakpoints.
    list_of_breakpoints : {float} The breakpoints of the fit lines. Always one shorter than list_of_alphas.

    Returns:
    --------
    list_of_lines : {list[pd.Series]} The lines.
    """

    # Gather the index selections to this list. Each fit has its own selection of the total indices.
    # The selections are separated by breakpoints in the fits. Also collect the list of line values
    # to its own list.
    list_of_index_selections = []
    list_of_lines = []
    for i, alpha in enumerate(list_of_alphas):

        # Define the selection (start&end) and apply it to all indices. Save the selected slice to a list.
        # For the start of the selection, first take 0, and then always index i-1 from breakpoints.
        # For the end of the selection, always take ith breakpoint, except for the final (take len(indices)==final index)
        selection_start = list_of_breakpoints[i-1] if i > 0 else 0
        selection_end = list_of_breakpoints[i] if i < len(list_of_breakpoints) else len(indices)
        index_selection = indices[(indices>=selection_start)&(indices<selection_end)]
        list_of_index_selections.append(index_selection)

        # Choose the constant term for the 1st order polynomial:
        if i == 0:
            line_const = const
        else:
            # Depending on the orientation of the previous line, the next line starts from the max or the 
            # min of the previous line.
            line_const = np.nanmax(list_of_lines[i-1]) if list_of_alphas[i-1] > 0 else np.nanmin(list_of_lines[i-1])

        # Generate the line and add it to the list of lines
        line = (list_of_index_selections[i] * alpha) + line_const
        list_of_lines.append(line)

    # Generate a list of datetime selection to index the lines
    list_of_datetimes = _generate_fits_datetimes(list_of_indices=list_of_index_selections, data_df=data_df, index_choice=index_choice)

    # Generate the list of series from list of lines (list of numpy arrays that contain the values of the lines)
    # and from the list of indices (which contain the corresponding x-values to the lines)
    list_of_series = [pd.Series(list_of_lines[i], index=list_of_datetimes[i]) for i in range(len(list_of_alphas))]

    return list_of_series


def _generate_fits_datetimes(list_of_indices:list, data_df:pd.DataFrame, index_choice:str):
    """

    Parameters:
    -----------
    list_of_indices : {list[pd.Series]} Fits generated from fit parameters.
    data_df : {pd.DataFrame} The dataframe that contains the selected data, indexed by time.
    index_choice : {str} Either 'counting_numbers' or 'time_s'

    Returns:
    -----------
    list_of_datetimes : {list[datetime]}
    """

    list_of_datetimes = []
    if index_choice=="counting_numbers":
        for indices in list_of_indices:
            datetimes_selection = data_df.loc[data_df[COUNTING_NUMBERS_COL_NAME].isin(indices)].index
            list_of_datetimes.append(datetimes_selection)
    else:
        for indices in list_of_indices:
            datetimes_selection = pd.to_datetime(indices, unit='s')

    return list_of_datetimes


def get_interpolated_timestamp(datetimes, break_point) -> pd.Timestamp:
    """
    Finds a timestamp from a series that relates to a floating-point index rather than integer.

    Parameters:
    -----------
    datetimes : {DatetimeIndex or similar}
    break_point : {float}

    Returns:
    ----------
    interpolated_timestamp : {pd.Timestamp}
    """

    # The "floor" of the index and the fractional part separately
    lower_index = int(break_point)
    fractional_part = break_point - lower_index

    # State the two timestamps to interpolate between
    lower_timestamp = datetimes[lower_index]
    try:
        upper_timestamp = datetimes[lower_index+1]
    except IndexError as ie:
        print(ie, fractional_part)
        upper_timestamp = lower_timestamp

    # Calculate the interpolated timestamp
    interpolated_timestamp = lower_timestamp + fractional_part * (upper_timestamp - lower_timestamp)

    return interpolated_timestamp


def search_first_peak(ints, window=None, threshold=None) -> tuple[float, int]:
    """
    Searches for a local maximum for a given window.

    Parameters:
    -----------
    ints : {array-like}
    window : {int}
    threshold : {float}

    Returns:
    ---------
    max_val : {float}
    max_idx : {int}
    """

    # Check that there are no nans
    if np.isnan(ints).any():
        raise ValueError("NaN values are not permitted!")

    # Default window length is 30 data points
    if window is None:
        window = 30

    # Default threshold value is very small
    if threshold is None:
        max_val = -1e5
    else:
        max_val = threshold

    warnings = 0
    threshold_hit = False
    for idx, val in enumerate(ints):

        # Just do nothing until we hit threshold
        if val < max_val and not threshold_hit:
            warnings = 0
            continue

        if val >= max_val:
            threshold_hit = True
            max_val = val
            max_idx = idx
            warnings = 0
        else:
            warnings += 1

        if warnings == window:
            return max_val, max_idx

    # If the peak was not found within the given window, return the 
    # values that were found. Unless the threshold was set too high, in which case
    # raise an exception.
    try:
        _ = max_idx
    except UnboundLocalError as ule:
        print(ule)
        raise ValueError("The parameter 'threshold' was set higher than any value in the intensity time series. Either set the threshold lower, or don't give it as an input.")
    return max_val, max_idx

