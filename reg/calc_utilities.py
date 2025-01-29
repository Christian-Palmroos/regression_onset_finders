
"""
Contains calculation utility functions and constants for linear regression model -based SEP event onset analysis
python package.
"""

__author__ = "Christian Palmroos"

import numpy as np
import pandas as pd

def resample_df(df, avg) -> pd.DataFrame:
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


def ints2log10(intensity, fill_style="bfill") -> pd.Series:
    """
    Converts intensities to log(intensity). Fills -infs with 
    fill_style.
    
    Parameters:
    -----------
    intensity : {pd.Series}
    fill_style : {str} either 'bfill' or 'ffill'
    
    Returns:
    ----------
    logints : {pd.Series}
    """

    VALID_FILL_STYLES = ("bfill", "ffill")

    # At the start check that the filling style is valid
    if fill_style not in VALID_FILL_STYLES:
        raise ValueError(f"Parameter fill_style must be in {VALID_FILL_STYLES}")

    # Takes the logarithm of the ints
    logints = np.log10(intensity)

    # There may be zeroes in the intensities, which get converted to -inf
    # Convert -infs to nan
    logints.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Finally fill all the nans with the selected style
    match fill_style:

        # For a reason I don't understand it's not possible to match-case against elements of 
        # VALID_FILL_STYLES here, and instead I'm forced to spell them out as strings. The current
        # implementation is inelegant and vexes me.
        case "bfill":
            logints.bfill(inplace=True)

        case "ffill":
            logints.ffill(inplace=True)

    return logints


def generate_fit_lines(indices, const, alpha1, alpha2, break_point) -> tuple[pd.Series, pd.Series]:
    """
    Generates two lines from fit parameters.
    
    Parameters:
    ----------
    indices : {array-like} The numerical indices of the data, the x-axis.
    const : {float} The constant of the first linear fit.
    alpha1 : {float} The slope of the first linear fit.
    alpha2 : {float} The slope of the second linear fit.
    break_point : {float} The point at which the gradient changes from alpha1 to alpha2.

    Returns:
    --------
    line1 : {pd.Series} the first line until break_point.
    line2 : {pd.Series} the second line from break_point to first peak.
    """

    # Working with indices, the breaking point of the two lines must
    # be an integer
    bp_int = int(break_point)

    # For the first line just take indices from start to the breakpoint
    indices_sel1 = indices[:bp_int]

    # For the second part we need two sets of indices; one running from 0 -> len(indices2)
    # to calculate the values, and the latter part of indices itself to index the values.
    indices_sel2 = indices[bp_int:]
    num_indices2 = len(indices_sel2)

    # The first line is generated very simply from start to breakpoint
    line1 = indices_sel1 * alpha1 + const

    # Depending on the orientation of the first line, line2 starts from the max or the min of the line1
    if alpha1 > 0:
        line2 = np.linspace(start=0, stop=num_indices2, num=num_indices2) * alpha2 + np.nanmax(line1)
    else:
        line2 = np.linspace(start=0, stop=num_indices2, num=num_indices2) * alpha2 + np.nanmin(line1)

    return pd.Series(line1, index=indices_sel1), pd.Series(line2, index=indices_sel2)


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
    upper_timestamp = datetimes[lower_index+1]

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
        if val < threshold and not threshold_hit:
            warnings = 0
            continue

        if val >= max_val:
            threshold_hit = True
            max_val = val
            warnings = 0
        else:
            warnings += 1

        if warnings == window:
            max_idx = idx-window
            return max_val, max_idx

    # If the peak was not found, return False and False
    return False, False
