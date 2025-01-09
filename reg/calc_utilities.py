
"""
Contains calculation utility functions and constants.
"""

__author__ = "Christian Palmroos"

import pandas as pd

def resample_df(df, avg):
    """
    Resamples a dataframe such that care is taken on the offset and origin of the data index.
    """

    if isinstance(avg,str):
        avg = pd.Timedelta(avg)

    copy_df = df.resample(rule=avg, origin="start", label="left").mean()

    # After resampling, a time offset must be added to preserve the correct alignment of the time bins
    copy_df.index = copy_df.index + pd.tseries.frequencies.to_offset(avg/2)

    return copy_df


def search_first_peak(ints, window=None, threshold=None):
    """
    Searches for a local maximum for a given window.
    
    ints : {array-like}
    
    window : {int}
    
    threshold : {float}
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
