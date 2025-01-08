
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