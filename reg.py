
"""
Contains the first development version for ramp regression and break regression algorithm -based
SEP event onset finding tools.
"""

__author__ = "Christian Palmroos"

import pandas as pd
import matplotlib.pyplot as plt

def ramp_regression():
    return 0

def break_regression():
    return 0

def quicklook(data, channel:str=None, resample:str=None, xlim:list=None):
    """
    Makes a quicklook plot of one or more channels for a given dataframe.
    
    data : dataframe
    
    channel : str, list
    
    resample : str
    
    xlim : list
    """

    STANDARD_FIGSIZE = (26,11)
    STANDARD_LEGENDSIZE = 26

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
        
    ax.legend(fontsize=STANDARD_LEGENDSIZE)
    
    plt.show()

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