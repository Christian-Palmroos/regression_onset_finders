
"""
Contains plotting utility functions and constants.
"""

__author__ = "Christian Palmroos"

import matplotlib.pyplot as plt

# Constants:
STANDARD_FIGSIZE = (26,11)

STANDARD_LEGENDSIZE = 26

STANDARD_TICK_LABELSIZE = 22

STANDARD__MAJOR_TICKLEN = 11
STANDARD_MINOR_TICKLEN = 8

STANDARD_MAJOR_TICKWIDTH = 2.8
STANDARD_MINOR_TICKWIDTH = 2.1

def set_standard_ticks(ax, labelsize:int=None):
    """
    Handles tickmarks, their sizes etc...
    """

    if labelsize is None:
        labelsize = STANDARD_TICK_LABELSIZE

    ax.tick_params(which="major", length=STANDARD__MAJOR_TICKLEN, width=STANDARD_MAJOR_TICKWIDTH, labelsize=labelsize)
    ax.tick_params(which="minor", length=STANDARD_MINOR_TICKLEN, width=STANDARD_MINOR_TICKWIDTH, labelsize=labelsize-5)