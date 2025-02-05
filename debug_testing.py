
"""
Testing grounds for debugging functions in the regression_onset_finders.
"""

__author__ = "Christian Palmroos"

import os
import pandas as pd

import reg

def main(filename:str):

    PATH = f"data{os.sep}"

    df = pd.read_csv(f"{PATH}{filename}", parse_dates=True, index_col="TimeUTC")

    results, l1, l2 = reg.workflow(data=df, channel="P1", window=100, threshold=2, diagnostics=True)

    results
    l1
    l2

if __name__ == "__main__":

    filenames = ["phys_data_2023-03-12_side0.csv"]

    index = 0

    main(filename = filenames[index])

