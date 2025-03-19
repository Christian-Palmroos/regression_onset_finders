
"""
Interactive user-interface module for selecting between SEPpy-based data loading and user-defined local data in the 
Regression-Onset-Tool notebook.
"""

__author__ = "Christian palmroos"


import ipywidgets as widgets

HANDLE_NAME = "Load data from: "
OPTIONS = ("SEPpy", "Local")
BUTTON_STYLES = ("success", "info", "warning", "")

TOGGLEBUTTON_TOOLTIPS = ("Select SEPpy for data loading",
                         "Load data from your own file")

data_file0 = widgets.Select(
                options=OPTIONS,
                value=OPTIONS[0],
                description=HANDLE_NAME,
                disabled=False
                )

data_file = widgets.ToggleButtons(
                options=OPTIONS,
                value=OPTIONS[0],
                description=HANDLE_NAME,
                button_style=BUTTON_STYLES[3],
                tooltips=TOGGLEBUTTON_TOOLTIPS,
                disabled=False
                )

