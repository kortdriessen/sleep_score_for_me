import numpy as np
import scipy
import pandas as pd
import tdt
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
import yaml
import re

import hypnogram as hp
from ecephys.utils import replace_outliers
import ecephys.plot as eplt
import ecephys.signal.timefrequency as tfr
import ecephys.signal.kd_utils as kd
import ecephys.signal.kd_plotting as kp
import ecephys.signal.kd_pax as kpx
import ecephys.xrsig.hypnogram_utils as xrhyp
import tdt_xarray as tx

from sklearn.decomposition import PCA
from neurodsp.plts.utils import check_ax
bp_def = dict(delta=(0.5, 4), theta=(4, 8), sigma = (11, 16), beta = (13, 30), low_gamma = (30, 55), high_gamma = (65, 90), omega=(300, 700))

hypno_colors = {
    "Wake": "forestgreen",
    "Brief-Arousal": "chartreuse",
    "Transition-to-NREM": "lightskyblue",
    "Transition-to-Wake": "palegreen",
    "NREM": "royalblue",
    "Transition-to-REM": "plum",
    "REM": "magenta",
    "Transition": "grey",
    "Art": "crimson",
    "Wake-art": "crimson",
    "Unsure": "white",
    }


def _infer_bout_start(df, bout):
    """Infer a bout's start time from the previous bout's end time.

    Parameters
    ----------
    h: DataFrame, (n_bouts, ?)
        Hypogram in Visbrain format with 'start_time'.
    row: Series
        A row from `h`, representing the bout that you want the start time of.

    Returns
    -------
    start_time: float
        The start time of the bout from `row`.
    """
    if bout.name == 0:
        start_time = 0.0
    else:
        start_time = df.loc[bout.name - 1].end_time

    return start_time

def load_hypno(path, st):
    """Load a Visbrain formatted hypnogram."""
    df = pd.read_csv(path, sep="\t", names=["state", "end_time"], comment="*")
    df["start_time"] = df.apply(lambda row: _infer_bout_start(df, row), axis=1)
    df["duration"] = df.apply(lambda row: row.end_time - row.start_time, axis=1)
    return to_datetime(df, st)


def to_datetime(df, start_datetime):
    df = df.copy()
    df["start_time"] = start_datetime + pd.to_timedelta(df["start_time"], "s")
    df["end_time"] = start_datetime + pd.to_timedelta(df["end_time"], "s")
    df["duration"] = pd.to_timedelta(df["duration"], "s")
    return hp.DatetimeHypnogram(df)