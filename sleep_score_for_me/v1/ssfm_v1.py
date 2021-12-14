import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import ecephys.signal.kd_utils as kd
import hypnogram as hp

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


"""
Functions Needed to do the actual scoring and build the hypnogram
-----------------------------------------------------------------
"""

def ssfm_v1(spg, emg_spg, nrem_percentiles=[50, 60, 65, 70], rem_percentiles=[60, 70, 80, 85], chan=2, f_range=slice(0, 100)):
    """Uses the metrics and scoring techniques of Watson et al., 2016 (Neuron) to score a chunk of data into NREM, REM, and Wake
    
    spg --> xarray spectrogram containing the channel you wish to use for scoring (can contain other channels as well, as long as 'chan' option is used)
    emg_spg --> spectrogram of emg, channel 1 is always selected out by default
    chan --> the channel you want to use (should choose a parietal EEG if possible)
    f_range --> the range of frequencies to use for the PCA
    percentiles --> these are the percentiles you want to see plotted in order to determine the NREM and REM threshold values"""

    # First we get the EMG Band: 
    emg_bands = kd.get_bp_set2(emg_spg, bands=bp_def, pandas=True)
    emg = emg_bands.xs(1, level='channel').omega
    
    # Then we get one channel in a given frequency range, and do the PCA
    spg_np = spg.sel(channel=chan, frequency=f_range).to_numpy().T
    pca = PCA(n_components=1)
    pca.fit(spg_np)
    pc = pca.transform(spg_np).flatten()

    # Then Theta Band: 
    theta_narrow = (5, 10)
    theta_wide = (2, 16) 
    thetaband = kd.get_bandpower(spg.sel(channel=chan), f_range=theta_narrow)/kd.get_bandpower(spg.sel(channel=chan), f_range=theta_wide)
    thetaband = (thetaband/emg)/emg

    # Now we can construct the Dataframe which will be used to assign states
    dt_ix = spg.datetime.values
    scoring_df = pd.Series(pc, index=dt_ix)
    scoring_df = scoring_df.to_frame(name='PC1')
    scoring_df['Theta'] = thetaband.values
    scoring_df['EMG'] = emg.values
    scoring_df['state'] = np.nan 
    
    # Now we need to figure out and set the thresholds for scoring out the NREM:
    hist, pcax = threshplot(pc, time=spg.datetime.values, percentiles=nrem_percentiles)
    
    nrem_threshold = float(input("Enter NREM Threshold: "))
    nrem_threshold = np.percentile(pc, nrem_threshold)

    #This is where we actually "score" the NREM based on a simple threshold value
    scoring_df.loc[scoring_df.PC1 >= nrem_threshold, 'state'] = 'NREM'
    
    # Now we need to set the threshold for REM::
    hist_rem, theta = threshplot(data=thetaband.values, time=spg.datetime.values, percentiles=rem_percentiles)

    rem_threshold = float(input("Enter REM Threshold: "))
    rem_threshold = np.percentile(thetaband.values, rem_threshold)

    #This is where we actually "score" the REM based on a simple threshold value
    scoring_df.loc[np.logical_and(scoring_df.Theta>=rem_threshold, scoring_df.state != 'NREM'), 'state'] = 'REM'
    
    #Now we just score the rest of the hypnogram as Wake:
    scoring_df.loc[np.logical_and(scoring_df.state!="NREM", scoring_df.state!='REM'), 'state'] = 'Wake'

    "At this point, all of the data is actually scored, and we can simply call build_hypno_for_me to get the start and end times and build the hypnogram"
    
    final_hypno = build_hypno_for_me(scoring_df['state'])
    m, d, g = plot_hypno_for_me(final_hypno, spg, emg_spg, bp_def)

    return hp.DatetimeHypnogram(final_hypno)


def build_hypno_for_me(states_and_times):
    """ states_and_times --> series with only the timepoints corresponding sleep states (i.e. data that has already been 'scored')"""
    
    # Get boolean series for each state
    nrem_bool = states_and_times == 'NREM'
    wake_bool = states_and_times == 'Wake'
    rem_bool = states_and_times == 'REM'

    # Use the boolean series to get start and end times for each state
    nrem_sne = starts_and_ends(nrem_bool)
    wake_sne = starts_and_ends(wake_bool)
    rem_sne = starts_and_ends(rem_bool)

    # Then we convert the start and end times for each state to a partial hypnogram
    nrem_hyp = pd.DataFrame(columns = ['state', 'end_time', 'start_time', 'duration'])
    nrem_hyp[['start_time', 'end_time']] = nrem_sne
    nrem_hyp['duration'] = nrem_hyp.end_time - nrem_hyp.start_time
    nrem_hyp['state'] = 'NREM'

    wake_hyp = pd.DataFrame(columns = ['state', 'end_time', 'start_time', 'duration'])
    wake_hyp[['start_time', 'end_time']] = wake_sne
    wake_hyp['duration'] = wake_hyp.end_time - wake_hyp.start_time
    wake_hyp['state'] = 'Wake'

    rem_hyp = pd.DataFrame(columns = ['state', 'end_time', 'start_time', 'duration'])
    rem_hyp[['start_time', 'end_time']] = rem_sne
    rem_hyp['duration'] = rem_hyp.end_time - rem_hyp.start_time
    rem_hyp['state'] = 'REM'

    #Then we concat those and sort by the start_time 
    fin_hypno = pd.concat([nrem_hyp, wake_hyp, rem_hyp])
    fin_hypno = fin_hypno.sort_values('start_time').reset_index(drop=True)
       
    return fin_hypno

def starts_and_ends(s, minimum_duration=np.timedelta64(3, 's')):
    start_times = np.empty(0)
    end_times = np.empty(0)
    period = s.index[1] - s.index[0]

    s_trues = s[s==True]
    ix = s_trues.index
    ix_counter = np.arange(0, len(ix))

    try:
        for i in ix_counter:
            if (ix[i] - period) != ix[i-1]:
                start_times = np.append(start_times, ix[i])
            if (ix[i] + period) != ix[i+1]:
                end_times = np.append(end_times, (ix[i] + period))
            elif np.logical_and((ix[i] + period) == ix[i+1], (ix[i] - period) == ix[i-1]):
                pass 
    except IndexError:
        print('passing indexing error')
        pass
    end_times = np.append(end_times, (ix[ix_counter.max()] + period))
    return [(start_time, end_time)
            for start_time, end_time in zip(start_times, end_times)
            if end_time >= (start_time + minimum_duration)]


"""
PLOTTING FUNCTIONS
------------------
"""

def threshplot(data, time=None, percentiles=[50, 60, 65, 70]):
    
    #%matplotlib inline
    # Plot the threshold options for scoring out the NREM: 
    f, h_ax = plt.subplots(figsize=(40, 15))
    h_ax = sns.histplot(data=data, bins=40, ax=h_ax)
    h_ax.axvline(np.percentile(data, percentiles[0]), color='magenta')
    h_ax.axvline(np.percentile(data, percentiles[1]), color='b')
    h_ax.axvline(np.percentile(data, percentiles[2]), color='forestgreen')
    h_ax.axvline(np.percentile(data, percentiles[3]), color='r')
    plt.show()
    
    f, lin_ax = plt.subplots(figsize=(40, 15))
    lin_ax = sns.lineplot(x=time, y=data, ax=lin_ax)
    lin_ax.axhline(np.percentile(data, percentiles[0]), color='magenta')
    lin_ax.axhline(np.percentile(data, percentiles[1]), color='b')
    lin_ax.axhline(np.percentile(data, percentiles[2]), color='forestgreen')
    lin_ax.axhline(np.percentile(data, percentiles[3]), color='r')
    plt.show()
    return h_ax, lin_ax

def plot_hypno_for_me(hypno, spg, emg_spg, bp_def, chan=2, smooth=False):
    fig, (m, d, g) = plt.subplots(ncols=1, nrows=3, figsize=(35,15))
    emg_spg = emg_spg.sel(channel=1)
    spg = spg.sel(channel=chan)
    
    #plot muscle activity
    emg_bp = kd.get_bandpower(emg_spg, bp_def['omega'])
    if smooth==True:
        emg_bp = kd.get_smoothed_da(emg_bp, smoothing_sigma=12)
    sns.lineplot(x=spg.datetime, y=emg_bp, color='black', ax=m) 
    shade_hypno_for_me(hypnogram=hypno, ax=m)
    m.set_title('Muscle Activity (Full Spectrum)')

    #plot delta power
    delta = kd.get_bandpower(spg, bp_def['delta'])
    if smooth==True:
        delta = kd.get_smoothed_da(delta, smoothing_sigma=12)
    sns.lineplot(x=delta.datetime, y=delta, color='black', ax=d)
    shade_hypno_for_me(hypnogram=hypno, ax=d)
    d.set_title('EEG-'+str(chan)+' Delta Bandpower')

    #plot gamma power
    gamma = kd.get_bandpower(spg, bp_def['high_gamma'])
    if smooth==True:
        gamma = kd.get_smoothed_da(gamma, smoothing_sigma=12)
    sns.lineplot(x=spg.datetime, y=gamma, color='black', ax=g)
    shade_hypno_for_me(hypnogram=hypno, ax=g)
    g.set_title('EEG-'+str(chan)+' Gamma Bandpower')

    return m, d, g

def shade_hypno_for_me(
    hypnogram, ax=None, xlim=None
):
    """Shade plot background using hypnogram state.

    Parameters
    ----------
    hypnogram: pandas.DataFrame
        Hypnogram with with state, start_time, end_time columns.
    ax: matplotlib.Axes, optional
        An axes upon which to plot.
    """
    xlim = ax.get_xlim() if (ax and not xlim) else xlim

    ax = check_ax(ax)
    for bout in hypnogram.itertuples():
        ax.axvspan(
            bout.start_time,
            bout.end_time,
            alpha=0.3,
            color=hypno_colors[bout.state],
            zorder=1000,
            ec="none",
        )

    ax.set_xlim(xlim)
    return ax

def compare_hypnos_for_me(spg, ssfm_hyp, your_hyp):
    f, (ssfm_hyp_ax, your_hyp_ax) = plt.subplots(nrows=2, ncols=1, figsize=(35, 15))
    spg = kd.get_bandpower(spg, (0.5, 4))
    ssfm_hyp_ax = sns.lineplot(x=spg.datetime, y=spg.sel(channel=2), ax=ssfm_hyp_ax)
    ssfm_hyp_ax.set_title('SSFM Hypnogram')
    your_hyp_ax = sns.lineplot(x=spg.datetime, y=spg.sel(channel=2), ax=your_hyp_ax)
    your_hyp_ax.set_title('Your Hypnogram')
    shade_hypno_for_me(ssfm_hyp, ax=ssfm_hyp_ax)
    shade_hypno_for_me(your_hyp, ax=your_hyp_ax)
    return ssfm_hyp_ax, your_hyp_ax