import numpy as np
import scipy
import pandas as pd
import tdt
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

import hypnogram as hp
import kd_analysis.main.kd_utils as kd
import kd_analysis.main.kd_plotting as kp
import kd_analysis.main.kd_hypno as kh
import kd_analysis.paxilline.pax_fin as kpx
import neurodsp.plts.utils as dspu
import sleep_score_for_me.v2 as ss
import sleep_score_for_me.utils as ssu

bp_def = dict(sub_delta = (0.5, 2), delta=(0.5, 4), theta=(4, 8), alpha = (8, 13), sigma = (11, 16), beta = (13, 30), low_gamma = (30, 55), high_gamma = (65, 90), omega=(300, 700))
import matplotlib
font = {'family' : 'serif',
        'weight' : 'bold',
        'size'   : 14}

matplotlib.rc('font', **font)

def simple_shaded_bp(bp,
                     hyp,
                     ax,
                     ss=12,
                     color='k',
                     linewidth=2):
    ax = sns.lineplot(x=bp.datetime, y=bp, ax=ax, color=color, linewidth=linewidth)
    kp.shade_hypno_for_me(hypnogram=hyp, ax=ax)
    return ax

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


def build_art_hypno(states_and_times):
    """ Builds the actual formatted hypnogram once the artifact detection has already been done. 
    states_and_times --> series with only the timepoints corresponding sleep states (i.e. data that has already been 'scored')
    """

    # Get boolean series for each state
    art_bool = states_and_times == 'Art'
    good_bool = states_and_times == 'Good'

    # Use the boolean series to get start and end times for each state
    art_sne = starts_and_ends(art_bool)
    good_sne = starts_and_ends(good_bool)

    # Then we convert the start and end times for each state to a partial hypnogram
    art_hyp = pd.DataFrame(columns = ['state', 'end_time', 'start_time', 'duration'])
    art_hyp[['start_time', 'end_time']] = art_sne
    art_hyp['duration'] = art_hyp.end_time - art_hyp.start_time
    art_hyp['state'] = 'Art'

    good_hyp = pd.DataFrame(columns = ['state', 'end_time', 'start_time', 'duration'])
    good_hyp[['start_time', 'end_time']] = good_sne
    good_hyp['duration'] = good_hyp.end_time - good_hyp.start_time
    good_hyp['state'] = 'Good'

    #Then we concat those and sort by the start_time 
    fin_hypno = pd.concat([art_hyp, good_hyp])
    fin_hypno = fin_hypno.sort_values('start_time').reset_index(drop=True)
       
    return fin_hypno

def artifactus_inspectus(spg, mspg, chan, ss=8, ptiles=(50, 75, 90, 95, 97, 99)):
    spg = spg.copy()
    mspg = mspg.copy()
    try:
        spg = spg.sel(channel=chan)
    except KeyError:
        pass
    
    bp_set = kd.get_bp_set2(spg, bp_def)
    omega = kd.get_bp_set2(mspg, bp_def).omega
    
    gamma = bp_set['low_gamma']
    delta = bp_set['delta']
    
    os = kd.get_smoothed_da(omega, smoothing_sigma=ss)
    gs = kd.get_smoothed_da(gamma, smoothing_sigma=ss)
    
    ds = kd.get_smoothed_da(delta, smoothing_sigma=ss)
    
    ogs = os*gs
    print(type(ds))
    fig, (d, g)  = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(40, 12))
    d = sns.lineplot(data=ds, ax=d, color='forestgreen')
    g = sns.lineplot(data=ogs, ax=g, color='steelblue')
    for p in ptiles:
        dp = np.percentile(ds, p)
        gp = np.percentile(ogs, p)
        d.axhline(y=dp, ls='--', color='k')
        d.set_title('Smoothed Delta Bandpower')
        g.axhline(y=gp, ls='--', color='k')
        g.set_title('(Smoothed Omega)*(Smoothed Gamma) "Bandpower"')
    plt.show()
    print(str(ptiles[0]))
    print(str(ptiles[1]))
    print(str(ptiles[2]))
    print(str(ptiles[3]))
    print(str(ptiles[4]))
    print(str(ptiles[5]))
    
    d_in = float(input("Enter Delta Threshold: "))
    d_cut = np.percentile(ds, d_in)
    g_in = float(input("Enter Omega-Gamma Threshold: "))
    g_cut = np.percentile(ogs, g_in)
    
    return ds, ogs, d_cut, g_cut

def artifactus_identicus(spg, mspg, bp_def, chan, ss=8):
    """
    Detects artifacts in a recording based on neural data and EMG, 
    returns a hypnogram which marks all timepoints as 'Good' or 'Art'
    """
    #First we need to properly allign the data so the datetime samples are evenly spaced.
    dt_original = spg.datetime.values
    start = dt_original.min()
    dt_freq = scipy.stats.mode(np.diff(dt_original)).mode[0]
    dt_freq = dt_freq / pd.to_timedelta(1, "ns")
    dt_freq = str(dt_freq)+'ns'
    new_dti = pd.date_range(start, periods=len(dt_original), freq=dt_freq)
    spg = spg.assign_coords(datetime=new_dti)
    mspg = mspg.assign_coords(datetime=new_dti)

    # Here we get the Bandpower sets and pull out the delta and omega from the data and muscle, respectively
    delta, omegamma, d_cut, o_cut = artifactus_inspectus(spg, mspg, chan=chan, ss=ss)

    #Here we build the DF for assigning the Artifact states:
    dt_ix = delta.datetime.values
    art_ser = pd.Series(delta, index=dt_ix)
    art_df = art_ser.to_frame(name='Delta')
    art_df['Omegamma'] = omegamma.values
    art_df['state'] = np.nan
    
    # Here we actually assign the artifact and good states for all timepoints
    art_df.loc[np.logical_and(art_df.Delta >= d_cut, art_df.Omegamma >= o_cut), 'state'] = 'Art'
    art_df.loc[art_df.state!='Art', 'state'] = 'Good'  
    
    #Now we just build the hypnogram itself and return it
    art_hypno = build_art_hypno(art_df['state'])
    art_hypno = hp.DatetimeHypnogram(art_hypno)
    
    fig, ax = plt.subplots(figsize=(40, 12))
    ax = simple_shaded_bp(delta, art_hypno, ax=ax, ss=None)
    
    return art_hypno

def artifactus_rejectus(spg, hypno):
    """Takes in an spg with a given channel, and returns a clean version with all artifacts cleaned, based on the hypnogram for the same channel
    """
    spg_states = kh.add_states(spg, hypno)
    good_spg = spg_states.where(spg_states.state == 'Good')
    return good_spg

