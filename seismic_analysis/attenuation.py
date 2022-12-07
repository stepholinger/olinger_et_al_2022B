import matplotlib.pyplot as plt
import datetime
import numpy as np
import obspy
from scipy.signal import find_peaks,hilbert,butter,filtfilt
from scipy.fft import rfft,rfftfreq



def estimate_Q(st,band,time_lims,ax=None):

    # filter seismic data
    st.filter('bandpass',freqmin=band[0],freqmax=band[1])
    st.trim(starttime = time_lims[0],endtime = time_lims[1])

    # calculate peak amplitude of envelope
    peak_idx = np.argmax(np.abs(st[0].data))
    peak_amp = st[0].data[peak_idx]

    # calculate 1/e of peak amplitude
    e_fold_amp = np.abs(peak_amp) * 1/np.e

    # find peaks in signal that exceed e-folding amplitude
    peaks = find_peaks(np.abs(st[0].data),height=e_fold_amp)

    # get e-folding time
    e_fold_time = st[0].times()[peaks[0][-1]] - st[0].times()[peak_idx]

    # estimate period of signal during e-folding time
    data = st[0].data[peak_idx:peaks[0][-1]] 
    f = rfftfreq(len(data), 1/st[0].stats.sampling_rate)
    spectra = abs(rfft(data))
    T = 1/f[np.argmax(spectra)]

    # calculate Q from e-folding time
    num_cycles = e_fold_time/T
    Q = num_cycles*np.pi

    # calculate envelope and filter envelope below dominant period
#     env = abs(hilbert(st[0].data))
#     fs = st[0].stats.sampling_rate
#     ny = fs/2
#     cutoff = 2/T
#     b, a = butter(4, cutoff/ny, 'low')
#     filt_env = filtfilt(b, a, env)
     
    # make plot
    if ax != None:
        plot_e_folding(st,peak_idx,peak_amp,e_fold_amp,e_fold_time,peaks,band,ax)

    return T,Q



# make plot showing e-folding values
def plot_e_folding(st,peak_idx,peak_amp,e_fold_amp,e_fold_time,peaks,band,ax):
    t = st[0].times()
    ax.plot(t,st[0].data,'k')
    ax.hlines(y=[-e_fold_amp,e_fold_amp],xmin=t[0],xmax=t[-1],color='r',linestyle='--')
    ax.vlines(x=t[peaks[0][-1]],ymin=-abs(peak_amp)*1.1,ymax=abs(peak_amp)*1.1,color='r',linestyle='--')
    ax.scatter(t[peaks[0]],st[0].data[peaks[0]],color='r')
    ax.scatter(t[peak_idx],peak_amp,s=1000,facecolor='gold',edgecolor='k',marker="*",zorder=3)
    ax.text(t[peak_idx]-200,peak_amp,"$A_{max}$",size=12)
    ax.text(t[peaks[0][-1]],st[0].data[peaks[0][-1]],"   $\dfrac{1}{e}A_{max}$",size=12)
    ax.text(t[peaks[0][-1]]-55,-abs(peak_amp)*1.25,r"$\tau = $" + str(np.round(e_fold_time/60)) + " mins",size=12)
    ax.set_ylim(-abs(peak_amp)*1.1,abs(peak_amp)*1.1)
    ax.set_xlim(t[0],t[-1])
    ax.tick_params(axis='y',labelsize=12)
    ax.set_ylabel("Velocity (m/s)",size=15)
    plt.xlabel("Time",size=15)
    return



# estimate duration from displacement and velocity seismograms
def estimate_ringdown(st,method,decay_amp=[]):
    data = st[0].data
    if method == "cumulative":
        cumulative_amp = np.cumsum(np.abs(data)-np.median(np.abs(data)))
        max_cumulative_amp = np.max(cumulative_amp)
        end_cumulative_amp = 0.95*max_cumulative_amp
        duration_bool = cumulative_amp < end_cumulative_amp
        end_idx = np.where(duration_bool == False)[0][0]
        duration = end_idx/st[0].stats.sampling_rate
    if method == "decay":
        max_amp = np.max(np.abs(data))
        max_idx = np.argmax(np.abs(data))
        win_size = 60*st[0].stats.sampling_rate
        windowed_data = np.reshape(data[:-1],(int(len(data[:-1])/win_size),int(win_size)))
        windowed_mean_amp = np.mean(np.abs(windowed_data),axis=1)
        duration_bool = windowed_mean_amp > decay_amp
        end_idx = np.where(duration_bool == True)[0][-1]
        duration = (end_idx*win_size - max_idx)/st[0].stats.sampling_rate
#         plt.plot(np.abs(data))
#         plt.hlines(decay_amp,0,len(data))
#         plt.show()
    return duration