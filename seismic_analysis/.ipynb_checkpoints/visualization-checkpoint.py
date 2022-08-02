import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext
from scipy.signal import spectrogram
import datetime
import obspy
import numpy as np



def make_spectrogram(st,starttime,endtime,low_freq,window_length,n_overlap):
    
    # get metadata
    station = st[0].stats.station
    channel = st[0].stats.channel

    # trim data to desired bounds
    st = st.trim(starttime=starttime,endtime=endtime)

    # make spectrogram
    f,t,s = spectrogram(st[0].data*1000, fs=st[0].stats.sampling_rate, nperseg=window_length,noverlap=n_overlap)
    
    # make plot
    fig,ax = plt.subplots(2,1,figsize=[20,15],gridspec_kw={'height_ratios': [1, 3]})
    
    # plot data
    hours = np.floor((endtime-starttime)/3600)+1
    ticks = [starttime.datetime + datetime.timedelta(seconds=hour*3600) for hour in range(int(hours))]
    times = [starttime.datetime + datetime.timedelta(seconds=s/st[0].stats.sampling_rate) for s in range(st[0].stats.npts)]
    ax[0].plot(times,st[0].filter("highpass",freq=low_freq).data*1000,'k')
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax[0].set_xticks(ticks)
    ax[0].grid(True)
    box = ax[0].get_position()
    box.y0 = box.y0 - 0.1
    box.y1 = box.y1 - 0.1
    ax[0].set_position(box)
    ax[0].set_ylabel("Velocity (mm/s)")
    ax[0].set_title(station+" "+channel+" (>"+str(low_freq)+" Hz)")
 
    # plot spectrogram
    times = [starttime + datetime.timedelta(seconds=time) for time in t]
    vrange = np.log10(np.max(s))-np.log10(np.min(s))
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax[1].get_shared_x_axes().join(ax[0],ax[1])
    ax[1].set_xticks(ticks)
    #ax[1].set_xlim([times[0]-datetime.timedelta(seconds=10),times[-1]+datetime.timedelta(seconds=10)])
    ax[1].set_yticks([-3,-2,-1,0,1])
    ax[1].set_yticklabels(["$10^{-3}$","$10^{-2}$","$10^{-1}$","$10^{0}$","$10^{1}$"])
    ax[1].set_ylabel("Frequency (Hz)")
    ax[1].set_xlabel("Time")
    plt.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.9, 0.1225, 0.025, 0.5])
    spec = ax[1].pcolor(times, np.log10(f), np.log10(s), vmin=np.log10(np.min(s))+vrange*0.5,vmax=np.log10(np.max(s)))
    cbar = plt.colorbar(spec,label="PSD ((mm/s)$^2$/Hz)",ticks=[-10,-8,-6,-4,-2,0],cax=cbar_ax)
    cbar.ax.set_yticklabels(['$10^{-10}$', '$10^{-8}$', '$10^{-6}$', '$10^{-4}$','$10^{-2}$','$10^0$']) 
    plt.savefig("outputs/figures/" + station + "_" + channel + "_spectrogram.png")