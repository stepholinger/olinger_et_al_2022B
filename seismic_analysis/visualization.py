import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext
from scipy.signal import spectrogram
import datetime
import obspy
import numpy as np    
from scipy.fft import rfft,rfftfreq



def plot_spectrogram(st,starttime,endtime,low_freq,window_length,n_overlap,resp):
    
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
    ax[0].tick_params(axis='both', which='major', labelsize=15)
    ax[0].grid(True)
    box = ax[0].get_position()
    box.y0 = box.y0 - 0.1
    box.y1 = box.y1 - 0.1
    ax[0].set_position(box)
    if resp == 'VEL':
        ax[0].set_ylabel("Velocity (mm/s)",fontsize=15)
    if resp == 'DISP':
        ax[0].set_ylabel("Displacement (mm)",fontsize=15)
    ax[0].set_title(station+" "+channel+" "+resp+" (>"+str(low_freq)+" Hz)",fontsize=20)
 
    # plot spectrogram
    times = [starttime + datetime.timedelta(seconds=time) for time in t]
    vrange = np.log10(np.max(s))-np.log10(np.min(s))
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax[1].get_shared_x_axes().join(ax[0],ax[1])
    ax[1].set_xticks(ticks)
    ax[1].tick_params(axis='both', which='major', labelsize=15)
    ax[1].set_xlim([times[0]-datetime.timedelta(seconds=10),times[-1]+datetime.timedelta(seconds=10)])
    ax[1].set_yticks([-3,-2,-1,0,1])
    ax[1].set_yticklabels(["$10^{-3}$","$10^{-2}$","$10^{-1}$","$10^{0}$","$10^{1}$"],fontsize=15)
    ax[1].set_ylabel("Frequency (Hz)",fontsize=15)
    ax[1].set_xlabel("Time",fontsize=15)
    plt.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.9, 0.1225, 0.025, 0.5])
    spec = ax[1].pcolormesh(times, np.log10(f[1:]), np.log10(s[1:,:]), vmin=np.log10(np.min(s))+vrange*0.5,vmax=np.log10(np.max(s)))
    cbar = plt.colorbar(spec,ticks=[-10,-8,-6,-4,-2,0],cax=cbar_ax)
    cbar.ax.set_yticklabels(['$10^{-10}$', '$10^{-8}$', '$10^{-6}$', '$10^{-4}$','$10^{-2}$','$10^0$']) 
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label("PSD ((mm/s)$^2$/Hz)",size=15)
    plt.savefig("outputs/figures/" + station + "_" + channel + "_" + resp +"_spectrogram.png")
    
    

def psd(signal,fs):
    s = rfft(signal)
    f = rfftfreq(len(signal),1/fs)
    power = np.square(np.abs(s))
    psd = power/(f[1]-f[0])
    return f,psd


    
def plot_spectra_and_timeseries(st_list,psd_list,f,low_cut,resp):
    
    # get indices corresponding to log-spaced frequency vector
    log_freq = np.logspace(np.log10(low_cut),np.log10(np.max(f)),500, endpoint = True)
    idx_list = []
    for freq in log_freq:
        diff = freq-f[1:]
        idx_list.append(np.argmin(np.abs(diff)))
    idx = np.unique(idx_list)

    # plot the spectra together
    fig = plt.figure(tight_layout=True,figsize=(12,7))
    gs = gridspec.GridSpec(2, 2)
    ax = [fig.add_subplot(gs[:, 0]),fig.add_subplot(gs[0, 1]),fig.add_subplot(gs[1, 1])]

    ax[0].plot(f[idx], psd_list[0][idx],c='darkorange')
    ax[0].set_yscale('log')
    ax[0].set_xscale('log')

    ax[0].plot(f[idx], psd_list[2][idx],c='purple')
    ax[0].set_yscale('log')
    ax[0].set_xscale('log')

    ax[0].plot(f[idx], psd_list[1][idx],alpha=0.4,c='darkorange')
    ax[0].set_yscale('log')
    ax[0].set_xscale('log')

    ax[0].plot(f[idx], psd_list[3][idx],alpha=0.4,c='purple')
    ax[0].set_yscale('log')
    ax[0].set_xscale('log')
    ax[0].set_ylim(1e-2,1e12)
    ax[0].set_xlim(low_cut,10)
    ax[0].set_xlabel('Frequency [Hz]')
    ax[0].set_title("Comparison of PSD",fontsize=15)

    # plot rift timeseries on the right panel
    fs = st_list[0][0].stats.sampling_rate
    trace_len = len(st_list[0][0].data)
    ax[1].plot(st_list[0][0].data*1000,'darkorange')
    ax[1].set_xlim(0,trace_len)
    ax[1].set_xticks((0,trace_len/4,trace_len/2,3*trace_len/4,trace_len))
    starttime = st_list[0][0].stats.starttime.datetime
    tick_space = datetime.timedelta(seconds=trace_len/4/fs)
    ticks = [starttime,starttime+tick_space,starttime+2*tick_space,starttime+3*tick_space,starttime+4*tick_space]
    ticklabels = [tick for tick in [tick.strftime("%H:%M") for tick in ticks[1:]]]
    ticklabels.insert(0,starttime.strftime("%Y-%m-%d\n%H:%M"))
    ax[1].set_xticklabels(ticklabels)
    ax[1].set_xlabel('Time')
    ax[1].set_title("May 9 Riftquake",fontsize=15)

    # plot scotia quake timeseries on the right panel
    trace_len = len(st_list[2][0].data)
    ax[2].plot(st_list[2][0].data*1000,'purple')
    ax[2].set_xlim(0,trace_len)
    ax[2].set_xticks((0,trace_len/4,trace_len/2,3*trace_len/4,trace_len))
    starttime = st_list[2][0].stats.starttime.datetime
    tick_space = datetime.timedelta(seconds=trace_len/4/fs)
    ticks = [starttime,starttime+tick_space,starttime+2*tick_space,starttime+3*tick_space,starttime+4*tick_space]
    ticklabels = [tick for tick in [tick.strftime("%H:%M") for tick in ticks[1:]]]
    ticklabels.insert(0,starttime.strftime("%Y-%m-%d\n%H:%M"))
    ax[2].set_xticklabels(ticklabels)
    ax[2].set_xlabel('Time')
    ax[2].set_title("Scotia Sea Earthquake",fontsize=15)

    ax[0].grid()
    ax[1].grid()
    ax[2].grid()

    if resp == "VEL":
        ax[0].set_ylabel('PSD [$(mm/s)^{2}/Hz$]')
        ax[1].set_ylabel('Velocity [mm/s]')
        ax[2].set_ylabel('Velocity [mm/s]')
        
    if resp == "DISP":
        ax[0].set_ylabel('PSD [$(mm)^{2}/Hz$]')
        ax[1].set_ylabel('Displacement [mm]')
        ax[2].set_ylabel('Displacement [mm]')

    plt.show()
    