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
    
    

def compute_psd(signal,fs):
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
    ax[0].set_title("A. Comparison of PSD",fontsize=15,loc='left')

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
    ax[1].set_title("B. May 9 Riftquake",fontsize=15,loc='left')

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
    ax[2].set_title("C. Scotia Sea Earthquake",fontsize=15,loc='left')

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

    plt.savefig('outputs/figures/spectra_comparison.png',dpi=200)
    
    

def plot_tilt_psd_ratio(f,psd_ratio):
    # get indices corresponding to log-spaced frequency vector
    log_freq = np.logspace(-4,np.log10(np.max(f)),400, endpoint = True)
    idx_list = []
    for freq in log_freq:
        diff = freq-f
        idx_list.append(np.argmin(np.abs(diff)))
    idx = np.unique(idx_list)

    # plot the psd ratio 
    fig,ax = plt.subplots(figsize=(10,7))
    ax.plot(f[idx],psd_ratio[idx],'k',zorder=1)
    ax.vlines(1/120,0,1e6,color='darkorange',linestyle='--',zorder=0)
    ax.text(1/180,1e2,'$f_{corner}$ = '+r'$\frac{1}{120}$s',rotation=90,size=15)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('Ratio of PSD (radial / predicted contribution by tilt)')
    ax.set_ylabel('Ratio')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylim(1e-4,1e5)
    plt.show()

    
def plot_ringdown(st,t_ring_est,t_ring_est_vect,amp_vect,t_ring_obs,decay_amp,resp):
    starttime = st[0].stats.starttime
    endtime = st[0].stats.endtime
    fig, ax = plt.subplots(figsize=(10,7))
    ax.plot(st[0].data*1000,'k',linewidth=1.5,zorder=2)
    trace_len = st[0].stats.npts
    ax.set_xlim(0,trace_len)
    fs = st[0].stats.sampling_rate
    num_hours = int(st[0].stats.npts / fs // 3600)
    ticks = [h*fs*3600 for h in range(num_hours+1)]
    ax.set_xticks(ticks)
    tick_times = [starttime.datetime + datetime.timedelta(hours = h+1) for h in range(num_hours)]
    ticklabels = [tick.strftime("%H:%M") for tick in tick_times]
    ticklabels.insert(0,starttime.strftime("%Y-%m-%d\n%H:%M"))
    ax.set_xticklabels(ticklabels)
    ax.grid(True)
    if resp =="VEL":
        ax.set_ylabel("Velocity (mm/s)")
    if resp =="DISP":
        ax.set_ylabel("Displacement (mm)")   
    ax.set_xlabel("Time")
    ax.tick_params(axis='both', which='major')
    ax.set_xlim(0,st[0].stats.npts)
    ax.set_title("Seismic data (" + st[0].stats.station + " " + st[0].stats.channel + ") and estimated ringdown time")
    [x.set_linewidth(1.5) for x in ax.spines.values()]

    fs = st[0].stats.sampling_rate
    ax.hlines(decay_amp*1000,0,st[0].stats.npts,'red',linestyle='--',zorder=0)
    max_time = np.argmax(np.abs(st[0].data))
#     ax.vlines([max_time,max_time+fs*t_ring_est,fs*t_ring_obs],-0.3,0.3,zorder=0,linestyle='--',linewidth=0.5,colors='grey')
#     ax.text(np.argmax(np.abs(st[0].data))-2800,-0.315,'$t_{A_{max}}$',fontsize=8)
#     ax.text(fs*t_ring_est-20000,-0.315,'Pred $t_{ring}$',fontsize=8)
#     ax.text(fs*t_ring_obs-20000,-0.315,'Obs $t_{ring}$',fontsize=8)
#     ax.text(trace_len+8000,decay_amp*1000,'$A_{max}e^{-\pi}$')
    ax.set_ylim(-0.275,0.275)
    decay_time_vect = [fs*s for s in t_ring_est_vect[0]]
    ax.plot(max_time+decay_time_vect,-amp_vect,color='C1',linestyle='--')
    ax.plot(max_time+decay_time_vect,amp_vect,color='C1',linestyle='--')
    decay_time_vect = [fs*s for s in t_ring_est_vect[1]]
    ax.plot(max_time+decay_time_vect,-amp_vect,color='C2',linestyle='--')
    ax.plot(max_time+decay_time_vect,amp_vect,color='C2',linestyle='--')
    plt.show()
    
    
    
def windowed_normalized_psd(st,win_size,freq,starttime,endtime):
    # iterate through time
    psd_list = []
    psd_max = []
    f_list = []
    for t in range(int((endtime-starttime)//win_size)):
        st_win = st.copy().trim(starttime=starttime+t*win_size,endtime=starttime+(t+1)*win_size)
        data = st_win[0].data

        # calculate spectra
        f, psd = compute_psd(data, fs)
        above_low_cut = [f>=freq[0]]
        below_high_cut = [f<=freq[1]]
        in_band = np.logical_and(above_low_cut,below_high_cut)[0]
        f_list.append(f[in_band])
        psd_list.append(psd[in_band])
        psd_max.append(np.max(psd[in_band]))

    # make a plot
    fig,ax = plt.subplots(2,1,figsize=(10,10))
    max_psd = np.max(psd_max)
    for i in range(len(psd_list)):
        psd_norm = psd_list[i]/max_psd
        ax[0].plot(f_list[i],psd_norm-0.1*i)
        win_start = starttime+i*win_size
        win_start_string = win_start.datetime.strftime("%H:%M:%S")
        win_end = starttime+(i+1)*win_size
        win_end_string = win_end.datetime.strftime("%H:%M:%S")
        ax[0].text(5.1,-0.1*i-0.01,win_start_string + "-" + win_end_string)
    #ax.set_xscale('log')
    #ax.set_yscale('log')
    ax[0].set_xlim(freq)
    ax[0].set_xlabel("Frequency (Hz)")
    ax[0].set_yticks([])
    ax[0].set_title("Normalized windowed PSD ("+st[0].stats.station+")")
    ax[1].plot(st[0].data,'k')
    trace_len = len(st[0].data)
    ax[1].set_xticks((0,trace_len/4,trace_len/2,3*trace_len/4,trace_len))
    starttime = st[0].stats.starttime.datetime
    tick_space = datetime.timedelta(seconds=trace_len/4/fs)
    ticks = [starttime,starttime+tick_space,starttime+2*tick_space,starttime+3*tick_space,starttime+4*tick_space]
    ticklabels = [tick for tick in [tick.strftime("%H:%M:%S") for tick in ticks[1:]]]
    ticklabels.insert(0,starttime.strftime("%Y-%m-%d\n%H:%M"))
    ax[1].set_xticklabels(ticklabels)
    ax[1].set_xlim(0,trace_len)
    plt.tight_layout()
    plt.show()
    
    
def particle_motion(st,pts_per_frame,components):
    c1 = st.select(component=components[0])[0].data
    c2 = st.select(component=components[1])[0].data
    norm_max = np.max([np.max(np.abs(c1)),np.max(np.abs(c2))])
    c1_norm = c1/norm_max
    c2_norm = c2/norm_max
    fig,ax = plt.subplots(2,1,figsize=(10,10))
    ax[1].set_xlim((-1,1))
    ax[1].set_ylim((-1,1))

    # plot the two components against each other
    c1_win = c1_norm[0:pts_per_frame]
    c2_win = c2_norm[0:pts_per_frame]
    particles = np.zeros(pts_per_frame,dtype=[("position", float , 2)])
    particles["position"][:,0] = c1_win
    particles["position"][:,1] = c2_win
    cmap = matplotlib.cm.get_cmap('plasma_r')
    colors = cmap(np.linspace(0, 1, num = pts_per_frame))
    scatter=ax[1].scatter(particles["position"][:,0], particles["position"][:,1],c=colors)
    ax[1].set_xlabel(components[0] + " component")
    ax[1].set_ylabel(components[1] + " component")

    # plot timeseries
    t = st[0].times("matplotlib")
    ax[0].plot(t,c2,'k')
    ax[0].set_xlim(t[0],t[-1])
    line = ax[0].axvline(t[pts_per_frame],-1,1,c='r')
    dateFmt = matplotlib.dates.DateFormatter('%H:%M:%S')
    ax[0].xaxis.set_major_formatter(dateFmt)
    ax[0].set_ylabel(components[1] + " displacement (m)")
    ax[0].set_title(st[0].stats.station + ' particle motion')
    
    def update(frame_number):
        c1_win = c1_norm[frame_number:frame_number+pts_per_frame]
        c2_win = c2_norm[frame_number:frame_number+pts_per_frame]
        particles["position"][:,0] = c1_win
        particles["position"][:,1] = c2_win
        scatter.set_offsets(particles["position"])
        line.set_xdata(t[frame_number+pts_per_frame])
        return scatter, 
    
    frames = len(c1)-pts_per_frame-1
    anim = animation.FuncAnimation(fig, update, frames=frames, interval=1)

    # saving to m4 using ffmpeg writer
    anim.save('particle_motion_' + st[0].stats.station + '_' + components[0] + '_' + components[1] + '.gif',writer="pillow")
    plt.close()