import numpy as np
import obspy
from obspy.signal.cross_correlation import correlate, xcorr_max
from geopy.distance import distance
import scipy
import matplotlib.pyplot as plt

def get_velocity(tr_local,tr_regional):

    # cross correlate with local station
    cc = correlate(tr_local,tr_regional,7000)
    shift,value = xcorr_max(cc)
    delay = shift/tr_regional.stats.sampling_rate

    # get station locations and calculate distance
    inv_local = obspy.read_inventory("data/XML/*"+tr_local.stats.station+"*")
    channel_local = tr_local.stats.network + "." + tr_local.stats.station + ".." + tr_local.stats.channel
    lat_lon_local = [inv_local.get_coordinates(channel_local)["latitude"],inv_local.get_coordinates(channel_local)["longitude"]]
    inv_regional = obspy.read_inventory("data/XML/*"+tr_regional.stats.station+"*")
    channel_regional = tr_regional.stats.network + "." + tr_regional.stats.station + ".." + tr_regional.stats.channel
    lat_lon_regional = [inv_regional.get_coordinates(channel_regional)["latitude"],inv_regional.get_coordinates(channel_regional)["longitude"]]
    station_distance = distance(lat_lon_local,lat_lon_regional).m
    velocity = station_distance/(-1*delay)
    return velocity



def get_characteristic_frequency(data,fs,band,spectra_method,char_freq_method):
    
    # calculate spectra using basic fft or welch's method
    if spectra_method == "fft":
        spectra = abs(np.fft.rfft(data))
        f = np.fft.rfftfreq(len(data), d=1/fs)
        power = np.square(spectra)
        psd = power/(f[1]-f[0])
    elif spectra_method == "welch":
        f,psd=scipy.signal.welch(data,fs=fs,nperseg=3000,noverlap=0)
        
    # just take the part of the spectra we're interested in
    above_low_cut = [f>band[0]]
    below_high_cut = [f<band[1]]
    in_band = np.logical_and(above_low_cut,below_high_cut)[0]
    f = f[in_band]
    psd = psd[in_band]
        
    if char_freq_method == "max":
        char_freq = f[np.argmax(psd)]
    elif char_freq_method == "mean":
        char_freq = np.sum(psd*f)/np.sum(psd)
    elif char_freq_method == "median":
        psd_cumsum = np.cumsum(psd)
        psd_sum = np.sum(psd)
        char_freq = f[np.argmin(np.abs(psd_cumsum-psd_sum/2))]
#     fig,ax = plt.subplots()
#     ax.plot(f,psd)
#     ax.set_xscale('log')
#     ax.set_yscale('log')
#     ax.vlines(char_freq,ymin=np.min(psd),ymax=np.max(psd))
#     plt.show()
    return char_freq
