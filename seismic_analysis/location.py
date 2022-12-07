import numpy as np
import glob
import os
import obspy
from obspy.signal.cross_correlation import correlate
from obspy.signal.cross_correlation import xcorr_max
from sklearn.decomposition import PCA
from pyproj import Proj,transform,Geod
from geopy import distance
from shapely import geometry
import geopandas as gpd
from datetime import datetime
from datetime import timedelta
import types
import time
from collections import Counter
import scipy
from scipy.signal import hilbert
import pathlib
import glob
import copy
import multiprocessing
import pickle
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle, ConnectionPatch
from seismic_analysis.directivity import get_velocity
import cartopy
import cartopy.crs as ccrs


def write_parameters(d):
    home_dir = str(pathlib.Path().absolute())
    with open(home_dir + "/outputs/locations/params.txt", 'w') as f:
        print(d.__dict__, file=f)

        
        
def get_files(l):
    files = glob.glob(l.data_path + "/PIG2/HHZ/*"+l.response+"*", recursive=True)
    files = [f.replace("PIG2","*") for f in files]
    files = [f.replace("HHZ","*H*") for f in files]
    files.sort()
    start_date = l.detection_times[0].strftime("%Y-%m-%d")
    start_index = [s for s in range(len(files)) if start_date in files[s]][0]
    end_date = l.detection_times[-1].strftime("%Y-%m-%d")
    end_index = [s for s in range(len(files)) if end_date in files[s]][0]+1
    files = files[start_index:end_index]
    return files



def make_results_object(l):
    baz_object = types.SimpleNamespace()
    baz_object.backazimuths = np.empty((l.num_detections),'float64')
    baz_object.backazimuths[:] = np.NaN
    baz_object.uncertainties = np.empty((l.num_detections),'float64')
    baz_object.uncertainties[:] = np.NaN
    return baz_object



def get_detection_times(ds):
    # extract times for each event in the dataset
    detection_times = []
    for event in ds.events:
        detection_times.append(event.origins[0].time.datetime)
    return np.array(detection_times)



def get_detections_today(l):
    current_date = datetime.strptime(l.f.split("/")[-1].split(".")[0],"%Y-%m-%d")
    bool_indices = np.logical_and(l.detection_times>=current_date,l.detection_times<current_date + timedelta(days=1))
    detections_today = l.detection_times[bool_indices]
    if sum(bool_indices) == 0:
        indices = [0,0]
    else:
        indices = [[i for i, x in enumerate(bool_indices) if x][0],[i for i, x in enumerate(bool_indices) if x][-1]+1]
    return detections_today, indices



def get_stations_to_use(st,l):
    available_stations = []
    for trace in st:
        available_stations.append(trace.stats.station)
    available_stations = np.unique(available_stations)
    stations_to_use = list(set(l.stations).intersection(set(available_stations)))
    return np.sort(stations_to_use)



def get_data_to_use(st_all,l):
    st = obspy.Stream()
    for s in l.stations:
        st += st_all.select(station=s)
    return st



def get_station_lon_lat(xml_path,networks,stations):
    stat_coords = []
    inv = obspy.read_inventory(xml_path + "/*")
    for network in networks:
        for s in stations:
            try:
                channel = network + "." + s + ".." + "BHZ"
                lat = inv.get_coordinates(channel)["latitude"]
                lon = inv.get_coordinates(channel)["longitude"]
            except:
                try:
                    channel = network + "." + s + ".." + "HHZ"
                    lat = inv.get_coordinates(channel)["latitude"]
                    lon = inv.get_coordinates(channel)["longitude"]
                except:
                    continue
            stat_coords.append([lon,lat])
    _, idx = np.unique(stat_coords,axis=0,return_index=True)
    stat_coords = np.array(stat_coords)[np.sort(idx)]
    return stat_coords



def get_crs_locations(lon_lat_coords,crs):
    # convert station coordinates to x and y and take average station location
    p2 = Proj(crs,preserve_units=False)
    p1 = Proj(proj='latlong',preserve_units=False)
    [stat_x,stat_y] = transform(p1,p2,lon_lat_coords[:,0],lon_lat_coords[:,1])
    return np.stack((stat_x,stat_y),axis=1)



def get_station_angles(l):
    station_angles = []
    for i in range(len(l.station_grid_coords)):
        x = (l.station_grid_coords[i,0]-l.array_centroid[0])
        y = (l.station_grid_coords[i,1]-l.array_centroid[1])
        angle = np.arctan2(y,x)*180/np.pi
        
        # subtract from 90 since the returned angles are in relation to 0 on the unit circle
        # we want them in relation to true north
        angle = 90-angle
        if angle < 0:
            angle = angle + 360
        
        station_angles.append(angle)
    return station_angles



def first_observed_arrival(st,l):
    # cross correlate all traces and find station with largest shift
    channels = ["Z","N","E"]
    first_stat_vector = []
    for chan in channels:
        st_chan = st.select(component=chan)
        shifts = np.zeros(len(st_chan))
        corrs = np.zeros(len(st_chan))
        for j in range(len(st_chan)):
            corr = correlate(st_chan[0], st_chan[j], l.max_shift)
            shift, correlation_coefficient = xcorr_max(corr,abs_max=True)
            shifts[j] = shift
            corrs[j] = correlation_coefficient
        stat_idx = np.argmax(shifts)
        first_stat_vector.append(st_chan[stat_idx].stats.station)
    counts = Counter(first_stat_vector).most_common(2)
    if len(counts) > 1:
        if counts[0][1] == counts[1][1]:
            first_stat = []
        else:
            first_stat = counts[0][0]
    else:
        first_stat = counts[0][0]
    print(first_stat)
    return first_stat



def check_data_quality(st,l):
    stations_to_remove = []
    for i in range(len(st)):
        snr = max(abs(st[i].data))/np.mean(abs(st[i].data))
        if snr < l.snr_threshold:
            stations_to_remove.append(st[i].stats.station)
    stations_to_remove = np.unique(stations_to_remove)
    for stat in stations_to_remove:
        for trace in st.select(station=stat):
            st.remove(trace)
    return st



def check_trace_length(st):
    uneven_length = 0
    first_length = len(st[0].data)
    for trace in st:
        if not len(trace) == first_length:
            uneven_length = 1
    return uneven_length



def compute_pca(st,l):
    # make array for storage of pca components
    first_component_vect = np.empty((0,2),"float64")

    # get mean amplitude for whole trace (average of both components)
    horz_data = np.transpose(np.concatenate(([st.select(component="E")[0].data],[st.select(component="N")[0].data])))

    # itertate through data in windows
    for n in range(l.num_steps):
        # get current window
        start_ind = n * l.slide * l.fs
        end_ind = start_ind + l.win_len*l.fs
        X = horz_data[start_ind:end_ind,:]
        # only progress if matrix of data is not empty
        if X.size > 0:
            # normalize and compute the PCA if staLta criteria is met
            if  np.mean(abs(X)) > l.stalta_threshold*np.mean(abs(horz_data)):
                # find component with max amplitude, normalize both traces by that max value, and compute PCA
                max_amp = np.amax(abs(X))
                X_norm = np.divide(X,max_amp)
                pca = PCA(n_components = 2)
                pca.fit(X_norm)

                # flip pca components based on station of first arrival
                if l.pca_correction == "radial":
                    first_components = correct_pca_radial(pca.components_[0,:],l)
                if l.pca_correction == "distance":
                    first_components = correct_pca_distance(pca.components_[0,:],l)
                if l.pca_correction == "sector":
                    first_components = correct_pca_sector(pca.components_[0,:],l)
                if l.pca_correction == "manual":
                    first_components = correct_pca_manual(pca.components_[0,:],l)
                    
                # save result
                first_component_vect = np.vstack((first_component_vect,first_components))

            else:
                # add zeros if we didn't run PCA on the window due to low STALTA
                first_component_vect = np.vstack((first_component_vect,[np.nan,np.nan]))
        else:
            # add zeros if we didn't run PCA on the window due to emptiness
            first_component_vect = np.vstack((first_component_vect,[np.nan,np.nan]))

    return first_component_vect



def angle_difference(angle_1,angle_2):
    diff_1 = abs(angle_1 - angle_2)
    diff_2 = 360 - diff_1
    return min(diff_1,diff_2)



def closest_station(baz,l):
    radial_diffs = []
    for i in range(len(l.stations)):
        radial_diffs.append(angle_difference(baz,l.station_angles[i]))
    station = l.stations[np.argmin(radial_diffs)]
    return station



def correct_pca_radial(pca_components,l):
    # get the backazimuth corresponding to the initial pca first components
    baz = 90 - np.arctan2(pca_components[1],pca_components[0])*180/np.pi
    if baz < 0:
        baz = baz + 360
        
    # get the other possible backazimuth
    if baz < 180:
        baz_180 = baz + 180
    if baz > 180:
        baz_180 = baz - 180

    # get the stations closest to these backazimuths
    predicted_station = closest_station(baz,l)
    predicted_station_180 = closest_station(baz_180,l)

    # check if the observed station of first arrival agrees with either of these predicted backazimuths
    if l.first_stat == predicted_station:
        corrected_pca_components = pca_components 
    if l.first_stat == predicted_station_180:
        corrected_pca_components = pca_components*-1
    else:
        corrected_pca_components = [np.nan,np.nan]
    return corrected_pca_components

             
    
def correct_pca_distance(pca_components,l):
    # get the backazimuth corresponding to the observed polarization direction
    baz = 90 - np.arctan2(pca_components[1],pca_components[0])*180/np.pi
    if baz < 0:
        baz = baz + 360

    # get distances to each station from array centroid
    station_distances = np.hypot(l.station_grid_coords[:,0]-l.array_centroid[0],l.station_grid_coords[:,1]-l.array_centroid[1])

    # calculate the distance from the array centroid to each station IN THE DIRECTION OF THE CALCULATED BACKAZIMUTH
    sector_1_stations = []
    sector_1_distances = []
    sector_2_stations = []    
    sector_2_distances = []

    for s in range(len(l.station_angles)):
        if angle_difference(l.station_angles[s],baz) < 90:
            sector_1_stations.append(l.stations[s])
            theta = abs(baz-l.station_angles[s])
            sector_1_distances.append(station_distances[s]*np.cos(theta*np.pi/180))
        if angle_difference(l.station_angles[s],baz) > 90:
            sector_2_stations.append(l.stations[s])
            theta = abs(baz-180-l.station_angles[s])
            sector_2_distances.append(station_distances[s]*np.cos(theta*np.pi/180))

    # now find farthest station from centroid (which would be first to see an incoming plane wave)
    if sector_1_distances:
        sector_1_first_arrival = sector_1_stations[np.argmax(sector_1_distances)]
    else:
        sector_1_first_arrival = []
    if sector_2_distances:
        sector_2_first_arrival = sector_2_stations[np.argmax(sector_2_distances)]
    else:
        sector_2_first_arrival = []
    if l.first_stat == sector_1_first_arrival:
        corrected_pca_components = pca_components 
    if l.first_stat == sector_2_first_arrival:
        corrected_pca_components = pca_components*-1
    else:
        corrected_pca_components = [np.nan,np.nan]
    return corrected_pca_components



def correct_pca_sector(pca_components,l):
    # get the backazimuth corresponding to the observed polarization direction
    baz = 90 - np.arctan2(pca_components[1],pca_components[0])*180/np.pi
    if baz < 0:
        baz = baz + 360
    
    # calculate the distance from the array centroid to each station IN THE DIRECTION OF THE CALCULATED BACKAZIMUTH
    sector_1_stations = []
    sector_2_stations = []    
    for s in range(len(l.station_angles)):
        if angle_difference(l.station_angles[s],baz) < 90:
            sector_1_stations.append(l.stations[s])
        if angle_difference(l.station_angles[s],baz) > 90:
            sector_2_stations.append(l.stations[s])

    if l.first_stat in sector_1_stations:
        corrected_pca_components = pca_components 
    if l.first_stat in sector_2_stations:
        corrected_pca_components = pca_components*-1
    else:
        corrected_pca_components = [np.nan,np.nan]
    return corrected_pca_components



def correct_pca_manual(pca_components,l):
    # get the backazimuth corresponding to the observed polarization direction
    baz = 90 - np.arctan2(pca_components[1],pca_components[0])*180/np.pi
    if baz < 0:
        baz = baz + 360
    if l.flip == True:
        corrected_pca_components = pca_components*-1
    elif l.flip == False:
        corrected_pca_components = pca_components 
    return corrected_pca_components



def calculate_event_baz(first_component_sums,norms):
    denom = np.sum(norms)
    avg_weighted_x = np.nansum(first_component_sums[:,0])/denom
    avg_weighted_y = np.nansum(first_component_sums[:,1])/denom
    event_baz = 90 - np.arctan2(avg_weighted_y,avg_weighted_x)*180/np.pi
    if event_baz < 0:
        event_baz = event_baz + 360
    return event_baz



def calculate_uncertainty(first_component_sums,norms):
    # rescale norms to get weight vector whose sum equals the number of windows for which we calculated PCA (this is necessary for the sqrt(-2log(R)) in the circular standard deviation calculation)
    weights = norms/(np.sum(norms)/np.sum([norms != 0]))
    normalized_first_component_sums = first_component_sums
    for n in range(len(normalized_first_component_sums)):
        if not np.sum(first_component_sums[n,0]) == 0:
            vect_len = np.sqrt(first_component_sums[n,0]*first_component_sums[n,0]+first_component_sums[n,1]*first_component_sums[n,1])
            normalized_first_component_sums[n,0] = first_component_sums[n,0]/vect_len
            normalized_first_component_sums[n,1] = first_component_sums[n,1]/vect_len
    event_uncertainty = (circular_stdev(normalized_first_component_sums,weights)*180/np.pi)
    return event_uncertainty



def circular_stdev(pca_components,weights):
    cos_sum = 0
    sin_sum = 0
    for i in range(len(pca_components)):
        if not np.isnan(pca_components[i,0]):
            cos = weights[i]*pca_components[i,0]
            cos_sum += cos
            sin = weights[i]*pca_components[i,1]
            sin_sum += sin
    cos_avg = cos_sum/np.sum(weights)
    sin_avg = sin_sum/np.sum(weights)
    R = np.sqrt(cos_avg*cos_avg+sin_avg*sin_avg)
    stdev = np.sqrt(-2*np.log(R))
    return stdev



def polarization_analysis(l):

    # get detection times from ASDF dataset and insert dummy time at the end for convenience
    detection_times_today, indices = get_detections_today(l)
    num_detections_today = len(detection_times_today)
    detection_times_today = np.append(detection_times_today,datetime(1970,1,1,0,0,0))

    # read all available data for the current day
    st = obspy.read(l.f)
    # get stations that are (1) available on current day and (2) within the user-specified list of desired stations
    l.stations = get_stations_to_use(st,l)

    # only keep the data from these stations for use in backaziumuth calculations
    st = get_data_to_use(st,l)
    st.filter("bandpass",freqmin=l.freq[0],freqmax=l.freq[1])

    # get geometrical parameters for the functional "array", which is made up of only the stations that are available
    l.station_lon_lat_coords = get_station_lon_lat(l.xml_path,l.network,l.stations)
    l.station_grid_coords = get_crs_locations(l.station_lon_lat_coords,l.crs)
    # update array centroid to reflect subset of stations being used for this particular event or keep it fixed
    if l.centroid == "moving":
        l.array_centroid = np.mean(l.station_grid_coords,axis=0)
        
    # get angles to each station that's desired and available on this particular day)
    l.station_angles = get_station_angles(l)

    # make containers for today's results
    event_baz_vect = np.empty((num_detections_today),'float64')
    event_baz_vect[:] = np.nan
    event_uncertainty_vect = np.empty((num_detections_today),'float64')
    event_uncertainty_vect[:] = np.nan
    
    # run polarization analysis for all events in the current file / on the current day
    for i in range(num_detections_today):
    
        # make arrays for storing PCA results
        all_first_components = np.empty((l.num_steps,2,len(l.stations)),"float64")
        all_first_components[:,:,:] = np.nan
    
        # get UTCDateTime and current date for convenience
        detection_utc_time = obspy.UTCDateTime(detection_times_today[i])
        l.current_detection = detection_utc_time
        
        # check if more than one event on current day; if so, read entire day. If not, just read the event.
        st_event = st.copy()
        st_event.trim(starttime=detection_utc_time,endtime=detection_utc_time+l.trace_len)
        st_event.taper(max_percentage=0.1, max_length=30.)
        
        # check for gaps and remove stations with bad data quality for this event
        start_time = st_event[0].stats.starttime
        if check_trace_length(st_event):
            print("Skipped event at "  + str(start_time) + " due to traces with uneven length\n")
            continue

        # check SNR and if all traces were removed due to poor snr, skip this event
        st_event = check_data_quality(st_event,l)
        if not st_event:
            print("Skipped event at "  + str(start_time) + " due to poor SNR on all stations\n")
            continue
            
        # loop through stations to get one trace from each to find earliest arrival
        l.first_stat = first_observed_arrival(st_event,l)
        if not l.first_stat:
            print("Skipped event at "  + str(start_time) + " due to indeterminate first arrival\n")
            continue
            
        # loop though stations to perform PCA on all windows in the event on each station's data
        for s in range(len(l.stations)):

            # check if there's data and skip current station if not
            if not st_event.select(station=l.stations[s]):
                continue
                
            # compute pca components for all windows in the event
            all_first_components[:,:,s] = compute_pca(st_event.select(station=l.stations[s]),l)
        
        # sum results (this is vector sum across stations of pca first components for each window)
        first_component_sums = np.nansum(all_first_components,axis=2)
        
        # take average weighted by norm of PCA component sums to get single mean event backazimuth
        norms = np.linalg.norm(first_component_sums,axis=1)
        if not np.sum(first_component_sums) == 0:
            event_baz_vect[i] = calculate_event_baz(first_component_sums,norms)
            event_uncertainty_vect[i] = calculate_uncertainty(first_component_sums,norms)
    if num_detections_today == 0:
         print("Finished with " + str(st[0].stats.starttime.date)+" (no detections) \n")   
    else:
        print("Finished with " + str(detection_times_today[0].date())+"\n")
    return event_baz_vect,event_uncertainty_vect,indices



def compute_backazimuths(l): 
    
    # get home directory path
    home_dir = str(pathlib.Path().absolute())
    
    # get centroid for the desired stations
    l.station_lon_lat_coords = get_station_lon_lat(l.xml_path,l.network,l.stations)
    l.station_grid_coords = get_crs_locations(l.station_lon_lat_coords,l.crs)
    l.array_centroid = np.mean(l.station_grid_coords,axis=0)

    # get all detection times
    l.num_detections = len(l.detection_times)
        
    # make object for storing pca vector sums and storing data to plot
    b = make_results_object(l)

    # write file with parameters for this run
    write_parameters(l)
    
    # make vector of all filenames
    files = get_files(l)
    print("Got all files...\n")
    
    # construct iterable list of detection parameter objects for imap
    inputs = []
    for f in files:
        l.f = f
        inputs.append(copy.deepcopy(l))
    print("Made inputs...\n")        
    # map inputs to polarization_analysis and save as each call finishes
    multiprocessing.freeze_support()
    p = multiprocessing.Pool(processes=l.n_procs)
    for result in p.imap_unordered(polarization_analysis,inputs):
        b.backazimuths[result[2][0]:result[2][1]] = result[0]
        b.uncertainties[result[2][0]:result[2][1]] = result[1]

        # open output file, save result vector, and close output file
        baz_file = open(l.filename+".pickle", "wb")
        pickle.dump(b, baz_file)
        baz_file.close()
        
    return b



def load_data(st,local_stations,regional_stations,components,freq,starttime,endtime):
    st_regional = obspy.Stream()
    for stat in regional_stations:
        for component in components: 
            st_regional = st_regional.append(st.select(station=stat,component=component)[0])
    st_regional.filter("bandpass",freqmin=freq[0],freqmax=freq[1])
    st_regional.trim(starttime=starttime,endtime=endtime)

    st_local = obspy.Stream()
    for station in local_stations:
        for component in components:
            st_local += st.select(station=station,component=component)
    st_local.filter("bandpass",freqmin=freq[0],freqmax=freq[1])
    st_local.trim(starttime=starttime,endtime=endtime)
    st_local.resample(st_regional[0].stats.sampling_rate)
    return st_local, st_regional



def get_station_grid_locations(origin_lat_lon,station_lat_lon):
    origin_lat = origin_lat_lon[0]
    origin_lon = origin_lat_lon[1]
    station_x_y = []
    for i in range(len(station_lat_lon)):
        x_dist = distance.distance([origin_lat,origin_lon],[origin_lat,station_lat_lon[i][1]]).m
        if station_lat_lon[i][1] < origin_lon:
            x_dist = -x_dist
        y_dist = distance.distance([origin_lat,origin_lon],[station_lat_lon[i][0],origin_lon]).m
        if station_lat_lon[i][0] < origin_lat:
            y_dist = -y_dist
        station_x_y.append([x_dist,y_dist])
    station_x_y = np.array(station_x_y)
    return station_x_y



def get_velocities(ref_station,st_local,local_stations,st_regional,regional_stations,components,local_velocity):
    velocities = []
    for station in local_stations:
        velocities.append(list(np.ones(len(components))*local_velocity))
    for station in regional_stations:
        station_velocities = []
        for component in components:
            tr_local = st_local.select(station=ref_station,component=component)[0]
            tr_regional = st_regional.select(station=station,component=component)[0]
            station_velocities.append(get_velocity(tr_local,tr_regional))
        velocities.append(station_velocities)
    return velocities



def get_grid(grid_length,grid_height,step,t0,t_step):
    x_vect = np.arange(0, grid_length, step)
    y_vect = np.arange(0, grid_height, step)
    t_vect = np.arange(t0,0,t_step)
    return x_vect,y_vect,t_vect



def get_arrival(tr_ref,tr,method=""):
# iterate through regional stations

    max_shift = len(tr_ref)*2
    local_env = obspy.Trace(np.abs(hilbert(tr_ref.data)))

    # cross correlate envelope with envelope of data from local station 
    if method == "envelope":
        env = obspy.Trace(np.abs(hilbert(tr.data)))
        cc = correlate(env,local_env,max_shift)
    else:
        cc = correlate(tr,tr_ref,max_shift)
    shift,xcorr_coef = xcorr_max(cc)
    arrival = shift/tr.stats.sampling_rate
    return arrival, xcorr_coef



def get_arrivals(ref_station,st,stations,components):
    arrivals = []
    xcorr_coefs = []
    for station in stations:
        station_arrivals = []
        station_xcorr_coef = []
        for component in components:
            tr_ref = st.select(station="PIG2",component=component)[0]
            tr = st.select(station=station,component=component)[0]
            arrival, xcorr_coef = get_arrival(tr_ref,tr,"envelope")
            station_arrivals.append(arrival)
            station_xcorr_coef.append(xcorr_coef)
        arrivals.append(station_arrivals)
        xcorr_coefs.append(station_xcorr_coef)
    return arrivals,xcorr_coefs



# def get_station_lat_lon(st):
#     channel = st[0].stats.component
#     st = st.select(component=channel)
#     coordinates = []
#     for tr in st:
#         inv = obspy.read_inventory("data/XML/*"+tr.stats.station+"*")
#         trace_id = tr.stats.network + "." + tr.stats.station + ".." + tr.stats.channel.split('H' + channel)[0] + "H" + channel
#         coordinates.append([inv.get_coordinates(trace_id)["latitude"],inv.get_coordinates(trace_id)["longitude"]])
#     return coordinates



# define function to predict synthetic arrival times
def travel_time(t0, x, y, velocity, sta_x, sta_y):
    dist = np.sqrt((sta_x - x)**2 + (sta_y - y)**2)
    tt = t0 + dist/velocity
    return tt



# function to compute residual sum of squares- averages residual across components for each station
def error(synth_arrivals,arrivals,weights,stat_dist):
    res = arrivals - synth_arrivals
    res = weights * res
    res = np.mean(res,axis=1)
    res_sqr = res**2
    mse = np.mean(res_sqr)
    rmse = np.sqrt(mse)
    return rmse



# define function to iterate through grid and calculate travel time residuals
def gridsearch(t0,x_vect,y_vect,station_x_y,velocities,arrivals,weights=[]):
    num_stat = np.shape(arrivals)[0]
    num_chan = np.shape(arrivals)[1]
    if len(weights) == 0:
        weights = np.ones(np.shape(arrivals))
    stat_x = station_x_y[:,0]
    stat_y = station_x_y[:,1]
    rmse_mat = np.zeros((len(t0),len(x_vect),len(y_vect)))
    for i in range(len(t0)):
        for j in range(len(x_vect)):
            for k in range(len(y_vect)):
                synth_arrivals = []
                for l in range(num_stat):
                    stat_synth_arrivals = []
                    for m in range(num_chan):
                        tt = travel_time(t0[i],x_vect[j],y_vect[k],velocities[l][m],stat_x[l],stat_y[l])
                        stat_dist = np.sqrt((stat_x[l] - x_vect[j])**2 + (stat_y[l] - y_vect[k])**2)
                        stat_synth_arrivals.append(tt)
                    synth_arrivals.append(stat_synth_arrivals)
                rmse = error(np.array(synth_arrivals),np.array(arrivals),weights,stat_dist)
                rmse_mat[i,j,k] = rmse
    return rmse_mat



def transform_imagery(file,dst_crs):
    with rasterio.open(file) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        ext = file[-4:]
        with rasterio.open(file.split(ext)[0] + "_" + dst_crs + ext, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)
                
                
                
def plot_landsat_tsx_and_location(landsat,tsx,backazimuth,rmse_mat,station_grid_coords,grid_axes_coords,vlims):

    # make figure and axes to plot on
    fig, ax = plt.subplots(figsize=(15,15),dpi=100)
    ax.axis('off')

    # iterate through tsx scenes
    for j in range(len(tsx)):
    
        # Construct axis to plot on
        ax.axis('off')
        axes_coords = np.array([1.2*j, 0, 1, 1])
        ax_image = fig.add_axes(axes_coords)

        # overlay multiple landsat images
        for i in range(len(landsat)):

            # read each band
            landsat_B2 = rasterio.open('data/LANDSAT/'+landsat[i]+'/'+landsat[i]+'_SR_B2_epsg:3245.TIF') #blue
            landsat_B3 = rasterio.open('data/LANDSAT/'+landsat[i]+'/'+landsat[i]+'_SR_B3_epsg:3245.TIF') #green
            landsat_B4 = rasterio.open('data/LANDSAT/'+landsat[i]+'/'+landsat[i]+'_SR_B4_epsg:3245.TIF') #red
            image_B2 = landsat_B2.read(1)
            image_B3 = landsat_B3.read(1)
            image_B4 = landsat_B4.read(1)

            # crop each band to 99th percentile of brightness
            image_B2[image_B2 > np.percentile(image_B2,99)] = np.percentile(image_B2,99)
            image_B3[image_B3 > np.percentile(image_B3,99)] = np.percentile(image_B3,99)
            image_B4[image_B4 > np.percentile(image_B4,99)] = np.percentile(image_B4,99)

            # combine bands into natural color image
            image_rgb = np.array([image_B2, image_B3, image_B4]).transpose(1,2,0)
            normalized_rgb = (image_rgb * (255 / np.max(image_rgb))).astype(np.uint8)

            # get bounds
            landsat_bounds = landsat_B2.bounds
            horz_len = landsat_bounds[2]-landsat_bounds[0]
            vert_len = landsat_bounds[3]-landsat_bounds[1]

            # display image
            if i == 0:
                ax_image.imshow(normalized_rgb,extent=[landsat_bounds[0],landsat_bounds[2],landsat_bounds[1],landsat_bounds[3]],interpolation='none')
            else:    
                # add alpha channel to image so it can be overlayed
                alpha_slice = np.array(normalized_rgb.shape)
                alpha_slice[2] = 1
                alpha_array = np.zeros(alpha_slice)
                alpha_array[np.nonzero(image_B2)[0],np.nonzero(image_B2)[1],0] = 255
                normalized_rgba = np.append(normalized_rgb,alpha_array,axis=2).astype(int)
                ax_image.imshow(normalized_rgba,extent=[landsat_bounds[0],landsat_bounds[2],landsat_bounds[1],landsat_bounds[3]],interpolation='none')

        # set overall plot bounds based on top landsat image
        plot_bounds = [landsat_bounds[0]+0.425*horz_len,landsat_bounds[2]-0.35*horz_len,landsat_bounds[1]+0.3*vert_len,landsat_bounds[3]-0.475*vert_len]

        # read tsx data
        directory = os.listdir('data/TSX/'+ tsx[j] + '/TSX-1.SAR.L1B/')[0]
        tsx_path = 'data/TSX/'+tsx[j]+'/TSX-1.SAR.L1B/'+ directory + '/IMAGEDATA/'
        raster = rasterio.open(glob.glob(tsx_path + '*epsg:3245.tif')[0])
        image = raster.read()

        # crop each band to 99th percentile of brightness
        image[image > np.percentile(image,99)] = np.percentile(image,99)

        # plot TSX imagery
        tsx_bounds = raster.bounds
        horz_len = tsx_bounds[2]-tsx_bounds[0]
        vert_len = tsx_bounds[3]-tsx_bounds[1]
        masked_image = np.ma.masked_where(image[0] == 0, image[0])
        ax_image.imshow(masked_image,extent=[tsx_bounds[0],tsx_bounds[2],tsx_bounds[1],tsx_bounds[3]],cmap='gray',vmin=vlims[j,0], vmax=vlims[j,1])

        # get corners of imagery extent
        p2 = Proj("EPSG:3245",preserve_units=False)
        p1 = Proj(proj='latlong',preserve_units=False)

        # plot location on second panel
        if j == 1:
            # plot gridsearch results
            grid_x = grid_axes_coords[0]
            grid_y = grid_axes_coords[1]
            cmap = plt.get_cmap('plasma')
            contour_x, contour_y = np.meshgrid(grid_x, grid_y)
            contours = ax_image.contour(contour_x, contour_y, np.log10(rmse_mat.T),levels=20,cmap=cmap,vmin=0.8,vmax=1.3)
            ax_image.clabel(contours, colors='k', inline_spacing=-10, fontsize=15)
            axes_coords = np.array([1.2*j+1.05, 0.025, 0.05, 0.95])
            c_axis = fig.add_axes(axes_coords)
            norm = Normalize(vmin=0.8,vmax=1.3)
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cbar = fig.colorbar(sm, cax=c_axis, orientation='vertical')
            cbar.ax.invert_yaxis()
            cbar.ax.tick_params(labelsize=15)
            cbar.ax.set_ylabel('log10(RMSE)',fontsize=25)
            
            # properly center the polar plot on the array centroid
            array_centroid = np.mean(station_grid_coords,axis=0)
            x_pos = (array_centroid[0]-plot_bounds[0])/(plot_bounds[1]-plot_bounds[0])
            y_pos = (array_centroid[1]-plot_bounds[2])/(plot_bounds[3]-plot_bounds[2])
            width = 0.3

            # make polar plot centered at array centroid
            ax_polar = fig.add_axes([1.2*j+x_pos-width/2,y_pos-width/2,width,width], projection = 'polar')
            ax_polar.set_theta_zero_location('N')
            ax_polar.set_theta_direction(-1)
            radius,bins = np.histogram(backazimuth[~np.isnan(backazimuth)]*np.pi/180,bins=np.linspace(0,2*np.pi,37))
            patches = ax_polar.bar(bins[:-1], radius, zorder=1, align='edge', width=np.diff(bins),facecolor=cmap(1),
                             edgecolor='black', fill=True, linewidth=1,alpha = .5)

            # Remove ylabels for area plots (they are mostly obstructive)
            ax_polar.set_yticks([])
            ax_polar.axis('off')
            
        # define, transform, and plot lat/lon grid
        lat = np.arange(-73,-76,-0.25)
        lon = np.arange(-98,-104,-0.5)
        x_lab_pos=[]
        y_lab_pos=[]
        line = np.linspace(-110,-90,100)
        for i in lat:
            line_x,line_y = transform(p1,p2,line,np.linspace(i,i,100))
            ax_image.plot(line_x,line_y,linestyle='--',dashes=(7, 10),linewidth=1,c='k',alpha=1)
            y_lab_pos.append(line_y[np.argmin(np.abs(line_x-plot_bounds[0]))])
        line = np.linspace(-80,-70,100)
        for i in lon:
            line_x,line_y = transform(p1,p2,np.linspace(i,i,100),line)
            ax_image.plot(line_x,line_y,linestyle='--',dashes=(7, 7),linewidth=1,c='k',alpha=1)
            x_lab_pos.append(line_x[np.argmin(np.abs(line_y-plot_bounds[2]))])

        # set ticks and labels for lat/lon grid
        ax_image.set_xticks(x_lab_pos)
        lonlabels = [str(lon[i]) + '$^\circ$' for i in range(len(lon))]
        ax_image.set_xticklabels(labels=lonlabels,fontsize=25)
        ax_image.set_xlabel("Longitude",fontsize=25)
        ax_image.set_yticks(y_lab_pos)
        latlabels = [str(lat[i]) + '$^\circ$' for i in range(len(lat))]
        ax_image.set_yticklabels(labels=latlabels,fontsize=25)
        ax_image.set_ylabel("Latitude",fontsize=25)
        ax_image.yaxis.set_label_coords(-0.05, 0.5)

        # plot station locations   
        axes_coords = np.array([1.2*j, 0, 1, 1])
        ax_stats = fig.add_axes(axes_coords)
        ax_stats.scatter(station_grid_coords[:,0],station_grid_coords[:,1],marker="^",c='black',s=400)

        # set axis limits and turn off labels for scatter axis
        ax_image.set_xlim([plot_bounds[0],plot_bounds[1]])
        ax_image.set_ylim([plot_bounds[2],plot_bounds[3]])
        ax_stats.set_xlim([plot_bounds[0],plot_bounds[1]])
        ax_stats.set_ylim([plot_bounds[2],plot_bounds[3]])
        ax_stats.axis('off')

    #     # plot grounding line
    #     grounding_line_file = "data/shapefiles/ASAID_GroundingLine_Continent.shp"
    #     grounding_lines = gpd.read_file(grounding_line_file)
    #     pig_mask = geometry.Polygon([(plot_bounds[0],plot_bounds[2]),
    #                         (plot_bounds[0],plot_bounds[3]),
    #                         (plot_bounds[1],plot_bounds[3]),
    #                         (plot_bounds[1],plot_bounds[2]),
    #                         (plot_bounds[0],plot_bounds[2])])
    #     pig_gdf = gpd.GeoDataFrame(geometry=[pig_mask],crs="EPSG:3245")
    #     pig_gdf = pig_gdf.to_crs(grounding_lines.crs)
    #     pig_grounding_line = grounding_lines.clip(pig_gdf)
    #     pig_grounding_line=pig_grounding_line.to_crs("EPSG:3245")
    #     pig_grounding_line.plot(linestyle='--',color='r',ax=ax_image)

        # add North arrow
        line_x,line_y = transform(p1,p2,np.linspace(-100.15,-100.15,100),np.linspace(-74.74,-74.7,100))
        ax_stats.plot(line_x,line_y,color='k',linewidth = 5)
        ax_stats.scatter(line_x[-1],line_y[-1],marker=(3,0,4),c='k',s=400)
        ax_stats.text(line_x[-1]-2000,line_y[-1]-3000,"N",color='k',fontsize=25)

        # add scale bar
        ax_stats.plot([plot_bounds[1]-25000,plot_bounds[1]-15000],[plot_bounds[2]+6000,plot_bounds[2]+6000],color='k',linewidth = 5)
        ax_stats.text(plot_bounds[1]-22500,plot_bounds[2]+4000,"10 km",color='k',fontsize=25)

        # add inset figure of antarctica
        if j == 0:
            world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
            ax_inset = fig.add_axes([-0.07,0.775,0.275,0.275],projection = ccrs.SouthPolarStereo())
            ax_inset.set_extent([-180, 180, -90, -65], crs=ccrs.PlateCarree())
            geom = geometry.box(minx=-103,maxx=-99,miny=-75.5,maxy=-74.5)
            ax_inset.add_geometries([geom], crs=ccrs.PlateCarree(), edgecolor='r',facecolor='none', linewidth=1)
            ax_inset.add_feature(cartopy.feature.OCEAN, facecolor='#A8C5DD', edgecolor='none')

        # add title
        file = os.listdir('data/TSX/' + tsx[j] + '/TSX-1.SAR.L1B/')[0]
        capture_string = '201205'+file.split('201205')[1].split('_')[0]
        capture_datetime = datetime.strptime(capture_string, "%Y%m%dT%H%M%S")
        ax_image.set_title(capture_datetime.strftime("%Y-%m-%d %H:%M:%S\n"),fontsize=35)
    
    # show plot
    plt.tight_layout()
    plt.savefig('outputs/figures/location.png',bbox_inches="tight")
    
    
    
def plot_imagery_seismic_location(background,tsx,plot_bounds,st,st_high,backazimuth,rmse_mat,station_grid_coords,grid_axes_coords,vlims):

    # make figure and axes to plot on
    fig, ax = plt.subplots(figsize=(15,15),dpi=100)
    ax.axis('off')

    # iterate through tsx scenes
    time_lims = []
    for j in range(len(tsx)):
    
        # Construct axis to plot on
        ax.axis('off')
        axes_coords = np.array([1.2*j, 0, 1, 1])
        ax_image = fig.add_axes(axes_coords)

        # overlay multiple landsat images
        if background[1] == 'landsat':
            landsat = background[0]
            for i in range(len(landsat)):

                # read each band
                landsat_B2 = rasterio.open('data/LANDSAT/'+landsat[i]+'/'+landsat[i]+'_SR_B2_epsg:3245.TIF') #blue
                landsat_B3 = rasterio.open('data/LANDSAT/'+landsat[i]+'/'+landsat[i]+'_SR_B3_epsg:3245.TIF') #green
                landsat_B4 = rasterio.open('data/LANDSAT/'+landsat[i]+'/'+landsat[i]+'_SR_B4_epsg:3245.TIF') #red
                image_B2 = landsat_B2.read(1)
                image_B3 = landsat_B3.read(1)
                image_B4 = landsat_B4.read(1)

                # crop each band to 99th percentile of brightness
                image_B2[image_B2 > np.percentile(image_B2,99)] = np.percentile(image_B2,99)
                image_B3[image_B3 > np.percentile(image_B3,99)] = np.percentile(image_B3,99)
                image_B4[image_B4 > np.percentile(image_B4,99)] = np.percentile(image_B4,99)

                # combine bands into natural color image
                image_rgb = np.array([image_B2, image_B3, image_B4]).transpose(1,2,0)
                normalized_rgb = (image_rgb * (255 / np.max(image_rgb))).astype(np.uint8)

                # get bounds
                landsat_bounds = landsat_B2.bounds
                horz_len = landsat_bounds[2]-landsat_bounds[0]
                vert_len = landsat_bounds[3]-landsat_bounds[1]

                # display image
                if i == 0:
                    ax_image.imshow(normalized_rgb,extent=[landsat_bounds[0],landsat_bounds[2],landsat_bounds[1],landsat_bounds[3]],interpolation='none',cmap='gray')
                else:    
                    # add alpha channel to image so it can be overlayed
                    alpha_slice = np.array(normalized_rgb.shape)
                    alpha_slice[2] = 1
                    alpha_array = np.zeros(alpha_slice)
                    alpha_array[np.nonzero(image_B2)[0],np.nonzero(image_B2)[1],0] = 255
                    normalized_rgba = np.append(normalized_rgb,alpha_array,axis=2).astype(int)
                    ax_image.imshow(normalized_rgba,extent=[landsat_bounds[0],landsat_bounds[2],landsat_bounds[1],landsat_bounds[3]],interpolation='none',cmap='gray')

        elif background[1] == 'tsx':
            tsx_background = background[0]

            # read tsx data
            directory = os.listdir('data/TSX/'+ tsx_background + '/TSX-1.SAR.L1B/')[0]
            tsx_path = 'data/TSX/'+tsx_background+'/TSX-1.SAR.L1B/'+ directory + '/IMAGEDATA/'
            raster = rasterio.open(glob.glob(tsx_path + '*epsg:3245.tif')[0])
            image = raster.read()

            # crop each band to 99th percentile of brightness
            image[image > np.percentile(image,99)] = np.percentile(image,99)

            # plot TSX imagery
            tsx_bounds = raster.bounds
            horz_len = tsx_bounds[2]-tsx_bounds[0]
            vert_len = tsx_bounds[3]-tsx_bounds[1]
            masked_image = np.ma.masked_where(image[0] == 0, image[0])
            ax_image.imshow(masked_image,extent=[tsx_bounds[0],tsx_bounds[2],tsx_bounds[1],tsx_bounds[3]],cmap='gray')

        # read tsx data
        directory = os.listdir('data/TSX/'+ tsx[j] + '/TSX-1.SAR.L1B/')[0]
        tsx_path = 'data/TSX/'+tsx[j]+'/TSX-1.SAR.L1B/'+ directory + '/IMAGEDATA/'
        raster = rasterio.open(glob.glob(tsx_path + '*epsg:3245.tif')[0])
        image = raster.read()

        # crop each band to 99th percentile of brightness
        image[image > np.percentile(image,99)] = np.percentile(image,99)

        # plot TSX imagery
        tsx_bounds = raster.bounds
        horz_len = tsx_bounds[2]-tsx_bounds[0]
        vert_len = tsx_bounds[3]-tsx_bounds[1]
        masked_image = np.ma.masked_where(image[0] == 0, image[0])
        ax_image.set_facecolor('k')
        ax_image.imshow(masked_image,extent=[tsx_bounds[0],tsx_bounds[2],tsx_bounds[1],tsx_bounds[3]],cmap='gray',vmin=vlims[j,0], vmax=vlims[j,1])

        # get corners of imagery extent
        p2 = Proj("EPSG:3245",preserve_units=False)
        p1 = Proj(proj='latlong',preserve_units=False)

        # plot location on second panel
        if j == 1:
            # plot gridsearch results
            grid_x = grid_axes_coords[0]
            grid_y = grid_axes_coords[1]
            cmap = plt.get_cmap('plasma')
            contour_x, contour_y = np.meshgrid(grid_x, grid_y)
            contours = ax_image.contour(contour_x, contour_y, np.log10(rmse_mat.T),levels=20,cmap=cmap,vmin=0.8,vmax=1.3,linewidths=3)
            ax_image.clabel(contours, colors='k', inline_spacing=-10, fontsize=25)
            axes_coords = np.array([1.2*j+1.05, 0.025, 0.05, 0.95])
            c_axis = fig.add_axes(axes_coords)
            norm = Normalize(vmin=0.8,vmax=1.3)
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cbar = fig.colorbar(sm, cax=c_axis, orientation='vertical')
            cbar.ax.invert_yaxis()
            cbar.ax.tick_params(labelsize=45)
            cbar.ax.set_ylabel('log10(RMSE)',fontsize=45)
            
            # properly center the polar plot on the array centroid
            array_centroid = np.mean(station_grid_coords,axis=0)
            x_pos = (array_centroid[0]-plot_bounds[0])/(plot_bounds[1]-plot_bounds[0])
            y_pos = (array_centroid[1]-plot_bounds[2])/(plot_bounds[3]-plot_bounds[2])
            width = 0.3

            # make polar plot centered at array centroid
            ax_polar = fig.add_axes([1.2*j+x_pos-width/2,y_pos-width/2,width,width], projection = 'polar')
            ax_polar.set_theta_zero_location('N')
            ax_polar.set_theta_direction(-1)
            baz = backazimuth[~np.isnan(backazimuth)]*np.pi/180
            ax_polar.arrow(baz,0,0,1,width=0.05,edgecolor='black',facecolor='black',lw=2,zorder=5)
            ax_polar.text(-0.7,0.7,r"$\theta$",color='k',fontsize=45)
#             radius,bins = np.histogram(backazimuth[~np.isnan(backazimuth)]*np.pi/180,bins=np.linspace(0,2*np.pi,37))
#             patches = ax_polar.bar(bins[:-1], radius, zorder=1, align='edge', width=np.diff(bins),facecolor='white',
#                              edgecolor='black', fill=True, linewidth=2,alpha = .5)

            # Remove ylabels for area plots (they are mostly obstructive)
            ax_polar.set_yticks([])
            ax_polar.axis('off')
            
            
        # define, transform, and plot lat/lon grid
        lat = np.arange(-73,-76,-0.25)
        lon = np.arange(-98,-104,-0.5)
        x_lab_pos=[]
        y_lab_pos=[]
        line = np.linspace(-110,-90,100)
        for i in lat:
            line_x,line_y = transform(p1,p2,line,np.linspace(i,i,100))
            ax_image.plot(line_x,line_y,linestyle='--',dashes=(7, 10),linewidth=1,c='k',alpha=1)
            y_lab_pos.append(line_y[np.argmin(np.abs(line_x-plot_bounds[0]))])
        line = np.linspace(-80,-70,100)
        for i in lon:
            line_x,line_y = transform(p1,p2,np.linspace(i,i,100),line)
            ax_image.plot(line_x,line_y,linestyle='--',dashes=(7, 7),linewidth=1,c='k',alpha=1)
            x_lab_pos.append(line_x[np.argmin(np.abs(line_y-plot_bounds[2]))])

        # set ticks and labels for lat/lon grid
        ax_image.set_xticks(x_lab_pos)
        lonlabels = [str(lon[i]) + '$^\circ$' for i in range(len(lon))]
        ax_image.set_xticklabels(labels=lonlabels,fontsize=45)
        ax_image.set_xlabel("Longitude",fontsize=45)
        ax_image.set_yticks(y_lab_pos)
        latlabels = [str(lat[i]) + '$^\circ$' for i in range(len(lat))]
        ax_image.set_yticklabels(labels=latlabels,fontsize=45)
        ax_image.set_ylabel("Latitude",fontsize=45)
        ax_image.yaxis.set_label_coords(-0.05, 0.5)

        # plot station locations   
        axes_coords = np.array([1.2*j, 0, 1, 1])
        ax_stats = fig.add_axes(axes_coords)
        ax_stats.scatter(station_grid_coords[:,0],station_grid_coords[:,1],marker="^",c='black',s=400)

        # set axis limits and turn off labels for scatter axis
        ax_image.set_xlim([plot_bounds[0],plot_bounds[1]])
        ax_image.set_ylim([plot_bounds[2],plot_bounds[3]])
        ax_stats.set_xlim([plot_bounds[0],plot_bounds[1]])
        ax_stats.set_ylim([plot_bounds[2],plot_bounds[3]])
        ax_stats.axis('off')

    #     # plot grounding line
    #     grounding_line_file = "data/shapefiles/ASAID_GroundingLine_Continent.shp"
    #     grounding_lines = gpd.read_file(grounding_line_file)
    #     pig_mask = geometry.Polygon([(plot_bounds[0],plot_bounds[2]),
    #                         (plot_bounds[0],plot_bounds[3]),
    #                         (plot_bounds[1],plot_bounds[3]),
    #                         (plot_bounds[1],plot_bounds[2]),
    #                         (plot_bounds[0],plot_bounds[2])])
    #     pig_gdf = gpd.GeoDataFrame(geometry=[pig_mask],crs="EPSG:3245")
    #     pig_gdf = pig_gdf.to_crs(grounding_lines.crs)
    #     pig_grounding_line = grounding_lines.clip(pig_gdf)
    #     pig_grounding_line=pig_grounding_line.to_crs("EPSG:3245")
    #     pig_grounding_line.plot(linestyle='--',color='r',ax=ax_image)

        # add North arrow
        line_x,line_y = transform(p1,p2,np.linspace(-100.15,-100.15,100),np.linspace(-74.74,-74.7,100))
        ax_stats.plot(line_x,line_y,color='k',linewidth = 5)
        ax_stats.scatter(line_x[-1],line_y[-1],marker=(3,0,4),c='k',s=400)
        ax_stats.text(line_x[-1]-2500,line_y[-1]-3500,"N",color='k',fontsize=45)

        # add scale bar
        ax_stats.plot([plot_bounds[1]-25000,plot_bounds[1]-15000],[plot_bounds[2]+6000,plot_bounds[2]+6000],color='k',linewidth = 5)
        ax_stats.text(plot_bounds[1]-24000,plot_bounds[2]+3400,"10 km",color='k',fontsize=45)

        # add inset figure of antarctica
        if j == 0:
            world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
            ax_inset = fig.add_axes([-0.155,0.7,0.35,0.35],projection = ccrs.SouthPolarStereo())
            ax_inset.set_extent([-180, 180, -90, -65], crs=ccrs.PlateCarree())
            geom = geometry.box(minx=-104.1,maxx=-98,miny=-76,maxy=-74)
            ax_inset.add_geometries([geom], crs=ccrs.PlateCarree(), edgecolor='r',facecolor='none', linewidth=1.5)
            ax_inset.add_feature(cartopy.feature.OCEAN, facecolor='#A8C5DD', edgecolor='none')

        # add title
        file = os.listdir('data/TSX/' + tsx[j] + '/TSX-1.SAR.L1B/')[0]
        capture_string = '201205'+file.split('201205')[1].split('_')[0]
        capture_datetime = datetime.strptime(capture_string, "%Y%m%dT%H%M%S")
        time_lims.append(capture_datetime)
        if j == 0:
            ax_image.set_title(".            A. TSX data at $t=t_1$",fontsize=60,pad=10,loc='left')
        else:
            ax_image.set_title("B. TSX data at $t=t_2$",fontsize=60,pad=10,loc='left')

    # display seismic data between two images
    starttime = obspy.UTCDateTime(time_lims[0])
    endtime = obspy.UTCDateTime(time_lims[1])
    num_ticks = (np.floor((endtime-starttime)/3600)+1)/12
    st = st.trim(starttime=starttime,endtime=endtime)
    ax_seismic = fig.add_axes([0,-0.625,2.2,0.5])
    ticks = [starttime.datetime + timedelta(seconds=tick*3600*12) for tick in range(int(num_ticks))]
    times = [starttime.datetime + timedelta(seconds=s/st[0].stats.sampling_rate) for s in range(st[0].stats.npts)]
    ax_seismic.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:00'))
    ax_seismic.set_xticks(ticks)
    ax_seismic.grid(True)
    ax_seismic.set_ylabel("Velocity (mm/s)",fontsize=45)
    ax_seismic.tick_params(axis='both', which='major', labelsize=45)
    ax_seismic.set_xlim([starttime,endtime])
    ax_seismic.set_title("C. PIG2 Z-component data from $t_1$ to $t_2$ (0.001-1 Hz)",fontsize=60,wrap=False,loc='left')
    ax_seismic.plot(times,st[0].data*1000,'k',linewidth=3,zorder=10)
    [x.set_linewidth(1.5) for x in ax_seismic.spines.values()]
    
    # highlight event window 
    starttime = obspy.UTCDateTime(2012,5,9,18)
    endtime = obspy.UTCDateTime(2012,5,9,20)
    ylims = ax_seismic.get_ylim()
    rect = Rectangle((starttime, ylims[0]), endtime-starttime, ylims[1]-ylims[0], linewidth=0, facecolor='r',alpha=0.1,zorder=0)
    ax_seismic.axvline(starttime,linestyle='--',dashes=(7, 7))
    ax_seismic.axvline(endtime,linestyle='--',dashes=(7, 7))
    ax_seismic.add_patch(rect)
    
    # display just the event (low frequency)
    ax_seismic2 = fig.add_axes([0,-0.625*2,2.2,0.5])
    starttime = obspy.UTCDateTime(2012,5,9,18)
    endtime = obspy.UTCDateTime(2012,5,9,20)
    st = st.trim(starttime=starttime,endtime=endtime)
    num_ticks = 9
    ticks = [starttime.datetime + timedelta(seconds=tick*900) for tick in range(int(num_ticks))]
    times = [starttime.datetime + timedelta(seconds=s/st[0].stats.sampling_rate) for s in range(st[0].stats.npts)]
    ax_seismic2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax_seismic2.set_ylabel("Velocity (mm/s)",fontsize=45)
    ax_seismic2.tick_params(axis='both', which='major', labelsize=45)
    ax_seismic2.set_xticks(ticks)
    ax_seismic2.grid(True)
    ax_seismic2.set_xlim([starttime,endtime])
    ax_seismic2.set_title("D. Rift event seismogram (0.001-1 Hz)",fontsize=60,loc='left')
    ax_seismic2.plot(times,st[0].data*1000,'k',linewidth=3)
    labels = ["05-09 18:00","18:15","18:30","18:45","19:00","19:15","19:30","19:45","20:00"]
    ax_seismic2.set_xticklabels(labels)
    [x.set_linewidth(1.5) for x in ax_seismic2.spines.values()]
    
    # draw lines connecting the highlighted area of first data plot to second plot
    line1_start = [starttime,ax_seismic2.get_ylim()[1]]
    line1_end = [starttime,ax_seismic.get_ylim()[0]]
    line2_start = [endtime,ax_seismic2.get_ylim()[1]]
    line2_end = [endtime,ax_seismic.get_ylim()[0]] 
    con1 = ConnectionPatch(xyA=line1_start, xyB=line1_end, coordsA='data', coordsB='data',axesA=ax_seismic2,axesB=ax_seismic,linestyle=(5,(7,7)))
    con2 = ConnectionPatch(xyA=line2_start, xyB=line2_end, coordsA='data', coordsB='data',axesA=ax_seismic2,axesB=ax_seismic,linestyle=(5,(7,7)))
    ax_seismic2.add_artist(con1)
    ax_seismic2.add_artist(con2)
    cons = [con1,con2]
    for con in cons:
        con.set_color('k')
        con.set_linewidth(1.5)
        
    # display just the event (high frequency)
    ax_seismic3 = fig.add_axes([0,-0.625*3,2.2,0.5])
    starttime = obspy.UTCDateTime(2012,5,9,18)
    endtime = starttime + 540
    st_high = st_high.trim(starttime=starttime,endtime=endtime)
    ax_seismic3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax_seismic3.set_yticks([-0.01,0,0.01])
    num_ticks = 11
    ticks = [starttime.datetime + timedelta(seconds=tick*60) for tick in range(int(num_ticks))]    
    times = [starttime.datetime + timedelta(seconds=s/st_high[0].stats.sampling_rate) for s in range(st_high[0].stats.npts)]
    ax_seismic3.set_xticks(ticks)
    ax_seismic3.grid(True)
    ax_seismic3.set_ylabel("Velocity (mm/s)",fontsize=45)
    ax_seismic3.tick_params(axis='both', which='major', labelsize=45)
    ax_seismic3.set_xlim([starttime,endtime])
    ax_seismic3.set_title("E. Rift event seismogram (>1 Hz)",fontsize=60,loc='left')
    ax_seismic3.plot(times,st_high[0].data*1000,'k',linewidth=3)
    labels = ["05-09 18:00","18:01","18:02","18:03","18:04","18:05","18:06","18:07","18:08","18:09","18:10"]
    ax_seismic3.set_xticklabels(labels)
    [x.set_linewidth(1.5) for x in ax_seismic3.spines.values()]
    
    # highlight event window
    ylims = ax_seismic2.get_ylim()
    rect = Rectangle((starttime, ylims[0]), endtime-starttime, ylims[1]-ylims[0], linewidth=0, facecolor='r',alpha=0.1,zorder=0)
    ax_seismic2.axvline(starttime,linestyle='--',dashes=(7, 7))
    ax_seismic2.axvline(endtime,linestyle='--',dashes=(7, 7))
    ax_seismic2.add_patch(rect)
    
    # draw lines connecting the highlighted area of first data plot to second plot
    line1_start = [starttime,ax_seismic3.get_ylim()[1]]
    line1_end = [starttime,ax_seismic2.get_ylim()[0]]
    line2_start = [endtime,ax_seismic3.get_ylim()[1]]
    line2_end = [endtime,ax_seismic2.get_ylim()[0]] 
    con1 = ConnectionPatch(xyA=line1_start, xyB=line1_end, coordsA='data', coordsB='data',axesA=ax_seismic3,axesB=ax_seismic2,linestyle=(5,(7,7)))
    con2 = ConnectionPatch(xyA=line2_start, xyB=line2_end, coordsA='data', coordsB='data',axesA=ax_seismic3,axesB=ax_seismic2,linestyle=(5,(7,7)))
    ax_seismic3.add_artist(con1)
    ax_seismic3.add_artist(con2)
    cons = [con1,con2]
    for con in cons:
        con.set_color('k')
        con.set_linewidth(1.5)
        
    
    # show plot
    plt.tight_layout(h_pad = 200)
    plt.savefig('outputs/figures/location.png',bbox_inches="tight")
    
    
def plot_imagery_and_data(background,tsx,plot_bounds,st,st_high,station_grid_coords,grid_axes_coords,vlims):

    # make figure and axes to plot on
    fig, ax = plt.subplots(figsize=(15,15),dpi=100)
    ax.axis('off')

    # iterate through tsx scenes
    time_lims = []
    for j in range(len(tsx)):
    
        # Construct axis to plot on
        ax.axis('off')
        axes_coords = np.array([1.2*j, 0, 1, 1])
        ax_image = fig.add_axes(axes_coords)

        # overlay multiple landsat images
        if background[1] == 'landsat':
            landsat = background[0]
            for i in range(len(landsat)):

                # read each band
                landsat_B2 = rasterio.open('data/LANDSAT/'+landsat[i]+'/'+landsat[i]+'_SR_B2_epsg:3245.TIF') #blue
                landsat_B3 = rasterio.open('data/LANDSAT/'+landsat[i]+'/'+landsat[i]+'_SR_B3_epsg:3245.TIF') #green
                landsat_B4 = rasterio.open('data/LANDSAT/'+landsat[i]+'/'+landsat[i]+'_SR_B4_epsg:3245.TIF') #red
                image_B2 = landsat_B2.read(1)
                image_B3 = landsat_B3.read(1)
                image_B4 = landsat_B4.read(1)

                # crop each band to 99th percentile of brightness
                image_B2[image_B2 > np.percentile(image_B2,99)] = np.percentile(image_B2,99)
                image_B3[image_B3 > np.percentile(image_B3,99)] = np.percentile(image_B3,99)
                image_B4[image_B4 > np.percentile(image_B4,99)] = np.percentile(image_B4,99)

                # combine bands into natural color image
                image_rgb = np.array([image_B2, image_B3, image_B4]).transpose(1,2,0)
                normalized_rgb = (image_rgb * (255 / np.max(image_rgb))).astype(np.uint8)

                # get bounds
                landsat_bounds = landsat_B2.bounds
                horz_len = landsat_bounds[2]-landsat_bounds[0]
                vert_len = landsat_bounds[3]-landsat_bounds[1]

                # display image
                if i == 0:
                    ax_image.imshow(normalized_rgb,extent=[landsat_bounds[0],landsat_bounds[2],landsat_bounds[1],landsat_bounds[3]],interpolation='none',cmap='gray')
                else:    
                    # add alpha channel to image so it can be overlayed
                    alpha_slice = np.array(normalized_rgb.shape)
                    alpha_slice[2] = 1
                    alpha_array = np.zeros(alpha_slice)
                    alpha_array[np.nonzero(image_B2)[0],np.nonzero(image_B2)[1],0] = 255
                    normalized_rgba = np.append(normalized_rgb,alpha_array,axis=2).astype(int)
                    ax_image.imshow(normalized_rgba,extent=[landsat_bounds[0],landsat_bounds[2],landsat_bounds[1],landsat_bounds[3]],interpolation='none',cmap='gray')

        elif background[1] == 'tsx':
            tsx_background = background[0]

            # read tsx data
            directory = os.listdir('data/TSX/'+ tsx_background + '/TSX-1.SAR.L1B/')[0]
            tsx_path = 'data/TSX/'+tsx_background+'/TSX-1.SAR.L1B/'+ directory + '/IMAGEDATA/'
            raster = rasterio.open(glob.glob(tsx_path + '*epsg:3245.tif')[0])
            image = raster.read()

            # crop each band to 99th percentile of brightness
            image[image > np.percentile(image,99)] = np.percentile(image,99)

            # plot TSX imagery
            tsx_bounds = raster.bounds
            horz_len = tsx_bounds[2]-tsx_bounds[0]
            vert_len = tsx_bounds[3]-tsx_bounds[1]
            masked_image = np.ma.masked_where(image[0] == 0, image[0])
            ax_image.imshow(masked_image,extent=[tsx_bounds[0],tsx_bounds[2],tsx_bounds[1],tsx_bounds[3]],cmap='gray')

        # read tsx data
        directory = os.listdir('data/TSX/'+ tsx[j] + '/TSX-1.SAR.L1B/')[0]
        tsx_path = 'data/TSX/'+tsx[j]+'/TSX-1.SAR.L1B/'+ directory + '/IMAGEDATA/'
        raster = rasterio.open(glob.glob(tsx_path + '*epsg:3245.tif')[0])
        image = raster.read()

        # crop each band to 99th percentile of brightness
        image[image > np.percentile(image,99)] = np.percentile(image,99)

        # plot TSX imagery
        tsx_bounds = raster.bounds
        horz_len = tsx_bounds[2]-tsx_bounds[0]
        vert_len = tsx_bounds[3]-tsx_bounds[1]
        masked_image = np.ma.masked_where(image[0] == 0, image[0])
        ax_image.set_facecolor('k')
        ax_image.imshow(masked_image,extent=[tsx_bounds[0],tsx_bounds[2],tsx_bounds[1],tsx_bounds[3]],cmap='gray',vmin=vlims[j,0], vmax=vlims[j,1])

        # get corners of imagery extent
        p2 = Proj("EPSG:3245",preserve_units=False)
        p1 = Proj(proj='latlong',preserve_units=False)
                      
        # define, transform, and plot lat/lon grid
        lat = np.arange(-73,-76,-0.25)
        lon = np.arange(-98,-104,-0.5)
        x_lab_pos=[]
        y_lab_pos=[]
        line = np.linspace(-110,-90,100)
        for i in lat:
            line_x,line_y = transform(p1,p2,line,np.linspace(i,i,100))
            ax_image.plot(line_x,line_y,linestyle='--',dashes=(7, 10),linewidth=1,c='k',alpha=1)
            y_lab_pos.append(line_y[np.argmin(np.abs(line_x-plot_bounds[0]))])
        line = np.linspace(-80,-70,100)
        for i in lon:
            line_x,line_y = transform(p1,p2,np.linspace(i,i,100),line)
            ax_image.plot(line_x,line_y,linestyle='--',dashes=(7, 7),linewidth=1,c='k',alpha=1)
            x_lab_pos.append(line_x[np.argmin(np.abs(line_y-plot_bounds[2]))])

        # set ticks and labels for lat/lon grid
        ax_image.set_xticks(x_lab_pos)
        lonlabels = [str(lon[i]) + '$^\circ$' for i in range(len(lon))]
        ax_image.set_xticklabels(labels=lonlabels,fontsize=45)
        ax_image.set_xlabel("Longitude",fontsize=45)
        ax_image.set_yticks(y_lab_pos)
        latlabels = [str(lat[i]) + '$^\circ$' for i in range(len(lat))]
        ax_image.set_yticklabels(labels=latlabels,fontsize=45)
        ax_image.set_ylabel("Latitude",fontsize=45)
        ax_image.yaxis.set_label_coords(-0.05, 0.5)

        # plot station locations   
        axes_coords = np.array([1.2*j, 0, 1, 1])
        ax_stats = fig.add_axes(axes_coords)
        ax_stats.scatter(station_grid_coords[:,0],station_grid_coords[:,1],marker="^",c='black',s=400)

        # set axis limits and turn off labels for scatter axis
        ax_image.set_xlim([plot_bounds[0],plot_bounds[1]])
        ax_image.set_ylim([plot_bounds[2],plot_bounds[3]])
        ax_stats.set_xlim([plot_bounds[0],plot_bounds[1]])
        ax_stats.set_ylim([plot_bounds[2],plot_bounds[3]])
        ax_stats.axis('off')

    #     # plot grounding line
    #     grounding_line_file = "data/shapefiles/ASAID_GroundingLine_Continent.shp"
    #     grounding_lines = gpd.read_file(grounding_line_file)
    #     pig_mask = geometry.Polygon([(plot_bounds[0],plot_bounds[2]),
    #                         (plot_bounds[0],plot_bounds[3]),
    #                         (plot_bounds[1],plot_bounds[3]),
    #                         (plot_bounds[1],plot_bounds[2]),
    #                         (plot_bounds[0],plot_bounds[2])])
    #     pig_gdf = gpd.GeoDataFrame(geometry=[pig_mask],crs="EPSG:3245")
    #     pig_gdf = pig_gdf.to_crs(grounding_lines.crs)
    #     pig_grounding_line = grounding_lines.clip(pig_gdf)
    #     pig_grounding_line=pig_grounding_line.to_crs("EPSG:3245")
    #     pig_grounding_line.plot(linestyle='--',color='r',ax=ax_image)

        # add North arrow
        line_x,line_y = transform(p1,p2,np.linspace(-100.15,-100.15,100),np.linspace(-74.74,-74.7,100))
        ax_stats.plot(line_x,line_y,color='k',linewidth = 5)
        ax_stats.scatter(line_x[-1],line_y[-1],marker=(3,0,4),c='k',s=400)
        ax_stats.text(line_x[-1]-2500,line_y[-1]-3500,"N",color='k',fontsize=45)

        # add scale bar
        ax_stats.plot([plot_bounds[1]-25000,plot_bounds[1]-15000],[plot_bounds[2]+6000,plot_bounds[2]+6000],color='k',linewidth = 5)
        ax_stats.text(plot_bounds[1]-24000,plot_bounds[2]+3400,"10 km",color='k',fontsize=45)

        # add inset figure of antarctica
        if j == 0:
            world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
            ax_inset = fig.add_axes([-0.155,0.7,0.35,0.35],projection = ccrs.SouthPolarStereo())
            ax_inset.set_extent([-180, 180, -90, -65], crs=ccrs.PlateCarree())
            geom = geometry.box(minx=-104.1,maxx=-98,miny=-76,maxy=-74)
            ax_inset.add_geometries([geom], crs=ccrs.PlateCarree(), edgecolor='r',facecolor='none', linewidth=1.5)
            ax_inset.add_feature(cartopy.feature.OCEAN, facecolor='#A8C5DD', edgecolor='none')

        # add title
        file = os.listdir('data/TSX/' + tsx[j] + '/TSX-1.SAR.L1B/')[0]
        capture_string = '201205'+file.split('201205')[1].split('_')[0]
        capture_datetime = datetime.strptime(capture_string, "%Y%m%dT%H%M%S")
        time_lims.append(capture_datetime)
        datestring = capture_datetime.strftime("%Y-%m-%d %H:%M")
        if j == 0:
            ax_image.set_title(".            A. TSX data at "+datestring,fontsize=60,pad=10,loc='left')
        else:
            ax_image.set_title("B. TSX data at "+datestring,fontsize=60,pad=10,loc='left')

    # display seismic data between two images
    starttime = obspy.UTCDateTime(time_lims[0])
    endtime = obspy.UTCDateTime(time_lims[1])
    num_ticks = (np.floor((endtime-starttime)/3600)+1)/12
    st = st.trim(starttime=starttime,endtime=endtime)
    ax_seismic = fig.add_axes([0,-0.625,2.2,0.5])
    ticks = [starttime.datetime + timedelta(seconds=tick*3600*12) for tick in range(int(num_ticks))]
    times = [starttime.datetime + timedelta(seconds=s/st[0].stats.sampling_rate) for s in range(st[0].stats.npts)]
    ax_seismic.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:00'))
    ax_seismic.set_xticks(ticks)
    ax_seismic.grid(True)
    ax_seismic.set_ylabel("Velocity (mm/s)",fontsize=45)
    ax_seismic.tick_params(axis='both', which='major', labelsize=45)
    ax_seismic.set_xlim([starttime,endtime])
    ax_seismic.set_title("C. PIG2 Z-component data from $t_1$ to $t_2$ (0.001-1 Hz)",fontsize=60,wrap=False,loc='left')
    ax_seismic.plot(times,st[0].data*1000,'k',linewidth=3,zorder=10)
    [x.set_linewidth(1.5) for x in ax_seismic.spines.values()]
    
    # display just the event (low frequency)
    ax_seismic2 = fig.add_axes([0,-0.625*2,2.2,0.5])
    starttime = obspy.UTCDateTime(2012,5,9,18)
    endtime = obspy.UTCDateTime(2012,5,9,20)
    st = st.trim(starttime=starttime,endtime=endtime)
    num_ticks = 9
    ticks = [starttime.datetime + timedelta(seconds=tick*900) for tick in range(int(num_ticks))]
    times = [starttime.datetime + timedelta(seconds=s/st[0].stats.sampling_rate) for s in range(st[0].stats.npts)]
    ax_seismic2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax_seismic2.set_ylabel("Velocity (mm/s)",fontsize=45)
    ax_seismic2.tick_params(axis='both', which='major', labelsize=45)
    ax_seismic2.set_xticks(ticks)
    ax_seismic2.grid(True)
    ax_seismic2.set_xlim([starttime,endtime])
    ax_seismic2.set_title("D. Rift event seismogram (0.001-1 Hz)",fontsize=60,loc='left')
    ax_seismic2.plot(times,st[0].data*1000,'k',linewidth=3)
    labels = ["05-09 18:00","18:15","18:30","18:45","19:00","19:15","19:30","19:45","20:00"]
    ax_seismic2.set_xticklabels(labels)
    [x.set_linewidth(1.5) for x in ax_seismic2.spines.values()]

        
    # display just the event (high frequency)
    ax_seismic3 = fig.add_axes([0,-0.625*3,2.2,0.5])
    starttime = obspy.UTCDateTime(2012,5,9,18)
    endtime = starttime + 540
    st_high = st_high.trim(starttime=starttime,endtime=endtime)
    ax_seismic3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax_seismic3.set_yticks([-0.01,0,0.01])
    num_ticks = 11
    ticks = [starttime.datetime + timedelta(seconds=tick*60) for tick in range(int(num_ticks))]    
    times = [starttime.datetime + timedelta(seconds=s/st_high[0].stats.sampling_rate) for s in range(st_high[0].stats.npts)]
    ax_seismic3.set_xticks(ticks)
    ax_seismic3.grid(True)
    ax_seismic3.set_ylabel("Velocity (mm/s)",fontsize=45)
    ax_seismic3.tick_params(axis='both', which='major', labelsize=45)
    ax_seismic3.set_xlim([starttime,endtime])
    ax_seismic3.set_title("E. Rift event seismogram (>1 Hz)",fontsize=60,loc='left')
    ax_seismic3.plot(times,st_high[0].data*1000,'k',linewidth=3)
    labels = ["05-09 18:00","18:01","18:02","18:03","18:04","18:05","18:06","18:07","18:08","18:09","18:10"]
    ax_seismic3.set_xticklabels(labels)
    [x.set_linewidth(1.5) for x in ax_seismic3.spines.values()]
        
    
    # show plot
    plt.tight_layout(h_pad = 200)
    plt.savefig('outputs/figures/imagery_and_seismic_data.png',bbox_inches="tight")