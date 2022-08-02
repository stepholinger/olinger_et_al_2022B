import os
import glob
import time
import obspy
from obspy.clients.fdsn.mass_downloader import RectangularDomain, Restrictions, MassDownloader



def download_data(data_path,starttime,endtime,networks,stations,channels):
    
    # set domain (just include the entire planet)
    domain = RectangularDomain(minlatitude=-90, maxlatitude=90, minlongitude=-180, maxlongitude=180.0)
    
    # iterate through networks, stations, and channels
    for network in networks:
        for station in stations:
            for channel in channels:     

                # set request parameters
                restrictions = Restrictions(
                    starttime=starttime,
                    endtime=endtime,
                    chunklength_in_sec=86400,
                    network=network, station=station, location="", channel=channel+"*",
                    location_priorities = ["01",],
                    channel_priorities = [channel+"Z",channel+"N",channel+"E"],
                    reject_channels_with_gaps=False,
                    minimum_length=0.0)

                mdl = MassDownloader(providers=["IRIS"])

                mdl.download(
                    domain=domain,
                    restrictions=restrictions, 
                    mseed_storage=data_path + "MSEED/raw/", 
                    stationxml_storage=(data_path + "XML/"))
                
                
                
def remove_ir(data_path,band,freq_lims,output_type):
    
    #make a list of all files in the raw folder
    raw_files = glob.glob(data_path + "MSEED/raw/*" + band + "*", recursive=True)
    raw_files.sort()

    #loop through all raw files
    for f in raw_files:

        #start timer
        t = time.time()

        #read in one data file
        st = obspy.read(f)

        #grab a couple useful variables from file metadata
        station = st[0].stats.station
        channel = st[0].stats.channel
        start_date = str(st[0].stats.starttime).split("T")[0]

        #specify output filename format
        out_path = data_path + "MSEED/no_IR/" + station + "/" + channel + "/" 
        out_file = out_path + start_date + "." + station + "." + channel + ".no_IR.MSEED"
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        #preprocess file
        st.detrend("demean")
        st.detrend("linear")
        st.taper(max_percentage=0.00025, max_length=20.)

        #read correct stationXML file
        XML_path = glob.glob(data_path + "XML/*" + station + "*.xml")[0]
        inv = obspy.read_inventory(XML_path)

        #remove instrument response
        st.remove_response(inventory=inv,pre_filt=freq_lims,output=output_type)

        #write new file
        st.write(out_file,format='MSEED')

        #end timer
        run_time = time.time() - t

        #give some output to check progress
        print("Response removed from " + f + " in " + str(run_time) + " seconds.")