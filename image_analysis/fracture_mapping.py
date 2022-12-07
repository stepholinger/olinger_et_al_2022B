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
import datetime
from datetime import timedelta
import types
import time
from collections import Counter
import scipy
from scipy.signal import hilbert,fftconvolve
from scipy.ndimage import label, gaussian_filter
from skimage import img_as_float
from skimage.morphology import reconstruction
import pathlib
import glob
import copy
import multiprocessing
import pickle
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.dates as mdates
from seismic_analysis.directivity import get_velocity
import cartopy
import cartopy.crs as ccrs


def read_tsx(scene):
    directory = os.listdir('data/TSX/'+ scene + '/TSX-1.SAR.L1B/')[0]
    tsx_path = 'data/TSX/'+ scene+'/TSX-1.SAR.L1B/'+ directory + '/IMAGEDATA/'
    raster = rasterio.open(glob.glob(tsx_path + '*epsg:3245.tif')[0])
    return raster



# function to clip a raster to a polygon in the same coordinant system
def clip(raster,bounds):
    # raster.bounds returns left, bottom, right, top
    # below is left, right, bottom, top
    clip_mask = geometry.Polygon([(bounds[0],bounds[2]),
                            (bounds[0],bounds[3]),
                            (bounds[1],bounds[3]),
                            (bounds[1],bounds[2]),
                            (bounds[0],bounds[2])])
    out_raster, out_transform = mask(raster, shapes=[clip_mask], crop=True)
    return out_raster
    
    
    
# function to align two images with cross correlation 
def align_images(image_1,image_2):
    
    # normalize inputs
    norm_1 = image_1/np.max(image_1)
    norm_2 = image_2/np.max(image_2)

    # demean inputs
    demean_norm_1 = norm_1 - np.mean(norm_1)
    demean_norm_2 = norm_2 - np.mean(norm_2)
    
    # compute autocorrelation of first input
    auto_corr = fftconvolve(demean_norm_1, demean_norm_1[::-1,::-1], mode='same')
    y_auto,x_auto = np.unravel_index(auto_corr.argmax(), auto_corr.shape)

    # compute correlation between each in put
    corr = fftconvolve(demean_norm_1, demean_norm_2[::-1,::-1], mode='same')
    y,x = np.unravel_index(corr.argmax(), corr.shape)

    # compute shift values
    y_shift = y-y_auto
    x_shift = x-x_auto
    
    return x_shift,y_shift



# function to threshold an image above a normalized value between 0 and 1
def binary_threshold(image,thresh):
    
    # convert to float and filter
    image = img_as_float(image)
    image = gaussian_filter(image, 1)

    # remove any exactly 0 values
    image[image == 0] = np.min(image[np.nonzero(image)])    

    # rescale image so pixel values are between 0 and 1
    image = image - np.min(image)
    image = image/np.max(image)
    
    # run reconstruction on the image (helps make bright areas more uniform)
    seed = np.copy(image)
    seed[1:-1, 1:-1] = 0
    dilated_image = reconstruction(seed, image, method='dilation')

    # binary thresholding
    dilated_image[dilated_image > thresh] = 1
    dilated_image[dilated_image < thresh] = 0
    return dilated_image



def plot_fracture_imagery(background,tsx,features,plot_bounds,vlims):

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
        ax_image.set_xticklabels(labels=lonlabels,fontsize=35)
        ax_image.set_xlabel("Longitude",fontsize=35)
        ax_image.set_yticks(y_lab_pos)
        latlabels = [str(lat[i]) + '$^\circ$' for i in range(len(lat))]
        ax_image.set_yticklabels(labels=latlabels,fontsize=35)
        ax_image.set_ylabel("Latitude",fontsize=35)
        ax_image.yaxis.set_label_coords(-0.05, 0.5)

        # plot any features on second panel (these should be masked arrays)
        for k in range(len(features)):
            feature = features[k][0]
            feature_extent = features[k][1]
            panel = features[k][2]
            feature_type = features[k][3]
            feature_color = features[k][4]
            if j == panel:
                if feature_type == "point":
                    ax_image.scatter(feature[0],feature[1],color=feature_color,s=1000,marker='*',edgecolors='k')
                if feature_type == "raster":
                    ax_image.imshow(feature,extent=feature_extent,cmap=feature_color,vmin=0,vmax=1)
        
        # plot station locations   
        axes_coords = np.array([1.2*j, 0, 1, 1])
        ax_stats = fig.add_axes(axes_coords)
#         ax_stats.scatter(station_grid_coords[:,0],station_grid_coords[:,1],marker="^",c='black',s=400)

        # set axis limits and turn off labels for scatter axis
        ax_image.set_xlim([plot_bounds[0],plot_bounds[1]])
        ax_image.set_ylim([plot_bounds[2],plot_bounds[3]])
        ax_stats.set_xlim([plot_bounds[0],plot_bounds[1]])
        ax_stats.set_ylim([plot_bounds[2],plot_bounds[3]])
        ax_stats.axis('off')

#         # plot grounding line
#         grounding_line_file = "data/shapefiles/ASAID_GroundingLine_Continent.shp"
#         grounding_lines = gpd.read_file(grounding_line_file)
#         pig_mask = geometry.Polygon([(plot_bounds[0],plot_bounds[2]),
#                             (plot_bounds[0],plot_bounds[3]),
#                             (plot_bounds[1],plot_bounds[3]),
#                             (plot_bounds[1],plot_bounds[2]),
#                             (plot_bounds[0],plot_bounds[2])])
#         pig_gdf = gpd.GeoDataFrame(geometry=[pig_mask],crs="EPSG:3245")
#         pig_gdf = pig_gdf.to_crs(grounding_lines.crs)
#         pig_grounding_line = grounding_lines.clip(pig_gdf)
#         pig_grounding_line=pig_grounding_line.to_crs("EPSG:3245")
#         pig_grounding_line.plot(linestyle='--',color='r',ax=ax_image)

        # add North arrow
        line_x,line_y = transform(p1,p2,np.linspace(-100.85,-100.85,100),np.linspace(-74.83,-74.80,100))
        ax_stats.plot(line_x,line_y,color='k',linewidth = 5)
        ax_stats.scatter(line_x[-1],line_y[-1],marker=(3,0,4),c='k',s=400)
        ax_stats.text(line_x[-1]-1000,line_y[-1]-2000,"N",color='k',fontsize=35)

        # add scale bar
        ax_stats.plot([plot_bounds[1]-8000,plot_bounds[1]-3000],[plot_bounds[2]+4500,plot_bounds[2]+4500],color='k',linewidth = 5)
        ax_stats.text(plot_bounds[1]-7000,plot_bounds[2]+3000,"5 km",color='k',fontsize=35)

        # add inset figure of antarctica
#         if j == 0:
#             world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
#             ax_inset = fig.add_axes([-0.1,0.8,0.275,0.275],projection = ccrs.SouthPolarStereo())
#             ax_inset.set_extent([-180, 180, -90, -65], crs=ccrs.PlateCarree())
#             geom = geometry.box(minx=-103,maxx=-99,miny=-75.5,maxy=-74.5)
#             ax_inset.add_geometries([geom], crs=ccrs.PlateCarree(), edgecolor='r',facecolor='none', linewidth=1)
#             ax_inset.add_feature(cartopy.feature.OCEAN, facecolor='#A8C5DD', edgecolor='none')

        # add title
        file = os.listdir('data/TSX/' + tsx[j] + '/TSX-1.SAR.L1B/')[0]
        capture_string = '201205'+file.split('201205')[1].split('_')[0]
        capture_datetime = datetime.datetime.strptime(capture_string, "%Y%m%dT%H%M%S")
        time_lims.append(capture_datetime)
        if j == 0:
            ax_image.set_title("TSX data from " + capture_datetime.strftime("%Y-%m-%d %H:%M:%S"),fontsize=45,pad=10)
        else:
            ax_image.set_title("TSX data from " + capture_datetime.strftime("%Y-%m-%d %H:%M:%S"),fontsize=45,pad=10)

    # show plot
    plt.tight_layout()
    plt.savefig('outputs/figures/fracture_extent.png',bbox_inches="tight")