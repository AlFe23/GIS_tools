# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 12:05:00 2023

@author: User
"""

from osgeo import gdal
import numpy as np
import os 
import rasterio 
#import GeoTools
import gc
import numpy.matlib

import scipy
import scipy.misc
from scipy import signal
from PIL import Image
import argparse
import cv2
import os

# Here we will give the path of the files tha tahs to be coregistered
Source_RGB = r"E:\GMATICS\IZS\050248726010_01\050248726010_01_P001_MUL\21JUL06100720-M2AS-050248726010_01_P001.TIF"
Pan = r"E:\GMATICS\IZS\050248726010_01\050248726010_01_P001_PAN\21JUL06100720-P2AS-050248726010_01_P001.TIF"
Source_google=r"E:\GMATICS\IZS\050248726010_01\Google_sat.tif"
def GSA_sharpening(prisma_img_path,sen_img_path,n,m,s):
    # n= first band for the PRISMA interval of interst
    # m = last band for PRISMA interval of interst
    # s = Sentinel Band corresponding to the PRISMA interval of interest
    pass
def parse_args():
    parser = argparse.ArgumentParser(description="Pansharpen and coregister WorldView imagery with Google Satellite reference.")
    parser.add_argument('--rgb', required=True, help='Path to multispectral (RGB) image')
    parser.add_argument('--pan', required=True, help='Path to panchromatic image')
    parser.add_argument('--google', required=True, help='Path to Google Satellite reference image')
    parser.add_argument('--out', required=True, help='Output prefix (output files will be named <prefix>_sharpened.tif, <prefix>_coreg.tif, etc.)')
    return parser.parse_args()


def main():
    args = parse_args()
    Source_RGB = args.rgb
    Pan = args.pan
    Source_google = args.google
    out_prefix = args.out

    HS = gdal.Open(prisma_img_path)  
    shape_prisma = np.shape(HS.GetRasterBand(1).ReadAsArray())
    No_bands = m-n+1
    HS_array = np.zeros((shape_prisma[0],shape_prisma[1],No_bands))
    k=0
    for i in range(n,m+1):
        HS_array[:,:,k] = HS.GetRasterBand(i).ReadAsArray()
        k = k+1 
    imageLR_LP = np.copy(HS_array)
    
    #import Sentinel-2 data     
    sen2   =  gdal.Open(sen_img_path)
    SEN = sen2.GetRasterBand(s)
    imageHR = SEN.ReadAsArray()
    #imageHR = imageHR/10000 trying to give the Sentinel image scaled to reflectance we still get the same output reflactance values as we give direct Sentinel image with reflactance valuse between 1-10000
    imageHR = np.asfortranarray(imageHR,dtype ='float64') 
    #imageHR = imageHR.astype('float64')
    
    ### Upsampling
    rows = np.shape(imageLR_LP)[0]
    cols = np.shape(imageLR_LP)[1]
    bands = np.shape(imageLR_LP)[2]
    ratio1 = np.shape(imageHR)[0]/np.shape(HS_array)[0]
    ratio1 = int(ratio1)
    imageLR = np.zeros((rows*ratio1,cols*ratio1,bands))
    
    for b in range(0,bands):
        imageLR[:,:,b] = cv2.resize(np.reshape(imageLR_LP[:,:,b],(rows,cols),order='F'), dsize=(cols*ratio1,rows*ratio1), interpolation=cv2.INTER_CUBIC)   # we choose the interpolation method hat gives the closest output to the imresize matlab function with default parameters
    
    ## Remove means from imageLR
    imageLR0 = np.zeros(np.shape(imageLR)) 
    for i in range(0,np.shape(imageLR)[2]):
        imageLR0[:,:,i] = imageLR[:,:,i] - np.mean(imageLR[:,:,i])
        
    ### Remove means from imageLR_LP 
    imageLR_LP0 = np.zeros(np.shape(HS_array))
    for i in range(0,np.shape(HS_array)[2]):
        imageLR_LP0[:,:,i] = imageLR_LP[:,:,i] - np.mean(imageLR_LP[:,:,i])
        
    
    
    '''REVIEW'''
    #############################################kk
    ### Sintetic intensity
    
    #percent by which the image is resized
    scale_factor = 1/ratio1
    
    #calculate the 50 percent of original dimensions
    width = int(imageHR.shape[1] * scale_factor)
    height = int(imageHR.shape[0] * scale_factor)
    dsize = (width, height)
    
    imageHR0 = imageHR - np.mean(imageHR)
    imageHR0 = cv2.resize(imageHR0,dsize,interpolation = cv2.INTER_AREA)  # we choose the interpolation method hat gives the closest output to the imresize matlab function with default parameters
    #imageHR0 = np.transpose(imageHR0)
    
    # other possible methods to be reviewed
    # imageHR0 = imageHR - np.mean(imageHR)  
    red = GSA_sharpening(Source_RGB, Pan, 0, 0, 0)
    print('Red bands processed.')
    green = GSA_sharpening(Source_RGB, Pan, 20, 24, 1)
    print('Green bands processed.')
    blue = GSA_sharpening(Source_RGB, Pan, 9, 17, 1)
    print('Blue bands processed.')

    combined = np.dstack((blue, green, red))

    import rasterio
    with rasterio.open(Pan) as src:
        transform = src.transform
        crs = src.crs
    data = combined
    data = np.transpose(data, (2, 0, 1))
    sharpened_path = out_prefix + '_sharpened.tif'
    with rasterio.open(sharpened_path, "w", driver="GTiff",
                       height=data.shape[1], width=data.shape[2],
                       count=data.shape[0], dtype=data.dtype,
                       crs=crs, transform=transform) as dst:
        dst.write(data)
    print(f'Sharpened image written to {sharpened_path}')
    # imageHR0 = ndimage.zoom(np.reshape(imageHR0[:,:],(3561,3825),order='F'),1/ratio1,prefilter=False)
        
    kwargs['path_out'] = out_prefix + '_coreg.tif'
    # Initialize COREG_LOCAL
    CRL = COREG_LOCAL(Source_google, sharpened_path, **kwargs)
    # Correct shifts and save output
    CRL.correct_shifts()
    print(f'Coregistered image written to {kwargs["path_out"]}')
    # from scipy import ndimage
    # imageHR0 = ndimage.zoom(imageHR0,1/ratio1,prefilter=False)
    main()
    ###   ESTIMATION ALPHA (replicate the function estimation alpha)
    
    I_MS = np.concatenate((imageLR_LP0,np.atleast_3d(np.ones(np.shape(HS_array[:,:,0])))),axis=2)  #np.atleast(B) converts B from 2 dimentional array to 3 dimentional array.
    I_PAN = imageHR0
    #type_estimation = 'global' #from the matlab fn implementation
    # lets repliceate the function estimation_alpha with type global
    IHc = np.reshape(I_PAN,(np.shape(I_PAN)[0]*np.shape(I_PAN)[1],1),order="F")
    ILRc = np.reshape(I_MS,(np.shape(I_MS)[0]*np.shape(I_MS)[1], np.shape(I_MS)[2]),order="F")
    alpha1= np.linalg.lstsq(ILRc,IHc)
    alpha = alpha1[0]
    alpha = np.reshape(alpha,(1,1,bands+1),order='F')
    I = np.sum(np.concatenate((imageLR0,np.atleast_3d(np.ones(np.shape(imageLR[:,:,0])))),axis=2)*np.tile(alpha,[np.shape(imageLR)[0],np.shape(imageLR)[1],1]),axis=2)
    
    ## Remove mean from I
    I0 = I - np.mean(I)    # np.mean(I)  gives a e-11 number, is it correct?????
    #I0 = I - mean(abs(I(:)));
    
    
    # Coefficient 
    g = np.ones((1, 1,No_bands+1))
    for i in range(0,No_bands):
        h = imageLR0[:,:,i]
        c = np.cov(np.transpose(np.reshape(np.transpose(I0),(-1,1))),np.transpose(np.reshape(np.transpose(h),(-1,1))))
        g[0,0,i+1] = c[0,1]/np.var(np.transpose(np.reshape(np.transpose(I0),-1)),ddof=1)
    print("g: ",np.shape(g))
    
    
    imageHR = imageHR - np.mean(imageHR)
    
    
    ##Detail extraction
    delta = imageHR - I0
    deltam = np.matlib.repmat(np.reshape(np.transpose(delta),(-1,1)),1,No_bands+1)
    
    
    ## Fusion 
    V = np.reshape(np.transpose(I0),(-1,1))
    for i in range (0,No_bands):
        h = imageLR0[:,:,i]
        V = np.concatenate((V,np.reshape(np.transpose(h),(-1,1))),axis=1)
    
    
    gm = np.zeros(np.shape(V))
    for i in range(0,np.shape(g)[2]):
        gm[:,i] = g[:,:,i]*np.ones((np.shape(imageLR)[0]*np.shape(imageLR)[1]))
    
    V_hat = V + deltam * gm
    
    ##Reshape fusi0on result
    I_Fus_GSA = np.reshape(V_hat[:,1:],(np.shape(imageLR)[0],np.shape(imageLR)[1],np.shape(imageLR)[2]),order="F")
    
    ## Final Mean Equalization
    for i in range (0,No_bands):
        h = I_Fus_GSA[:,:,i] 
        I_Fus_GSA[:,:,i] = h - np.mean(h) + np.mean(imageLR[:,:,i])
            
    I_Fus_GSA[imageLR==0]= 0
    
    return I_Fus_GSA


red = GSA_sharpening(Source_RGB,Pan,0,0,0)
print('Red bands processed.')
green = GSA_sharpening(Source_RGB,Pan,20,24,1)
print('Green bands processed.')
blue = GSA_sharpening(Source_RGB,Pan,9,17,1)
print('Blue bands processed.')

combined = np.dstack((blue, green, red))

import numpy as np
import rasterio
from rasterio.transform import from_origin

with rasterio.open(Pan) as src:
    transform = src.transform
    crs = src.crs
data = combined
data = np.transpose(data,(2,0,1))
with rasterio.open("GSA_sharpened_new.tif", "w", driver="GTiff",
                   height=data.shape[1], width=data.shape[2],
                   count=data.shape[0], dtype=data.dtype,
                   crs=crs, transform=transform) as dst:
    dst.write(data)


# #Converting array to raster
# sen_georaster = GeoTools.readGeoRaster(sen_img_path)
# georef_info = GeoTools.get_gdal_geotransform(sen_img_path)
# #array2georaster(combined,sen_georaster, 'Sharpened_GSA')
# sen_georaster.npdtype=np.float64
# array = combined[:][:][:] 
# array = array.astype('float64')
# array_georef = sen_georaster
# array_georef.nodata = np.float64(0)
# array_georef.array = array[:,:,:]
# GeoTools.writeGeoRaster(array_georef, "D:\Prisma_Test\Data_fusion_sharpening\GSA\Sharpened_prisma.tif")
#array2georaster1(imageHR,sen_georaster,'imageHR')    

# for coregistration after fusion 
import joblib
import os
from osgeo import ogr, osr, gdal
import os
#os.environ["PROJ_LIB"] = r"C:\Users\User\anaconda3\envs\coregister_env\Library\share\proj"
from arosics import COREG_LOCAL
# Parameters for co-registration
kwargs = {
    'grid_res': 20,
    'window_size': (200, 200),
    'q': False,     # Quiet mode
    'max_shift': 10, # Increased max shift
    'r_b4match': 1,       # Use band 2 (red band)
    's_b4match': 1,
    'max_iter': 20,  # Increase the number of iterations to 10
    'align_grids': True,  # Enable grid alignment
    'fmt_out': "GTiff",  # Specify GeoTIFF as output format
    'ignore_errors': True,
    'CPUs': 16
}

kwargs['path_out'] = os.path.splitext(Pan)[0] + "GSA_corg.tif"

# Initialize COREG_LOCAL
CRL = COREG_LOCAL(Source_google, "GSA_sharpened_new.tif", **kwargs)

# Correct shifts and save output
CRL.correct_shifts()



# for coregistration Before fusion 
RGB=r"F:\GMATICS\IZS\050248726010_01\050248726010_01_P001_MUL\21JUL06100720-M2AS-050248726010_01_P001.TIF"
Pan=r"F:\GMATICS\IZS\050248726010_01\050248726010_01_P001_PAN\21JUL06100720-P2AS-050248726010_01_P001.TIF"
google_data=r"F:\GMATICS\IZS\050248726010_01\Google_sat.tif"
import rasterio
from rasterio.enums import Resampling
import numpy as np

# --- Load the high-resolution Pan image ---
with rasterio.open(Pan) as pan_src:
    pan_data = pan_src.read(1)  # Read the first (or only) band
    pan_transform = pan_src.transform
    pan_crs = pan_src.crs
    pan_shape = pan_data.shape  # (rows, cols)

# --- Open the lower-resolution RGB image ---
with rasterio.open(RGB) as rgb_src:
    rgb_data_resampled = rgb_src.read(
        out_shape=(
            rgb_src.count,        # number of bands (e.g., 3 for RGB)
            pan_shape[0],         # new height (match PAN)
            pan_shape[1]          # new width (match PAN)
        ),
        resampling=Resampling.bilinear  # can also use Resampling.cubic
    )
    rgb_transform_resampled = rgb_src.transform
rgb_resampled = rgb_data_resampled.transpose(1, 2, 0)  # shape: (H, W, Bands)
# Now `RGB` is resampled to match `pan_data` size
# - pan_data
# - google_data

import numpy as np

# Ensure PAN has a third dimension to match RGB's format
pan_data_expanded = pan_data[:, :, np.newaxis]  # shape: (H, W, 1)

# Stack RGB and PAN along the last axis (bands)
rgb_pan_stacked = np.concatenate((rgb_resampled[:, :, :3], pan_data_expanded), axis=2)  # shape: (H, W, 4)


# to save the stack
import rasterio

# Get CRS and transform from reference (e.g., PAN image)
with rasterio.open(Pan) as src:
    crs = src.crs
    transform = src.transform

# Save to file
with rasterio.open(r"E:\GMATICS\IZS\050248726010_01\RGB_PAN_STACK.tif", "w", driver="GTiff",
                   height=rgb_pan_stacked.shape[0], width=rgb_pan_stacked.shape[1],
                   count=rgb_pan_stacked.shape[2], dtype=rgb_pan_stacked.dtype,
                   crs=crs, transform=transform) as dst:
    dst.write(rgb_pan_stacked.transpose(2, 0, 1))



import joblib
import os
from osgeo import ogr, osr, gdal
import os
#os.environ["PROJ_LIB"] = r"C:\Users\User\anaconda3\envs\coregister_env\Library\share\proj"
from arosics import COREG_LOCAL
# Parameters for co-registration
kwargs = {
    'grid_res': 20,
    'window_size': (200, 200),
    'q': False,     # Quiet mode
    'max_shift': 10, # Increased max shift
    'r_b4match': 1,       # Use band 2 (red band)
    's_b4match': 1,
    'max_iter': 20,  # Increase the number of iterations to 10
    'align_grids': True,  # Enable grid alignment
    'fmt_out': "GTiff",  # Specify GeoTIFF as output format
    'CPUs': 16,
    'ignore_errors': True
}

# Coregistrazione utilizzando la google satellite
kwargs = {
    'grid_res': 1, #♥5, #1.2,px res #50 meters #200 meters
    'window_size': (300, 300), #(256, 256),
    'max_shift': 15, #♠10
    'max_iter': 25, #20
    'resamp_alg_calc': 'nearest',
    'resamp_alg_deshift': 'nearest',
    'nodata': [0,0],
    'v': True,
    'fmt_out': 'GTIFF',
    'max_points': 10000, #3000
    'align_grids': True, # False,
    'CPUs': 16,
    'r_b4match': 1, #1 red 2 green 3 blue
    's_b4match': 1, #5red  3 green 2 blue
}


kwargs['path_out'] = os.path.splitext(Pan)[0] + "_corg_simpleresample.tif"

# Initialize COREG_LOCAL
CRL = COREG_LOCAL(google_data, r"F:\GMATICS\IZS\050248726010_01\RGB_PAN_STACK.tif", **kwargs)

# Correct shifts and save output
CRL.correct_shifts()




