#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 18:55:11 2025

Modified by: [Your Name]
Description: Process one or several Level-2 PRISMA products contained in ZIP files.
"""

import os
import zipfile
import h5py
import numpy as np
from osgeo import gdal, osr
import subprocess
import argparse
import sys



# ---------------------------
# Level-2 PRISMA Reading Functions with Band Removal
# ---------------------------
def prismaL2_read(filename):
    """
    Reads a Level-2 PRISMA product (reflectance) from an HDF5 file.
    ...
    (Function code remains unchanged)
    """
    with h5py.File(filename, 'r') as f:
        # Read reflectance cubes from HCO product
        vnir_cube = f['HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/VNIR_Cube'][:]
        swir_cube = f['HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/SWIR_Cube'][:]
        # Read PCO cube
        pco_cube = f['HDFEOS/SWATHS/PRS_L2D_PCO/Data Fields/Cube'][:]
        
        # Read geolocation fields
        hco_lat = f['HDFEOS/SWATHS/PRS_L2D_HCO/Geolocation Fields/Latitude'][:]
        hco_lon = f['HDFEOS/SWATHS/PRS_L2D_HCO/Geolocation Fields/Longitude'][:]
        pco_lat = f['HDFEOS/SWATHS/PRS_L2D_PCO/Geolocation Fields/Latitude'][:]
        pco_lon = f['HDFEOS/SWATHS/PRS_L2D_PCO/Geolocation Fields/Longitude'][:]
        
        # Read CW arrays from KDP_AUX (same as Level-1)
        cw_vnir = f['KDP_AUX/Cw_Vnir_Matrix'][:]
        cw_swir = f['KDP_AUX/Cw_Swir_Matrix'][:]
        
        # Read Reflectance conversion factors
        L2ScaleVnirMin = f.attrs['L2ScaleVnirMin']
        L2ScaleVnirMax = f.attrs['L2ScaleVnirMax']
        L2ScaleSwirMin = f.attrs['L2ScaleSwirMin']
        L2ScaleSwirMax = f.attrs['L2ScaleSwirMax']
        L2ScalePanMin = f.attrs['L2ScalePanMin']
        L2ScalePanMax = f.attrs['L2ScalePanMax']

    # --- DN → riflettanza -------------------------------------------------
    vnir_cube = L2ScaleVnirMin + (vnir_cube.astype(np.float32) *
                  (L2ScaleVnirMax - L2ScaleVnirMin) / 65535.0)
    
    swir_cube = L2ScaleSwirMin + (swir_cube.astype(np.float32) *
                  (L2ScaleSwirMax - L2ScaleSwirMin) / 65535.0)
    
    pco_cube  = L2ScalePanMin  + (pco_cube.astype(np.float32)  *
                  (L2ScalePanMax  - L2ScalePanMin ) / 65535.0)

    
    # Reorder the cubes from (rows, bands, cols) to (rows, cols, bands)
    vnir_cube_bip = np.transpose(vnir_cube, (0, 2, 1))
    swir_cube_bip = np.transpose(swir_cube, (0, 2, 1))
    
    # Rotate 270° counterclockwise (i.e. 90° clockwise)
    vnir_cube_bip = np.rot90(vnir_cube_bip, k=-1, axes=(0, 1))
    swir_cube_bip = np.rot90(swir_cube_bip, k=-1, axes=(0, 1))
    
    # Remove unwanted bands as in L1:
    VNIR_cube_clean = np.delete(vnir_cube_bip, [0, 1, 2], axis=2)
    SWIR_cube_clean = np.delete(swir_cube_bip, [171, 172], axis=2)
    SWIR_cube_clean = np.delete(SWIR_cube_clean, [0, 1, 2, 3], axis=2)
    
    # Concatenate VNIR and SWIR cubes along the band axis:
    concatenated_cube = np.concatenate((SWIR_cube_clean, VNIR_cube_clean), axis=2)
    concatenated_cube = concatenated_cube[:, :, ::-1]  # Reverse band order if desired.
    
    # Process the CW arrays similarly:
    cw_vnir = cw_vnir[:, 99:162][:, ::-1]
    cw_swir = cw_swir[:, 81:252][:, ::-1]
    cw_swir_clean = np.delete(cw_swir, [0, 1, 2, 3], axis=1)
    concatenated_cw = np.concatenate((cw_vnir, cw_swir_clean), axis=1)
    
    # Rotate the panchromatic (PCO) cube.
    pco_cube = np.rot90(pco_cube, k=-1, axes=(0, 1))
    
    return concatenated_cube, concatenated_cw, hco_lat, hco_lon, pco_cube, pco_lat, pco_lon





def extract_he5_from_zip(zip_path, extract_to=None):
    """
    Extracts the first .he5 file found in the ZIP archive.
    """
    if extract_to is None:
        extract_to = os.path.dirname(zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file in zip_ref.namelist():
            if file.endswith('.he5'):
                zip_ref.extract(file, extract_to)
                return os.path.join(extract_to, file)
    raise FileNotFoundError("No .he5 file found in the ZIP archive.")

def prismaL2_read_from_zip(zip_path, extract_to=None):
    """
    Extracts a Level-2 PRISMA .he5 file from a ZIP archive and reads its data.
    """
    he5_file = extract_he5_from_zip(zip_path, extract_to)
    result = prismaL2_read(he5_file)
    # Optionally, remove the extracted file after processing:
    # os.remove(he5_file)
    return result

# ---------------------------
# GeoTIFF Saving Functions
# ---------------------------
def save_as_geotiff_multichannel(data, output_file, latitude_vnir, longitude_vnir, channel_wavelengths=None):
    """
    Saves a multichannel array as a GeoTIFF with EPSG:4326 projection.
    """
    data_float32 = data.astype(np.float32)
    rotated = np.rot90(data_float32, k=1, axes=(0, 1))
    height, width, bands = rotated.shape
    
    temp_file = 'temp_output.tif'
    vrt_file = 'temp_output.vrt'
    lat_file = 'latitude.tif'
    lon_file = 'longitude.tif'
    
    driver = gdal.GetDriverByName('GTiff')
    lat_ds = driver.Create(lat_file, latitude_vnir.shape[1], latitude_vnir.shape[0], 1, gdal.GDT_Float32)
    lon_ds = driver.Create(lon_file, longitude_vnir.shape[1], longitude_vnir.shape[0], 1, gdal.GDT_Float32)
    lat_ds.GetRasterBand(1).WriteArray(latitude_vnir.astype(np.float32))
    lon_ds.GetRasterBand(1).WriteArray(longitude_vnir.astype(np.float32))
    lat_ds = None
    lon_ds = None
    
    dataset = driver.Create(temp_file, width, height, bands, gdal.GDT_Float32)
    for i in range(bands):
        band = dataset.GetRasterBand(i+1)
        band.WriteArray(rotated[:, :, i])
        if channel_wavelengths is not None and i < len(channel_wavelengths):
            band.SetDescription(f'Wavelength: {channel_wavelengths[i]:.2f} nm')
            band.SetMetadataItem('WAVELENGTH', f'{channel_wavelengths[i]:.2f}')
    dataset.FlushCache()
    dataset = None
    
    vrt_options = gdal.TranslateOptions(format='VRT')
    gdal.Translate(vrt_file, temp_file, options=vrt_options)
    
    vrt_ds = gdal.Open(vrt_file, gdal.GA_Update)
    vrt_ds.SetMetadata({
        'X_DATASET': lon_file,
        'X_BAND': '1',
        'Y_DATASET': lat_file,
        'Y_BAND': '1',
        'PIXEL_OFFSET': '0',
        'LINE_OFFSET': '0',
        'PIXEL_STEP': '1',
        'LINE_STEP': '1'
    }, 'GEOLOCATION')
    
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    vrt_ds.SetProjection(srs.ExportToWkt())
    
    description = ("Product generated by Alvise Ferrari for School of Aerospace Engineering, "
                   "La Sapienza, under CLEAR-UP license. No liability for improper use.")
    vrt_ds.SetMetadataItem('DESCRIPTION', description)
    vrt_ds = None
    
    subprocess.run([
        'gdalwarp',
        '-geoloc',
        '-t_srs', 'EPSG:4326',
        vrt_file,
        output_file
    ], check=True)
    
    os.remove(temp_file)
    os.remove(vrt_file)
    os.remove(lat_file)
    os.remove(lon_file)

def save_as_geotiff_single_band(data, output_file, latitude_vnir, longitude_vnir):
    """
    Saves a single-band array as a GeoTIFF with EPSG:4326 projection.
    """
    data_float32 = data.astype(np.float32)
    rotated = np.rot90(data_float32, k=1, axes=(0, 1))
    height, width = rotated.shape  # rotated now is 2D
    
    temp_file = 'temp_output.tif'
    vrt_file = 'temp_output.vrt'
    lat_file = 'latitude.tif'
    lon_file = 'longitude.tif'
    
    driver = gdal.GetDriverByName('GTiff')
    lat_ds = driver.Create(lat_file, latitude_vnir.shape[1], latitude_vnir.shape[0], 1, gdal.GDT_Float32)
    lon_ds = driver.Create(lon_file, longitude_vnir.shape[1], longitude_vnir.shape[0], 1, gdal.GDT_Float32)
    lat_ds.GetRasterBand(1).WriteArray(latitude_vnir.astype(np.float32))
    lon_ds.GetRasterBand(1).WriteArray(longitude_vnir.astype(np.float32))
    lat_ds = None
    lon_ds = None
    
    dataset = driver.Create(temp_file, width, height, 1, gdal.GDT_Float32)
    band = dataset.GetRasterBand(1)
    band.WriteArray(rotated)
    dataset.FlushCache()
    dataset = None
    
    vrt_options = gdal.TranslateOptions(format='VRT')
    gdal.Translate(vrt_file, temp_file, options=vrt_options)
    
    vrt_ds = gdal.Open(vrt_file, gdal.GA_Update)
    vrt_ds.SetMetadata({
        'X_DATASET': lon_file,
        'X_BAND': '1',
        'Y_DATASET': lat_file,
        'Y_BAND': '1',
        'PIXEL_OFFSET': '0',
        'LINE_OFFSET': '0',
        'PIXEL_STEP': '1',
        'LINE_STEP': '1'
    }, 'GEOLOCATION')
    
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    vrt_ds.SetProjection(srs.ExportToWkt())
    
    description = ("Product generated by Alvise Ferrari for School of Aerospace Engineering, "
                   "La Sapienza, under CLEAR-UP license. No liability for improper use.")
    vrt_ds.SetMetadataItem('DESCRIPTION', description)
    vrt_ds = None
    
    subprocess.run([
        'gdalwarp',
        '-geoloc',
        '-t_srs', 'EPSG:4326',
        vrt_file,
        output_file
    ], check=True)
    
    os.remove(temp_file)
    os.remove(vrt_file)
    os.remove(lat_file)
    os.remove(lon_file)

# ---------------------------
# Processing Function
# ---------------------------
def process_prisma_level2(input_zip, output_dir):
    """
    Runs the complete processing for a Level-2 PRISMA product.
    
    Parameters:
      input_zip: Path to the input Level-2 PRISMA ZIP file.
      output_dir: Directory where the output GeoTIFF files will be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Read the Level-2 data from the ZIP file.
    (concatenated_cube, concatenated_cw, hco_lat, hco_lon, 
     pco_cube, pco_lat, pco_lon) = prismaL2_read_from_zip(input_zip)
    
    # Create a channel-to-wavelength mapping using the central row of the concatenated CW array.
    central_row = concatenated_cw[concatenated_cw.shape[0] // 2, :]
    channel_names = [f"Channel_{i+1}" for i in range(central_row.size)]
    channel_wavelength_mapping = dict(zip(channel_names, central_row))
    channel_wavelengths = [channel_wavelength_mapping[f"Channel_{i+1}"] for i in range(len(channel_names))]
    
    # Build a base name from the input zip file.
    base_name = os.path.splitext(os.path.basename(input_zip))[0]
    
    # Define output filenames.
    reflectance_out = os.path.join(output_dir, f"{base_name}_reflectance.tif")
    pco_out = os.path.join(output_dir, f"{base_name}_pco.tif")
    
    print("Saving reflectance data to:", reflectance_out)
    save_as_geotiff_multichannel(concatenated_cube, reflectance_out, hco_lat, hco_lon, channel_wavelengths=channel_wavelengths)
    
    print("Saving panchromatic data to:", pco_out)
    save_as_geotiff_single_band(pco_cube, pco_out, pco_lat, pco_lon)
    
    print(f"Processing of {input_zip} complete.")

# ---------------------------
# Main Block for Command-Line and Editor Execution
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process one or several Level-2 PRISMA products and generate GeoTIFF outputs."
    )
    parser.add_argument("input_path", nargs="?", default=None,
                        help="Path to the input Level-2 PRISMA ZIP file or directory containing ZIP files")
    parser.add_argument("output_dir", nargs="?", default=None,
                        help="Output directory for the GeoTIFF files")
    args = parser.parse_args()
    
    # Use default paths if no command-line arguments are provided.
    if args.input_path is None or args.output_dir is None:
        print("No command-line arguments provided. Using default values.")
        # Replace these default paths with ones valid on your system:
        #default_input_path = r"D:\Lavoro\GMATICS\IRIDE_ForestMonitoring\phase2\IMMAGINI_PRISMA"
        #default_output_dir = r"D:\Lavoro\GMATICS\IRIDE_ForestMonitoring\phase2\IMMAGINI_PRISMA\output"
        default_input_path = r"D:\Lavoro\GMATICS\IRIDE_ForestMonitoring\phase2\test_prisma_read_img"
        default_output_dir = r"D:\Lavoro\GMATICS\IRIDE_ForestMonitoring\phase2\test_prisma_read_img\output"
        input_path = default_input_path
        output_dir = default_output_dir
    else:
        input_path = args.input_path
        output_dir = args.output_dir
    
    # Check if the input path is a directory or a single file
    if os.path.isdir(input_path):
        # Loop over all ZIP files in the directory.
        for file in os.listdir(input_path):
            if file.lower().endswith('.zip'):
                zip_file_path = os.path.join(input_path, file)
                print("Processing file:", zip_file_path)
                process_prisma_level2(zip_file_path, output_dir)
    else:
        process_prisma_level2(input_path, output_dir)
