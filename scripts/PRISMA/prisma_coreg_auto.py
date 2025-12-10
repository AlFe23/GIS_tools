# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 12:47:27 2025

@author: ferra
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 18:55:11 2025

Modified to also download and mosaic Sentinel-2 products for coregistration.

@author: ferra
"""

import os
import zipfile
import h5py
import numpy as np
from osgeo import gdal, osr
import subprocess
import argparse
import sys
import requests
import json
import glob
import shutil
from arosics import COREG_LOCAL

# ---------------------------
# Level-2 PRISMA Reading Functions with Band Removal
# ---------------------------
def prismaL2_read(filename):
    """
    Reads a Level-2 PRISMA product (reflectance) from an HDF5 file.
    
    Expected fields (note the space in "Data Fields"):
      - HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/VNIR_Cube
      - HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/SWIR_Cube
      - HDFEOS/SWATHS/PRS_L2D_PCO/Data Fields/Cube
      - HDFEOS/SWATHS/PRS_L2D_HCO/Geolocation Fields/Latitude
      - HDFEOS/SWATHS/PRS_L2D_HCO/Geolocation Fields/Longitude
      - HDFEOS/SWATHS/PRS_L2D_PCO/Geolocation Fields/Latitude
      - HDFEOS/SWATHS/PRS_L2D_PCO/Geolocation Fields/Longitude
      
    The reflectance values are already in physical units.
    Band removal is performed as for Level-1:
      - VNIR: remove bands [0,1,2]
      - SWIR: remove bands [171, 172] then remove the first 4 bands.
      
    Additionally, the panchromatic (PCO) cube is rotated by 90° clockwise.
    
    Returns:
      - concatenated_cube: Combined reflectance cube from HCO (VNIR and SWIR) after reordering.
      - concatenated_cw: Concatenated CW array for VNIR and SWIR.
      - hco_lat, hco_lon: Latitude and longitude arrays from the HCO geolocation fields.
      - pco_cube: The rotated panchromatic cube from the PCO product.
      - pco_lat, pco_lon: Latitude and longitude arrays from the PCO geolocation fields.
    """
    with h5py.File(filename, 'r') as f:
        vnir_cube = f['HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/VNIR_Cube'][:]
        swir_cube = f['HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/SWIR_Cube'][:]
        pco_cube = f['HDFEOS/SWATHS/PRS_L2D_PCO/Data Fields/Cube'][:]
        hco_lat = f['HDFEOS/SWATHS/PRS_L2D_HCO/Geolocation Fields/Latitude'][:]
        hco_lon = f['HDFEOS/SWATHS/PRS_L2D_HCO/Geolocation Fields/Longitude'][:]
        pco_lat = f['HDFEOS/SWATHS/PRS_L2D_PCO/Geolocation Fields/Latitude'][:]
        pco_lon = f['HDFEOS/SWATHS/PRS_L2D_PCO/Geolocation Fields/Longitude'][:]
        cw_vnir = f['KDP_AUX/Cw_Vnir_Matrix'][:]
        cw_swir = f['KDP_AUX/Cw_Swir_Matrix'][:]
    
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

DEFAULT_COREG_KWARGS = {
    'grid_res': 50,
    'window_size': (256, 256),
    'q': False,
    'max_shift': 20,
    'r_b4match': 3,
    's_b4match': 21,
    'max_iter': 10,
    'align_grids': True,
    'fmt_out': "GTiff",
    'ignore_errors': True
}

def reproject_if_needed(reference_path, target_path, work_dir):
    """Ensure target uses same CRS as reference."""
    ref_ds = gdal.Open(reference_path)
    target_ds = gdal.Open(target_path)
    if ref_ds is None or target_ds is None:
        raise RuntimeError("Impossibile aprire reference o target per la coregistrazione.")

    ref_proj = ref_ds.GetProjection()
    target_proj = target_ds.GetProjection()
    ref_ds = None
    target_ds = None

    ref_sr = osr.SpatialReference()
    ref_sr.ImportFromWkt(ref_proj)
    target_sr = osr.SpatialReference()
    target_sr.ImportFromWkt(target_proj)

    if ref_sr.IsSame(target_sr):
        return target_path

    os.makedirs(work_dir, exist_ok=True)
    reprojected_path = os.path.join(work_dir, f"reproj_{os.path.basename(target_path)}")
    print(f"Riproiezione di {os.path.basename(target_path)} verso il CRS della Sentinel ...")
    gdal.Warp(reprojected_path, target_path, dstSRS=ref_sr.ExportToWkt())
    return reprojected_path

def coregister_prisma_to_sentinel(reference_image_path, prisma_image_path, output_dir, extra_kwargs=None):
    """Coregistra il GeoTIFF PRISMA sul mosaico Sentinel usando AROSICS."""
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(prisma_image_path))[0]
    final_output_path = os.path.join(output_dir, f"{base_name}_coregistered.tif")

    kwargs = DEFAULT_COREG_KWARGS.copy()
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    kwargs['projectDir'] = output_dir
    kwargs['path_out'] = final_output_path

    updated_target = reproject_if_needed(reference_image_path, prisma_image_path, output_dir)

    CRL = COREG_LOCAL(reference_image_path, updated_target, **kwargs)
    CRL.correct_shifts()

    if updated_target != prisma_image_path and os.path.exists(updated_target):
        try:
            os.remove(updated_target)
        except OSError:
            pass
    print(f"Coregistrazione completata: {final_output_path}")
    return final_output_path

# ---------------------------
# GeoTIFF Saving Functions for PRISMA
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
# Processing Function for PRISMA
# ---------------------------
def process_prisma_level2(input_zip, output_dir):
    """
    Runs the complete processing for a Level-2 PRISMA product.
    
    Parameters:
      input_zip: Path to the input Level-2 PRISMA ZIP file.
      output_dir: Directory where the output GeoTIFF files will be saved.
      
    Returns the HCO latitude and longitude arrays (used later for AOI computation).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    (concatenated_cube, concatenated_cw, hco_lat, hco_lon, 
     pco_cube, pco_lat, pco_lon) = prismaL2_read_from_zip(input_zip)
    
    # Create a channel-to-wavelength mapping using the central row of the concatenated CW array.
    central_row = concatenated_cw[concatenated_cw.shape[0] // 2, :]
    channel_names = [f"Channel_{i+1}" for i in range(central_row.size)]
    channel_wavelength_mapping = dict(zip(channel_names, central_row))
    channel_wavelengths = [channel_wavelength_mapping[f"Channel_{i+1}"] for i in range(len(channel_names))]
    
    base_name = os.path.splitext(os.path.basename(input_zip))[0]
    reflectance_out = os.path.join(output_dir, f"{base_name}_reflectance.tif")
    pco_out = os.path.join(output_dir, f"{base_name}_pco.tif")
    
    print("Saving reflectance data to:", reflectance_out)
    save_as_geotiff_multichannel(concatenated_cube, reflectance_out, hco_lat, hco_lon, channel_wavelengths=channel_wavelengths)
    
    print("Saving panchromatic data to:", pco_out)
    save_as_geotiff_single_band(pco_cube, pco_out, pco_lat, pco_lon)
    
    print("PRISMA processing complete.")
    return reflectance_out, pco_out, hco_lat, hco_lon

# ---------------------------
# Sentinel-2 Download and Mosaic Functions
# ---------------------------
def compute_aoi_wkt(hco_lat, hco_lon):
    lat_min, lat_max = float(np.min(hco_lat)), float(np.max(hco_lat))
    lon_min, lon_max = float(np.min(hco_lon)), float(np.max(hco_lon))
    return f"POLYGON(({lon_min} {lat_min}, {lon_min} {lat_max}, {lon_max} {lat_max}, {lon_max} {lat_min}, {lon_min} {lat_min}))"

def download_s2_from_cdse(aoi_wkt, start_date, end_date, max_cloud=30, user="", password="", output_dir=""):
    """
    Searches for and downloads a Sentinel-2 L2A product from the Copernicus Data Space
    that intersects the given AOI (WKT) and lies within the specified time window.
    
    Returns the path to the downloaded ZIP file.
    """
    catalogue_url = "https://catalogue.dataspace.copernicus.eu/odata/v1"
    collection = "SENTINEL-2"
    product_type = "S2MSI2A"
    search_query = (
        f"{catalogue_url}/Products?$filter=Collection/Name eq '{collection}' "
        f"and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and "
        f"att/OData.CSC.StringAttribute/Value eq '{product_type}') and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;{aoi_wkt}') and "
        f"ContentDate/Start gt {start_date} and ContentDate/Start lt {end_date} and "
        f"Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and "
        f"att/OData.CSC.DoubleAttribute/Value le {max_cloud})&$top=400&$expand=Attributes"
    )
    response = requests.get(search_query)
    response.raise_for_status()
    results = response.json()["value"]
    if not results:
        raise RuntimeError("No Sentinel-2 products found for the given query.")

    # For simplicity, pick the product with the lowest cloud cover.
    def get_cloud_cover(attrs):
        for a in attrs:
            if a["Name"] == "cloudCover":
                return float(a["Value"])
        return 1000.0

    results.sort(key=lambda r: get_cloud_cover(r["Attributes"]))
    chosen = results[0]
    product_id = chosen["Id"]
    product_name = chosen["Name"]
    print(f"Chosen Sentinel-2 product: {product_name} (cloudCover={get_cloud_cover(chosen['Attributes'])})")

    # Authenticate with CDSE
    auth_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    auth_data = {
        "client_id": "cdse-public",
        "grant_type": "password",
        "username": "alvise.ferrari@uniroma1.it",
        "password": "GR4wQJ62zduP_D",
    }
    auth_resp = requests.post(auth_url, data=auth_data, verify=True, allow_redirects=False)
    if auth_resp.status_code != 200:
        raise RuntimeError(f"Authentication failed: {auth_resp.text}")
    access_token = json.loads(auth_resp.text)["access_token"]

    # Get the download URL
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {access_token}"})
    url_value = f"{catalogue_url}/Products({product_id})/$value"
    init_resp = session.get(url_value, allow_redirects=False)
    if "Location" not in init_resp.headers:
        raise RuntimeError("Download URL not found.")
    download_url = init_resp.headers["Location"]

    # Download the ZIP file
    os.makedirs(output_dir, exist_ok=True)
    local_zip = os.path.join(output_dir, f"{product_name}.zip")
    print(f"Downloading Sentinel-2 product to {local_zip} ...")
    with session.get(download_url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("Content-Length", 0))
        chunk_size = 2_000_000
        downloaded = 0
        with open(local_zip, "wb") as f_out:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f_out.write(chunk)
                    downloaded += len(chunk)
                    pct = 100.0 * downloaded / total_size
                    print(f"Downloaded: {pct:0.2f}%", end="\r")
    print("\nDownload complete.")
    return local_zip

def s2_mosaic(src_safe_dir, out_tif, epsg="EPSG:32632"):
    """
    Create un mosaico Sentinel partendo da una cartella che contiene
    uno o più prodotti .SAFE oppure da una singola cartella .SAFE.
    Se è presente un solo prodotto, viene comunque scritto il GeoTIFF di output.
    """
    bands = ["B02", "B03", "B04", "B08"]
    vrts = []
    if os.path.isdir(src_safe_dir) and src_safe_dir.endswith(".SAFE"):
        safe_dirs = [src_safe_dir]
    else:
        safe_dirs = glob.glob(os.path.join(src_safe_dir, "*.SAFE"))

    if not safe_dirs:
        raise RuntimeError(f"Nessuna cartella .SAFE trovata in {src_safe_dir}")

    for safe in safe_dirs:
        granules = glob.glob(os.path.join(safe, "GRANULE", "*", "IMG_DATA"))
        if not granules:
            continue
        img_dir = granules[0]
        band_files = []
        for b in bands:
            pattern = os.path.join(img_dir, "R10m", f"*_{b}_*.jp2")
            files = glob.glob(pattern)
            if files:
                band_files.append(files[0])
        if band_files:
            vrt_name = os.path.join(src_safe_dir, os.path.basename(safe) + ".vrt")
            gdal.BuildVRT(vrt_name, band_files, separate=True)
            vrts.append(vrt_name)
    if not vrts:
        raise RuntimeError("Nessuna banda Sentinel trovata per il mosaico.")
    warp_opts = gdal.WarpOptions(format="GTiff", srcNodata=0, dstNodata=0,
                                 options=["COMPRESS=LZW", "TILED=YES"],
                                 dstSRS=epsg, xRes=10, yRes=10)
    gdal.Warp(out_tif, vrts, options=warp_opts)
    # Clean up VRT files
    for vrt in vrts:
        os.remove(vrt)
    print(f"Mosaic created: {out_tif}")

def prepare_sentinel_mosaic(hco_lat, hco_lon, prisma_output_dir, args):
    """
    Gestisce il download (opzionale) o l'uso di SAFE locali e crea il mosaico Sentinel.
    """
    sentinel_safe_dir = args.sentinel_safe_dir
    sentinel_output_dir = args.s2_output_dir or (sentinel_safe_dir if sentinel_safe_dir else os.path.join(prisma_output_dir, "Sentinel"))
    os.makedirs(sentinel_output_dir, exist_ok=True)

    if sentinel_safe_dir:
        if not os.path.isdir(sentinel_safe_dir):
            raise RuntimeError(f"La cartella specificata non esiste: {sentinel_safe_dir}")
        print(f"Uso delle cartelle SAFE già presenti in {sentinel_safe_dir}")
    else:
        if not args.download_sentinel:
            raise RuntimeError("Occorre specificare --download-sentinel oppure --sentinel-safe-dir.")
        aoi_wkt = compute_aoi_wkt(hco_lat, hco_lon)
        print("AOI derivata dal PRISMA:", aoi_wkt)
        s2_zip = download_s2_from_cdse(aoi_wkt,
                                       args.s2_start_date,
                                       args.s2_end_date,
                                       max_cloud=args.s2_max_cloud,
                                       user=args.s2_user,
                                       password=args.s2_password,
                                       output_dir=sentinel_output_dir)
        sentinel_safe_dir = os.path.join(sentinel_output_dir, "extracted_SAFE")
        os.makedirs(sentinel_safe_dir, exist_ok=True)
        with zipfile.ZipFile(s2_zip, 'r') as zip_ref:
            zip_ref.extractall(sentinel_safe_dir)
        print(f"Prodotto Sentinel estratto in {sentinel_safe_dir}")

    sentinel_mosaic_out = args.s2_mosaic_path or os.path.join(sentinel_output_dir, "sentinel_mosaic.tif")
    s2_mosaic(sentinel_safe_dir, sentinel_mosaic_out)
    return sentinel_mosaic_out

# ---------------------------
# Combined Processing Function for PRISMA and Sentinel-2
# ---------------------------
def process_prisma_and_sentinel(prisma_zip, prisma_output_dir, s2_output_dir, s2_start_date, s2_end_date, s2_max_cloud, s2_user, s2_password):
    """
    Processes a PRISMA product and automatically downloads and mosaics a Sentinel-2 product
    for coregistration.
    """
    reflectance_path, _, hco_lat, hco_lon = process_prisma_level2(prisma_zip, prisma_output_dir)

    sentinel_args = argparse.Namespace(
        download_sentinel=True,
        sentinel_safe_dir=None,
        s2_start_date=s2_start_date,
        s2_end_date=s2_end_date,
        s2_max_cloud=s2_max_cloud,
        s2_user=s2_user,
        s2_password=s2_password,
        s2_output_dir=s2_output_dir,
        s2_mosaic_path=os.path.join(s2_output_dir, "sentinel_mosaic.tif")
    )
    sentinel_mosaic = prepare_sentinel_mosaic(hco_lat, hco_lon, prisma_output_dir, sentinel_args)
    coregister_prisma_to_sentinel(sentinel_mosaic, reflectance_path, prisma_output_dir)

    print("Sentinel-2 processing and coregistration complete.")

# ---------------------------
# Main Block for Command-Line and Editor Execution
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a Level-2 PRISMA product and optionally download and mosaic Sentinel-2 images for coregistration."
    )
    parser.add_argument("input_zip", nargs="?", default=None,
                        help="Path to the input Level-2 PRISMA ZIP file")
    parser.add_argument("output_dir", nargs="?", default=None,
                        help="Output directory for the PRISMA GeoTIFF files")
    parser.add_argument("--download-sentinel", action="store_true",
                        help="Download e mosaicare automaticamente Sentinel-2 per la coregistrazione")
    parser.add_argument("--sentinel-safe-dir", type=str, default=None,
                        help="Cartella contenente uno o più .SAFE già scaricati (alternativa al download)")
    parser.add_argument("--s2_start_date", type=str, default="2025-03-01",
                        help="Start date for Sentinel-2 search (YYYY-MM-DD)")
    parser.add_argument("--s2_end_date", type=str, default="2025-03-31",
                        help="End date for Sentinel-2 search (YYYY-MM-DD)")
    parser.add_argument("--s2_max_cloud", type=float, default=30,
                        help="Maximum cloud cover for Sentinel-2 products")
    parser.add_argument("--s2_user", type=str, default="",
                        help="Username for Sentinel-2 download authentication")
    parser.add_argument("--s2_password", type=str, default="",
                        help="Password for Sentinel-2 download authentication")
    parser.add_argument("--s2_output_dir", type=str, default=None,
                        help="Output directory for Sentinel-2 products")
    parser.add_argument("--s2_mosaic_path", type=str, default=None,
                        help="Percorso esplicito per il mosaico Sentinel risultante")
    parser.add_argument("--coreg-output-dir", type=str, default=None,
                        help="Cartella in cui salvare i PRISMA coregistrati (default: output_dir)")
    args = parser.parse_args()
    
    # Use default values if not provided.
    if args.input_zip is None or args.output_dir is None:
        print("No command-line arguments provided. Using default values for PRISMA processing.")
        # Replace these paths with ones valid on your system.
        default_input_zip = r"D:\Lavoro\GMATICS\IRIDE_ForestMonitoring\phase2\IMMAGINI_PRISMA\PRS_L2D_STD_20190609100304_20190609100308_0001.zip"
        default_output_dir = r"D:\Lavoro\GMATICS\IRIDE_ForestMonitoring\phase2\IMMAGINI_PRISMA"
        input_zip = default_input_zip
        output_dir = default_output_dir
    else:
        input_zip = args.input_zip
        output_dir = args.output_dir

    # Process the PRISMA product.
    reflectance_path, _, hco_lat, hco_lon = process_prisma_level2(input_zip, output_dir)
    
    sentinel_mosaic_path = None
    if args.download_sentinel or args.sentinel_safe_dir:
        try:
            sentinel_mosaic_path = prepare_sentinel_mosaic(hco_lat, hco_lon, output_dir, args)
        except RuntimeError as err:
            print(f"Errore durante la preparazione della Sentinel: {err}")
    
    if sentinel_mosaic_path:
        coreg_dir = args.coreg_output_dir or output_dir
        coregister_prisma_to_sentinel(sentinel_mosaic_path, reflectance_path, coreg_dir)
    else:
        print("Nessuna Sentinel fornita: coregistrazione non eseguita.")
    
    print("All processing complete.")


#python prisma_l2d_reader.py r"D:\Lavoro\GMATICS\IRIDE_ForestMonitoring\phase2\IMMAGINI_PRISMA\PRS_L2D_STD_20190609100304_20190609100308_0001.zip" r"D:\Lavoro\GMATICS\IRIDE_ForestMonitoring\phase2\Output\IMMAGINI_PRISMA" --download-sentinel --s2_user alvise.ferrari@uniroma1.it --s2_password GR4wQJ62zduP_D --s2_output_dir r"D:\Lavoro\GMATICS\IRIDE_ForestMonitoring\phase2\Output\IMMAGINI_PRISMA"
