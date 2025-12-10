import os
import zipfile
import h5py
import numpy as np
from osgeo import gdal, osr
import subprocess
import argparse
import sys

def prisma_read(filename):
    """
    Reads PRISMA L1 data from an HDF5 file.
    
    Returns:
      - concatenated_cube: combined radiance cube (rows x cols x bands)
      - concatenated_cw: concatenated array of central wavelengths (CW)
      - concatenated_fwhm: concatenated array of FWHM values
      - rgb_image: RGB image derived from selected bands
      - vnir_cube_bip: VNIR data in BIP format
      - swir_cube_bip: SWIR data in BIP format
      - latitude_vnir, longitude_vnir: geolocation arrays for VNIR
      - latitude_swir, longitude_swir: geolocation arrays for SWIR
    """
    with h5py.File(filename, 'r') as f:
        vnir_cube_DN = f['HDFEOS/SWATHS/PRS_L1_HCO/Data Fields/VNIR_Cube'][:]
        swir_cube_DN = f['HDFEOS/SWATHS/PRS_L1_HCO/Data Fields/SWIR_Cube'][:]
    
        cw_vnir = f['KDP_AUX/Cw_Vnir_Matrix'][:]
        fwhm_vnir = f['KDP_AUX/Fwhm_Vnir_Matrix'][:]
        cw_swir = f['KDP_AUX/Cw_Swir_Matrix'][:]
        fwhm_swir = f['KDP_AUX/Fwhm_Swir_Matrix'][:]
    
        offset_swir = f.attrs['Offset_Swir']
        scaleFactor_swir = f.attrs['ScaleFactor_Swir']
        offset_vnir = f.attrs['Offset_Vnir']
        scaleFactor_vnir = f.attrs['ScaleFactor_Vnir']
        
        latitude_vnir = f['HDFEOS/SWATHS/PRS_L1_HCO/Geolocation Fields/Latitude_VNIR'][:]
        longitude_vnir = f['HDFEOS/SWATHS/PRS_L1_HCO/Geolocation Fields/Longitude_VNIR'][:]
        latitude_swir = f['HDFEOS/SWATHS/PRS_L1_HCO/Geolocation Fields/Latitude_SWIR'][:]
        longitude_swir = f['HDFEOS/SWATHS/PRS_L1_HCO/Geolocation Fields/Longitude_SWIR'][:]
    
    # Convert DN to radiance and adjust units
    swir_cube_rads = (swir_cube_DN / scaleFactor_swir) - offset_swir
    vnir_cube_rads = (vnir_cube_DN / scaleFactor_vnir) - offset_vnir
    swir_cube_rads = swir_cube_rads * 0.1
    vnir_cube_rads = vnir_cube_rads * 0.1
    
    # Convert from BIL to BIP (transpose axes)
    vnir_cube_bip = np.transpose(vnir_cube_rads, (0, 2, 1))
    swir_cube_bip = np.transpose(swir_cube_rads, (0, 2, 1))
    
    # Rotate 270° counterclockwise (equivalent to 90° clockwise)
    vnir_cube_bip = np.rot90(vnir_cube_bip, k=-1, axes=(0, 1))
    swir_cube_bip = np.rot90(swir_cube_bip, k=-1, axes=(0, 1))
    
    # Remove unwanted bands:
    # VNIR: remove the first 3 bands; SWIR: remove bands 172, 173 and the first 4 bands.
    VNIR_cube_clean = np.delete(vnir_cube_bip, [0, 1, 2], axis=2)
    SWIR_cube_clean = np.delete(swir_cube_bip, [171, 172], axis=2)
    SWIR_cube_clean = np.delete(SWIR_cube_clean, [0, 1, 2, 3], axis=2)
    
    # Process CW and FWHM arrays
    cw_vnir = cw_vnir[:, 99:162][:, ::-1]
    fwhm_vnir = fwhm_vnir[:, 99:162][:, ::-1]
    cw_swir = cw_swir[:, 81:252][:, ::-1]
    fwhm_swir = fwhm_swir[:, 81:252][:, ::-1]
    cw_swir_clean = np.delete(cw_swir, [0, 1, 2, 3], axis=1)
    fwhm_swir_clean = np.delete(fwhm_swir, [0, 1, 2, 3], axis=1)
    
    # Concatenate VNIR and SWIR cubes and reverse the order of bands
    concatenated_cube = np.concatenate((SWIR_cube_clean, VNIR_cube_clean), axis=2)
    concatenated_cube = concatenated_cube[:, :, ::-1]
    
    # Concatenate CW and FWHM arrays
    concatenated_cw = np.concatenate((cw_vnir, cw_swir_clean), axis=1)
    concatenated_fwhm = np.concatenate((fwhm_vnir, fwhm_swir_clean), axis=1)
    
    # Create an RGB image using selected bands (e.g., bands 30, 20, and 8)
    red_channel = concatenated_cube[:, :, 29]
    green_channel = concatenated_cube[:, :, 19]
    blue_channel = concatenated_cube[:, :, 7]
    
    red_normalized = (red_channel - red_channel.min()) / (red_channel.max() - red_channel.min())
    green_normalized = (green_channel - green_channel.min()) / (green_channel.max() - green_channel.min())
    blue_normalized = (blue_channel - blue_channel.min()) / (blue_channel.max() - blue_channel.min())
    
    rgb_image = np.stack([red_normalized, green_normalized, blue_normalized], axis=-1)
    
    return (concatenated_cube, concatenated_cw, concatenated_fwhm, rgb_image,
            vnir_cube_bip, swir_cube_bip, latitude_vnir, longitude_vnir, latitude_swir, longitude_swir)

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

def prisma_read_from_zip(zip_path, extract_to=None):
    """
    Extracts a PRISMA L1 .he5 file from a ZIP archive and reads its data.
    """
    he5_file = extract_he5_from_zip(zip_path, extract_to)
    result = prisma_read(he5_file)
    # Uncomment the following line if you want to remove the extracted file after processing:
    # os.remove(he5_file)
    return result

def save_as_geotiff_multichannel(data, output_file, latitude_vnir, longitude_vnir, channel_wavelengths=None):
    """
    Saves a multichannel array as a GeoTIFF with EPSG:4326 projection.
    
    Optionally, if channel_wavelengths is provided (a list with one wavelength per band),
    the wavelength is added to each band's metadata.
    """
    data_float32 = data.astype(np.float32)
    # Revert the rotation applied in prisma_read()
    data_float32 = np.rot90(data_float32, k=1, axes=(0, 1))
    
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
    
    dataset = driver.Create(temp_file, data.shape[1], data.shape[0], data.shape[2], gdal.GDT_Float32)
    for i in range(data.shape[2]):
        band = dataset.GetRasterBand(i+1)
        band.WriteArray(data_float32[:, :, i])
        # If channel wavelengths are provided, add the wavelength to the band metadata.
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
    data_float32 = np.rot90(data_float32, k=1, axes=(0, 1))
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
    
    dataset = driver.Create(temp_file, data.shape[1], data.shape[0], 1, gdal.GDT_Float32)
    band = dataset.GetRasterBand(1)
    band.WriteArray(data_float32)
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

# # --- Example Usage ---

# # Path to the ZIP file containing the PRISMA L1 .he5 file.
# l1_zip = "D:\Lavoro\Assegno_Ricerca_Sapienza\CLEAR_UP\CH4_detection\PRISMA\img_PRISMA\Italia\Malagrotta\PRS_L1_STD_OFFL_20240806100501_20240806100506_0001.zip"

# # Read PRISMA L1 data from the ZIP file.
# (rads_cube, cw_array, fwhm_array, rgb_image,
#  vnir_cube, swir_cube, lat_vnir, lon_vnir, lat_swir, longitude_swir) = prisma_read_from_zip(l1_zip)

# # Create channel-to-wavelength mapping.
# # Assume cw_array has shape (n_rows, n_channels) and use the central row.
# central_row = cw_array[cw_array.shape[0] // 2, :]
# channel_names = [f"Channel_{i+1}" for i in range(central_row.size)]
# channel_wavelength_mapping = dict(zip(channel_names, central_row))

# # print("Channel to wavelength mapping:")
# # for channel, wavelength in channel_wavelength_mapping.items():
# #     print(f"{channel}: {wavelength:.2f} nm")

# # Create a list of wavelengths in the order corresponding to the bands in the GeoTIFF.
# # (Assuming the order in the concatenated cube matches the order of the CW channels.)
# channel_wavelengths = [channel_wavelength_mapping[f"Channel_{i+1}"] for i in range(len(channel_names))]

# # Save the radiance cube as a multichannel GeoTIFF and embed channel wavelength metadata.
# save_as_geotiff_multichannel(rads_cube, "output_rads.tif", lat_vnir, lon_vnir, channel_wavelengths=channel_wavelengths)

# # Save the RGB image as a multichannel GeoTIFF.
# save_as_geotiff_multichannel(rgb_image, "output_rgb.tif", lat_vnir, lon_vnir, channel_wavelengths=None)

# # Optionally, save a single band (e.g., the red channel) as a GeoTIFF.
# save_as_geotiff_single_band(rgb_image[:, :, 0], "output_red.tif", lat_vnir, lon_vnir)


# ---------------------------
# Processing Function for L1 Product
# ---------------------------
def process_prisma_l1(input_zip, output_dir):
    """
    Runs the complete processing for a PRISMA L1 product.
    
    Parameters:
      input_zip: Path to the input PRISMA L1 ZIP file.
      output_dir: Directory where the output GeoTIFF files will be saved.
    
    The function:
      - Extracts and reads the L1 product.
      - Generates a channel-to-wavelength mapping using the central row of the CW array.
      - Saves the radiance cube as a multichannel GeoTIFF with embedded wavelength metadata.
      - Saves the RGB image as a multichannel GeoTIFF.
      - Saves a single band (red channel) as a GeoTIFF.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    (rads_cube, cw_array, fwhm_array, rgb_image,
     vnir_cube, swir_cube, lat_vnir, lon_vnir, lat_swir, lon_swir) = prisma_read_from_zip(input_zip)
    
    # Create channel-to-wavelength mapping using the central row of the CW array.
    central_row = cw_array[cw_array.shape[0] // 2, :]
    channel_names = [f"Channel_{i+1}" for i in range(central_row.size)]
    channel_wavelength_mapping = dict(zip(channel_names, central_row))
    channel_wavelengths = [channel_wavelength_mapping[f"Channel_{i+1}"] for i in range(len(channel_names))]
    
    base_name = os.path.splitext(os.path.basename(input_zip))[0]
    radiance_out = os.path.join(output_dir, f"{base_name}_radiance.tif")
    rgb_out = os.path.join(output_dir, f"{base_name}_rgb.tif")
    red_out = os.path.join(output_dir, f"{base_name}_red.tif")
    
    print("Saving radiance data to:", radiance_out)
    save_as_geotiff_multichannel(rads_cube, radiance_out, lat_vnir, lon_vnir, channel_wavelengths=channel_wavelengths)
    
    print("Saving RGB image to:", rgb_out)
    save_as_geotiff_multichannel(rgb_image, rgb_out, lat_vnir, lon_vnir, channel_wavelengths=None)
    
    # print("Saving red channel to:", red_out)
    # save_as_geotiff_single_band(rgb_image[:, :, 0], red_out, lat_vnir, lon_vnir)
    
    print("Processing complete.")

# ---------------------------
# Main Block for Command-Line and Editor Execution
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a PRISMA L1 product and generate GeoTIFF outputs."
    )
    parser.add_argument("input_zip", nargs="?", default=None,
                        help="Path to the input PRISMA L1 ZIP file")
    parser.add_argument("output_dir", nargs="?", default=None,
                        help="Output directory for the GeoTIFF files")
    args = parser.parse_args()
    
    if args.input_zip is None or args.output_dir is None:
        print("No command-line arguments provided. Using default values.")
        # Replace these with valid paths on your system.
        default_input_zip = r"D:\Lavoro\GMATICS\IRIDE_ForestMonitoring\phase2\codici_ph2\PRS_L1_STD_OFFL_20240806100501_20240806100506_0001.zip"
        default_output_dir = r"D:\Lavoro\GMATICS\IRIDE_ForestMonitoring\phase2\codici_ph2"
        input_zip = default_input_zip
        output_dir = default_output_dir
    else:
        input_zip = args.input_zip
        output_dir = args.output_dir
    
    process_prisma_l1(input_zip, output_dir)