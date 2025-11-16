"""
GSA (Gram-Schmidt Adaptive) pansharpening implementation for WorldView imagery.
"""

from osgeo import gdal
import numpy as np
import cv2
import rasterio
from typing import Tuple, Optional

def GSA_sharpening(ms_path: str, pan_path: str, output_path: Optional[str] = None) -> np.ndarray:
    """
    Perform GSA pansharpening on multispectral and panchromatic images.
    
    Args:
        ms_path: Path to the multispectral image
        pan_path: Path to the panchromatic image
        output_path: Optional path to save the output. If None, only returns the array
    
    Returns:
        np.ndarray: Pansharpened image array
    """
    # Read multispectral and pan images
    with rasterio.open(ms_path) as ms_src:
        ms_data = ms_src.read()
        ms_transform = ms_src.transform
        ms_crs = ms_src.crs
        ms_meta = ms_src.meta

    with rasterio.open(pan_path) as pan_src:
        pan_data = pan_src.read(1)
        pan_shape = pan_data.shape

    # Resample MS to PAN resolution
    ms_data_resampled = np.zeros((ms_data.shape[0], pan_shape[0], pan_shape[1]))
    for band in range(ms_data.shape[0]):
        ms_data_resampled[band] = cv2.resize(ms_data[band], 
                                            (pan_shape[1], pan_shape[0]), 
                                            interpolation=cv2.INTER_CUBIC)

    # GSA algorithm implementation
    imageLR = np.transpose(ms_data_resampled, (1, 2, 0))
    imageHR = pan_data

    # Remove means
    imageLR0 = np.zeros_like(imageLR)
    for i in range(imageLR.shape[2]):
        imageLR0[:,:,i] = imageLR[:,:,i] - np.mean(imageLR[:,:,i])
    
    imageHR0 = imageHR - np.mean(imageHR)

    # Estimate alpha coefficients
    IHc = imageHR0.reshape(-1, 1)
    ILRc = np.column_stack([imageLR0.reshape(-1, imageLR0.shape[2]), 
                           np.ones(imageHR0.size)])
    alpha = np.linalg.lstsq(ILRc, IHc, rcond=None)[0]

    # Calculate intensity
    I = np.dot(np.column_stack([imageLR0.reshape(-1, imageLR0.shape[2]), 
                               np.ones((imageHR0.size, 1))]), alpha)
    I = I.reshape(imageHR0.shape)
    I0 = I - np.mean(I)

    # Calculate injection gains
    g = np.ones((imageLR0.shape[2] + 1, 1))
    for i in range(imageLR0.shape[2]):
        h = imageLR0[:,:,i].reshape(-1)
        c = np.cov(I0.reshape(-1), h)[0,1]
        g[i] = c / np.var(I0.reshape(-1))

    # Detail extraction and injection
    delta = imageHR - I0
    delta_m = delta.reshape(-1, 1)
    
    V = np.column_stack([I0.reshape(-1, 1)] + 
                       [imageLR0[:,:,i].reshape(-1, 1) for i in range(imageLR0.shape[2])])
    
    gm = np.tile(g.T, (V.shape[0], 1))
    V_hat = V + delta_m * gm

    # Reshape and finalize
    I_Fus_GSA = V_hat[:,1:].reshape(imageLR.shape)
    
    # Final mean adjustment
    for i in range(imageLR.shape[2]):
        I_Fus_GSA[:,:,i] = (I_Fus_GSA[:,:,i] - np.mean(I_Fus_GSA[:,:,i]) + 
                           np.mean(imageLR[:,:,i]))

    # Handle nodata
    I_Fus_GSA[imageLR == 0] = 0

    if output_path:
        # Update metadata for output
        ms_meta.update({
            'height': I_Fus_GSA.shape[0],
            'width': I_Fus_GSA.shape[1],
            'transform': ms_transform,
            'count': I_Fus_GSA.shape[2]
        })
        
        # Save to file
        with rasterio.open(output_path, 'w', **ms_meta) as dst:
            for i in range(I_Fus_GSA.shape[2]):
                dst.write(I_Fus_GSA[:,:,i], i+1)

    return I_Fus_GSA

def process_worldview_rgb(ms_path: str, pan_path: str, output_path: str) -> None:
    """
    Process WorldView RGB bands with pansharpening.
    
    Args:
        ms_path: Path to the multispectral image
        pan_path: Path to the panchromatic image
        output_path: Path to save the pansharpened output
    """
    # Process each band
    result = GSA_sharpening(ms_path, pan_path)
    
    # Get georeferencing info from pan image
    with rasterio.open(pan_path) as src:
        transform = src.transform
        crs = src.crs
    
    # Save the result
    data = np.transpose(result, (2, 0, 1))
    with rasterio.open(output_path, "w", driver="GTiff",
                      height=data.shape[1], width=data.shape[2],
                      count=data.shape[0], dtype=data.dtype,
                      crs=crs, transform=transform) as dst:
        dst.write(data)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Perform GSA pansharpening on WorldView imagery")
    parser.add_argument('--ms', required=True, help='Path to multispectral image')
    parser.add_argument('--pan', required=True, help='Path to panchromatic image')
    parser.add_argument('--out', required=True, help='Output path for pansharpened image')
    
    args = parser.parse_args()
    
    process_worldview_rgb(args.ms, args.pan, args.out)