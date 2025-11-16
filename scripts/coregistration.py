"""
Image coregistration functionality using AROSICS library.
"""

from arosics import COREG_LOCAL
from typing import Dict, Optional
import rasterio
import numpy as np

def create_coregistration_params(grid_res: int = 20,
                               window_size: tuple = (200, 200),
                               max_shift: int = 10,
                               band_number: int = 1,
                               **kwargs) -> Dict:
    """
    Create parameters for coregistration.
    
    Args:
        grid_res: Resolution of the grid (in pixels)
        window_size: Size of the moving window
        max_shift: Maximum allowed shift in pixels
        band_number: Band to use for matching (1-based index)
        **kwargs: Additional parameters for AROSICS
    
    Returns:
        Dict of parameters for COREG_LOCAL
    """
    params = {
        'grid_res': grid_res,
        'window_size': window_size,
        'q': False,
        'max_shift': max_shift,
        'r_b4match': band_number,
        's_b4match': band_number,
        'max_iter': 20,
        'align_grids': True,
        'fmt_out': "GTiff",
        'ignore_errors': True,
        'CPUs': 16
    }
    
    # Update with any additional parameters
    params.update(kwargs)
    
    return params

def coregister_images(reference_path: str,
                     target_path: str,
                     output_path: str,
                     params: Optional[Dict] = None) -> None:
    """
    Coregister target image to reference image using AROSICS.
    
    Args:
        reference_path: Path to the reference image
        target_path: Path to the image to be coregistered
        output_path: Path where to save the coregistered image
        params: Optional dictionary of parameters for COREG_LOCAL
    """
    if params is None:
        params = create_coregistration_params()
    
    # Set output path in parameters
    params['path_out'] = output_path
    
    # Initialize COREG_LOCAL
    CRL = COREG_LOCAL(reference_path, target_path, **params)
    
    # Perform coregistration
    CRL.correct_shifts()

def prepare_stack_for_coregistration(rgb_path: str,
                                   pan_path: str,
                                   output_path: str) -> str:
    """
    Create a stacked image from RGB and PAN for coregistration.
    
    Args:
        rgb_path: Path to RGB image
        pan_path: Path to panchromatic image
        output_path: Path to save the stacked output
    
    Returns:
        Path to the stacked image
    """
    # Read the high-resolution Pan image
    with rasterio.open(pan_path) as pan_src:
        pan_data = pan_src.read(1)
        pan_transform = pan_src.transform
        pan_crs = pan_src.crs
        pan_shape = pan_data.shape

    # Open and resample the RGB image
    with rasterio.open(rgb_path) as rgb_src:
        rgb_data_resampled = rgb_src.read(
            out_shape=(
                rgb_src.count,
                pan_shape[0],
                pan_shape[1]
            ),
            resampling=rasterio.enums.Resampling.bilinear
        )

    # Prepare the stacked data
    rgb_resampled = rgb_data_resampled.transpose(1, 2, 0)
    pan_expanded = pan_data[:, :, np.newaxis]
    stacked = np.concatenate((rgb_resampled, pan_expanded), axis=2)

    # Save the stacked result
    with rasterio.open(output_path, "w",
                      driver="GTiff",
                      height=stacked.shape[0],
                      width=stacked.shape[1],
                      count=stacked.shape[2],
                      dtype=stacked.dtype,
                      crs=pan_crs,
                      transform=pan_transform) as dst:
        dst.write(stacked.transpose(2, 0, 1))

    return output_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Coregister images using AROSICS")
    parser.add_argument('--reference', required=True, help='Path to reference image')
    parser.add_argument('--target', required=True, help='Path to target image')
    parser.add_argument('--out', required=True, help='Output path')
    parser.add_argument('--grid-res', type=int, default=20, help='Grid resolution')
    parser.add_argument('--max-shift', type=int, default=10, help='Maximum shift')
    parser.add_argument('--window-size', type=int, nargs=2, default=[200, 200], 
                      help='Window size (height width)')
    
    args = parser.parse_args()
    
    params = create_coregistration_params(
        grid_res=args.grid_res,
        window_size=tuple(args.window_size),
        max_shift=args.max_shift
    )
    
    coregister_images(args.reference, args.target, args.out, params)