"""
Example script demonstrating the use of WorldView image enhancement tools.
This example shows different ways to use the pansharpening and coregistration functionality.
"""

import os
from pathlib import Path
import rasterio
import numpy as np
from scripts.gsa_pansharpening import process_worldview_rgb
from scripts.coregistration import coregister_images
from scripts.worldview_enhance import enhance_worldview_image

def create_sample_data(output_dir: str) -> tuple:
    """Create synthetic test data for demonstration."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create sample data paths
    ms_path = os.path.join(output_dir, "sample_ms.tif")
    pan_path = os.path.join(output_dir, "sample_pan.tif")
    ref_path = os.path.join(output_dir, "sample_reference.tif")
    
    # Create sample multispectral image (3 bands, lower resolution)
    ms_data = np.random.rand(3, 100, 100).astype(np.float32)
    with rasterio.open(
        ms_path, 'w',
        driver='GTiff',
        height=100, width=100,
        count=3,
        dtype=ms_data.dtype,
        crs='+proj=utm +zone=33 +datum=WGS84',
        transform=rasterio.transform.from_bounds(0, 0, 100, 100, 100, 100)
    ) as dst:
        dst.write(ms_data)
    
    # Create sample panchromatic image (1 band, higher resolution)
    pan_data = np.random.rand(400, 400).astype(np.float32)
    with rasterio.open(
        pan_path, 'w',
        driver='GTiff',
        height=400, width=400,
        count=1,
        dtype=pan_data.dtype,
        crs='+proj=utm +zone=33 +datum=WGS84',
        transform=rasterio.transform.from_bounds(0, 0, 100, 100, 400, 400)
    ) as dst:
        dst.write(pan_data, 1)
    
    # Create sample reference image
    ref_data = np.random.rand(400, 400).astype(np.float32)
    with rasterio.open(
        ref_path, 'w',
        driver='GTiff',
        height=400, width=400,
        count=1,
        dtype=ref_data.dtype,
        crs='+proj=utm +zone=33 +datum=WGS84',
        transform=rasterio.transform.from_bounds(0, 0, 100, 100, 400, 400)
    ) as dst:
        dst.write(ref_data, 1)
    
    return ms_path, pan_path, ref_path

def main():
    # Setup
    output_dir = "example_data"
    ms_path, pan_path, ref_path = create_sample_data(output_dir)
    
    print("1. Pansharpening only example:")
    sharp_output = os.path.join(output_dir, "pansharpened.tif")
    process_worldview_rgb(ms_path, pan_path, sharp_output)
    print(f"   Output saved to: {sharp_output}")
    
    print("\n2. Coregistration only example:")
    coreg_output = os.path.join(output_dir, "coregistered.tif")
    coregister_images(ref_path, ms_path, coreg_output)
    print(f"   Output saved to: {coreg_output}")
    
    print("\n3. Complete pipeline example (pansharpen then coregister):")
    enhance_worldview_image(
        ms_path=ms_path,
        pan_path=pan_path,
        reference_path=ref_path,
        output_prefix=os.path.join(output_dir, "enhanced"),
        coregister_first=False
    )
    print("   Outputs saved with prefix 'enhanced'")
    
    print("\n4. Complete pipeline example (coregister then pansharpen):")
    enhance_worldview_image(
        ms_path=ms_path,
        pan_path=pan_path,
        reference_path=ref_path,
        output_prefix=os.path.join(output_dir, "enhanced_alt"),
        coregister_first=True
    )
    print("   Outputs saved with prefix 'enhanced_alt'")

if __name__ == "__main__":
    main()