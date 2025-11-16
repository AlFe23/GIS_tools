"""
Main pipeline for WorldView image enhancement using pansharpening and coregistration.
Allows for running the complete pipeline or individual steps.
"""

import os
from typing import Optional
from gsa_pansharpening import process_worldview_rgb
from coregistration import coregister_images, create_coregistration_params, prepare_stack_for_coregistration

def enhance_worldview_image(ms_path: str,
                          pan_path: str,
                          reference_path: str,
                          output_prefix: str,
                          coregister_first: bool = False,
                          skip_pansharpening: bool = False,
                          skip_coregistration: bool = False) -> None:
    """
    Enhance WorldView imagery through pansharpening and/or coregistration.
    
    Args:
        ms_path: Path to multispectral image
        pan_path: Path to panchromatic image
        reference_path: Path to reference image (e.g., Google Satellite)
        output_prefix: Prefix for output files
        coregister_first: Whether to coregister before pansharpening
        skip_pansharpening: Skip the pansharpening step
        skip_coregistration: Skip the coregistration step
    """
    if not skip_pansharpening and not skip_coregistration:
        if coregister_first:
            # 1. Stack MS and PAN
            print("Preparing image stack for coregistration...")
            stack_path = f"{output_prefix}_stack.tif"
            prepare_stack_for_coregistration(ms_path, pan_path, stack_path)
            
            # 2. Coregister stack
            print("Coregistering stacked image...")
            coreg_stack_path = f"{output_prefix}_coreg_stack.tif"
            coregister_images(reference_path, stack_path, coreg_stack_path)
            
            # 3. Pansharpen using coregistered images
            print("Applying pansharpening...")
            process_worldview_rgb(ms_path, pan_path, f"{output_prefix}_enhanced.tif")
        else:
            # 1. Pansharpen
            print("Applying pansharpening...")
            sharp_path = f"{output_prefix}_sharp.tif"
            process_worldview_rgb(ms_path, pan_path, sharp_path)
            
            # 2. Coregister
            print("Coregistering pansharpened image...")
            coregister_images(reference_path, sharp_path, f"{output_prefix}_enhanced.tif")
    
    elif not skip_pansharpening:
        # Only pansharpen
        print("Applying pansharpening...")
        process_worldview_rgb(ms_path, pan_path, f"{output_prefix}_sharp.tif")
    
    elif not skip_coregistration:
        # Only coregister
        print("Coregistering image...")
        coregister_images(reference_path, ms_path, f"{output_prefix}_coreg.tif")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Enhance WorldView imagery through pansharpening and coregistration"
    )
    
    parser.add_argument('--ms', help='Path to multispectral image')
    parser.add_argument('--pan', help='Path to panchromatic image')
    parser.add_argument('--reference', help='Path to reference image (e.g., Google Satellite)')
    parser.add_argument('--out-prefix', required=True, 
                       help='Output prefix for generated files')
    parser.add_argument('--coregister-first', action='store_true',
                       help='Perform coregistration before pansharpening')
    parser.add_argument('--skip-pansharpening', action='store_true',
                       help='Skip the pansharpening step')
    parser.add_argument('--skip-coregistration', action='store_true',
                       help='Skip the coregistration step')
    
    args = parser.parse_args()
    
    # Validate arguments based on selected operations
    if not args.skip_pansharpening and (not args.ms or not args.pan):
        parser.error("--ms and --pan are required when pansharpening is enabled")
    
    if not args.skip_coregistration and not args.reference:
        parser.error("--reference is required when coregistration is enabled")
    
    enhance_worldview_image(
        ms_path=args.ms,
        pan_path=args.pan,
        reference_path=args.reference,
        output_prefix=args.out_prefix,
        coregister_first=args.coregister_first,
        skip_pansharpening=args.skip_pansharpening,
        skip_coregistration=args.skip_coregistration
    )