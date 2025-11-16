#!/usr/bin/env python3
"""CLI wrapper for scripts/mosaic_rasters.py

This script provides a friendly command-line entrypoint and a few extra
flags for common options.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the scripts directory is importable so we can import mosaic_rasters
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mosaic_rasters import mosaic_rasters  # type: ignore


def parse_args():
    p = argparse.ArgumentParser(description="Mosaic multiple rasters into a single GeoTIFF")
    p.add_argument("inputs", nargs="+", help="Input files or glob patterns (e.g. data/*.tif)")
    p.add_argument("output", help="Output GeoTIFF path")
    p.add_argument("--crs", help="Target CRS (e.g. EPSG:32632)")
    p.add_argument("--res", help="Target resolution (single number or 'x,y')")
    p.add_argument("--method", default="first", help="Merge method (first/last/min/max)")
    p.add_argument("--nodata", type=float, help="Set output nodata value")
    p.add_argument("--dtype", help="Force output dtype (e.g. float32)")
    p.add_argument("--compress", default="LZW")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    res_arg = None
    if args.res:
        parts = args.res.split(",")
        res_arg = float(parts[0]) if len(parts) == 1 else (float(parts[0]), float(parts[1]))

    out = mosaic_rasters(
        args.inputs,
        args.output,
        target_crs=args.crs,
        res=res_arg,
        merge_method=args.method,
        nodata=args.nodata,
        dtype=args.dtype,
        compress=args.compress,
        overwrite=args.overwrite,
    )
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
