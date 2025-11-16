#!/usr/bin/env python3
"""
Robust raster mosaicking utility.

Features
- Accepts multiple raster inputs (globs or explicit paths).
- Reprojects inputs to a target CRS (optional) using WarpedVRT.
- Uses rasterio.merge to produce a single mosaic.
- Supports common merge methods and writes a tiled/compressed GeoTIFF.

This module focuses on being pragmatic and robust for many satellite image
formats. It relies on rasterio (and optionally rioxarray/xarray for more
advanced use-cases).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union
import logging

import rasterio
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.vrt import WarpedVRT
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _expand_inputs(inputs: Iterable[str]) -> List[Path]:
    paths: List[Path] = []
    for p in inputs:
        for match in Path().glob(p) if any(ch in p for ch in "*?[]") else [Path(p)]:
            if match.exists():
                paths.append(match)
            else:
                # if a literal path doesn't exist, warn and skip
                logger.warning("Input not found: %s", match)
    if not paths:
        raise FileNotFoundError("No input rasters found from provided patterns")
    return paths


def _open_sources(paths: List[Path]):
    for p in paths:
        yield rasterio.open(p)


def mosaic_rasters(
    inputs: Iterable[str],
    out_path: Union[str, Path],
    target_crs: Optional[Union[str, dict]] = None,
    res: Optional[Union[float, Tuple[float, float]]] = None,
    merge_method: str = "first",
    nodata: Optional[float] = None,
    dtype: Optional[str] = None,
    compress: str = "LZW",
    tiled: bool = True,
    bigtiff: str = "IF_SAFER",
    overwrite: bool = False,
    resampling: Resampling = Resampling.nearest,
) -> Path:
    """Create a mosaic from multiple raster inputs.

    Args:
        inputs: Iterable of input path patterns or file paths.
        out_path: Output GeoTIFF path.
        target_crs: Optional target CRS (e.g., "EPSG:32633"). If omitted, the
            CRS of the first readable input is used.
        res: Target resolution in units of target CRS. Either scalar or (xres,yres).
        merge_method: Passed to rasterio.merge.merge (e.g. 'first', 'last', 'min', 'max').
        nodata: Output nodata value.
        dtype: Forced output dtype (e.g. 'float32'). If None, inferred from inputs.
        compress: TIFF compression (default: LZW).
        tiled: Create tiled GeoTIFF.
        bigtiff: BIGTIFF option passed to rasterio (IF_SAFER by default).
        overwrite: Overwrite existing output.
        resampling: Resampling enum used when warping inputs.

    Returns:
        Path to the written mosaic file.
    """
    out_path = Path(out_path)
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {out_path}. Use overwrite=True to replace.")

    paths = _expand_inputs(inputs)
    srcs = []
    try:
        for src in _open_sources(paths):
            srcs.append(src)

        if not srcs:
            raise RuntimeError("No readable input rasters")

        # pick target CRS
        if target_crs is None:
            first_crs = srcs[0].crs
            if first_crs is None:
                raise RuntimeError("First input has no CRS; set target_crs explicitly")
            target_crs = first_crs

        # Prepare VRTs for each source (reproject on-the-fly if needed)
        vrts = []
        for src in srcs:
            if src.crs is None or src.crs != rasterio.crs.CRS.from_user_input(target_crs):
                vrt = WarpedVRT(src, crs=target_crs, resampling=resampling)
                vrts.append(vrt)
            else:
                vrts.append(src)

        # merge; rasterio.merge will compute an adequate output shape/transform
        merge_kwargs = {"method": merge_method}
        if res is not None:
            # rasterio.merge expects res as (xres, yres) or scalar
            merge_kwargs["res"] = res
        if nodata is not None:
            merge_kwargs["nodata"] = nodata

        mosaic_arr, out_transform = merge(vrts, **merge_kwargs)

        # mosaic_arr is (bands, h, w)
        out_meta = vrts[0].meta.copy()
        out_meta.update(
            {
                "driver": "GTiff",
                "height": mosaic_arr.shape[1],
                "width": mosaic_arr.shape[2],
                "transform": out_transform,
                "crs": rasterio.crs.CRS.from_user_input(target_crs),
                "compress": compress,
                "tiled": bool(tiled),
                "BIGTIFF": bigtiff,
            }
        )

        if dtype:
            out_meta["dtype"] = dtype
            mosaic_arr = mosaic_arr.astype(dtype)
        else:
            # ensure dtype is writeable (use numpy dtype of array)
            out_meta["dtype"] = mosaic_arr.dtype.name

        if nodata is not None:
            out_meta["nodata"] = nodata

        # Ensure parent dir exists
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Write output
        with rasterio.open(out_path, "w", **out_meta) as dst:
            dst.write(mosaic_arr)

        logger.info("Wrote mosaic: %s", out_path)
        return out_path
    finally:
        for obj in srcs:
            try:
                obj.close()
            except Exception:
                pass


if __name__ == "__main__":
    # Small CLI for quick use (kept minimal; see mosaic_cli for richer options)
    import argparse

    parser = argparse.ArgumentParser(description="Create a mosaic from multiple rasters.")
    parser.add_argument("inputs", nargs="+", help="Input raster files or glob patterns")
    parser.add_argument("output", help="Output GeoTIFF path")
    parser.add_argument("--crs", help="Target CRS (e.g. EPSG:32632)")
    parser.add_argument("--res", help="Target resolution (single number or 'x,y')")
    parser.add_argument("--method", default="first", help="Merge method (first/last/min/max)")
    parser.add_argument("--nodata", type=float, help="Set nodata value for output")
    parser.add_argument("--dtype", help="Force output dtype (e.g. float32)")
    parser.add_argument("--compress", default="LZW")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    res_arg = None
    if args.res:
        parts = args.res.split(",")
        res_arg = float(parts[0]) if len(parts) == 1 else (float(parts[0]), float(parts[1]))

    mosaic_rasters(
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
