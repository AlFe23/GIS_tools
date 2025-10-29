#!/usr/bin/env python3
"""
Zarr → GeoTIFF converter for geospatial arrays.

Features
- Opens consolidated/non-consolidated Zarr
- Auto-picks main variable (or specify with --var)
- Handles 2D (y,x) or 3D (band,y,x)
- Reads coords (x/y or lon/lat) and CF grid_mapping
- Writes CRS if present (or override with --crs)
- Optional windowing by bounds (--bounds xmin ymin xmax ymax)
- Controls dtype, nodata, tiling, compression, BIGTIFF
- Creates overviews (--pyramids) for faster QGIS display
"""
import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import rioxarray  # noqa: F401 - registers xarray accessor
import xarray as xr

try:
    import rasterio
    from rasterio.enums import Resampling
except Exception:  # pragma: no cover - rasterio optional
    rasterio = None  # not critical if only writing via rioxarray


def _pick_data_var(ds: xr.Dataset, prefer: Optional[str] = None) -> xr.DataArray:
    if prefer and prefer in ds:
        da = ds[prefer]
        if da.ndim in (2, 3):
            return da

    # helpful common names
    candidates = ["data", "dem", "elevation", "band", "image", "array"]
    for name in candidates:
        if name in ds and ds[name].ndim in (2, 3):
            return ds[name]

    # fallback: first 2D/3D var
    for key, value in ds.data_vars.items():
        if value.ndim in (2, 3):
            return value

    raise RuntimeError("No 2D/3D data variable found. Use --var to select one.")


def _ensure_spatial_dims(da: xr.DataArray) -> xr.DataArray:
    dims = {dim.lower(): dim for dim in da.dims}
    # Prefer x/y, else lon/lat, else try best guess
    if "x" in dims and "y" in dims:
        da = da.rio.set_spatial_dims(x_dim=dims["x"], y_dim=dims["y"], inplace=False)
    elif "lon" in dims and "lat" in dims:
        da = (
            da.rename({dims["lon"]: "x", dims["lat"]: "y"})
            .rio.set_spatial_dims("x", "y", inplace=False)
        )
    else:
        # heuristic: last two dims are (y,x)
        if da.ndim >= 2:
            ydim, xdim = da.dims[-2], da.dims[-1]
            da = da.rio.set_spatial_dims(x_dim=xdim, y_dim=ydim, inplace=False)
        else:
            raise RuntimeError("Cannot infer spatial dims (need a 2D or 3D array).")
    return da


def _extract_crs_from_grid_mapping(ds: xr.Dataset, da: xr.DataArray) -> Optional[str]:
    grid_mapping_name = da.attrs.get("grid_mapping")
    if not grid_mapping_name or grid_mapping_name not in ds:
        return None
    grid_mapping_var = ds[grid_mapping_name]
    attr_candidates = [
        "spatial_ref",
        "crs_wkt",
        "crs",
        "proj4",
        "proj4text",
        "projjson",
    ]
    for key in attr_candidates:
        val = grid_mapping_var.attrs.get(key)
        if val:
            return val

    epsg_candidates = [
        "epsg",
        "EPSG",
        "epsg_code",
        "proj:epsg",
        "proj:code",
        "grid_mapping_name",
    ]
    for key in epsg_candidates:
        val = grid_mapping_var.attrs.get(key)
        if val:
            text = str(val)
            if not text.upper().startswith("EPSG:") and text.isdigit():
                return f"EPSG:{text}"
            return text
    return None


def _write_crs_if_known(
    da: xr.DataArray, ds: xr.Dataset, override_crs: Optional[str]
) -> xr.DataArray:
    if override_crs:
        return da.rio.write_crs(override_crs, inplace=False)

    def _first_truthy(*values: Optional[str]) -> Optional[str]:
        for value in values:
            if value:
                return str(value)
        return None

    # gather candidates from various sources
    candidate = _first_truthy(
        _extract_crs_from_grid_mapping(ds, da),
        *[
            var.attrs.get(key)
            for name, var in ds.data_vars.items()
            if var.ndim == 0
            for key in ("spatial_ref", "crs_wkt", "crs", "proj4", "projjson")
        ],
        da.attrs.get("crs"),
        da.attrs.get("spatial_ref"),
        da.attrs.get("crs_wkt"),
        da.encoding.get("crs") if isinstance(da.encoding, dict) else None,
    )

    if not candidate:
        for coord in da.coords.values():
            candidate = _first_truthy(
                coord.attrs.get("crs"),
                coord.attrs.get("spatial_ref"),
                coord.attrs.get("crs_wkt"),
                coord.attrs.get("epsg"),
                coord.attrs.get("EPSG"),
            )
            if candidate:
                break

    if not candidate:
        candidate = _first_truthy(
            da.attrs.get("epsg"),
            da.attrs.get("EPSG"),
            da.attrs.get("proj:epsg"),
            da.attrs.get("proj:code"),
            da.attrs.get("epsg_code"),
            ds.attrs.get("crs"),
            ds.attrs.get("spatial_ref"),
            ds.attrs.get("crs_wkt"),
            ds.attrs.get("projjson"),
            ds.attrs.get("proj:epsg"),
        )

    if not candidate and hasattr(da, "rio"):
        try:
            _ = da.rio.crs  # triggers lazy resolution; may be None
            candidate = da.rio.crs
        except Exception:  # pragma: no cover - best effort
            candidate = None

    if candidate:
        try:
            return da.rio.write_crs(candidate, inplace=False)
        except Exception:  # pragma: no cover - fall back to EPSG lookup
            pass

    for key in ("epsg", "EPSG", "proj:epsg", "proj:code"):
        val = da.attrs.get(key) or ds.attrs.get(key)
        if val:
            epsg_code = str(val)
            if not epsg_code.upper().startswith("EPSG:") and epsg_code.isdigit():
                epsg_code = f"EPSG:{epsg_code}"
            return da.rio.write_crs(epsg_code, inplace=False)

    return da  # If still unknown, leave unset; user may pass --crs


def _apply_bounds_window(
    da: xr.DataArray, bounds: Optional[Tuple[float, float, float, float]]
) -> xr.DataArray:
    if not bounds:
        return da
    xmin, ymin, xmax, ymax = bounds
    return da.rio.clip_box(minx=xmin, miny=ymin, maxx=xmax, maxy=ymax)


def _maybe_to_dtype(da: xr.DataArray, dtype: Optional[str]) -> xr.DataArray:
    if not dtype:
        return da
    try:
        return da.astype(dtype)
    except Exception as exc:  # pragma: no cover - conversion errors
        raise RuntimeError(f"Failed to cast to dtype {dtype}: {exc}") from exc


def _parse_bounds(bounds: Optional[str]) -> Optional[Tuple[float, float, float, float]]:
    if not bounds:
        return None
    parts = [part.strip() for part in bounds.replace(",", " ").split()]
    if len(parts) != 4:
        raise ValueError("--bounds must have 4 numbers: xmin ymin xmax ymax")
    return tuple(float(part) for part in parts)  # type: ignore


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a geospatial Zarr store to a GeoTIFF readable in QGIS."
    )
    parser.add_argument(
        "zarr",
        help="Path to Zarr store (folder ending with .zarr or any directory with Zarr metadata).",
    )
    parser.add_argument("out_tif", help="Output GeoTIFF path.")
    parser.add_argument(
        "--var", help="Name of the data variable to export (auto-detected if omitted)."
    )
    parser.add_argument("--crs", help="Force/override CRS (e.g., 'EPSG:32632').")
    parser.add_argument("--bounds", help="Crop bounds: 'xmin ymin xmax ymax'.")
    parser.add_argument("--dtype", help="Cast output to dtype (e.g., float32, int16).")
    parser.add_argument("--nodata", type=float, help="Set output nodata value.")
    parser.add_argument(
        "--compress", default="LZW", help="GeoTIFF compression (default: LZW)."
    )
    parser.add_argument(
        "--tiled",
        default="YES",
        choices=["YES", "NO"],
        help="Create a tiled GeoTIFF (default: YES).",
    )
    parser.add_argument(
        "--bigtiff",
        default="IF_SAFER",
        choices=["YES", "NO", "IF_NEEDED", "IF_SAFER"],
        help="BIGTIFF option (default: IF_SAFER).",
    )
    parser.add_argument(
        "--pyramids", action="store_true", help="Build overviews (pyramids) after writing."
    )
    parser.add_argument(
        "--resampling",
        default="nearest",
        choices=[
            "nearest",
            "bilinear",
            "cubic",
            "lanczos",
            "average",
            "mode",
            "min",
            "max",
        ],
        help="Resampling for overviews (default: nearest).",
    )
    parser.add_argument(
        "--consolidated",
        action="store_true",
        help="Open Zarr as consolidated metadata if available.",
    )
    args = parser.parse_args()

    zarr_path = Path(args.zarr)
    if not zarr_path.exists():
        sys.exit(f"Zarr path not found: {zarr_path}")

    ds = xr.open_zarr(zarr_path.as_posix(), consolidated=args.consolidated)
    data = _pick_data_var(ds, args.var)
    data = _ensure_spatial_dims(data)
    data = _write_crs_if_known(data, ds, args.crs)

    bounds = _parse_bounds(args.bounds)
    if bounds:
        data = _apply_bounds_window(data, bounds)

    if data.ndim == 3:
        band_dim, y_dim, x_dim = data.dims[-3], data.dims[-2], data.dims[-1]
        if data.dims != (band_dim, y_dim, x_dim):
            data = data.transpose(band_dim, y_dim, x_dim)
    elif data.ndim != 2:
        raise RuntimeError(f"Unsupported array shape {data.shape}; need 2D or 3D.")

    data = _maybe_to_dtype(data, args.dtype)
    if args.nodata is not None:
        data = data.rio.write_nodata(args.nodata, inplace=False)

    data = data.compute()
    data.rio.to_raster(
        args.out_tif,
        tiled=(args.tiled == "YES"),
        compress=args.compress,
        BIGTIFF=args.bigtiff,
    )

    if args.pyramids and rasterio is not None:
        resampling_map = {
            "nearest": Resampling.nearest,
            "bilinear": Resampling.bilinear,
            "cubic": Resampling.cubic,
            "lanczos": Resampling.lanczos,
            "average": Resampling.average,
            "mode": Resampling.mode,
            "min": Resampling.min,
            "max": Resampling.max,
        }
        with rasterio.open(args.out_tif, "r+") as dst:
            factors = [2, 4, 8, 16, 32]
            dst.build_overviews(factors, resampling_map[args.resampling])
            dst.update_tags(ns="rio_overview", resampling=args.resampling)

    print(f"Wrote {args.out_tif}")


if __name__ == "__main__":
    main()
