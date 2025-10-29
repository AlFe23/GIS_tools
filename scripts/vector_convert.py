#!/usr/bin/env python3
"""
Universal vector converter (read/write common geospatial formats).
- Preserves attributes and geometry
- Optional reprojection (to CRS)
- Optional metrics: area_m2, perimeter_m (computed in projected CRS)
- Optional geometry fixes, explode multiparts, force 2D
- Optional bbox + attribute filtering
- Auto-chooses driver from output extension; supports layer names

Examples
--------
Basic:
    python scripts/vector_convert.py input.gpkg out.parquet

Reproject + add metrics (auto-UTM for metrics):
    python scripts/vector_convert.py input.shp out.gpkg --to-crs EPSG:32632 --add-metrics

Limit to bbox + explode multiparts + fix geometries:
    python scripts/vector_convert.py input.geojson out.shp --bbox 12.2 41.7 12.7 42.0 --explode --fix

Write a specific layer from a GPKG and name target layer:
    python scripts/vector_convert.py input.gpkg out.gpkg --in-layer my_src --out-layer my_dst

Write GeoJSON with WGS84 and drop Z:
    python scripts/vector_convert.py input.parquet out.geojson --to-crs EPSG:4326 --force-2d
"""

import argparse
import warnings
from pathlib import Path

import geopandas as gpd
import shapely
from pyproj import CRS

# --- drivers by extension ---
EXT_TO_DRIVER = {
    ".shp": "ESRI Shapefile",
    ".gpkg": "GPKG",
    ".geojson": "GeoJSON",
    ".json": "GeoJSON",
    ".parquet": "PARQUET",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert vector formats, preserving attributes, with optional metrics."
    )
    parser.add_argument("input", help="Input vector path (SHP/GPKG/GeoJSON/Parquet, etc.)")
    parser.add_argument("output", help="Output path; driver inferred from extension unless --driver set.")
    parser.add_argument("--in-layer", help="Layer to read (for multi-layer sources like GPKG).")
    parser.add_argument("--out-layer", help="Layer name for multi-layer outputs (e.g., GPKG).")
    parser.add_argument("--driver", help="Explicit output driver (e.g., 'GPKG', 'ESRI Shapefile').")
    parser.add_argument("--to-crs", help="Reproject output to this CRS (e.g., 'EPSG:32632').")
    parser.add_argument("--encoding", default=None, help="Output encoding (e.g., 'UTF-8', 'LATIN1') for SHP.")
    parser.add_argument("--explode", action="store_true", help="Explode multipart geometries to singlepart.")
    parser.add_argument("--fix", action="store_true", help="Attempt to fix invalid geometries (buffer(0)).")
    parser.add_argument("--force-2d", action="store_true", help="Drop Z (force 2D) if present.")
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("xmin", "ymin", "xmax", "ymax"),
        help="Filter by bounding box in source CRS (before reprojection).",
    )
    parser.add_argument(
        "--where",
        help="Pandas query string to filter attributes (e.g., \"POP>10000 and TYPE=='city'\").",
    )
    parser.add_argument("--add-metrics", action="store_true", help="Add area_m2 and perimeter_m fields.")
    parser.add_argument(
        "--metrics-crs",
        help="CRS to use for metrics (projected). If omitted, auto-choose a suitable UTM.",
    )
    parser.add_argument(
        "--parquet-compression",
        default="snappy",
        help="Compression for GeoParquet (default: snappy).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output if it exists.")
    return parser.parse_args()


def infer_driver_from_ext(path: Path) -> str:
    ext = path.suffix.lower()
    if ext not in EXT_TO_DRIVER:
        raise SystemExit(f"Cannot infer driver from extension '{ext}'. Use --driver.")
    return EXT_TO_DRIVER[ext]


def read_any(input_path: Path, layer: str | None) -> gpd.GeoDataFrame:
    if input_path.suffix.lower() == ".parquet":
        gdf = gpd.read_parquet(input_path)
    else:
        gdf = gpd.read_file(input_path, layer=layer)
    if gdf.empty:
        warnings.warn("Input GeoDataFrame is empty.")
    if gdf.crs is None:
        warnings.warn("Input has no CRS. Consider using --to-crs to set target CRS explicitly.")
    return gdf


def drop_z(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    def _to_2d(geom):
        if geom is None or geom.is_empty:
            return geom
        try:
            return shapely.force_2d(geom)
        except Exception:
            return shapely.geometry.shape(shapely.geometry.mapping(geom))

    gdf = gdf.copy()
    gdf.geometry = gdf.geometry.apply(_to_2d)
    return gdf


def fix_geoms(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    gdf.geometry = gdf.geometry.buffer(0)
    return gdf


def explode_multipart(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return gdf.explode(index_parts=False, ignore_index=True)


def auto_metrics_crs(gdf: gpd.GeoDataFrame) -> CRS:
    """Choose a projected CRS suitable for area/perimeter metrics."""
    try:
        if gdf.crs is None or CRS.from_user_input(gdf.crs).is_geographic:
            g_wgs = (
                gdf.to_crs("EPSG:4326")
                if gdf.crs is not None
                else gdf.set_crs("EPSG:4326", allow_override=True)
            )
        else:
            g_wgs = gdf.to_crs("EPSG:4326")
        centroid = g_wgs.unary_union.centroid
        lon, lat = float(centroid.x), float(centroid.y)
        zone = int((lon + 180) // 6) + 1
        epsg = 32600 + zone if lat >= 0 else 32700 + zone
        return CRS.from_epsg(epsg)
    except Exception:
        return CRS.from_epsg(3857)


def add_metrics(gdf: gpd.GeoDataFrame, metrics_crs: CRS) -> gpd.GeoDataFrame:
    projected = gdf.to_crs(metrics_crs)
    gdf = gdf.copy()
    gdf["area_m2"] = projected.geometry.area
    gdf["perimeter_m"] = projected.geometry.length
    return gdf


def filter_bbox(gdf: gpd.GeoDataFrame, bbox: list[float]) -> gpd.GeoDataFrame:
    xmin, ymin, xmax, ymax = bbox
    return gdf.cx[xmin:xmax, ymin:ymax]


def write_any(
    gdf: gpd.GeoDataFrame,
    output: Path,
    driver: str | None,
    layer: str | None,
    encoding: str | None,
    parquet_compression: str,
    overwrite: bool,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        raise SystemExit(f"Output exists: {output}. Use --overwrite to replace.")

    ext = output.suffix.lower()
    if ext == ".parquet":
        gdf.to_parquet(output, compression=parquet_compression, index=False)
        return

    resolved_driver = driver or infer_driver_from_ext(output)
    kwargs: dict[str, object] = {}
    if resolved_driver == "ESRI Shapefile" and encoding:
        kwargs["encoding"] = encoding

    gdf.to_file(output, layer=layer, driver=resolved_driver, **kwargs)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    gdf = read_any(input_path, args.in_layer)

    if args.bbox:
        gdf = filter_bbox(gdf, args.bbox)

    if args.where:
        gdf = gdf.query(args.where)

    if args.fix:
        gdf = fix_geoms(gdf)
    if args.force_2d:
        gdf = drop_z(gdf)
    if args.explode:
        gdf = explode_multipart(gdf)

    if args.to_crs:
        gdf = gdf.to_crs(args.to_crs)

    if args.add_metrics:
        if gdf.crs is None and not args.metrics_crs:
            raise SystemExit("Cannot add metrics: input CRS unknown. Use --to-crs or --metrics-crs.")
        metrics_crs = (
            CRS.from_user_input(args.metrics_crs)
            if args.metrics_crs
            else auto_metrics_crs(gdf if gdf.crs else gdf.set_crs("EPSG:4326", allow_override=True))
        )
        gdf = add_metrics(gdf, metrics_crs)

    write_any(
        gdf,
        output_path,
        args.driver,
        args.out_layer,
        args.encoding,
        args.parquet_compression,
        args.overwrite,
    )
    print(f"✅ Wrote: {output_path}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
