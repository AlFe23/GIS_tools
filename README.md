# GIS Tools

A collection of utilities for working with geospatial datasets. The project is currently scaffolded with example scripts and documentation placeholders.

## Structure

- `scripts/`: Python scripts for data conversion and processing.
- `examples/`: Example notebooks or walkthroughs demonstrating how to use the tools.

## Getting Started

1. Create the conda environment:
   ```bash
   conda env create -f environment.yml
   ```
2. Activate the environment:
   ```bash
   conda activate gis-tools
   ```
3. Install any additional dependencies by editing `requirements.txt`.

## Scripts

- **`scripts/zarr_to_geotiff.py`**  
  Converts 2D/3D geospatial arrays stored in Zarr datasets into GeoTIFF rasters.
- **`scripts/vector_convert.py`**  
  Universal vector-format converter supporting Shapefile, GeoPackage, GeoJSON, and GeoParquet.

### `zarr_to_geotiff.py` options

| Flag | Description |
| --- | --- |
| `zarr` | Path to the source Zarr store. |
| `out_tif` | Output GeoTIFF path. |
| `--var` | Name of the data variable to export (auto-detects if omitted). |
| `--crs` | Override CRS written to the GeoTIFF (e.g. `EPSG:4326`). |
| `--bounds` | Crop bounds `xmin ymin xmax ymax`. |
| `--dtype` | Cast output to a specific dtype (e.g. `float32`, `int16`). |
| `--nodata` | Set the output nodata value. |
| `--compress` | GeoTIFF compression algorithm (`LZW` default). |
| `--tiled` | Create tiled GeoTIFF (`YES` default). |
| `--bigtiff` | BIGTIFF option (`IF_SAFER` default). |
| `--pyramids` | Build overviews for faster map display. |
| `--resampling` | Resampling method for overviews (`nearest` default). |
| `--consolidated` | Open Zarr with consolidated metadata. |

### Usage examples

Convert a Zarr store into a tiled, compressed GeoTIFF:

```bash
python scripts/zarr_to_geotiff.py \
  /data/dem.zarr \
  outputs/dem.tif
```

Select a specific variable, crop by bounds, and set nodata:

```bash
python scripts/zarr_to_geotiff.py \
  /data/mosaic.zarr \
  outputs/mosaic_subset.tif \
  --var ndvi \
  --bounds -120.5 35.0 -120.0 35.5 \
  --nodata -9999
```

Generate overviews with cubic resampling and enforce EPSG:32632:

 ```bash
 python scripts/zarr_to_geotiff.py \
  /data/scene.zarr \
  outputs/scene.tif \
  --crs EPSG:32632 \
  --pyramids \
  --resampling cubic
 ```

### `vector_convert.py` options

| Flag | Description |
| --- | --- |
| `input` | Source vector dataset path. |
| `output` | Target dataset path (driver inferred from extension). |
| `--in-layer` | Layer name to read from multi-layer sources (e.g. GeoPackage). |
| `--out-layer` | Target layer name for multi-layer outputs. |
| `--driver` | Explicit output driver name. |
| `--to-crs` | Reproject output to the specified CRS. |
| `--encoding` | Output encoding for Shapefiles. |
| `--explode` | Explode multipart geometries into single-part features. |
| `--fix` | Attempt to fix invalid geometries (`buffer(0)`). |
| `--force-2d` | Drop Z values from geometries. |
| `--bbox` | Bounding box filter `xmin ymin xmax ymax` in source CRS. |
| `--where` | Attribute filter using a pandas query string. |
| `--add-metrics` | Add `area_m2` and `perimeter_m` fields. |
| `--metrics-crs` | CRS used for metrics (auto-picked UTM if omitted). |
| `--parquet-compression` | Compression codec for GeoParquet outputs. |
| `--overwrite` | Replace existing output file. |

### `vector_convert.py` usage examples

Convert a Shapefile into GeoParquet:

```bash
python scripts/vector_convert.py data/input.shp outputs/input.parquet
```

Reproject, add area/perimeter, and write to GeoPackage:

```bash
python scripts/vector_convert.py \
  data/roads.shp \
  outputs/roads.gpkg \
  --to-crs EPSG:32632 \
  --add-metrics
```

Filter by bounding box, fix geometries, drop Z, and save as GeoJSON:

```bash
python scripts/vector_convert.py \
  data/features.parquet \
  outputs/features.geojson \
  --bbox 12.2 41.7 12.7 42.0 \
  --fix \
  --force-2d
```

## Contributing

Pull requests are welcome. Please open an issue first to discuss any major changes.
