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

## Contributing

Pull requests are welcome. Please open an issue first to discuss any major changes.
