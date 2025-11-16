# GIS Tools

A collection of utilities for working with geospatial datasets. The project includes tools for data conversion, processing, and image enhancement.

## Structure

- `scripts/`: Python scripts for data conversion and processing.
- `examples/`: Example notebooks and scripts demonstrating how to use the tools.

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

## Available Tools

### Image Enhancement Tools

The repository includes three main scripts for WorldView image enhancement:

1. **`scripts/gsa_pansharpening.py`**: Implements the GSA (Gram-Schmidt Adaptive) pansharpening algorithm
2. **`scripts/coregistration.py`**: Handles image coregistration using AROSICS
3. **`scripts/worldview_enhance.py`**: Complete pipeline combining pansharpening and coregistration

#### Basic Usage

**Pansharpening only:**
```bash
python scripts/gsa_pansharpening.py \
  --ms path/to/multispectral.tif \
  --pan path/to/panchromatic.tif \
  --out path/to/output.tif
```

**Coregistration only:**
```bash
python scripts/coregistration.py \
  --reference path/to/reference.tif \
  --target path/to/target.tif \
  --out path/to/output.tif \
  --grid-res 20 \
  --max-shift 10
```

**Complete pipeline:**
```bash
python scripts/worldview_enhance.py \
  --ms path/to/multispectral.tif \
  --pan path/to/panchromatic.tif \
  --reference path/to/reference.tif \
  --out-prefix path/to/output
```

#### Advanced Usage

The pipeline can be customized to:
- Run coregistration before pansharpening: `--coregister-first`
- Skip pansharpening: `--skip-pansharpening`
- Skip coregistration: `--skip-coregistration`

Example with coregistration first:
```bash
python scripts/worldview_enhance.py \
  --ms path/to/multispectral.tif \
  --pan path/to/panchromatic.tif \
  --reference path/to/reference.tif \
  --out-prefix path/to/output \
  --coregister-first
```

## Scripts

- **`scripts/zarr_to_geotiff.py`**  
  Converts 2D/3D geospatial arrays stored in Zarr datasets into GeoTIFF rasters.
- **`scripts/vector_convert.py`**  
  Universal vector-format converter for Shapefile, GeoPackage, GeoJSON, and GeoParquet. It preserves attributes, optionally repairs geometries, reprojects data, filters features, and can append metric fields in meters using an automatically selected projected CRS.

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

#### Flusso di base

Convertire uno shapefile in GeoParquet mantenendo attributi e CRS:

```bash
python scripts/vector_convert.py data/input.shp outputs/input.parquet
```

Se l’origine contiene più layer (GeoPackage, FileGDB), specificare il layer di ingresso/uscita:

```bash
python scripts/vector_convert.py data/source.gpkg outputs/roads.gpkg --in-layer roads --out-layer roads_clean
```

#### Reproiezione e metriche

Calcolare area e perimetro (m²/m) in un CRS proiettato: se non viene indicato `--metrics-crs`, lo script sceglie automaticamente la zona UTM più adatta in base al baricentro del dataset.

```bash
python scripts/vector_convert.py \
  data/landuse.geojson \
  outputs/landuse_metrics.gpkg \
  --to-crs EPSG:32632 \
  --add-metrics
```

Per forzare un CRS specifico per i calcoli:

```bash
python scripts/vector_convert.py \
  data/landuse.geojson \
  outputs/landuse_metrics.gpkg \
  --add-metrics \
  --metrics-crs EPSG:3035
```

#### Pulizia geometrie e filtraggio

Riparare geometrie non valide con `--fix`, ridurre a 2D (rimozione quota) e dividere multipart in feature singole:

```bash
python scripts/vector_convert.py \
  data/features.parquet \
  outputs/features_2d.geojson \
  --fix \
  --force-2d \
  --explode
```

Filtrare per bounding box nello stesso CRS della sorgente e applicare un filtro attributi via Pandas:

```bash
python scripts/vector_convert.py \
  data/points.gpkg \
  outputs/points_filtered.gpkg \
  --bbox 12.20 41.70 12.70 42.00 \
  --where "population > 10000 and status == 'city'"
```

#### Formati e codifiche

Salvare GeoJSON in WGS84 forzando la reproiezione:

```bash
python scripts/vector_convert.py data/buildings.parquet outputs/buildings.geojson --to-crs EPSG:4326
```

Esportare in shapefile con encoding Latin-1 e sovrascrivere file esistente:

```bash
python scripts/vector_convert.py \
  data/zoning.gpkg \
  outputs/zoning.shp \
  --encoding LATIN1 \
  --overwrite
```

## Contributing

Pull requests are welcome. Please open an issue first to discuss any major changes.
