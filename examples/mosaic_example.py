"""Small example that creates two synthetic GeoTIFFs and mosaics them.

Run from the repository root:
    python examples/mosaic_example.py
"""
from pathlib import Path
import tempfile
import numpy as np
import rasterio
from rasterio.transform import from_origin
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from mosaic_rasters import mosaic_rasters  # type: ignore


def _create_test_tif(path, arr, transform, crs="EPSG:4326", dtype="float32", nodata=None):
    h, w = arr.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype=dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        if nodata is not None:
            dst.nodata = nodata
        dst.write(arr.astype(dtype), 1)


def main():
    tmp = Path(tempfile.mkdtemp(prefix="mosaic_example_"))
    a1 = np.ones((100, 100), dtype="float32") * 10
    a2 = np.ones((100, 100), dtype="float32") * 20

    t1 = from_origin(-1.0, 1.0, 0.01, 0.01)
    t2 = from_origin(-0.5, 0.5, 0.01, 0.01)

    p1 = tmp / "t1.tif"
    p2 = tmp / "t2.tif"
    out = tmp / "mosaic.tif"

    _create_test_tif(p1, a1, t1)
    _create_test_tif(p2, a2, t2)

    print("Created test rasters:", p1, p2)
    mosaic_rasters([str(p1), str(p2)], str(out), target_crs="EPSG:4326", overwrite=True)
    print("Mosaic written to:", out)


if __name__ == "__main__":
    main()
