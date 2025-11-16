"""Basic test for mosaic functionality (creates small rasters and mosaics them).

This test is a lightweight sanity check and requires rasterio to be
installed in the environment running tests.
"""
import tempfile
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_origin
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from mosaic_rasters import mosaic_rasters  # type: ignore


def _create(path, data, transform, crs="EPSG:4326"):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype.name,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data, 1)


def test_mosaic_tmp(tmp_path):
    # create two overlapping rasters
    a = np.ones((20, 20), dtype="uint8") * 5
    b = np.ones((20, 20), dtype="uint8") * 10
    t1 = from_origin(0, 20, 1, 1)
    t2 = from_origin(10, 10, 1, 1)

    p1 = tmp_path / "a.tif"
    p2 = tmp_path / "b.tif"
    out = tmp_path / "mosaic.tif"

    _create(p1, a, t1)
    _create(p2, b, t2)

    mosaic_rasters([str(p1), str(p2)], str(out), target_crs="EPSG:4326", overwrite=True)

    assert out.exists()
