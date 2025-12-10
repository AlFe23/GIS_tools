import numpy as np
from osgeo import gdal
import scipy.ndimage

import numpy as np
from osgeo import gdal
import scipy.ndimage

def apply_gdal_sieve(
    input_raster_path: str,
    output_raster_path: str,
    mmu_area_m2: float,
    pixel_resolution: float,
    connectivity: int = 8,
    min_width: int = 2,
    min_width_required: bool = True
):
    """
    Apply the GDAL sieve filter to remove small features from a raster and
    optionally enforce a minimum width for each group, preserving input
    datatype, compression, NoData and metadata.
    """
    # 1) Open input and read basic info + metadata
    input_ds = gdal.Open(input_raster_path)
    if input_ds is None:
        raise RuntimeError(f"Cannot open {input_raster_path}")
    src_band = input_ds.GetRasterBand(1)
    band_desc = src_band.GetDescription()
    
    # recupera NoData, metadati e statistiche
    nodata = src_band.GetNoDataValue()
    ds_md = input_ds.GetMetadata()
    band_md = src_band.GetMetadata()
    stats = src_band.GetStatistics(True, True)  # [min, max, mean, std]

    gt = input_ds.GetGeoTransform()
    proj = input_ds.GetProjection()
    xsize = input_ds.RasterXSize
    ysize = input_ds.RasterYSize

    # 2) calcola soglia in pixel
    mmu_pixels = int(mmu_area_m2 / (pixel_resolution ** 2))

    # 3) crea output con le stesse caratteristiche
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(
        output_raster_path,
        xsize, ysize, 1,
        gdal.GDT_Int16,
        options=["COMPRESS=LZW"]
    )
    out_ds.SetGeoTransform(gt)
    out_ds.SetProjection(proj)
    out_ds.SetMetadata(ds_md)

    out_band = out_ds.GetRasterBand(1)
    out_band.SetDescription(band_desc)
    out_band.SetMetadata(band_md)
    out_band.SetNoDataValue(nodata)
    # ripristina statistiche
    out_band.SetStatistics(stats[0], stats[1], stats[2], stats[3])

    # 4) applica il filtro sieve
    gdal.SieveFilter(
        src_band, None, out_band,
        mmu_pixels, connectivity
    )

    # 5) opzionale: enforce minimum width
    if min_width_required:
        arr = out_band.ReadAsArray()
        labeled, cnt = scipy.ndimage.label(arr != nodata, structure=np.ones((3, 3)))
        for fid in range(1, cnt+1):
            mask = (labeled == fid)
            rows, cols = np.where(mask)
            h = rows.max() - rows.min() + 1
            w = cols.max() - cols.min() + 1
            if h < min_width or w < min_width:
                arr[mask] = nodata
        # riscrive e salva
        out_band.WriteArray(arr)

    # 6) chiudi e forza scrittura su disco
    out_band.FlushCache()
    out_ds.FlushCache()

    input_ds = None
    out_ds = None


# Usage example:
input_raster = r"D:\Lavoro\GMATICS\IRIDE_ForestMonitoring\phase2\S40701\abruzzo\result_merged.tif"
output_raster_mmu = r"D:\Lavoro\GMATICS\IRIDE_ForestMonitoring\phase2\S40701\abruzzo\result_merged_mmu.tif"
#output_raster_mmu_mw = r"D:\Lavoro\GMATICS\IRIDE_ForestMonitoring\phase2\results_gmatics\S40701\abruzzo\result_merged_mmu_mw.tif"





mmu_area = 5000  # MMU in square meters
pixel_res = 10   # Resolution in meters (for example, 10m per pixel)
min_width=2

# Call the function with min_width_required set to True (enforce the minimum width)
apply_gdal_sieve(input_raster, output_raster_mmu, mmu_area, pixel_res, min_width, min_width_required=False)
#apply_gdal_sieve(input_raster, output_raster_mmu_mw, mmu_area, pixel_res, min_width, min_width_required=True)



