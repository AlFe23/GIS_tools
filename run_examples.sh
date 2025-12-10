#!/usr/bin/env bash
# Utility entry point for running common GIS_tools scripts with example arguments.
# Activate the conda environment before running: `conda activate gis_tools`.

set -euo pipefail

echo "Running GSA pansharpening example..."
python scripts/gsa_pansharpening.py \
  --ms /mnt/c/Users/AlviseFerrari/Documents/GMATICS_laptop/MAPOSAT/F3/img_a_catalogo/MAXAR_WV/25EUSI-1694-06_182058/25EUSI-1694-06_182058_MUL/WorldView2_2024_11_03_10_13_18/24NOV03101314-M2AS-25EUSI-1694-06_182058.TIF \
  --pan /mnt/c/Users/AlviseFerrari/Documents/GMATICS_laptop/MAPOSAT/F3/img_a_catalogo/MAXAR_WV/25EUSI-1694-06_182058/25EUSI-1694-06_182058_PAN/24NOV03101314-P2AS-25EUSI-1694-06_182058.TIF \
  --out /mnt/c/Users/AlviseFerrari/Documents/GMATICS_laptop/MAPOSAT/F3/img_a_catalogo/MAXAR_WV/25EUSI-1694-06_182058/25EUSI-1694-06_182058_MUL/WorldView2_2024_11_03_10_13_18/24NOV03101314-M2AS-25EUSI-1694-06_182058_pansharp.tif

echo "Running GSA pansharpening example (29 Aug 2024 acquisition)..."
python scripts/gsa_pansharpening.py \
  --ms /mnt/c/Users/AlviseFerrari/Documents/GMATICS_laptop/MAPOSAT/F3/img_a_catalogo/MAXAR_WV/25EUSI-1694-06_182058/25EUSI-1694-06_182058_MUL/WorldView2_2024_08_29_09_57_12/24AUG29095707-M2AS-25EUSI-1694-06_182058.TIF \
  --pan /mnt/c/Users/AlviseFerrari/Documents/GMATICS_laptop/MAPOSAT/F3/img_a_catalogo/MAXAR_WV/25EUSI-1694-06_182058/25EUSI-1694-06_182058_PAN/24AUG29095707-P2AS-25EUSI-1694-06_182058.TIF \
  --out /mnt/c/Users/AlviseFerrari/Documents/GMATICS_laptop/MAPOSAT/F3/img_a_catalogo/MAXAR_WV/25EUSI-1694-06_182058/25EUSI-1694-06_182058_MUL/WorldView2_2024_08_29_09_57_12/24AUG29095707-M2AS-25EUSI-1694-06_182058_pansharp.tif
