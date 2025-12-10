import os
from osgeo import ogr, osr, gdal
from arosics import COREG_LOCAL

# Percorso dell'immagine di riferimento
reference_image_path = r"D:\Lavoro\GMATICS\IRIDE_ForestMonitoring\phase2\IMMAGINI_PRISMA\output\Campania\S2B_MSIL2A_20220815T094549_N0400_R079_T33TVF_20220815T113129.tif"

# Lista delle immagini target da coregistrare
target_images = [
    r"D:\Lavoro\GMATICS\IRIDE_ForestMonitoring\phase2\IMMAGINI_PRISMA\output\Campania\PRS_L2D_STD_20220829095830_20220829095834_0001_reflectance.tif",
]

# Directory di output per le immagini coregistrate
output_dir = r"D:\Lavoro\GMATICS\IRIDE_ForestMonitoring\phase2\IMMAGINI_PRISMA\output\Campania"
os.makedirs(output_dir, exist_ok=True)

# Parametri base per la coregistrazione.
# Importante: impostiamo projectDir pari a output_dir in modo che il file venga scritto nella cartella desiderata.
base_kwargs = {
    'grid_res': 50,
    'window_size': (256, 256),
    'projectDir': output_dir,  # Impostiamo qui la directory di output
    'q': False,               # Modalità silenziosa
    'max_shift': 15,          # Aumento dello shift massimo
    'r_b4match': 3,           # Uso della banda 2 (banda rossa)
    's_b4match': 21,
    'max_iter': 10,           # Incremento del numero di iterazioni a 10
    'align_grids': True,      # Abilitazione dell'allineamento a griglia
    'fmt_out': "GTiff",       # Output in formato GeoTIFF
    'ignore_errors': True
}

def reproject_if_needed(ref_path, target_path, output_directory):
    """Verifica il CRS delle immagini e, se necessario, riproietta target_path nel CRS della referenza."""
    ref_ds = gdal.Open(ref_path)
    target_ds = gdal.Open(target_path)
    if ref_ds is None or target_ds is None:
        raise Exception("Impossibile aprire uno dei file.")

    # Ottieni le proiezioni in formato WKT
    ref_proj = ref_ds.GetProjection()
    target_proj = target_ds.GetProjection()

    # Crea oggetti SpatialReference
    ref_sr = osr.SpatialReference()
    ref_sr.ImportFromWkt(ref_proj)
    target_sr = osr.SpatialReference()
    target_sr.ImportFromWkt(target_proj)

    # Se i CRS non sono uguali, riproietta l'immagine target
    if not ref_sr.IsSame(target_sr):
        reprojected_target_path = os.path.join(output_directory, "reproj_" + os.path.basename(target_path))
        print(f"Riproiezione di {os.path.basename(target_path)} nel CRS della referenza...")
        gdal.Warp(reprojected_target_path, target_path, dstSRS=ref_sr.ExportToWkt())
        return reprojected_target_path
    else:
        return target_path

# Ciclo principale
if __name__ == "__main__":
    for target_image in target_images:
        try:
            # Verifica e, se necessario, riproietta l'immagine target
            updated_target_image = reproject_if_needed(reference_image_path, target_image, output_dir)

            # Costruiamo il percorso di output desiderato.
            base_name = os.path.basename(updated_target_image)
            base_no_ext, _ = os.path.splitext(base_name)
            final_output_path = os.path.join(output_dir, base_no_ext + '_coregistered.tif')
            
            # Copia dei parametri base e impostazione del percorso di output definitivo
            kwargs = base_kwargs.copy()
            kwargs['path_out'] = final_output_path
            
            # Inizializza COREG_LOCAL con i parametri definiti
            CRL = COREG_LOCAL(reference_image_path, updated_target_image, **kwargs)
            
            # Esegue la coregistrazione: il file verrà salvato direttamente in final_output_path
            CRL.correct_shifts()
            
            if os.path.exists(final_output_path):
                print(f"Coregistrazione completata per {os.path.basename(target_image)}. Output salvato in {final_output_path}")
            else:
                print("Errore: il file di output non è stato creato.")
            
            # Se è stato creato un file intermedio (riproiettato), lo eliminiamo
            if updated_target_image != target_image:
                try:
                    os.remove(updated_target_image)
                    print(f"File intermedio {updated_target_image} eliminato.")
                except Exception as err:
                    print(f"Impossibile eliminare il file intermedio {updated_target_image}: {err}")
                    
        except Exception as e:
            print(f"Errore durante l'elaborazione di {os.path.basename(target_image)}: {e}")
