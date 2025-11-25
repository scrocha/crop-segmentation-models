import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
from rasterio.features import shapes
import rasterio
import os
import glob
import zipfile
from tqdm import tqdm
from pathlib import Path

MASKS_DIR = "./dados/mascaras_campo_verde_sam2"
PATCHES_DIR = "./dados/patches_campo_verde"
OUTPUT_SHP = "./dados/sam2/campo_verde_mascaras.shp"
AREA_MIN = 100  # metros quadrados


def converter_npz_para_shp():
    if not os.path.exists(MASKS_DIR):
        print(f"Diretório de máscaras não encontrado: {MASKS_DIR}")
        return

    if not os.path.exists(PATCHES_DIR):
        print(f"Diretório de patches não encontrado: {PATCHES_DIR}")
        return

    npz_files = glob.glob(os.path.join(MASKS_DIR, "*.npz"))
    if not npz_files:
        print(f"Nenhum arquivo NPZ encontrado em {MASKS_DIR}")
        return

    print(f"Encontrados {len(npz_files)} arquivos NPZ para processar")

    todas_geometrias = []
    crs_final = None

    for npz_path in tqdm(npz_files, desc="Processando arquivos NPZ"):
        try:
            npz_name = Path(npz_path).stem
            if npz_name.endswith('_masks'):
                patch_name = npz_name[:-6]
            else:
                patch_name = npz_name

            patch_path = os.path.join(PATCHES_DIR, f"{patch_name}.tif")

            if not os.path.exists(patch_path):
                print(f"Patch não encontrado: {patch_path}")
                continue

            with rasterio.open(patch_path) as src:
                transform = src.transform
                crs = src.crs
                if crs_final is None:
                    crs_final = crs

            data = np.load(npz_path)
            if 'masks' in data:
                masks = data['masks']
            else:
                masks = data[list(data.keys())[0]]

            for mask_id, mask_array in enumerate(masks):
                mask_binary = mask_array.astype(np.uint8)
                polygons = list(
                    shapes(mask_binary, mask=mask_binary, transform=transform)
                )

                for i, (polygon, value) in enumerate(polygons):
                    if value == 1:  # Região da máscara
                        try:
                            geom = Polygon(polygon['coordinates'][0])

                            if not geom.is_valid:
                                geom = geom.buffer(0)

                            if geom.is_valid and geom.area >= AREA_MIN:
                                todas_geometrias.append(
                                    {
                                        'geometry': geom,
                                        'mask_id': f"{patch_name}_{mask_id}_{i}",
                                        'patch_orig': patch_name,
                                        'area_ha': geom.area / 10000,
                                    }
                                )

                        except Exception as e:
                            continue

        except Exception as e:
            print(f"Erro ao processar {npz_path}: {e}")
            continue

    if not todas_geometrias:
        print("Nenhuma geometria válida foi gerada!")
        return

    print(f"Total de polígonos gerados: {len(todas_geometrias)}")

    geometries = [g['geometry'] for g in todas_geometrias]
    attributes = {
        'mask_id': [g['mask_id'] for g in todas_geometrias],
        'patch_orig': [g['patch_orig'] for g in todas_geometrias],
        'area_ha': [g['area_ha'] for g in todas_geometrias],
    }

    gdf = gpd.GeoDataFrame(attributes, geometry=geometries, crs=crs_final)

    os.makedirs(os.path.dirname(OUTPUT_SHP), exist_ok=True)

    gdf.to_file(OUTPUT_SHP)

    print(f"Polígonos salvos: {len(gdf)}")
    print(f"Área total: {gdf['area_ha'].sum():.2f} hectares")
    print(f"Área média: {gdf['area_ha'].mean():.2f} hectares")
    print(f"Shapefile salvo: {OUTPUT_SHP}")

def zip_files():
    name = OUTPUT_SHP[:-4]

    for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
        file_path = f"{OUTPUT_SHP[:-4]}{ext}"
        if not os.path.exists(file_path):
            print(f"Arquivo {file_path} não encontrado para zipar.")
            return

    with zipfile.ZipFile(f"{name}.zip", 'w') as zipf:
        for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
            file_path = f"{OUTPUT_SHP[:-4]}{ext}"
            zipf.write(file_path, arcname=os.path.basename(file_path))


if __name__ == "__main__":
    converter_npz_para_shp()
    zip_files()
