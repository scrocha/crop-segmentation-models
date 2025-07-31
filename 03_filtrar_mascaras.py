import geopandas as gpd
import rasterio
from rasterio.mask import mask
import numpy as np
import os
from tqdm import tqdm
import zipfile
import argparse

CLASSES_MAPBIOMAS_AGRICULTURA = [
    18,  # Agricultura
    19,  # Agricultura Temporária
    39,  # Soja
    20,  # Cana
    40,  # Arroz
    62,  # Algodão
    41,  # Outras Culturas Temporárias
    36,  # Lavoura Perenne
    46,  # Café
    47,  # Citrus
    35,  # Dendê
    48,  # Outras Culturas Perenes
]


def get_agriculture_coverage(geometry, mapbiomas_raster):
    """
    Calcula o percentual de cobertura de classes agrícolas dentro de uma geometria.
    """
    try:
        with rasterio.open(mapbiomas_raster) as src:
            out_image, out_transform = mask(
                src, [geometry], crop=True, filled=True
            )

            if out_image.ndim == 3:
                out_image = out_image[0]

            total_pixels = np.count_nonzero(out_image)
            if total_pixels == 0:
                return 0.0

            agri_pixels = np.isin(
                out_image, CLASSES_MAPBIOMAS_AGRICULTURA
            ).sum()

            return (agri_pixels / total_pixels) * 100

    except Exception as e:
        if "Input shapes do not overlap raster" not in str(e):
            print(f"\n⚠️  Aviso ao processar geometria: {e}")
        return 0.0


def filtrar_mascaras(
    input_shp,
    output_shp,
    mapbiomas_raster,
    area_min_ha,
    area_max_ha,
    agri_pct_min,
):
    if not os.path.exists(input_shp):
        raise FileNotFoundError(
            f"❌ Shapefile de entrada não encontrado: {input_shp}"
        )
    if not os.path.exists(mapbiomas_raster):
        raise FileNotFoundError(
            f"❌ Raster MapBiomas não encontrado: {mapbiomas_raster}"
        )

    print("🔄 Iniciando filtragem de máscaras...")
    gdf = gpd.read_file(input_shp)
    initial_count = len(gdf)
    print(f"   - Polígonos iniciais: {initial_count}")

    gdf_filtrado = gdf[
        (gdf['area_ha'] >= area_min_ha) & (gdf['area_ha'] <= area_max_ha)
    ].copy()
    count_after_area = len(gdf_filtrado)
    print(f"   - Polígonos restantes: {count_after_area}")

    if gdf_filtrado.empty:
        return

    with rasterio.open(mapbiomas_raster) as src:
        if gdf_filtrado.crs != src.crs:
            gdf_filtrado = gdf_filtrado.to_crs(src.crs)

    coberturas = list()
    for geom in tqdm(
        gdf_filtrado.geometry, desc="Analisando cobertura MapBiomas"
    ):
        cobertura = get_agriculture_coverage(geom, mapbiomas_raster)
        coberturas.append(cobertura)

    gdf_filtrado['agri_pct'] = coberturas

    gdf_final = gdf_filtrado[gdf_filtrado['agri_pct'] >= agri_pct_min].copy()
    count_after_mapbiomas = len(gdf_final)
    print(f"   - Polígonos restantes: {count_after_mapbiomas}")

    if gdf_final.empty:
        return

    os.makedirs(os.path.dirname(output_shp), exist_ok=True)
    gdf_final.to_file(output_shp)


def main():
    INPUT_SHP = "./dados/sam2/campo_verde_mascaras.shp"
    OUTPUT_SHP_FILTRADO = "./dados/sam2/campo_verde_mascaras_filtradas.shp"
    MAPBIOMAS_RASTER = "./dados/mapbiomas_campo_verde.tif"

    AREA_MIN_HA = 15.0  # Área mínima de um talhão (ex: 15 hectares)
    AREA_MAX_HA = 200.0  # Área máxima de um talhão (ex: 200 hectares)
    AGRI_PCT_MIN = (
        80.0  # Mínimo de 80% do polígono deve ser agricultura no MapBiomas
    )

    filtrar_mascaras(
        input_shp=INPUT_SHP,
        output_shp=OUTPUT_SHP_FILTRADO,
        mapbiomas_raster=MAPBIOMAS_RASTER,
        area_min_ha=AREA_MIN_HA,
        area_max_ha=AREA_MAX_HA,
        agri_pct_min=AGRI_PCT_MIN,
    )

    name = OUTPUT_SHP_FILTRADO[:-4]

    for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
        file_path = f"{OUTPUT_SHP_FILTRADO[:-4]}{ext}"
        if not os.path.exists(file_path):
            print(f"⚠️ Arquivo {file_path} não encontrado para zipar.")
            return

    with zipfile.ZipFile(f"{name}.zip", 'w') as zipf:
        for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
            file_path = f"{OUTPUT_SHP_FILTRADO[:-4]}{ext}"
            zipf.write(file_path, arcname=os.path.basename(file_path))


if __name__ == "__main__":
    main()
