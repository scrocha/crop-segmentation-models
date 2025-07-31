# 08_gerar_vetores_mapbiomas_simplificado.py
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import Polygon
import numpy as np
import os
from tqdm import tqdm

RASTER_MAPBIOMAS = "./dados/mapbiomas_campo_verde.tif"
OUTPUT_DIR = "./dados/mapbiomas_vetorizado"

GRUPOS_MAPBIOMAS = {
    "agricultura_temporaria": [
        19,
        20,
        39,
        40,
        41,
        62,
    ],  # lavoura temporária e suas subclasses
    "agricultura_perene": [
        36,
        35,
        46,
        47,
        48,
    ],  # lavoura perene e suas subclasses
    "pastagem": [15],  # pastagem
    "silvicultura": [9],  # silvicultura
    "floresta": [1, 3, 4, 5, 6, 49],  # floresta e subclasses
    "vegetacao": [10, 11, 12, 32, 29, 50],  # vegetação herbácea e arbustiva
    "area_urbanizada": [24],  # área urbana
    "area_nao_vegetada": [
        22,
        23,
        25,
        30,
    ],  # áreas não vegetadas (praias, mineração etc.)
    "mosaico_usos": [21],  # mosaico de usos
    "corpos_dagua": [26, 31, 33],  # rios, lagos, aquicultura
    "nao_observado": [27],  # não observado
}


def vetorizar_classes(raster_path, class_ids, group_name, output_dir):
    print(f"\n🔄 Processando grupo: {group_name}")

    try:
        with rasterio.open(raster_path) as src:
            mapbiomas_array = src.read(1)

            mascara_binaria = np.isin(mapbiomas_array, class_ids).astype(
                np.uint8
            )

            if np.sum(mascara_binaria) == 0:
                print(
                    f"   - ⚠️ Nenhuma área encontrada para o grupo '{group_name}'. Pulando."
                )
                return

            results = (
                {'properties': {'raster_val': v}, 'geometry': s}
                for i, (s, v) in enumerate(
                    shapes(
                        mascara_binaria,
                        mask=mascara_binaria,
                        transform=src.transform,
                    )
                )
            )

            geoms = list(results)
            if not geoms:
                print(
                    f"   - ⚠️ Não foi possível gerar polígonos para o grupo '{group_name}'."
                )
                return

            gdf = gpd.GeoDataFrame.from_features(geoms, crs=src.crs)

            gdf['classe'] = group_name

            gdf['pixels'] = np.sum(mascara_binaria)
            gdf['area_m2'] = gdf['pixels'] * 100
            gdf['area_ha'] = gdf['area_m2'] / 10000

            output_path = os.path.join(
                output_dir, f"mapbiomas_{group_name}.shp"
            )
            gdf.to_file(output_path)

            print(f"   - ✅ Shapefile salvo em: {output_path}")
            print(f"   - Polígonos gerados: {len(gdf)}")

    except Exception as e:
        print(f"   - ❌ Erro ao processar o grupo '{group_name}': {e}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for nome_grupo, ids_de_classe in GRUPOS_MAPBIOMAS.items():
        vetorizar_classes(
            RASTER_MAPBIOMAS, ids_de_classe, nome_grupo, OUTPUT_DIR
        )

    print("\nProcessamento concluído!")


if __name__ == "__main__":
    main()
