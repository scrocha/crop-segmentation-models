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
    "agricultura": [18, 19, 39, 20, 40, 62, 41, 36, 46, 47, 35, 48],
    "pastagem": [15],
    "floresta": [1, 3, 4, 5, 6, 49],
    "silvicultura": [9],
    "area_urbanizada": [24],
    "corpos_dagua": [26, 33, 31],
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
            gdf['area_ha'] = gdf.geometry.area / 10000

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
