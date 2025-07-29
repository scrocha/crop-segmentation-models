import geopandas as gpd
import rasterio
import pandas as pd
from rasterio.mask import mask
import numpy as np
import os
from tqdm import tqdm


def calcular_ndvi_stats_avancado(geometry, patch_path):
    try:
        with rasterio.open(patch_path) as src:
            out_image, out_transform = mask(
                src, [geometry], crop=True, filled=False
            )

            red = out_image[0].astype(float)
            nir = out_image[2].astype(float)

            ndvi = np.ma.masked_invalid((nir - red) / (nir + red + 1e-6))

            pixels_validos = ndvi[~ndvi.mask].compressed()

            if pixels_validos.size < 10:
                return None

            mean_ndvi = np.mean(pixels_validos)
            std_ndvi = np.std(pixels_validos)

            cv_ndvi = (std_ndvi / mean_ndvi) * 100 if mean_ndvi != 0 else 0

            p10_ndvi = np.percentile(pixels_validos, 10)
            p90_ndvi = np.percentile(pixels_validos, 90)

            stats = {
                'ndvi_mean': float(mean_ndvi),
                'ndvi_std': float(std_ndvi),
                'ndvi_cv': float(cv_ndvi),
                'ndvi_p10': float(p10_ndvi),
                'ndvi_p90': float(p90_ndvi),
            }

            return stats

    except Exception as e:
        if "Input shapes do not overlap raster" not in str(e):
            print(f"\n❌ Erro ao processar geometria: {e}")
        return None


def main():
    input_shp = "./dados/campo_verde_mascaras_filtradas.shp"
    patches_dir = "./dados/patches_campo_verde"
    output_shp = "./dados/campo_verde_talhoes_com_heterogeneidade.shp"

    print(f"Iniciando análise de heterogeneidade para: {input_shp}")

    gdf = gpd.read_file(input_shp)

    if 'patch_orig' not in gdf.columns:
        raise ValueError(
            "A coluna 'patch_orig' não foi encontrada no shapefile."
        )

    stats_list = list()

    for idx, row in tqdm(
        gdf.iterrows(), total=gdf.shape[0], desc="Analisando talhões"
    ):
        patch_name = row['patch_orig']
        patch_path = os.path.join(patches_dir, f"{patch_name}.tif")

        if not os.path.exists(patch_path):
            stats_list.append(dict())
            continue

        stats = calcular_ndvi_stats_avancado(row['geometry'], patch_path)
        stats_list.append(stats if stats is not None else dict())

    stats_df = pd.DataFrame(stats_list, index=gdf.index)
    gdf = gdf.join(stats_df)

    gdf.dropna(subset=['ndvi_mean'], inplace=True)

    print("\nAnálise concluída.")

    print("Estatísticas de Heterogeneidade e Saúde dos Talhões:")
    colunas_stats = [
        'ndvi_mean',
        'ndvi_std',
        'ndvi_cv',
        'ndvi_p10',
        'ndvi_p90',
    ]
    print(gdf[colunas_stats].describe().round(3))

    gdf.to_file(output_shp)
    print(f"\nShapefile com análise avançada salvo em: {output_shp}")


if __name__ == "__main__":
    main()
