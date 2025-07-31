import geopandas as gpd
import rasterio
from rasterio.mask import mask
import numpy as np
import os
from tqdm import tqdm
import zipfile

INPUT_SHP = "./dados/sam2/campo_verde_mascaras_filtradas.shp"
IMAGEM_RASTER = "./dados/campo_verde_merged_clip.tif"
OUTPUT_SHP = "./dados/sam2/campo_verde_talhoes_com_heterogeneidade.shp"

BANDA_VERMELHO = 3
BANDA_NIR = 1


def calcular_ndvi_stats(geometry, raster_path):
    try:
        with rasterio.open(raster_path) as src:
            out_image, _ = mask(src, [geometry], crop=True, filled=False)

            if out_image.size == 0 or out_image[0].size == 0:
                return None

            red = out_image[BANDA_VERMELHO - 1].astype(np.float32)
            nir = out_image[BANDA_NIR - 1].astype(np.float32)

            ndvi = np.ma.masked_invalid((nir - red) / (nir + red + 1e-6))

            pixels_validos = ndvi[
                ~ndvi.mask & (ndvi >= -1) & (ndvi <= 1)
            ].compressed()

            if pixels_validos.size < 10:
                return None
            stats = {
                'ndvi_mean': np.mean(pixels_validos),
                'ndvi_std': np.std(pixels_validos),
                'ndvi_cv': (
                    (np.std(pixels_validos) / np.mean(pixels_validos)) * 100
                    if np.mean(pixels_validos) != 0
                    else 0
                ),
                'ndvi_p10': np.percentile(pixels_validos, 10),
                'ndvi_p90': np.percentile(pixels_validos, 90),
            }
            return stats

    except Exception:
        return None


def main():
    if not os.path.exists(INPUT_SHP) or not os.path.exists(IMAGEM_RASTER):
        print(
            "❌ ERRO: Arquivo de entrada (SHP ou TIF) não encontrado. Verifique os caminhos."
        )
        return

    gdf = gpd.read_file(INPUT_SHP)

    with rasterio.open(IMAGEM_RASTER) as src:
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)

    print(
        f"🚀 Processando {len(gdf)} talhões para calcular estatísticas NDVI..."
    )
    stats_list = [
        calcular_ndvi_stats(row.geometry, IMAGEM_RASTER)
        for _, row in tqdm(
            gdf.iterrows(), total=gdf.shape[0], desc="Analisando talhões"
        )
    ]

    stats_df = gpd.GeoDataFrame(stats_list, index=gdf.index)
    gdf = gdf.join(stats_df)

    valores_validos = gdf['ndvi_mean'].notna().sum()
    print(
        f"\n✅ Análise concluída. {valores_validos} de {len(gdf)} talhões processados com sucesso."
    )

    if valores_validos == 0:
        print("⚠️ Nenhum talhão produziu estatísticas válidas.")
        return

    gdf.dropna(subset=['ndvi_mean'], inplace=True)

    print("\n📊 Resumo das Estatísticas NDVI Calculadas:")
    colunas_stats = [
        'ndvi_mean',
        'ndvi_std',
        'ndvi_cv',
        'ndvi_p10',
        'ndvi_p90',
    ]
    print(gdf[colunas_stats].describe().round(3))

    gdf.to_file(OUTPUT_SHP)
    print(
        f"\n💾 Shapefile com análise de heterogeneidade salvo em: {OUTPUT_SHP}"
    )

    for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
        file_path = f"{OUTPUT_SHP[:-4]}{ext}"
        if not os.path.exists(file_path):
            print(f"⚠️ Arquivo {file_path} não encontrado para zipar.")
            return

    with zipfile.ZipFile(f"{OUTPUT_SHP[:-4]}.zip", 'w') as zipf:
        for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
            file_path = f"{OUTPUT_SHP[:-4]}{ext}"
            zipf.write(file_path, arcname=os.path.basename(file_path))


if __name__ == "__main__":
    main()
