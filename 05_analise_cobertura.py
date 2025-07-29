# 07_avaliar_cobertura.py
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import numpy as np
import os

# Use a mesma lista de classes do script de filtragem para consistência
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

def calcular_area_agricola_total(aoi_shp_path, mapbiomas_raster_path):
    aoi_gdf = gpd.read_file(aoi_shp_path)

    with rasterio.open(mapbiomas_raster_path) as src:
        if aoi_gdf.crs != src.crs:
            aoi_gdf = aoi_gdf.to_crs(src.crs)

        try:
            out_image, out_transform = mask(src, aoi_gdf.geometry, crop=True)
        except ValueError as e:
            if "Input shapes do not overlap raster" in str(e):
                print(
                    "❌ ERRO: O shapefile da sua Área de Interesse (AOI) não se sobrepõe ao raster do MapBiomas."
                )
                return 0
            raise e

        pixel_area_m2 = src.res[0] * src.res[1]

        agri_pixel_count = np.isin(
            out_image, CLASSES_MAPBIOMAS_AGRICULTURA
        ).sum()

        total_agri_area_ha = (agri_pixel_count * pixel_area_m2) / 10000

    print(
        f"   - Área agrícola total na AOI (referência MapBiomas): {total_agri_area_ha:.2f} ha"
    )
    return total_agri_area_ha


def calcular_metricas_segmentacao(filtered_shp_path, area_total_agricola_ha):

    gdf_filtrado = gpd.read_file(filtered_shp_path)

    if gdf_filtrado.empty:
        print(
            "   - Shapefile de máscaras filtradas está vazio. Métricas serão zero."
        )
        return {"recall": 0, "precision": 0}

    if (
        'area_ha' not in gdf_filtrado.columns
        or 'agri_pct' not in gdf_filtrado.columns
    ):
        raise ValueError(
            "O shapefile filtrado deve conter as colunas 'area_ha' e 'agri_pct' do script 04."
        )

    gdf_filtrado['agri_area_ha'] = gdf_filtrado['area_ha'] * (
        gdf_filtrado['agri_pct'] / 100
    )
    area_segmentada_corretamente_ha = gdf_filtrado['agri_area_ha'].sum()

    area_total_segmentada_ha = gdf_filtrado['area_ha'].sum()

    print(
        f"   - Área total dos polígonos segmentados e filtrados: {area_total_segmentada_ha:.2f} ha"
    )
    print(
        f"   - Desses, a área que é de fato agricultura: {area_segmentada_corretamente_ha:.2f} ha"
    )

    recall = (
        (area_segmentada_corretamente_ha / area_total_agricola_ha) * 100
        if area_total_agricola_ha > 0
        else 0
    )

    precision = (
        (area_segmentada_corretamente_ha / area_total_segmentada_ha) * 100
        if area_total_segmentada_ha > 0
        else 0
    )

    return {"recall": recall, "precision": precision}


def main():
    AOI_SHP = "./dados/campo_verde.geojson"
    FILTERED_SHP = "./dados/campo_verde_mascaras_filtradas.shp"
    MAPBIOMAS_RASTER = "./dados/mapbiomas_10m_collection2_integration_v1-classification_2022.tif"

    area_total_ha = calcular_area_agricola_total(AOI_SHP, MAPBIOMAS_RASTER)

    if area_total_ha == 0:
        print("Não há área agrícola na AOI para calcular as métricas.")
        return

    metricas = calcular_metricas_segmentacao(FILTERED_SHP, area_total_ha)

    print("\n" + "=" * 60)
    print("📊 RELATÓRIO DE DESEMPENHO DA SEGMENTAÇÃO")
    print("=" * 60)
    print(f"Recall (Taxa de Detecção): {metricas['recall']:.2f}%")
    print(
        "   - Significado: Seu método foi capaz de segmentar {recall:.2f}% de toda a área agrícola disponível na região.".format(
            recall=metricas['recall']
        )
    )
    print("-" * 60)
    print(f"Precisão (Acurácia da Segmentação): {metricas['precision']:.2f}%")
    print(
        "   - Significado: {precision:.2f}% da área total que você segmentou corresponde de fato a classes de agricultura.".format(
            precision=metricas['precision']
        )
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
