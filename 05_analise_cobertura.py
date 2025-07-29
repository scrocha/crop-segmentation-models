import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.features import rasterize
import numpy as np
import os

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


def get_pixel_area_m2(src: rasterio.io.DatasetReader) -> float:
    pixel_area_m2 = abs(src.res[0] * src.res[1])

    if src.crs.is_geographic:
        lat_centro = (src.bounds.bottom + src.bounds.top) / 2
        fator_correcao = np.cos(np.radians(lat_centro))
        metros_por_grau = 111320  # Valor médio

        pixel_area_m2 = abs(
            (src.res[0] * metros_por_grau * fator_correcao)
            * (src.res[1] * metros_por_grau)
        )

    return pixel_area_m2


def calcular_area_agricola_total(
    aoi_shp_path: str, mapbiomas_raster_path: str
) -> float:
    aoi_gdf = gpd.read_file(aoi_shp_path)

    with rasterio.open(mapbiomas_raster_path) as src:
        if aoi_gdf.crs != src.crs:
            aoi_gdf = aoi_gdf.to_crs(src.crs)

        try:
            out_image, _ = mask(src, aoi_gdf.geometry, crop=True)
        except ValueError:
            return 0

        pixel_area_m2 = get_pixel_area_m2(src)
        agri_pixel_count = np.isin(
            out_image, CLASSES_MAPBIOMAS_AGRICULTURA
        ).sum()

        return (agri_pixel_count * pixel_area_m2) / 10000


def calcular_metricas_segmentacao(
    filtered_shp_path: str,
    mapbiomas_raster_path: str,
    area_total_agricola_ha: float,
) -> dict:
    gdf_filtrado = gpd.read_file(filtered_shp_path)

    with rasterio.open(mapbiomas_raster_path) as src:
        if gdf_filtrado.crs != src.crs:
            gdf_filtrado = gdf_filtrado.to_crs(src.crs)

        mapbiomas_array = src.read(1)
        mascara_referencia_agri = np.isin(
            mapbiomas_array, CLASSES_MAPBIOMAS_AGRICULTURA
        )

        geometries = gdf_filtrado.geometry
        mascara_segmentacao = rasterize(
            geometries,
            out_shape=mapbiomas_array.shape,
            transform=src.transform,
            fill=0,
            dtype='uint8',
        ).astype(bool)

        interseccao = mascara_segmentacao & mascara_referencia_agri

        pixel_area_ha = get_pixel_area_m2(src) / 10000

        area_segmentada_corretamente_ha = np.sum(interseccao) * pixel_area_ha
        area_total_segmentada_ha = np.sum(mascara_segmentacao) * pixel_area_ha

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
    MAPBIOMAS_RASTER = "./dados/mapbiomas_campo_verde.tif"

    print("Iniciando análise de cobertura da segmentação...")

    area_total_ha = calcular_area_agricola_total(AOI_SHP, MAPBIOMAS_RASTER)
    if area_total_ha == 0:
        print(
            "❌ ERRO CRÍTICO: Área agrícola de referência na AOI é zero. Verifique a sobreposição dos seus arquivos."
        )
        return

    metricas = calcular_metricas_segmentacao(
        FILTERED_SHP, MAPBIOMAS_RASTER, area_total_ha
    )

    print("\n" + "=" * 60)
    print("📊 RELATÓRIO DE DESEMPENHO DA SEGMENTAÇÃO")
    print("=" * 60)
    print(f"Área Agrícola de Referência (na AOI): {area_total_ha:.2f} ha")
    print("-" * 60)
    print(f"Recall (Cobertura): {metricas['recall']:.2f}%")
    print(f"Precisão (Qualidade): {metricas['precision']:.2f}%")
    print("=" * 60)
    print("\nLembretes:")
    print(
        "  - Recall: De toda a agricultura que existe, quantos % seu método encontrou."
    )
    print(
        "  - Precisão: De tudo que seu método encontrou, quantos % eram de fato agricultura."
    )


if __name__ == "__main__":
    main()
