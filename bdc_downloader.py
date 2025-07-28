# -*- coding: utf-8 -*-
"""
Módulo bdc_downloader

Fornece funções para consumir a API STAC do BDC usando pystac-client, baixar assets
satélite e processar imagens, além de obter geometrias de estados e municípios brasileiros via geobr.
"""

import zipfile
from pystac_client.item_search import Item
from pystac_client import Client
import geopandas as gpd
from geobr import read_state, read_municipality
from urllib.request import urlretrieve
from datetime import datetime, timedelta
from typing import Iterator, List, Dict, Optional, Union
import os
from pathlib import Path
import rasterio
from rasterio.mask import mask
import concurrent.futures

BDC_API_URL = "https://data.inpe.br/bdc/stac/v1"  # URL base da API STAC do BDC


def get_state_data(
    code: str, year: int = 2020, simplified: bool = True
) -> gpd.GeoDataFrame:
    """
    Retorna GeoDataFrame com a geometria do estado brasileiro especificado.

    Args:
        code: Código/federativo (sigla) do estado, ex: 'SP'.
        year: Ano de referência para os dados (padrão: 2020).
        simplified: Se True, retorna geometria simplificada (padrão: True).

    Returns:
        GeoDataFrame contendo a forma do estado.

    Exemplo:
        >>> estados = get_state_data('SP')
        >>> estados.plot()
    """
    return read_state(code_state=code, year=year, simplified=simplified)


def get_municipality_data(
    code: str, year: int = 2020, simplified: bool = True
) -> gpd.GeoDataFrame:
    """
    Retorna GeoDataFrame com a geometria do município brasileiro especificado.

    Args:
        code: Código IBGE do município.
        year: Ano de referência para os dados (padrão: 2020).
        simplified: Se True, retorna geometria simplificada (padrão: True).

    Returns:
        GeoDataFrame contendo a forma do município.

    Exemplo:
        >>> municipios = get_municipality_data('3550308')
        >>> municipios.plot()
    """
    return read_municipality(code_muni=code, year=year, simplified=simplified)


def get_stac_client(url: Optional[str] = None) -> Client:
    """
    Cria e retorna um cliente STAC para a API do BDC (ou outra informada).

    Args:
        url: URL da API STAC (opcional, padrão: BDC).

    Returns:
        Client: Instância do cliente pystac-client.

    Exemplo:
        >>> client = get_stac_client()
        >>> client = get_stac_client("https://outro-stac.com/api/v1")
    """
    return Client.open(url or BDC_API_URL)


def get_available_collections(client: Optional[Client] = None) -> List[str]:
    """
    Retorna os IDs de todas as coleções disponíveis na API STAC.

    Args:
        client: Cliente STAC configurado (opcional).

    Returns:
        Lista de strings com os IDs das coleções.

    Exemplo:
        >>> get_available_collections()
        >>> client = get_stac_client()
        >>> get_available_collections(client)
    """
    if client is None:
        client = get_stac_client()
    return [col.id for col in client.get_collections()]


def get_collection_metadata(
    collection_id: str, client: Optional[Client] = None
) -> Dict:
    """
    Recupera metadados de uma coleção específica.

    Args:
        collection_id: ID da coleção.
        client: Cliente STAC configurado (opcional).

    Returns:
        Dicionário contendo informações como título, descrição, extensão, licenças, provedores e assets.

    Exemplo:
        >>> get_collection_metadata('CBERS4_MUX')
    """
    if client is None:
        client = get_stac_client()
    col = client.get_collection(collection_id)
    return {
        "id": col.id,
        "title": col.title,
        "description": col.description,
        "extent": col.extent,
        "license": col.license,
        "providers": col.providers,
        "links": col.links,
        "assets": col.item_assets,
    }


def get_collection_assets_metadata(
    collection_id: str, client: Optional[Client] = None
) -> Dict[str, Dict]:
    """
    Retorna metadados dos assets definidos em uma coleção STAC.

    Args:
        collection_id: ID da coleção.
        client: Cliente STAC configurado (opcional).

    Returns:
        Dicionário mapeando cada asset a um sub-dicionário com título, descrição, tipo de mídia e papéis.

    Exemplo:
        >>> get_collection_assets_metadata('CBERS4_MUX')
    """
    if client is None:
        client = get_stac_client()
    col = client.get_collection(collection_id)
    meta: Dict[str, Dict] = {}
    for key, asset in col.item_assets.items():
        meta[key] = {
            "title": asset.title,
            "description": asset.description,
            "media_type": asset.media_type,
            "roles": asset.roles,
        }
    return meta


def get_collection_items(
    collection_id: str, client: Optional[Client] = None
) -> Iterator[Item]:
    """
    Retorna um iterador sobre todos os items de uma coleção STAC.

    Args:
        collection_id: ID da coleção.
        client: Cliente STAC configurado (opcional).

    Returns:
        Iterador de objetos Item.

    Exemplo:
        >>> for item in get_collection_items('CBERS4_MUX'):
        ...     print(item.id)
    """
    if client is None:
        client = get_stac_client()
    return client.get_collection(collection_id).get_items()


def get_collection_available_dates(
    collection_id: str, client: Optional[Client] = None
) -> List[datetime]:
    """
    Lista todas as datas em que há items disponíveis para a coleção.

    Args:
        collection_id: ID da coleção.
        client: Cliente STAC configurado (opcional).

    Returns:
        Lista de objetos datetime ordenada.

    Raises:
        ValueError: Se a coleção não contiver nenhum item.

    Exemplo:
        >>> get_collection_available_dates('CBERS4_MUX')
    """
    if client is None:
        client = get_stac_client()
    items = get_collection_items(collection_id, client)
    try:
        first = next(items)
    except StopIteration:
        raise ValueError(f"Nenhum item na coleção {collection_id}")

    search = client.search(
        collections=[collection_id],
        intersects=first.geometry,
        bbox=first.bbox,
        fields={"include": ["properties.datetime"]},
        sortby=[{"field": "properties.datetime", "direction": "asc"}],
    )
    dates = {
        datetime.fromisoformat(item.properties["datetime"])
        for item in search.items()
        if item.properties.get("datetime")
    }
    return sorted(dates)


def search_stac_items(
    collection: str,
    geometry: Optional[dict] = None,
    bbox: Optional[List[float]] = None,
    datetime_range: Optional[str] = None,
    cloud_cover_lt: Optional[float] = None,
    limit: Optional[int] = None,
    client: Optional[Client] = None,
    **kwargs,
) -> List[Item]:
    """
    Busca items em uma coleção STAC com filtros opcionais, como nuvens e área de interesse.

    Args:
        collection: ID da coleção.
        geometry: GeoJSON para filtro espacial (intersects).
        bbox: Lista [minx, miny, maxx, maxy] para filtro espacial.
        datetime_range: String ISO de intervalo de tempo (ex: '2020-01-01/2020-12-31').
        cloud_cover_lt: Filtra assets com cobertura de nuvens menor que este valor.
        limit: Número máximo de items a retornar.
        client: Cliente STAC configurado (opcional).
        **kwargs: Parâmetros adicionais de busca.

    Returns:
        Lista de objetos Item resultantes da busca.

    Exemplo:
        >>> items = search_stac_items('CBERS4_MUX', cloud_cover_lt=10, limit=5)
        >>> for item in items:
        ...     print(item.id)
    """
    if client is None:
        client = get_stac_client()
    params = {
        "collections": [collection],
        "datetime": datetime_range,
        **kwargs,
    }
    if cloud_cover_lt is not None:
        params["query"] = {"eo:cloud_cover": {"lt": cloud_cover_lt}}
    if geometry:
        params["intersects"] = geometry
    if bbox:
        params["bbox"] = bbox

    items = client.search(**params).item_collection()
    return items[:limit] if limit else items


def download_item_assets(
    item: Item,
    output_dir: str,
    valid_assets: Optional[List[str]] = None,
    max_workers: int = 5,
) -> None:
    """
    Baixa os assets especificados de um item STAC. Se um asset for um arquivo .zip,
    ele será baixado, extraído e o .zip original será removido.

    Args:
        item: Objeto Item STAC contendo os assets.
        output_dir: Diretório onde os arquivos serão salvos.
        valid_assets: Lista de chaves de assets a serem baixados (ou None para todos).
        max_workers: Número máximo de threads para download paralelo.
    """
    dest_folder = Path(output_dir) / item.id
    dest_folder.mkdir(parents=True, exist_ok=True)

    assets_to_download = [
        (key, asset)
        for key, asset in item.assets.items()
        if valid_assets is None or key in valid_assets
    ]

    def _download_and_process_asset(key, asset):
        """Função auxiliar para baixar e, se necessário, extrair um asset."""
        asset_filename = Path(asset.href).name
        download_path = dest_folder / asset_filename

        try:
            urlretrieve(asset.href, str(download_path))
            print(f"Download do asset '{key}' concluído.")

            # Verifica se o asset é um arquivo zip e o extrai
            if (
                asset.media_type == 'application/zip'
                or download_path.suffix.lower() == '.zip'
            ):
                with zipfile.ZipFile(download_path, 'r') as zip_ref:
                    zip_ref.extractall(dest_folder)

                # Remove o arquivo .zip após a extração
                os.remove(download_path)

        except Exception as e:
            print(f"ERRO ao baixar ou processar o asset '{key}': {e}")

    # Usa um pool de threads para baixar os assets em paralelo
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:
        futures = [
            executor.submit(_download_and_process_asset, key, asset)
            for key, asset in assets_to_download
        ]
        concurrent.futures.wait(futures)


def merge_bands_to_multiband(
    item_id: str,
    output_dir: str,
    band_files: List[str],
    output_filename: Optional[str] = None,
) -> None:
    """
    Mescla arquivos de bandas individuais em um único GeoTIFF multibanda.

    Args:
        item_id: Identificador do item (nome da pasta onde estão as bandas).
        output_dir: Diretório base onde as bandas estão salvas.
        band_files: Lista de nomes de arquivos de banda a serem mesclados.
        output_filename: Nome do arquivo de saída (padrão: '<item_id>_multiband.tif').

    Exemplo:
        >>> merge_bands_to_multiband('item123', './dados', ['BAND1.tif', 'BAND2.tif'])
    """
    base = Path(output_dir) / item_id
    # Garante que estamos pegando apenas arquivos, não subdiretórios
    paths = [
        base / f
        for f in band_files
        if (base / f).is_file() and f.endswith(('.tif', '.tiff'))
    ]
    paths.sort()
    paths = paths[::-1]

    if not paths:
        print(f"Nenhum arquivo de banda .tif encontrado para unir em {base}")
        return

    try:
        srcs = [rasterio.open(str(p)) for p in paths]
        meta = srcs[0].meta.copy()
        meta.update(count=len(srcs))

        out_name = output_filename or f"{item_id}_multiband.tif"
        out_path = base / out_name

        with rasterio.open(str(out_path), "w", **meta) as dst:
            for idx, src in enumerate(srcs, start=1):
                dst.write(src.read(1), idx)

    except Exception as e:
        print(f"ERRO ao unir as bandas: {e}")
    finally:
        for src in srcs:
            src.close()


def mask_raster_with_geobr_polygon(
    raster_path: str,
    geodf: gpd.GeoDataFrame,
    output_path: Optional[str] = None,
) -> None:
    """
    Aplica máscara de polígonos (ex: estados ou municípios do geobr) a um raster e salva o resultado.

    Args:
        raster_path: Caminho para o arquivo raster de entrada.
        geodf: GeoDataFrame com a(s) geometria(s) de máscara.
        output_path: Caminho de saída (se None, adiciona sufixo '_masked').

    Exemplo:
        >>> estados = get_state_data('SP')
        >>> mask_raster_with_geobr_polygon('imagem.tif', estados, 'imagem_masked.tif')
    """
    if not Path(raster_path).exists():
        print(
            f"ERRO: Arquivo raster para mascarar não encontrado em {raster_path}"
        )
        return

    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
    if geodf.crs != raster_crs:
        geodf = geodf.to_crs(raster_crs)

    geoms = list(geodf.geometry)
    with rasterio.open(raster_path) as src:
        out_img, out_tf = mask(src, geoms, crop=True)
        out_meta = src.meta.copy()
        out_meta.update(
            {
                "height": out_img.shape[1],
                "width": out_img.shape[2],
                "transform": out_tf,
            }
        )
        if not output_path:
            base, ext = os.path.splitext(raster_path)
            output_path = f"{base}_masked{ext}"
        with rasterio.open(output_path, "w", **out_meta) as dst:
            dst.write(out_img)
