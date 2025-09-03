# Análise e Segmentação de Imagens de Satélite para Agricultura com SAM2

Este repositório contém um pipeline completo para identificar e analisar talhões agrícolas a partir de imagens de satélite. O processo utiliza dados do **Brazil Data Cube (BDC)**, segmentação de imagens com o modelo **Segment Anything Model 2 (SAM2)** e validação com dados do **MapBiomas**.

## Visão Geral do Pipeline

O objetivo principal é automatizar a detecção de áreas agrícolas em uma região de interesse, filtrar os resultados com base em critérios de área e cobertura de solo, e analisar a heterogeneidade e a precisão da segmentação.

O pipeline é composto pelos seguintes passos:

1.  **Download e Pré-processamento de Dados**: Baixa imagens de satélite e dados geográficos da área de interesse.
2.  **Segmentação de Imagens**: Utiliza o modelo SAM2 para gerar máscaras de segmentação nos talhões.
3.  **Pós-processamento e Agregação**: Converte as máscaras em formato vetorial (shapefile).
4.  **Filtragem de Máscaras**: Seleciona apenas os polígonos que correspondem a áreas agrícolas com base em dados do MapBiomas.
5.  **Análise de Heterogeneidade**: Calcula o NDVI para avaliar a variabilidade dentro de cada talhão.
6.  **Análise de Cobertura**: Avalia a performance da segmentação comparando os resultados com dados de referência.

## Como Executar o Pipeline

### 1\. Pré-requisitos

Antes de começar, certifique-se de que você tem **Python 3.8+** e **uv** instalados. Se você não tiver o `uv`, pode instalá-lo com:

```bash
pip install uv
```

### 2\. Instalação de Dependências

Primeiro, crie um ambiente virtual e instale as dependências do projeto.

```bash
# Crie um ambiente virtual na pasta .venv
uv venv

# Ative o ambiente virtual
# No Linux/macOS
source .venv/bin/activate
# No Windows
.venv\Scripts\activate

# Instale as dependências do requirements.txt
uv pip install -r requirements.txt
```

### 3\. Execução dos Módulos

Os scripts são numerados na ordem em que devem ser executados. Siga os passos abaixo:

**Passo 0: Download e Preparação dos Dados**
Execute o notebook `00_download_data.ipynb` em um ambiente Jupyter. Ele irá baixar as imagens de satélite da região de "Campo Verde - MT", criar patches (recortes menores da imagem) e obter os dados de referência do MapBiomas.

**Passo 1: Segmentação com SAM2**
Execute o script `01_segmentation_with_sam2.py` para aplicar o modelo de segmentação SAM2 sobre os patches. As máscaras resultantes serão salvas como arquivos `.npz`.

```bash
uv run python 01_segmentation_with_sam2.py
```

**Passo 2: Agregação das Máscaras em Shapefile**
O script `02_agregar_npz_em_shp.py` converte as máscaras `.npz` em um único arquivo vetorial (shapefile).

```bash
uv run python 02_agregar_npz_em_shp.py
```

**Passo 3: Filtragem dos Polígonos Segmentados**
Utilize o `03_filtrar_mascaras.py` para filtrar os polígonos com base em área e no percentual de cobertura agrícola, usando os dados do MapBiomas.

```bash
uv run python 03_filtrar_mascaras.py
```

**Passo 4: Análise de Heterogeneidade (NDVI)**
O script `04_analise_heterogeniedade.py` calcula estatísticas de NDVI para cada talhão filtrado.

```bash
uv run python 04_analise_heterogeniedade.py
```

**Passo 5: Análise de Cobertura e Desempenho**
Finalmente, execute `05_analise_cobertura.py` para calcular métricas de precisão e recall, avaliando a performance do pipeline.

```bash
uv run python 05_analise_cobertura.py
```

## Descrição dos Módulos

### Módulos Principais do Pipeline

  * **`00_download_data.ipynb`**:

      * Utiliza o `bdc_downloader.py` para buscar e baixar mosaicos de imagens do satélite Sentinel-2.
      * Define a área de interesse (município de Campo Verde - MT).
      * Recorta as imagens para a área de interesse e as divide em patches menores para otimizar a segmentação.
      * Baixa e recorta os dados do MapBiomas para a mesma área.

  * **`01_segmentation_with_sam2.py`**:

      * Carrega o modelo SAM2 pré-treinado (`facebook/sam2.1-hiera-base-plus`).
      * Instancia o `SAM2AutomaticMaskGenerator` para gerar máscaras de forma automática em cada patch de imagem.
      * Salva as máscaras geradas em formato `.npz`.

  * **`02_agregar_npz_em_shp.py`**:

      * Lê todos os arquivos `.npz` contendo as máscaras.
      * Converte cada máscara em um polígono vetorial usando `rasterio.features.shapes`.
      * Agrega todos os polígonos em um único shapefile com `geopandas`.

  * **`03_filtrar_mascaras.py`**:

      * Aplica filtros para remover polígonos com área fora de um intervalo pré-definido (mínimo e máximo em hectares).
      * Para cada polígono, calcula a porcentagem de cobertura de classes agrícolas com base no raster do MapBiomas.
      * Mantém apenas os polígonos com um alto percentual de cobertura agrícola (ex: \> 80%).

  * **`04_analise_heterogeniedade.py`**:

      * Calcula o NDVI (Índice de Vegetação por Diferença Normalizada) para cada talhão usando as bandas do infravermelho próximo e vermelho.
      * Gera estatísticas como média, desvio padrão e percentis do NDVI para identificar a variabilidade interna de cada área.

  * **`05_analise_cobertura.py`**:

      * Calcula a área total agrícola na região de interesse usando o MapBiomas como "verdade terrestre".
      * Compara a área total segmentada e filtrada com a área de referência para calcular métricas de **recall** (cobertura) e **precisão** (qualidade), avaliando a eficácia do pipeline.

### Módulos de Suporte

  * **`bdc_downloader.py`**: Módulo utilitário que abstrai a comunicação com a API STAC do Brazil Data Cube, facilitando a busca e o download de imagens de satélite.
  * **`sam2/` (diretório)**: Contém a implementação do **Segment Anything Model 2**, incluindo o preditor de imagens (`sam2_image_predictor.py`) e o gerador automático de máscaras (`automatic_mask_generator.py`), que são os componentes centrais para a etapa de segmentação.
