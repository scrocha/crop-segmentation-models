import torch
import numpy as np
import os
import rasterio
import glob
from tqdm import tqdm
import time

from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

MODEL_ID = "facebook/sam2.1-hiera-base-plus"

INPUT_DIR = "./dados/patches_campo_verde"
OUTPUT_DIR = "./dados/mascaras_campo_verde"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def segmentar_patches_via_hf_id():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Diretório de saída: {OUTPUT_DIR}")

    try:
        predictor = SAM2ImagePredictor.from_pretrained(MODEL_ID)

        sam2_model = predictor.model

        mask_generator = SAM2AutomaticMaskGenerator(
            model=sam2_model,
            points_per_side=64,
            points_per_batch=128,
            pred_iou_thresh=0.80,
            stability_score_thresh=0.90,
            stability_score_offset=1.0,
            mask_threshold=0.0,
            box_nms_thresh=0.6,
            crop_n_layers=1,
            crop_nms_thresh=0.6,
            crop_overlap_ratio= 128 / 1024,  # 128 pixels de sobreposição em um patch de 1024 pixels
            crop_n_points_downscale_factor=2,
            min_mask_region_area=150000,  # 150k pixels
            use_m2m=True,
        )

    except Exception as e:
        print(f"❌ Erro ao carregar ou configurar o modelo: {e}")
        return

    image_paths = glob.glob(os.path.join(INPUT_DIR, "*.tif"))
    if not image_paths:
        print(f"Nenhum arquivo .tif encontrado no diretório: {INPUT_DIR}")
        return

    print(f"Encontrados {len(image_paths)} patches para processar.")

    for image_path in tqdm(
        image_paths, desc="Segmentando patches"
    ):
        base_name = os.path.basename(image_path)
        output_filename = f"{os.path.splitext(base_name)[0]}_masks.npz"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        try:
            with rasterio.open(image_path) as src:
                image_np = src.read()
                image_np = np.transpose(image_np, (1, 2, 0))  # CHW -> HWC

            masks = mask_generator.generate(image_np)

            if not masks:
                continue

            mask_arrays = [mask['segmentation'] for mask in masks]

            stacked_masks = np.stack(mask_arrays, axis=0)
            np.savez_compressed(output_path, masks=stacked_masks)

        except Exception as e:
            print(f"\n❌ Erro ao processar o arquivo {base_name}: {e}")
            continue

    print("\nProcessamento concluído!")
    print(f"Arquivos .npz salvos em: {OUTPUT_DIR}")


if __name__ == "__main__":
    segmentar_patches_via_hf_id()
