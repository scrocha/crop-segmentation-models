import torch
import numpy as np
import os
import rasterio
import glob
from tqdm import tqdm
from transformers import pipeline
from PIL import Image

MODEL_ID = "facebook/sam-vit-base"
INPUT_DIR = "./dados/patches_campo_verde"
OUTPUT_DIR = f"./dados/mascaras_campo_verde/{MODEL_ID.split('/')[-1]}"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def segmentar_patches_com_pipeline():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        mask_generator = pipeline(
            "mask-generation",
            model=MODEL_ID,
            device=DEVICE,
        )

    except Exception as e:
        print(f"❌ Erro ao carregar a pipeline: {e}")
        return

    image_paths = glob.glob(os.path.join(INPUT_DIR, "*.tif"))
    if not image_paths:
        print(f"Nenhum arquivo .tif encontrado no diretório: {INPUT_DIR}")
        return

    print(f"Encontrados {len(image_paths)} patches para processar.")

    for image_path in tqdm(
        image_paths, desc=f"Segmentando com {MODEL_ID.split('/')[-1]}"
    ):
        base_name = os.path.basename(image_path)
        output_filename = f"{os.path.splitext(base_name)[0]}_masks.npz"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        if os.path.exists(output_path):
            continue

        try:
            with rasterio.open(image_path) as src:
                image_np = src.read()
                image_np = np.transpose(image_np, (1, 2, 0))
            image_pil = Image.fromarray(image_np)

            outputs = mask_generator(
                image_pil, points_per_batch=128*2, pred_iou_thresh=0.7
            )

            masks = outputs["masks"]

            if not masks:
                continue

            min_area_pixels = 5000
            filtered_masks = [
                mask for mask in masks if np.sum(mask) >= min_area_pixels
            ]

            if not filtered_masks:
                continue

            masks_final = np.stack(filtered_masks, axis=0).astype(np.uint8)
            np.savez_compressed(output_path, masks=masks_final)

        except Exception as e:
            print(f"\n❌ Erro ao processar o arquivo {base_name}: {e}")
            continue

    print("\nProcessamento concluído!")
    print(f"Arquivos .npz salvos em: {OUTPUT_DIR}")


if __name__ == "__main__":
    segmentar_patches_com_pipeline()
