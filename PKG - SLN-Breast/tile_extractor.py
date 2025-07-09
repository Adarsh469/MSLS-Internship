# tile_extractor.py
import openslide
import os
import cv2
import numpy as np
from tqdm import tqdm

def is_tissue(tile, threshold=0.8):
    gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
    tissue_ratio = 1.0 - (np.count_nonzero(thresh) / thresh.size)
    return tissue_ratio > threshold

def extract_tiles(svs_path, output_dir, tile_size=512, level=0, tissue_threshold=0.5):
    slide = openslide.OpenSlide(svs_path)
    slide_id = os.path.splitext(os.path.basename(svs_path))[0]

    width, height = slide.level_dimensions[level]

    os.makedirs(os.path.join(output_dir, slide_id), exist_ok=True)

    tile_coords = []

    for y in tqdm(range(0, height, tile_size)):
        for x in range(0, width, tile_size):
            tile = slide.read_region((x, y), level, (tile_size, tile_size)).convert("RGB")
            tile_np = np.array(tile)

            if is_tissue(tile_np, tissue_threshold):
                tile_path = os.path.join(output_dir, slide_id, f"tile_{x}_{y}.png")
                cv2.imwrite(tile_path, cv2.cvtColor(tile_np, cv2.COLOR_RGB2BGR))
                tile_coords.append((slide_id, x, y, tile_path))

    return tile_coords

import pandas as pd

all_tile_coords = []

svs_folder = "SLN-Breast"  # ✔️ This is where your .svs files actually are
output_dir = "tiles"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(svs_folder):
    if filename.endswith(".svs"):
        svs_path = os.path.join(svs_folder, filename)
        print(f"Processing {filename}...")
        tile_coords = extract_tiles(
            svs_path=svs_path,
            output_dir=output_dir,
            tile_size=512,
            level=0,
            tissue_threshold=0.5
        )
        all_tile_coords.extend(tile_coords)

# Save all tile coordinates from all slides
df = pd.DataFrame(all_tile_coords, columns=["slide_id", "x", "y", "tile_path"])
df.to_csv("tile_coords.csv", index=False)


