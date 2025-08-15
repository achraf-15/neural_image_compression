import os
import glob
import random
import argparse
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import numpy as np
from tqdm import tqdm

# --------------------
# Helper functions
# --------------------
def add_quantization_noise(img, levels=256):
    np_img = np.array(img).astype(np.float32) / 255.0
    noise = np.random.uniform(-0.5 / levels, 0.5 / levels, np_img.shape)
    np_img = np.clip(np_img + noise, 0, 1)
    return Image.fromarray((np_img * 255).astype(np.uint8))

def is_saturated(img, threshold=0.95):
    np_img = np.array(img).astype(np.float32) / 255.0
    saturation = np.max(np_img, axis=2) - np.min(np_img, axis=2)
    return (saturation > threshold).mean() > 0.05

def random_downsample_crop(img, target_size=256, min_factor=0.75):
    w, h = img.size
    factor = random.uniform(min_factor, 1.0)
    new_w, new_h = int(w * factor), int(h * factor)
    if new_w < target_size or new_h < target_size:
        return None
    img_resized = img.resize((new_w, new_h), Image.BICUBIC)
    left = random.randint(0, new_w - target_size)
    top = random.randint(0, new_h - target_size)
    img_cropped = img_resized.crop((left, top, left + target_size, top + target_size))
    return img_cropped

# --------------------
# Main preprocessing
# --------------------
def preprocess_images(input_dir, output_dir, target_size=256, min_factor=0.75, saturation_thresh=0.95, seed=None, overwrite=False):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_images = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))

    for img_path in tqdm(all_images, desc="Preprocessing images"):
        try:
            img = Image.open(img_path).convert('RGB')
        except (UnidentifiedImageError, OSError):
            print(f"[WARN] Skipping corrupted file: {img_path}")
            continue

        # Discard if too saturated
        if is_saturated(img, threshold=saturation_thresh):
            continue

        # Discard if too small
        if min(img.size) * min_factor < target_size:
            continue

        # Add quantization noise
        img = add_quantization_noise(img)

        # Random downsample + crop
        img_cropped = random_downsample_crop(img, target_size=target_size, min_factor=min_factor)
        if img_cropped is None:
            continue

        save_path = output_dir / img_path.name
        if save_path.exists() and not overwrite:
            continue

        img_cropped.save(save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess images for dataset.")
    parser.add_argument("--input_dir", type=str, default="./data/coco_val2017", help="Directory with original images.")
    parser.add_argument("--output_dir", type=str, default="./data/coco_preprocessed", help="Directory to save preprocessed images.")
    parser.add_argument("--target_size", type=int, default=256, help="Final cropped image size.")
    parser.add_argument("--min_factor", type=float, default=0.75, help="Minimum downsample factor before cropping.")
    parser.add_argument("--saturation_thresh", type=float, default=0.95, help="Threshold for saturation filtering.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files in output_dir.")
    args = parser.parse_args()

    preprocess_images(
        args.input_dir,
        args.output_dir,
        target_size=args.target_size,
        min_factor=args.min_factor,
        saturation_thresh=args.saturation_thresh,
        seed=args.seed,
        overwrite=args.overwrite
    )