#!/usr/bin/env python3
"""
COCO Annotation to Mask Image Converter (Shapely version)

This script converts COCO format annotations, including support for complex objects
with multiple polygons and holes, back to mask images where pixels are labeled
with class IDs:
- 0: background
- 1: cell
- 2: clump
- 3: noise
- 4: off-focus
- 5: joint cell

It leverages extra annotations generated during COCO creation to accurately
handle multi-part segmentations and interior voids (holes).

Usage:
    python coco2masks.py --coco_file /path/to/annotations.json --output_dir /path/to/output_masks [--workers N]
"""

import os
import json
import numpy as np
import tifffile
import argparse
import multiprocessing
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import create_segmentation_mask from object_classif_loader
from object_classif_loader import create_segmentation_mask


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for COCO-to-mask conversion.

    Returns:
        argparse.Namespace: Parsed arguments with attributes:
            coco_file (str): Path to the COCO annotations JSON file.
            output_dir (str): Directory where mask images will be saved.
            workers (int): Number of parallel workers (None=1, 0=all cores, >0 specific).
    """
    parser = argparse.ArgumentParser(
        description='Convert COCO annotations (shapely polygons) to mask images.'
    )
    parser.add_argument(
        '-c', '--coco_file', type=str, required=True,
        help='Path to COCO annotations JSON file'
    )
    parser.add_argument(
        '-o', '--output_dir', type=str, required=True,
        help='Directory to save the generated mask images'
    )
    parser.add_argument(
        '-w', '--workers', type=int, default=0,
        help='Number of parallel workers: None=1, 0=all cores, >0 specific number'
    )
    return parser.parse_args()


def load_coco_annotations(coco_file: str) -> dict:
    """
    Load COCO annotations from a JSON file.

    Args:
        coco_file (str): Path to the COCO annotations JSON file.

    Returns:
        dict: Parsed COCO JSON content.
    """
    with open(coco_file, 'r') as f:
        return json.load(f)


def create_mask_from_annotations(
    coco_data: dict,
    image_id: int,
    width: int,
    height: int
) -> np.ndarray:
    """
    Create a mask image from annotations for a single image.

    Args:
        coco_data (dict): COCO JSON content.
        image_id (int): ID of the image to process.
        width (int): Width of the target mask image.
        height (int): Height of the target mask image.

    Returns:
        np.ndarray: 2D array of shape (height, width) with uint8 labels.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    anns = [ann for ann in coco_data["annotations"] if ann["image_id"] == image_id]
    for ann in anns:
        cat_id = ann["category_id"]
        seg = ann.get("segmentation")
        seg_types = ann.get("segmentation_types")
        if not seg:
            continue
        full_mask = create_segmentation_mask(seg, seg_types, height, width)
        if full_mask is not None:
            binary = full_mask.numpy()
            cat_mask = binary * cat_id
            mask = np.where(cat_mask > 0, cat_mask, mask)
    return mask


def _process_single_image(
    img_info: dict,
    coco_data: dict,
    output_dir: str
) -> None:
    """
    Process one image: create mask and write to output.

    Args:
        img_info (dict): COCO image info dict with id, file_name, width, height.
        coco_data (dict): Full COCO JSON content.
        output_dir (str): Directory where mask will be saved.
    """
    image_id = img_info["id"]
    width = img_info["width"]
    height = img_info["height"]
    file_name = img_info["file_name"]
    mask = create_mask_from_annotations(coco_data, image_id, width, height)
    out_path = os.path.join(output_dir, file_name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tifffile.imwrite(out_path, mask)


def process_coco_annotations(
    coco_data: dict,
    output_dir: str,
    workers: int = None
) -> None:
    """
    Process all images in COCO annotations and save mask images in parallel.

    Args:
        coco_data (dict): COCO JSON content.
        output_dir (str): Directory where mask images will be saved.
        workers (int): Number of parallel workers (None=1, 0=all cores, >0 specific).
    """
    os.makedirs(output_dir, exist_ok=True)
    if workers is None:
        num_workers = 1
    elif workers == 0:
        num_workers = multiprocessing.cpu_count()
    else:
        num_workers = workers

    images = coco_data["images"]
    if num_workers == 1:
        for img in tqdm(images, desc="Creating masks"):
            _process_single_image(img, coco_data, output_dir)
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(_process_single_image, img, coco_data, output_dir)
                for img in images
            ]
            for _ in tqdm(as_completed(futures), total=len(futures), desc="Creating masks"):
                pass


def main() -> None:
    """
    Entry point for COCO-to-mask conversion script.

    Parses arguments, loads COCO annotations, and processes all images.
    """
    args = parse_arguments()
    coco = load_coco_annotations(args.coco_file)
    process_coco_annotations(coco, args.output_dir, args.workers)
    print(f"Created mask images for {len(coco['images'])} images in {args.output_dir}")


if __name__ == "__main__":
    main()
