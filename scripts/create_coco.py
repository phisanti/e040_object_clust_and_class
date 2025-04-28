#!/usr/bin/env python3
"""
COCO Annotation Extractor for Classified Objects

This script extracts COCO format annotations from classified object images.
It processes TIFF images where pixels are labeled with class IDs:
- 0: background
- 1: cell
- 2: clump
- 3: noise
- 4: off-focus
- 5: joint cell

The script generates a COCO-format JSON file containing image information,
object annotations with bounding boxes, and category definitions.

Usage:
    python extract_coco_annotations.py --input_dir /path/to/images --output_file /path/to/annotations.json
"""

import os
import json
import numpy as np
import datetime

import tifffile
import argparse
from tqdm import tqdm
from pathlib import Path
from scipy.ndimage import label
from skimage import measure
from shapely.geometry import Polygon, MultiPolygon

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Extract COCO annotations from classified object images.')
    parser.add_argument('-i', '--input_dir', type=str, 
                        default="../../Semantic_bac_segment/data/tmp_stacks/objects_images/",
                        help='Path to directory containing classified object TIFF images')
    parser.add_argument('-o', '--output_file', type=str, 
                        default="./data/objects_annotations/annotations.json",
                        help='Path to save the COCO annotations JSON file')
    return parser.parse_args()


def initialize_coco_structure():
    return {
        "info": {
            "description": "Cell Object Dataset",
            "url": "",
            "version": "1.0",
            "year": datetime.datetime.now().year,
            "contributor": "",
            "date_created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "licenses": [
            {
                "id": 1,
                "name": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
                "url": "https://creativecommons.org/licenses/by/4.0/"
            }
        ],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "cell", "supercategory": "cell"},
            {"id": 2, "name": "clump", "supercategory": "cell"},
            {"id": 3, "name": "noise", "supercategory": "artifact"},
            {"id": 4, "name": "off-focus", "supercategory": "artifact"},
            {"id": 5, "name": "joint cell", "supercategory": "cell"}
        ]
    }


def get_tiff_files(input_dir):
    """Get a sorted list of all TIFF files in the input directory."""
    tiff_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.tif', '.tiff'))])
    return tiff_files

def create_sub_mask_annotation(sub_mask, image_id, category_id, annotation_id, is_crowd):
    """
    Generate a COCO-style annotation for a single object mask, including support for occlusions and holes.

        Args:
        sub_mask (np.ndarray): Binary mask for the object (2D array).
        image_id (int): ID of the image containing the object.
        category_id (int): COCO category ID for the object.
        annotation_id (int): Unique annotation ID.
        is_crowd (int): COCO iscrowd flag.

    Returns:
        dict: COCO-style annotation dictionary with segmentation, bbox, area, etc., or None if no valid polygons found.

    General strategy:
        - The function extracts all contours from the binary mask using skimage's find_contours.
        - Each contour is converted to a polygon, simplified, and filtered for validity.
        - All valid polygons are collected and used to compute the segmentation, bounding box, and area.
        - **A ring of 0s (padding) is added around the mask before contour extraction.**
          This is necessary to ensure that objects touching the image border (even at multiple or disconnected points)
          are properly closed and detected as valid polygons.

    Occluded objects:
        - If an object is split into multiple visible parts (due to occlusion), each part is detected as a separate contour.
        - All such parts are included as separate polygons in the segmentation, allowing for multi-part object annotation.

    Holes:
        - Holes within objects are detected as inner contours with opposite orientation (counter-clockwise).
        - These are included in the segmentation list and can be distinguished by their orientation.
        - This enables downstream tools to correctly subtract holes from the filled object mask.

    Why:
        - This approach is necessary to accurately represent complex biological objects that may be occluded or contain holes (e.g., vacuoles, artifacts).
        - Properly encoding both visible parts and holes ensures that segmentation masks and downstream analyses (e.g., mask rasterization, visualization) are correct and biologically meaningful.
    """
    # Add a ring of 0s (padding) around the mask
    padded_mask = np.pad(sub_mask, pad_width=1, mode='constant', constant_values=0)
    # Find contours (boundary lines) around each sub-mask
    contours = measure.find_contours(padded_mask, 0.5)
    segmentations = []
    segment_types = []
    polygons = []

    for contour in contours:
        # Flip from (row, col) to (x, y) and subtract the padding pixel
        contour = np.array([(col - 1, row - 1) for row, col in contour])
        poly = Polygon(contour)
        if not poly.is_valid or poly.is_empty:
            continue
        poly = poly.simplify(1.0, preserve_topology=False)
        if poly.is_empty or not poly.is_valid:
            continue
        # Determine orientation: 0 for normal (clockwise), 1 for hole (counter-clockwise)
        orientation = 0 if poly.exterior.is_ccw is False else 1
        polygons.append(poly)
        segmentation = np.array(poly.exterior.coords).ravel().tolist()
        if len(segmentation) >= 6:
            segmentations.append(segmentation)
            segment_types.append(orientation)
    if not polygons:
        return None
    multi_poly = MultiPolygon(polygons)
    x, y, max_x, max_y = multi_poly.bounds
    width = max_x - x
    height = max_y - y
    bbox = [x, y, width, height]
    area = multi_poly.area
    annotation = {
        'segmentation': segmentations,
        'segmentation_types': segment_types,  
        'iscrowd': is_crowd,
        'image_id': image_id,
        'category_id': category_id,
        'id': annotation_id,
        'bbox': bbox,
        'area': area
    }
    return annotation

def process_image(args):
    img, image_id, tiff_file, annotation_id, max_classes = args
    coco_annotations = []
    height, width = img.shape
    image_info = {
        "id": image_id,
        "file_name": tiff_file,
        "width": width,
        "height": height,
        "date_captured": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "license": 1
    }
    class_ids = np.unique(img)
    class_ids = class_ids[class_ids > 0]
    for class_id in class_ids:
        if class_id > max_classes:
            continue
        class_mask = (img == class_id)
        labeled_mask, num_objects = measure.label(class_mask, connectivity=1, return_num=True)
        for obj_idx in range(1, num_objects + 1):
            sub_mask = (labeled_mask == obj_idx).astype(np.uint8)
            if np.sum(sub_mask) < 10:
                continue
            annotation = create_sub_mask_annotation(
                sub_mask, image_id, int(class_id), annotation_id, is_crowd=0
            )
            if annotation is not None:
                annotation['id'] = annotation_id
                coco_annotations.append(annotation)
                annotation_id += 1
    return image_info, coco_annotations


def create_coco_annotations(input_dir, output_file, max_classes):
    """
    Extract COCO annotations from classified object images.
    
    Args:
        input_dir: Path to directory containing classified object TIFF images
        output_file: Path to save the COCO annotations JSON file
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Initialize COCO format structure
    coco_data = initialize_coco_structure()
    
    # List all TIFF files
    tiff_files = get_tiff_files(input_dir)
    
    if not tiff_files:
        print(f"No TIFF files found in {input_dir}")
        return
    
    annotation_id = 1
    
    # Process each image
    for image_id, tiff_file in enumerate(tqdm(tiff_files, desc="Processing images", position=0, leave=True), 1):
        file_path = os.path.join(input_dir, tiff_file)
        try:
            # Load the image
            img = tifffile.imread(file_path)
            args = (img, image_id, tiff_file, annotation_id, max_classes)
            image_info, coco_annotations = process_image(args)
            coco_data["images"].append(image_info)
            coco_data["annotations"].extend(coco_annotations)
            annotation_id += len(coco_annotations)
                
        except Exception as e:
            print(f"Error processing {tiff_file}: {e}")
    
    # Save annotations
    save_annotations(coco_data, output_file)
    
    print(f"Processed {len(tiff_files)} images with {annotation_id-1} annotations")
    print(f"COCO annotations saved to {output_file}")


def save_annotations(coco_data, output_file):
    """Save the COCO annotations to a JSON file."""
    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=2)


def main():
    """Main function to run the script."""
    MAX_CLASSES=5 # For this project, the code works with 5 classes
    args = parse_arguments()
    create_coco_annotations(args.input_dir, args.output_file, MAX_CLASSES)


if __name__ == "__main__":
    main()