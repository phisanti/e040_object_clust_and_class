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
import tifffile
import argparse
from tqdm import tqdm
from pathlib import Path
from scipy.ndimage import label
from skimage import measure

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
    """Initialize the COCO format data structure."""
    return {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "cell"},
            {"id": 2, "name": "clump"},
            {"id": 3, "name": "noise"},
            {"id": 4, "name": "off-focus"},
            {"id": 5, "name": "joint cell"}
        ]
    }


def get_tiff_files(input_dir):
    """Get a sorted list of all TIFF files in the input directory."""
    tiff_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.tif', '.tiff'))])
    return tiff_files


def process_image(img, image_id, tiff_file, annotation_id, coco_data):
    """Process a single image and extract annotations for all objects."""
    from scipy.ndimage import label
    
    # Add image info
    height, width = img.shape
    coco_data["images"].append({
        "id": image_id,
        "file_name": tiff_file,
        "width": width,
        "height": height
    })
    
    # Find unique object class IDs (excluding background class 0)
    class_ids = np.unique(img)
    class_ids = class_ids[class_ids > 0]
    
    # Process each class
    for class_id in class_ids:
        if class_id > 5:  # Skip invalid class IDs
            continue
            
        # Create binary mask for this class
        class_mask = (img == class_id)
        
        # Label connected components to identify individual objects
        labeled_mask, num_objects = label(class_mask)
        
        # Process each individual object within this class
        for obj_idx in range(1, num_objects + 1):
            # Create mask for this specific object
            object_mask = (labeled_mask == obj_idx)
            
            # Find contours/bounds
            y_indices, x_indices = np.where(object_mask)
            if len(y_indices) == 0:
                continue
                
            # Bounding box: [x, y, width, height]
            x_min, y_min = np.min(x_indices), np.min(y_indices)
            x_max, y_max = np.max(x_indices), np.max(y_indices)
            bbox = [int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1)]
            
            # Area
            area = int(np.sum(object_mask))
            
            # Get contours
            contours = measure.find_contours(object_mask.astype(np.uint8), 0.5)
            
            segmentation = []
            for contour in contours:
                # Flip xy to yx for COCO format and flatten
                contour = np.fliplr(contour).flatten().tolist()
                # COCO requires at least 6 points (3 xy coordinates)
                if len(contour) >= 6:
                    segmentation.append(contour)
            
            # If no valid contours found, create a simple box contour
            if not segmentation:
                # Create a simple box contour
                box_contour = [
                    x_min, y_min,
                    x_max, y_min,
                    x_max, y_max,
                    x_min, y_max
                ]
                segmentation.append(box_contour)
            
            # Add annotation
            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": int(class_id),
                "bbox": bbox,
                "segmentation": segmentation,
                "area": area,
                "iscrowd": 0
            })
            
            annotation_id += 1
    
    return annotation_id




def create_coco_annotations(input_dir, output_file):
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
            annotation_id = process_image(img, image_id, tiff_file, annotation_id, coco_data)
                
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
    args = parse_arguments()
    create_coco_annotations(args.input_dir, args.output_file)


if __name__ == "__main__":
    main()
