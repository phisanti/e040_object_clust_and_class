import os
import json
import numpy as np
import tifffile
from pycocotools import mask as mask_utils

from typing import Dict, List, Tuple, Optional, Any, Union, Callable



import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def determine_polygon_orientation(polygon):
    """
    Determines if a polygon is a positive area (1) or hole (0) based on its orientation.
    
    Args:
        polygon (list): Polygon coordinates in flat format [x1, y1, x2, y2, ...]
        
    Returns:
        int: 1 if a positive area (clockwise), 0 if a hole (counter-clockwise)
    """
    # Convert flat list to coordinate pairs
    points = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]
    if len(points) < 3:
        return 1  # Default to positive area if not enough points
        
    # Calculate signed area using Shoelace formula
    signed_area = sum((points[i][0] * points[(i+1) % len(points)][1]) - 
                      (points[i][1] * points[(i+1) % len(points)][0]) 
                      for i in range(len(points)))
    
    # Clockwise (negative signed area) is positive segment (1)
    # Counter-clockwise (positive signed area) is hole (0)
    return 1 if signed_area < 0 else 0


def create_segmentation_mask(segmentation, segmentation_types, img_height, img_width):
    """
    Create a binary mask from segmentation polygons, properly handling holes and multiple-segments. 
    Note, post_mask/hole_mask and final_mask are inverted shape (H, W).
    
    Args:
        segmentation (list): List of segmentation polygons
        segmentation_types (list): List indicating if each polygon is positive (1) or hole (0)
        img_height (int): Height of the image
        img_width (int): Width of the image
    
    Returns:
        torch.Tensor: Binary mask with holes properly handled
    """
    if not isinstance(segmentation, list) or len(segmentation) == 0:
        return None

    # If segmentation_types is None, infer from polygon orientation
    if segmentation_types is None:
        segmentation_types = [determine_polygon_orientation(seg) for seg in segmentation]
        
        # Ensure at least one segment is a positive area (type 1)
        if 1 not in segmentation_types and len(segmentation_types) > 0:
            segmentation_types[0] = 1  # Force first segment to be positive

    final_mask = None
    
    if segmentation_types and len(segmentation_types) == len(segmentation):
        # Handle multi-part segmentation with holes
        positive_segments = [seg for i, seg in enumerate(segmentation) if segmentation_types[i] == 1]
        hole_segments = [seg for i, seg in enumerate(segmentation) if segmentation_types[i] == 0]
        
        if positive_segments:
            # Convert positive segments to RLEs and merge
            pos_rles = mask_utils.frPyObjects(positive_segments, img_height, img_width)
            merged_pos_rle = mask_utils.merge(pos_rles)
            pos_mask = mask_utils.decode(merged_pos_rle)  # shape: (H, W)
            
            if hole_segments:
                # Convert hole segments to RLEs and merge
                hole_rles = mask_utils.frPyObjects(hole_segments, img_height, img_width)
                merged_hole_rle = mask_utils.merge(hole_rles)
                hole_mask = mask_utils.decode(merged_hole_rle)  # shape: (H, W)
                
                # Subtract hole mask from positive mask
                final_mask = pos_mask - hole_mask
            else:
                final_mask = pos_mask
        else:
            return None
    else:
        # Simple segmentation (single part or no type info) - treat as positive
        rles = mask_utils.frPyObjects(segmentation, img_height, img_width)
        final_mask = mask_utils.decode(mask_utils.merge(rles))  # shape: (H, W)
    
    if final_mask is not None:
        return torch.from_numpy(final_mask).float()
    return None

class ObjectClassifDataset(Dataset):
    def __init__(
        self,
        annotations: List[Dict],
        image_dir: str,
        image_info: Dict[int, Dict],
        transform: Optional[Callable] = None,
        segment_objects: bool = True,
        resize_dim: int = 64,
        dynamic_resizing: bool = False,
        return_idx: bool = False
    ):
        self.annotations = annotations
        self.image_dir = image_dir
        self.image_info = image_info
        self.transform = transform
        self.resize_dim = resize_dim
        self.image_cache = {}
        self.segment_objects = segment_objects
        self.dynamic_resizing = dynamic_resizing
        self.return_idx = return_idx
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        image_id = annotation["image_id"]
        category_id = annotation["category_id"]
        bbox = annotation["bbox"]
        segmentation = annotation["segmentation"]
        segmentation_types = annotation.get("segmentation_types", None)
        annot_id = annotation["id"]
        
        # Get image info
        img_info = self.image_info[image_id]
        file_name = img_info["file_name"]
        
        # Load image (with caching)
        if image_id not in self.image_cache:
            img_path = os.path.join(self.image_dir, file_name)
            # Convert to torch tensor immediately after loading
            self.image_cache[image_id] = torch.from_numpy(tifffile.imread(img_path)).float()
            
            # Limit cache size to prevent memory issues
            if len(self.image_cache) > 10:
                remove_id = next(iter(self.image_cache))
                if remove_id != image_id:
                    del self.image_cache[remove_id]

        image = self.image_cache[image_id]

        # Extract object using bbox [x, y, width, height]
        x, y, w, h = [int(v) for v in bbox]
        obj_img = image[y:y+h, x:x+w]
        
        # Apply segmentation mask if requested
        if self.segment_objects and segmentation:
            # Create mask using the helper function
            full_mask = create_segmentation_mask(
                segmentation, 
                segmentation_types, 
                img_info["height"], 
                img_info["width"]
            )
            
            if full_mask is not None:
                # Crop mask to bbox
                mask = full_mask[y:y+h, x:x+w]
                
                # If image has more dimensions than mask, expand mask for broadcasting
                if obj_img.dim() == 3 and mask.dim() == 2:
                    mask = mask.unsqueeze(-1)
                
                # Apply mask to object image
                obj_img = obj_img * mask

        # Add channel dimension if needed
        if obj_img.dim() == 2:
            obj_img = obj_img.unsqueeze(0)  # [H,W] -> [1,H,W]
        
        # Get current dimensions
        c, h, w = obj_img.shape
        
        # Resize logic based on dynamic_resizing flag
        if self.dynamic_resizing:
            # Only resize if object is larger than resize_dim in any dimension
            if h > self.resize_dim or w > self.resize_dim:
                # Calculate scaling to maintain aspect ratio (only downscale, never upscale)
                scale = self.resize_dim / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                
                # Apply resize
                obj_img = obj_img.unsqueeze(0)  # [C,H,W] -> [1,C,H,W]
                obj_img = F.interpolate(obj_img, size=(new_h, new_w), mode='bilinear', align_corners=False)
                obj_img = obj_img.squeeze(0)  # [1,C,H,W] -> [C,H,W]
            
            # Pad to target size
            pad_h = max(0, self.resize_dim - obj_img.shape[1])
            pad_w = max(0, self.resize_dim - obj_img.shape[2])
            
            if pad_h > 0 or pad_w > 0:
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
                
                obj_img = F.pad(obj_img, (pad_left, pad_right, pad_top, pad_bottom), 
                            mode='constant', value=0)
        else:
            # Original resizing logic - always resize to maintain aspect ratio
            obj_img = obj_img.unsqueeze(0)  # [C,H,W] -> [1,C,H,W]
            
            # Calculate scaling to maintain aspect ratio
            scale = min(self.resize_dim / obj_img.shape[-2], self.resize_dim / obj_img.shape[-1])
            new_h, new_w = int(obj_img.shape[-2] * scale), int(obj_img.shape[-1] * scale)
            obj_img = F.interpolate(obj_img, size=(new_h, new_w), mode='bilinear', align_corners=False)
            
            # Pad if needed
            if new_h < self.resize_dim or new_w < self.resize_dim:
                pad_h = self.resize_dim - new_h
                pad_w = self.resize_dim - new_w
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
                
                obj_img = F.pad(obj_img, (pad_left, pad_right, pad_top, pad_bottom), 
                            mode='constant', value=0)
            
            # Remove batch dimension
            obj_img = obj_img.squeeze(0)
        
        if self.transform:
            obj_img = self.transform(obj_img)

        if self.return_idx:
            return {'image': obj_img, 'label': category_id, 'id': annot_id}
        else:
            return {'image': obj_img, 'label': category_id}

class ObjectClassifDatasetCreator:
    """
    Creates datasets for object classification from COCO-format annotations.
    
    Args:
        image_dir (str): Directory containing source images
        annotation_file (str): Path to COCO-format annotation JSON file
        val_ratio (float): Ratio of data to use for validation (0.0-1.0)
        resize_dim (int): Target dimension for resized objects (NxN)
    """
    
    def __init__(
        self,
        image_dir: str,
        annotation_file: str,
        val_ratio: float = 0.2,
        resize_dim: int = 64
    ):
        self.image_dir = image_dir
        self.annotation_file = annotation_file
        self.val_ratio = val_ratio
        self.resize_dim = resize_dim
        
        # Load and process annotations
        self._load_annotations()
        
    def _load_annotations(self):
        """Load and process the annotation file"""
        with open(self.annotation_file, 'r') as f:
            data = json.load(f)
            
        self.annotations = data["annotations"]
        self.categories = {cat["id"]: cat["name"] for cat in data["categories"]}
        
        # Create image info lookup
        self.image_info = {img["id"]: img for img in data["images"]}
        
        # Shuffle annotations for random split
        np.random.shuffle(self.annotations)
        
    def create_datasets(
        self,
        train_transform: Optional[Callable] = None,
        val_transform: Optional[Callable] = None,
        batch_size: int = 32,
        dynamic_resizing: bool = False,
        segment_objects: bool = True,
        return_idx: bool = False
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create training and validation datasets/dataloaders.
        
        Args:
            train_transform: Transformations to apply to training data
            val_transform: Transformations to apply to validation data
            batch_size: Batch size for dataloaders
            dynamic_resizing: If True, only resize objects larger than resize_dim
            segment_objects: If True, applies segmentation masks to objects
            return_idx: If True, returns the index of the object in the original image
        
        Returns:
            Tuple containing training and validation DataLoaders
        """
        # Split annotations
        split_idx = int(len(self.annotations) * (1 - self.val_ratio))
        train_annotations = self.annotations[:split_idx]
        val_annotations = self.annotations[split_idx:]
        
        # Create datasets
        train_dataset = ObjectClassifDataset(
            train_annotations,
            self.image_dir,
            self.image_info,
            transform=train_transform,
            resize_dim=self.resize_dim,
            dynamic_resizing=dynamic_resizing,
            segment_objects=segment_objects,
            return_idx=return_idx
        )
        
        val_dataset = ObjectClassifDataset(
            val_annotations,
            self.image_dir,
            self.image_info,
            transform=val_transform,
            resize_dim=self.resize_dim,
            dynamic_resizing=dynamic_resizing,
            segment_objects=segment_objects,
            return_idx=return_idx
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def get_categories(self) -> Dict[int, str]:
        """Return the category mapping"""
        return self.categories


if __name__ == "__main__":
    import argparse
    from tqdm import tqdm
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test ObjectClassifDatasetCreator")
    parser.add_argument("--image_dir", type=str, default="../Semantic_bac_segment/data/source_restored/", 
                        help="Directory containing source images")
    parser.add_argument("--annotation_file", type=str, default="./data/objects_annotations/annotations.json", 
                        help="Path to COCO-format annotation JSON file")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation ratio")
    parser.add_argument("--resize_dim", type=int, default=64, help="Target dimension for objects")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    args = parser.parse_args()
    
    # Create dataset
    print(f"Creating dataset from {args.annotation_file}")
    dataset_creator = ObjectClassifDatasetCreator(
        args.image_dir,
        args.annotation_file,
        val_ratio=args.val_ratio,
        resize_dim=args.resize_dim
    )
    
    # Get categories
    categories = dataset_creator.get_categories()
    print(f"Categories: {categories}")
    
    # Create dataloaders
    train_loader, val_loader = dataset_creator.create_datasets(
        train_transform=None,
        val_transform=None,
        batch_size=args.batch_size
    )
    
    print(f"Train dataset size: {len(train_loader.dataset)} objects")
    print(f"Validation dataset size: {len(val_loader.dataset)} objects")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Test a training epoch
    print("\nRunning a test training epoch...")
    category_counts = {cat_id: 0 for cat_id in categories.keys()}
    
    for batch_idx, (images, labels) in enumerate(tqdm(train_loader)):
        # Count occurrences of each category
        for label in labels:
            category_counts[label.item()] += 1
            
        # Print info for first batch
        if batch_idx == 0:
            print(f"\nFirst batch shape: {images.shape}")
            print(f"First batch labels: {labels[:5]}...")
            
        # Break after 10 batches for quick testing
        if batch_idx >= 9:
            break
    
    print("\nCategory distribution in sampled batches:")
    for cat_id, count in category_counts.items():
        if count > 0:
            print(f"  {categories[cat_id]}: {count} objects")
    
    # Test a single validation batch
    print("\nTesting validation loader...")
    val_images, val_labels = next(iter(val_loader))
    print(f"Validation batch shape: {val_images.shape}")
    print(f"Validation batch labels: {val_labels[:5]}...")
    
    print("\nTest completed successfully!")