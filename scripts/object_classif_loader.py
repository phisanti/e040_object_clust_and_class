import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import tifffile

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class ObjectClassifDataset(Dataset):
    def __init__(
        self,
        annotations: List[Dict],
        image_dir: str,
        image_info: Dict[int, Dict],
        transform: Optional[Callable] = None,
        resize_dim: int = 64
    ):
        self.annotations = annotations
        self.image_dir = image_dir
        self.image_info = image_info
        self.transform = transform
        self.resize_dim = resize_dim
        self.image_cache = {}  # Cache for loaded images
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        image_id = annotation["image_id"]
        category_id = annotation["category_id"]
        bbox = annotation["bbox"]
        
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
                # Remove oldest item from cache (first in dictionary)
                remove_id = next(iter(self.image_cache))
                if remove_id != image_id:
                    del self.image_cache[remove_id]

        image = self.image_cache[image_id]

        # Extract object using bbox [x, y, width, height]
        x, y, w, h = [int(v) for v in bbox]
        obj_img = image[y:y+h, x:x+w]

        # Add channel dimension if needed
        if obj_img.dim() == 2:
            obj_img = obj_img.unsqueeze(0)  # [H,W] -> [1,H,W]
        
        # Resize
        if obj_img.shape[-2] != self.resize_dim or obj_img.shape[-1] != self.resize_dim:
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
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create training and validation datasets/dataloaders.
        
        Args:
            train_transform: Transformations to apply to training data
            val_transform: Transformations to apply to validation data
            batch_size: Batch size for dataloaders
            
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
            resize_dim=self.resize_dim
        )
        
        val_dataset = ObjectClassifDataset(
            val_annotations,
            self.image_dir,
            self.image_info,
            transform=val_transform,
            resize_dim=self.resize_dim
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