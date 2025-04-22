import os
import argparse
import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import sys

sys.path.append(os.path.expanduser("~/switchdrive/boeck_lab_projects/e025_EverFocus"))
sys.path.append(os.path.expanduser("~/switchdrive/boeck_lab_projects/HiTMicTools/src"))
sys.path.append(os.path.expanduser("~/switchdrive/boeck_lab_projects//src"))

from HiTMicTools.model_components.focus_restorer import FocusRestorer
from HiTMicTools.model_arch.nafnet import NAFNet
import json

MODEL_CONFIG_PATH = '/Users/santiago/switchdrive/boeck_lab_projects/HiTMicTools/models/bf_models/NAFNet-medium-5-scaled01-ssim-patch256-bf_best_config.json'
MODEL_PATH = '/Users/santiago/switchdrive/boeck_lab_projects/HiTMicTools/models/bf_models/NAFNet-medium-5-scaled01-ssim-patch256-bf_best_model.pth'

def difference_of_gaussians(image_path, output_dir, channel, sigma, progress=None, restore=False):
    """
    Apply difference of gaussians to a TIFF image.
    
    Args:
        image_path: Path to the input TIFF image
        output_dir: Directory to save the processed image
        channel: Channel to extract (0-based index)
        sigma: Sigma value for Gaussian blur
        progress: tqdm progress bar instance
        restore: Whether to apply focus restoration
    """
    try:
        # Read the image
        img = tifffile.imread(image_path)
        
        # Extract the specified channel if image is multi-channel
        if img.ndim > 2 and img.shape[-3] > channel:
            img = img[channel] if img.ndim == 3 else img[:, channel, :, :] if img.ndim == 4 else img
        
        # Convert to 32-bit float
        img = img.astype(np.float32)
        
        # Apply Gaussian blur
        blurred = gaussian_filter(img, sigma=sigma)
        
        # Compute difference of gaussians (original / blurred)
        dog = np.divide(img, blurred, out=np.zeros_like(img), where=blurred!=0)
        
        # Apply focus restoration if requested
        if restore:
            with open(MODEL_CONFIG_PATH) as json_file:
                config = json.load(json_file)
            
            restore_model = NAFNet(**config['model_args'])
            focus_restorer = FocusRestorer(
                model_path=MODEL_PATH,
                model_graph=restore_model,
                patch_size=256,
                overlap_ratio=0.2,
                scale_method="range01",
                half_precision=True,
                scaler_args={"pmin": 1, "pmax": 99.8}
            )
            
            dog = focus_restorer.predict(dog, batch_size=1)
            output_path = os.path.join(output_dir, f"restored_{os.path.basename(image_path)}")
        else:
            output_path = os.path.join(output_dir, f"dog_{os.path.basename(image_path)}")
        
        # Save the result
        tifffile.imwrite(output_path, dog)
        
        if progress:
            progress.update(1)
            
        return True
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        if progress:
            progress.update(1)
        return False

def main():
    parser = argparse.ArgumentParser(description='Apply Difference of Gaussians to TIFF images')
    parser.add_argument('input_dir', type=str, help='Input directory containing TIFF images')
    parser.add_argument('output_dir', type=str, help='Output directory for processed images')
    parser.add_argument('--channel', type=int, default=0, help='Channel to extract (0-based index)')
    parser.add_argument('--sigma', type=float, default=5.0, help='Sigma value for Gaussian blur')
    parser.add_argument('--restore', action='store_true', help='Apply focus restoration after background removal')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get list of TIFF files including subfolders
    tiff_files = []
    for root, _, files in os.walk(args.input_dir):
        for file in files:
            if file.lower().endswith(('.tif', '.tiff')):
                tiff_files.append(os.path.join(root, file))
    
    if not tiff_files:
        print(f"No TIFF files found in {args.input_dir}")
        return
    
    print(f"Found {len(tiff_files)} TIFF files. Processing...")
    
    # Process files with progress bar
    with tqdm(total=len(tiff_files)) as progress:
        for image_path in tiff_files:
            difference_of_gaussians(image_path, args.output_dir, args.channel, args.sigma, progress, args.restore)
    
    print(f"Processing complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
