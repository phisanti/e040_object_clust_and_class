import os
import argparse
import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
from skimage.transform import resize

sys.path.append(os.path.expanduser("~/switchdrive/boeck_lab_projects/e025_EverFocus"))
sys.path.append(os.path.expanduser("~/switchdrive/boeck_lab_projects/HiTMicTools/src"))
sys.path.append(os.path.expanduser("~/switchdrive/boeck_lab_projects//src"))

from HiTMicTools.model_components.focus_restorer import FocusRestorer
from HiTMicTools.model_arch.nafnet import NAFNet
import json

MODEL_CONFIG_PATH = '/Users/santiago/switchdrive/boeck_lab_projects/HiTMicTools/models/bf_models/NAFNet-medium-5-scaled01-ssim-patch256-bf_best_config.json'
MODEL_PATH = '/Users/santiago/switchdrive/boeck_lab_projects/HiTMicTools/models/bf_models/NAFNet-medium-5-scaled01-ssim-patch256-bf_best_model.pth'
MODEL_CONFIG_PATH2 = '/Users/santiago/Desktop/NAFNet-tiny-5-augmentation-scaled01-ssim-patch256-aug-bf_best_config.json'
MODEL_PATH2 = '/Users/santiago/Desktop/NAFNet-tiny-5-augmentation-scaled01-ssim-patch256-aug-bf_best_model.pth'

def compare_models(image_path, output_dir, channel, sigma, progress=None):
    """
    Process an image with two models and compare results.
    
    Args:
        image_path: Path to the input TIFF image
        output_dir: Directory to save the processed image
        channel: Channel to extract (0-based index)
        sigma: Sigma value for Gaussian blur
        progress: tqdm progress bar instance
    """
    try:
        # Read the image
        img = tifffile.imread(image_path)
        
        # Extract the specified channel if image is multi-channel
        if img.ndim > 2 and img.shape[-3] > channel:
            img = img[channel] if img.ndim == 3 else img[:, channel, :, :] if img.ndim == 4 else img
        
        # Convert to 32-bit float
        img = img.astype(np.float32)
        
        # Apply Gaussian blur for background correction
        blurred = gaussian_filter(img, sigma=sigma)
        
        # Compute difference of gaussians (original / blurred)
        dog = np.divide(img, blurred, out=np.zeros_like(img), where=blurred!=0)
        
        # Initialize models
        # Model 1
        with open(MODEL_CONFIG_PATH) as json_file:
            config1 = json.load(json_file)
        
        restore_model1 = NAFNet(**config1['model_args'])
        focus_restorer1 = FocusRestorer(
            model_path=MODEL_PATH,
            model_graph=restore_model1,
            patch_size=256,
            overlap_ratio=0.2,
            scale_method="range01",
            half_precision=True,
            scaler_args={"pmin": 1, "pmax": 99.8}
        )
        
        # Model 2
        with open(MODEL_CONFIG_PATH2) as json_file:
            config2 = json.load(json_file)
        
        restore_model2 = NAFNet(**config2['model_args'])
        focus_restorer2 = FocusRestorer(
            model_path=MODEL_PATH2,
            model_graph=restore_model2,
            patch_size=256,
            overlap_ratio=0.2,
            scale_method="range01",
            half_precision=True,
            scaler_args={"pmin": 1, "pmax": 99.8}
        )
        
        # Apply both models
        restored1 = focus_restorer1.predict(dog, batch_size=1)
        restored2 = focus_restorer2.predict(dog, batch_size=1)
        
        # Compute difference between restored images
        difference = np.abs(restored1 - restored2)
        
        # Create multipanel figure
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        
        # Normalize images for display
        def normalize_for_display(img):
            img_min, img_max = np.percentile(img, [1, 99.8])
            if img_min == img_max:
                return np.zeros_like(img)  # Return zeros if the image is constant
            return np.clip((img - img_min) / (img_max - img_min), 0, 1)
        
        # Display images
        axes[0].imshow(normalize_for_display(img), cmap='gray')
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        axes[1].imshow(normalize_for_display(dog), cmap='gray')
        axes[1].set_title('Background Corrected')
        axes[1].axis('off')
        
        axes[2].imshow(normalize_for_display(restored1), cmap='gray')
        axes[2].set_title('Model 1')
        axes[2].axis('off')
        
        axes[3].imshow(normalize_for_display(restored2), cmap='gray')
        axes[3].set_title('Model 2')
        axes[3].axis('off')
        
        axes[4].imshow(normalize_for_display(difference), cmap='hot')
        axes[4].set_title('Difference')
        axes[4].axis('off')
        
        plt.tight_layout()
        
        # Save the figure
        output_filename = f"restorecomp_{os.path.splitext(os.path.basename(image_path))[0]}.png"
        output_path = os.path.join(output_dir, output_filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        if progress:
            progress.update(1)
            
        return True
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        if progress:
            progress.update(1)
        return False

def main():
    parser = argparse.ArgumentParser(description='Compare two restoration models on TIFF images')
    parser.add_argument('input_dir', type=str, help='Input directory containing TIFF images')
    parser.add_argument('output_dir', type=str, help='Output directory for processed images')
    parser.add_argument('--channel', type=int, default=0, help='Channel to extract (0-based index)')
    parser.add_argument('--sigma', type=float, default=5.0, help='Sigma value for Gaussian blur')
    
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
            compare_models(image_path, args.output_dir, args.channel, args.sigma, progress)
    
    print(f"Processing complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
