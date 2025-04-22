#!/usr/bin/env python3
import os
import time
import random
import argparse
import numpy as np
import subprocess
from pathlib import Path
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor
import tifffile
from jetraw_tools.jetraw_tiff import JetrawTiff

def get_random_images(folder_path, num_images=1, extension='.ome.p.tiff'):
    """Get random images using find command for efficiency"""
    try:
        # Use find command to get a list of files (much faster for large directories)
        cmd = f"find '{folder_path}' -name '*{extension}' -type f"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error running find command: {result.stderr}")
            return []
            
        # Split the output into lines and filter empty lines
        all_files = [Path(f) for f in result.stdout.strip().split('\n') if f]
        
        if not all_files:
            return []
            
        # Take random samples
        return random.sample(all_files, min(num_images, len(all_files)))
    except Exception as e:
        print(f"Error finding files in {folder_path}: {e}")
        return []

def extract_channels(image_path, output_path, channels=[0, 1]):
    """Extract specified channels from a jetraw tiff without loading the entire file"""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Open the jetraw tiff file
        tiff = JetrawTiff()
        tiff.open(str(image_path), "r")
        
        # Get image dimensions
        height = tiff.height
        width = tiff.width
        num_pages = tiff.pages
        
        # Check if we have enough pages for the requested channels
        max_channel = max(channels)
        if max_channel >= num_pages:
            print(f"Warning: Image {image_path} has {num_pages} pages, but channel {max_channel} was requested")
            tiff.close()
            return False
        
        # Extract the requested channels
        extracted_channels = []
        for channel in channels:
            if channel < num_pages:
                img_channel = tiff.read_page(channel)
                extracted_channels.append(img_channel)
        
        # Close the jetraw tiff
        tiff.close()
        
        # Save the extracted channels as a regular ome.tiff
        if extracted_channels:
            tifffile.imwrite(
                output_path,
                np.stack(extracted_channels),
                photometric='minisblack',
                metadata={'axes': 'CYX'}
            )
            return True
        
        return False
    
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False

def process_folder(folder, output_dir, num_images_per_folder=1):
    """Process a single folder to extract random images"""
    try:
        random_images = get_random_images(folder, num_images_per_folder)
        results = []
        
        for img_path in random_images:
            # Create output path with .ome.tiff extension
            rel_path = img_path.relative_to(folder) if img_path.is_relative_to(folder) else Path(img_path.name)
            output_path = Path(output_dir) / rel_path.with_suffix('').with_suffix('.ome.tiff')
            
            # Extract and save channels
            success = extract_channels(img_path, output_path)
            results.append((img_path, success))
            
        return results
    except Exception as e:
        print(f"Error processing folder {folder}: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description='Extract channels from Jetraw TIFF images')
    parser.add_argument('--input', type=str, default='/Volumes/RINF-QNAP/', 
                        help='Input directory containing .ome.p.tiff files')
    parser.add_argument('--output', type=str, default='./data/sample_images', 
                        help='Output directory for processed images')
    parser.add_argument('--num_images', type=int, default=5, 
                        help='Total number of images to process')
    parser.add_argument('--max_workers', type=int, default=4, 
                        help='Maximum number of parallel workers')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)
    # Get all folders in the input directory
    input_path = Path(args.input)
    # Filter out folders with '@' in their name
    folders = [f for f in input_path.iterdir() if f.is_dir() and '@' not in f.name]
    
    if not folders:
        print(f"No valid folders found in {args.input}")
        return
    
    print(f"Found {len(folders)} valid folders (excluding those with '@' in name)")
    
    # Calculate how many images to get from each folder
    num_folders = len(folders)
    images_per_folder = max(1, args.num_images // num_folders)
    # Process folders in parallel
    # Original parallel code (commented out)
    # with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
    #     futures = [executor.submit(process_folder, folder, args.output, images_per_folder) 
    #               for folder in folders]
    #     
    #     # Use tqdm to show progress
    #     processed_images = []
    #     for future in tqdm(futures, total=len(futures), desc="Processing folders"):
    #         processed_images.extend(future.result())
    #         
    #         # Stop if we've processed enough images
    #         if len(processed_images) >= args.num_images:
    #             break

    # Non-parallel version for debugging
    print(f"DEBUG: Processing {len(folders)} folders sequentially for debugging")
    print(f"DEBUG: Target is {args.num_images} images, {images_per_folder} per folder")
    
    processed_images = []
    for folder_idx, folder in enumerate(tqdm(folders, desc="Processing folders")):
        print(f"DEBUG: Processing folder {folder_idx+1}/{len(folders)}: {folder}")
        start_time = time.time()
        
        results = process_folder(folder, args.output, images_per_folder)
        
        elapsed = time.time() - start_time
        print(f"DEBUG: Folder {folder} completed in {elapsed:.2f}s, found {len(results)} images")
        
        processed_images.extend(results)
        
        # Stop if we've processed enough images
        if len(processed_images) >= args.num_images:
            print(f"DEBUG: Reached target of {args.num_images} images, stopping early")
            break

    # Print summary
    successful = sum(1 for _, success in processed_images if success)
    print(f"Processed {len(processed_images)} images, {successful} successful")
    
    # List the processed images
    print("\nProcessed images:")
    for img_path, success in processed_images[:10]:  # Show only first 10
        status = "✓" if success else "✗"
        print(f"{status} {img_path}")
    
    if len(processed_images) > 10:
        print(f"... and {len(processed_images) - 10} more")

if __name__ == "__main__":
    main()
