#!/usr/bin/env python3
"""
Overlay experiment images with transparency using OpenCV and NumPy
"""

import cv2
import numpy as np
from pathlib import Path
import sys

def overlay_images(image_paths, output_file='comparison_overlay.png', alpha=0.5):
    """
    Overlay multiple images with transparency
    
    Args:
        image_paths: List of image file paths
        output_file: Output filename
        alpha: Transparency (0.5 = 50% transparent)
    """
    if len(image_paths) < 2:
        print("[!] Need at least 2 images to overlay")
        return False
    
    print(f"\n[*] Loading {len(image_paths)} images...")
    
    # Load first image as base
    base_img = cv2.imread(str(image_paths[0]))
    if base_img is None:
        print(f"[!] Could not load: {image_paths[0]}")
        return False
    
    print(f"  [+] Base: {Path(image_paths[0]).name}")
    height, width = base_img.shape[:2]
    
    # Convert to float for blending
    result = base_img.astype(float)
    
    # Overlay each additional image
    for i, img_path in enumerate(image_paths[1:], 1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [!] Could not load: {img_path}")
            continue
        
        print(f"  [+] Overlay {i}: {Path(img_path).name}")
        
        # Resize if dimensions don't match
        if img.shape[:2] != (height, width):
            print(f"    [*] Resizing from {img.shape[:2]} to {(height, width)}")
            img = cv2.resize(img, (width, height))
        
        # Blend with transparency
        img_float = img.astype(float)
        result = cv2.addWeighted(result, 1-alpha, img_float, alpha, 0)
    
    # Convert back to uint8
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    # Save result
    cv2.imwrite(output_file, result)
    print(f"\n[SUCCESS] Overlay saved: {output_file}")
    return True


def overlay_experiments(exp_names, plot_type='DRT', alpha=0.5):
    """
    Overlay specific experiments
    
    Args:
        exp_names: List of experiment names
        plot_type: 'DRT' or 'EIS'
        alpha: Transparency level
    """
    outputs_dir = Path('outputs')
    image_paths = []
    
    for exp_name in exp_names:
        exp_dir = outputs_dir / exp_name
        if not exp_dir.exists():
            print(f"[!] Experiment not found: {exp_name}")
            continue
        
        # Find overlay image
        overlay_file = exp_dir / f"{exp_name}_{plot_type}_overlay.png"
        if not overlay_file.exists():
            print(f"[!] Overlay not found: {overlay_file}")
            continue
        
        image_paths.append(overlay_file)
    
    if len(image_paths) < 2:
        print("[!] Need at least 2 valid images")
        return False
    
    # Generate output filename
    exp_names_short = '_vs_'.join([e[:15] for e in exp_names])
    output_file = f"opencv_overlay_{exp_names_short}_{plot_type}.png"
    
    return overlay_images(image_paths, output_file, alpha)


def main():
    """
    Usage examples:
    python overlay_images_opencv.py M1_M1-5psi-07162025 M1-r2_M1r2-5psi-07172025
    python overlay_images_opencv.py M1_M1-5psi-07162025 M1-r2_M1r2-5psi-07172025 M1-r3_M1r3-5psi-07182025
    """
    if len(sys.argv) < 3:
        print("Usage: python overlay_images_opencv.py <experiment1> <experiment2> [experiment3...]")
        print("\nExample:")
        print('  python overlay_images_opencv.py "M1_M1-5psi-07162025" "M1-r2_M1r2-5psi-07172025"')
        return
    
    experiments = sys.argv[1:]
    
    print("="*70)
    print("OpenCV Image Overlay with Transparency")
    print("="*70)
    print(f"Experiments: {', '.join(experiments)}")
    print(f"Alpha (transparency): 0.5")
    print()
    
    # Create DRT overlay
    print("\n--- DRT Overlay ---")
    overlay_experiments(experiments, plot_type='DRT', alpha=0.5)
    
    # Create EIS overlay
    print("\n--- Nyquist (EIS) Overlay ---")
    overlay_experiments(experiments, plot_type='EIS', alpha=0.5)
    
    print("\n" + "="*70)
    print("[COMPLETE] Check output files!")
    print("="*70)


if __name__ == "__main__":
    main()



