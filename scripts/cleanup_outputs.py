#!/usr/bin/env python3
"""
Clean up individual condition plots, keep only overlay plots
"""

from pathlib import Path
import os

def cleanup_outputs():
    """Remove individual condition plots, keep overlays"""
    outputs_dir = Path('outputs')
    
    if not outputs_dir.exists():
        print("[!] Outputs directory not found")
        return
    
    deleted_count = 0
    kept_count = 0
    
    # Process each experiment folder
    for exp_dir in outputs_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        
        print(f"\n[*] Processing: {exp_dir.name}")
        
        # Process files in the experiment folder
        for file in exp_dir.iterdir():
            if file.is_file():
                filename = file.name
                
                # Keep overlay plots and conditions.json
                if ('_DRT_overlay.png' in filename or 
                    '_EIS_overlay.png' in filename or
                    'conditions.json' in filename):
                    kept_count += 1
                    print(f"  [KEEP] {filename}")
                else:
                    # Delete individual condition plots
                    try:
                        os.remove(file)
                        deleted_count += 1
                        print(f"  [DEL]  {filename}")
                    except Exception as e:
                        print(f"  [ERR]  Could not delete {filename}: {e}")
    
    print("\n" + "="*70)
    print(f"[SUCCESS] Cleanup complete!")
    print(f"  Deleted: {deleted_count} files")
    print(f"  Kept:    {kept_count} files")
    print("="*70)

if __name__ == "__main__":
    cleanup_outputs()



