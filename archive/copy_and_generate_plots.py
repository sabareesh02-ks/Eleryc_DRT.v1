#!/usr/bin/env python3
"""
Copy CSV data and generate condition-specific PNG plots
From separated_by_current folder to outputs folder
"""

import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json

def create_nyquist_plot(csv_file, output_png, experiment_name, condition):
    """Create Nyquist plot from EIS CSV file"""
    try:
        df = pd.read_csv(csv_file)
        
        # Find Z_real and Z_imag columns (case insensitive)
        z_real_col = None
        z_imag_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if 'z_re' in col_lower or 'zre' in col_lower or 're(z)' in col_lower or 'z_real' in col_lower:
                z_real_col = col
            if 'z_im' in col_lower or 'zim' in col_lower or 'im(z)' in col_lower or '-im(z)' in col_lower or 'z_imag' in col_lower:
                z_imag_col = col
        
        if not z_real_col or not z_imag_col:
            print(f"  [!] Could not find Z_real/Z_imag columns in {csv_file.name}")
            return False
        
        # Create plot
        plt.figure(figsize=(8, 6))
        plt.plot(df[z_real_col], -df[z_imag_col], 'o-', linewidth=2, markersize=4, color='#1e40af')
        plt.xlabel('Z_real (Ω)', fontsize=12, fontweight='bold')
        plt.ylabel('-Z_imag (Ω)', fontsize=12, fontweight='bold')
        plt.title(f'{experiment_name}\nNyquist Plot @ {condition}', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(output_png, dpi=150, bbox_inches='tight')
        plt.close()
        return True
        
    except Exception as e:
        print(f"  [X] Error creating Nyquist plot: {e}")
        plt.close()
        return False


def create_drt_plot(csv_file, output_png, experiment_name, condition):
    """Create DRT plot from DRT CSV file"""
    try:
        df = pd.read_csv(csv_file)
        
        # Usually first column is frequency/time, second is DRT
        x_col = df.columns[0]
        y_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        
        # Create plot
        plt.figure(figsize=(8, 6))
        plt.plot(df[x_col], df[y_col], 'o-', linewidth=2, markersize=4, color='#06b6d4')
        plt.xlabel(x_col, fontsize=12, fontweight='bold')
        plt.ylabel('DRT Value', fontsize=12, fontweight='bold')
        plt.title(f'{experiment_name}\nDRT @ {condition}', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
        plt.tight_layout()
        plt.savefig(output_png, dpi=150, bbox_inches='tight')
        plt.close()
        return True
        
    except Exception as e:
        print(f"  [X] Error creating DRT plot: {e}")
        plt.close()
        return False


def process_experiment(source_dir, output_dir):
    """Process one experiment folder"""
    exp_name = source_dir.name
    print(f"\n[*] Processing: {exp_name}")
    
    # Create output directory if it doesn't exist
    output_exp_dir = output_dir / exp_name
    output_exp_dir.mkdir(exist_ok=True)
    
    conditions = set()
    
    # Process DRT files
    drt_dir = source_dir / 'drt'
    if drt_dir.exists():
        for csv_file in drt_dir.glob('*.csv'):
            # Extract condition from filename (e.g., M1_12A_DRT.csv -> 12A)
            filename = csv_file.stem
            parts = filename.split('_')
            condition = None
            for part in parts:
                if 'A' in part and any(c.isdigit() for c in part):
                    condition = part
                    break
            
            if condition:
                conditions.add(condition)
                output_png = output_exp_dir / f"{exp_name}_{condition}_DRT.png"
                if create_drt_plot(csv_file, output_png, exp_name, condition):
                    print(f"  [+] Created DRT plot: {condition}")
    
    # Process EIS files
    eis_dir = source_dir / 'eis'
    if eis_dir.exists():
        for csv_file in eis_dir.glob('*.csv'):
            # Extract condition from filename
            filename = csv_file.stem
            parts = filename.split('_')
            condition = None
            for part in parts:
                if 'A' in part and any(c.isdigit() for c in part):
                    condition = part
                    break
            
            if condition:
                conditions.add(condition)
                output_png = output_exp_dir / f"{exp_name}_{condition}_Nyquist.png"
                if create_nyquist_plot(csv_file, output_png, exp_name, condition):
                    print(f"  [+] Created Nyquist plot: {condition}")
    
    # Copy overlay PNG files if they exist
    for png_file in source_dir.glob('*.png'):
        dest_png = output_exp_dir / png_file.name
        if not dest_png.exists():
            shutil.copy2(png_file, dest_png)
            print(f"  [+] Copied overlay: {png_file.name}")
    
    # Save conditions metadata
    if conditions:
        conditions_file = output_exp_dir / 'conditions.json'
        with open(conditions_file, 'w') as f:
            json.dump(sorted(list(conditions)), f)
        print(f"  [+] Saved conditions: {sorted(conditions)}")
    
    return len(conditions)


def main():
    """Main function"""
    print("=" * 70)
    print("  COPY AND GENERATE COMPARISON PLOTS")
    print("  From separated_by_current to outputs folder")
    print("=" * 70)
    print()
    
    source_base = Path('separated_by_current')
    output_base = Path('outputs')
    
    if not source_base.exists():
        print("[ERROR] 'separated_by_current' folder not found!")
        return
    
    output_base.mkdir(exist_ok=True)
    
    # Get all experiment folders
    exp_folders = [d for d in source_base.iterdir() if d.is_dir()]
    total = len(exp_folders)
    processed = 0
    total_conditions = 0
    
    print(f"Found {total} experiments to process")
    print()
    
    for exp_dir in exp_folders:
        num_conditions = process_experiment(exp_dir, output_base)
        if num_conditions > 0:
            processed += 1
            total_conditions += num_conditions
    
    print()
    print("=" * 70)
    print(f"[SUCCESS] Complete!")
    print(f"   Processed: {processed}/{total} experiments")
    print(f"   Generated: {total_conditions} condition sets")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Refresh your browser")
    print("  2. Click the 'Compare' tab")
    print("  3. Select experiments and conditions")
    print("  4. Click 'Show Comparison'")
    print("  5. Enjoy your comparison plots!")
    print()


if __name__ == "__main__":
    main()

