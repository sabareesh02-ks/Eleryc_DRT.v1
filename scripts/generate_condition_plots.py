#!/usr/bin/env python3
"""
Automatic Condition Plot Generator for Eleryc
Splits overlay plots into individual condition plots for comparison
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
import re

def detect_conditions_from_data(df):
    """
    Detect different conditions (currents) from the dataframe
    Looks for columns or groups that indicate different conditions
    """
    # Standard conditions to always check for
    standard_conditions = ['4A', '8A', '12A', '16A', '20A', '24A']
    conditions = []
    
    # Common column patterns that indicate conditions
    # Adjust these based on your actual data structure
    condition_indicators = ['Current', 'current', 'A', 'Condition', 'condition', 'I']
    
    # Check if there's a condition column
    for col in df.columns:
        if any(indicator in col for indicator in condition_indicators):
            # Extract unique conditions
            detected = df[col].unique().tolist()
            # Convert to standard format (e.g., 12 -> 12A)
            conditions = []
            for c in detected:
                c_str = str(c)
                if not c_str.endswith('A') and any(char.isdigit() for char in c_str):
                    conditions.append(f"{c_str}A")
                else:
                    conditions.append(c_str)
            return conditions, col
    
    # If no condition column, check if columns themselves indicate conditions
    # e.g., "Z_real_12A", "Z_real_15A"
    condition_pattern = {}
    for col in df.columns:
        # Look for patterns like _12A, _15A, etc.
        if 'A' in col and any(c.isdigit() for c in col):
            parts = col.split('_')
            for part in parts:
                if 'A' in part and any(c.isdigit() for c in part):
                    # Extract just the number+A part
                    match = re.search(r'(\d+\.?\d*)A', part)
                    if match:
                        condition = match.group(0)
                        if condition not in condition_pattern:
                            condition_pattern[condition] = []
                        condition_pattern[condition].append(col)
    
    if condition_pattern:
        return list(condition_pattern.keys()), 'column_based'
    
    # If nothing detected, return standard conditions as fallback
    return standard_conditions, None


def create_nyquist_plot(df, condition, output_file, experiment_name):
    """Create Nyquist plot for a specific condition"""
    try:
        plt.figure(figsize=(8, 6))
        
        # Find Z_real and Z_imag columns
        z_real_col = None
        z_imag_col = None
        
        for col in df.columns:
            if 'z_real' in col.lower() or 'zre' in col.lower():
                z_real_col = col
            if 'z_imag' in col.lower() or 'zim' in col.lower():
                z_imag_col = col
        
        if z_real_col and z_imag_col:
            plt.plot(df[z_real_col], -df[z_imag_col], 'o-', linewidth=2, markersize=6)
            plt.xlabel('Z_real (Ω)', fontsize=12)
            plt.ylabel('-Z_imag (Ω)', fontsize=12)
            plt.title(f'{experiment_name} - Nyquist Plot @ {condition}', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.axis('equal')
            plt.tight_layout()
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
            return True
        else:
            print(f"  ⚠ Could not find Z_real/Z_imag columns")
            plt.close()
            return False
            
    except Exception as e:
        print(f"  ✗ Error creating Nyquist plot: {e}")
        plt.close()
        return False


def create_drt_plot(df, condition, output_file, experiment_name):
    """Create DRT plot for a specific condition"""
    try:
        plt.figure(figsize=(8, 6))
        
        # Find frequency/time and DRT columns
        x_col = df.columns[0]  # Usually frequency or time
        y_col = None
        
        for col in df.columns[1:]:
            if 'drt' in col.lower() or 'gamma' in col.lower():
                y_col = col
                break
        
        if not y_col:
            y_col = df.columns[1]
        
        plt.plot(df[x_col], df[y_col], 'o-', linewidth=2, markersize=6)
        plt.xlabel(x_col, fontsize=12)
        plt.ylabel('DRT Value', fontsize=12)
        plt.title(f'{experiment_name} - DRT @ {condition}', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        return True
            
    except Exception as e:
        print(f"  ✗ Error creating DRT plot: {e}")
        plt.close()
        return False


def process_experiment(exp_dir):
    """Process one experiment folder"""
    exp_name = exp_dir.name
    print(f"\n📁 Processing: {exp_name}")
    
    # Find CSV files
    drt_csv = None
    nyquist_csv = None
    
    for file in exp_dir.glob('*.csv'):
        if 'drt' in file.name.lower():
            drt_csv = file
        elif 'eis' in file.name.lower() or 'nyquist' in file.name.lower():
            nyquist_csv = file
    
    if not drt_csv and not nyquist_csv:
        print("  ⚠ No CSV files found")
        return []
    
    conditions_info = {}
    
    # Process Nyquist data
    if nyquist_csv:
        print(f"  📊 Processing Nyquist data...")
        try:
            df = pd.read_csv(nyquist_csv)
            conditions, condition_col = detect_conditions_from_data(df)
            
            print(f"  ✓ Found conditions: {conditions}")
            
            for condition in conditions:
                # Filter data for this condition
                if condition_col and condition_col != 'column_based':
                    condition_df = df[df[condition_col] == condition]
                else:
                    condition_df = df
                
                # Generate plot
                output_file = exp_dir / f"{exp_name}_{condition}_Nyquist.png"
                if create_nyquist_plot(condition_df, condition, output_file, exp_name):
                    print(f"  ✓ Created: {condition}_Nyquist.png")
                    if condition not in conditions_info:
                        conditions_info[condition] = {}
                    conditions_info[condition]['nyquist'] = f"{exp_name}_{condition}_Nyquist.png"
        
        except Exception as e:
            print(f"  ✗ Error processing Nyquist: {e}")
    
    # Process DRT data
    if drt_csv:
        print(f"  📊 Processing DRT data...")
        try:
            df = pd.read_csv(drt_csv)
            conditions, condition_col = detect_conditions_from_data(df)
            
            for condition in conditions:
                # Filter data for this condition
                if condition_col and condition_col != 'column_based':
                    condition_df = df[df[condition_col] == condition]
                else:
                    condition_df = df
                
                # Generate plot
                output_file = exp_dir / f"{exp_name}_{condition}_DRT.png"
                if create_drt_plot(condition_df, condition, output_file, exp_name):
                    print(f"  ✓ Created: {condition}_DRT.png")
                    if condition not in conditions_info:
                        conditions_info[condition] = {}
                    conditions_info[condition]['drt'] = f"{exp_name}_{condition}_DRT.png"
        
        except Exception as e:
            print(f"  ✗ Error processing DRT: {e}")
    
    # Save conditions metadata
    if conditions_info:
        metadata_file = exp_dir / "conditions.json"
        with open(metadata_file, 'w') as f:
            json.dump(list(conditions_info.keys()), f)
        print(f"  ✓ Saved conditions metadata")
    
    return list(conditions_info.keys())


def main():
    """Main function"""
    print("=" * 70)
    print("  AUTOMATIC CONDITION PLOT GENERATOR")
    print("  Splits overlay plots into individual condition plots")
    print("=" * 70)
    print()
    
    base_dir = Path('outputs')
    
    if not base_dir.exists():
        print("❌ Error: 'outputs' folder not found!")
        return
    
    experiments = [d for d in base_dir.iterdir() if d.is_dir()]
    total = len(experiments)
    processed = 0
    
    print(f"Found {total} experiments to process")
    print()
    
    all_conditions = set()
    
    for exp_dir in experiments:
        conditions = process_experiment(exp_dir)
        if conditions:
            all_conditions.update(conditions)
            processed += 1
    
    print()
    print("=" * 70)
    print(f"✅ Completed! Processed {processed}/{total} experiments")
    print(f"📊 Conditions found: {sorted(all_conditions)}")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Check the 'outputs' folders for individual condition PNG files")
    print("  2. Refresh your experiment viewer")
    print("  3. Use the Comparison tab to compare conditions!")
    print()


if __name__ == "__main__":
    main()

