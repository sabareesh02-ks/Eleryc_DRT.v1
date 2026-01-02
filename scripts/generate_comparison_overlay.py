#!/usr/bin/env python3
"""
Generate TRUE overlay comparison plots
Combines multiple experiments/conditions into single overlaid plots
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Color palette matching the original overlays (red, blue, green, purple)
COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']

def create_drt_comparison(experiments_data, output_file):
    """
    Create DRT comparison overlay plot - just plot raw CSV data
    experiments_data: [(exp_name, condition, csv_file), ...]
    """
    plt.figure(figsize=(12, 8))
    plt.style.use('default')
    
    for idx, (exp_name, condition, csv_file) in enumerate(experiments_data):
        try:
            # Skip first 2 header rows (L,R) and use row 3 as column names
            df = pd.read_csv(csv_file, skiprows=2)
            
            # Find tau and gamma columns - just use raw data from CSV
            tau_col = None
            gamma_col = None
            
            for col in df.columns:
                col_lower = col.lower()
                if col_lower == 'tau' or 'tau' in col_lower:
                    tau_col = col
                if col_lower == 'gamma' or 'gamma' in col_lower:
                    gamma_col = col
            
            # Fallback: use first two numeric columns
            if not tau_col or not gamma_col:
                tau_col = df.columns[0]
                gamma_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
            
            color = COLORS[idx % len(COLORS)]
            # Plot raw data directly - no calculations
            plt.plot(df[tau_col], df[gamma_col], 'o-', 
                    linewidth=2.5, markersize=4,
                    color=color, 
                    label=f'{exp_name} @ {condition}')
        except Exception as e:
            print(f"Error processing {exp_name}: {e}")
            continue
    
    plt.xlabel('τ (s)', fontsize=13)
    plt.ylabel('Y (Ω)', fontsize=13)
    plt.title('DRT Comparison', fontsize=15, pad=15)
    plt.xscale('log')
    plt.grid(True, alpha=0.4, which='both')
    plt.legend(fontsize=11, loc='upper right', frameon=True, fancybox=True, shadow=False)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[+] Created DRT comparison: {output_file}")


def create_nyquist_comparison(experiments_data, output_file):
    """
    Create Nyquist comparison overlay plot - just plot raw CSV data
    experiments_data: [(exp_name, condition, csv_file), ...]
    """
    plt.figure(figsize=(12, 9))
    plt.style.use('default')
    
    for idx, (exp_name, condition, csv_file) in enumerate(experiments_data):
        try:
            df = pd.read_csv(csv_file)
            
            # Find Z_real and Z_imag columns - just use raw data from CSV
            z_real_col = None
            z_imag_col = None
            
            for col in df.columns:
                col_lower = col.lower()
                if 'mu_z_re' in col_lower:
                    z_real_col = col
                if 'mu_z_im' in col_lower:
                    z_imag_col = col
            
            # Fallback to other column names if mu_Z columns not found
            if not z_real_col or not z_imag_col:
                for col in df.columns:
                    col_lower = col.lower()
                    if 'z_re' in col_lower or 'zre' in col_lower or 're(z)' in col_lower or 'z_real' in col_lower:
                        z_real_col = col
                    if 'z_im' in col_lower or 'zim' in col_lower or 'im(z)' in col_lower or 'z_imag' in col_lower:
                        z_imag_col = col
            
            if not z_real_col or not z_imag_col:
                print(f"Could not find Z columns in {csv_file}")
                print(f"Available columns: {df.columns.tolist()}")
                continue
            
            color = COLORS[idx % len(COLORS)]
            # Plot raw data directly - no calculations
            plt.plot(df[z_real_col], -df[z_imag_col], 'o-', 
                    linewidth=2.5, markersize=4,
                    color=color,
                    label=f'{exp_name} @ {condition}')
        except Exception as e:
            print(f"Error processing {exp_name}: {e}")
            continue
    
    plt.xlabel('Z_re (Ω)', fontsize=13)
    plt.ylabel('-Z_im (Ω)', fontsize=13)
    plt.title('Nyquist Comparison', fontsize=15, pad=15)
    plt.grid(True, alpha=0.4)
    plt.axis('equal')
    plt.legend(fontsize=11, loc='upper right', frameon=True, fancybox=True, shadow=False)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[+] Created Nyquist comparison: {output_file}")


def main():
    """
    Usage: python generate_comparison_overlay.py exp1:cond1 exp2:cond2 ...
    Example: python generate_comparison_overlay.py "M1_M1-5psi-07162025:4A" "M1-r2_M1r2-5psi-07172025:4A"
    """
    if len(sys.argv) < 2:
        print("Usage: python generate_comparison_overlay.py exp1:cond1 exp2:cond2 ...")
        print('Example: python generate_comparison_overlay.py "M1_M1-5psi-07162025:4A" "M1-r2_M1r2-5psi-07172025:4A"')
        return
    
    base_dir = Path('separated_by_current')
    output_dir = Path('outputs/comparisons')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    drt_data = []
    eis_data = []
    
    # Parse arguments
    for arg in sys.argv[1:]:
        try:
            exp_name, condition = arg.split(':')
            exp_dir = base_dir / exp_name
            
            # Find DRT CSV (try both naming patterns)
            drt_dir = exp_dir / 'drt'
            found_drt = False
            for csv_file in drt_dir.glob(f'*{condition}*.csv'):
                if 'DRT' in csv_file.name.upper():
                    drt_data.append((exp_name, condition, csv_file))
                    found_drt = True
                    break
            
            if not found_drt:
                for csv_file in drt_dir.glob(f'*DRT*{condition}*.csv'):
                    drt_data.append((exp_name, condition, csv_file))
                    break
            
            # Find EIS CSV (try both naming patterns)
            eis_dir = exp_dir / 'eis'
            found_eis = False
            for csv_file in eis_dir.glob(f'*{condition}*.csv'):
                if 'EIS' in csv_file.name.upper():
                    eis_data.append((exp_name, condition, csv_file))
                    found_eis = True
                    break
            
            if not found_eis:
                for csv_file in eis_dir.glob(f'*EIS*{condition}*.csv'):
                    eis_data.append((exp_name, condition, csv_file))
                    break
                
        except Exception as e:
            print(f"Error parsing {arg}: {e}")
            continue
    
    if not drt_data and not eis_data:
        print("No data found!")
        return
    
    # Generate unique filename (use all experiments, prioritize drt_data)
    all_experiments = drt_data if drt_data else eis_data
    comparison_name = '_vs_'.join([f"{exp}_{cond}" for exp, cond, _ in all_experiments])
    if len(comparison_name) > 150:
        comparison_name = f"comparison_{len(all_experiments)}experiments"
    
    # Create comparison plots
    if drt_data:
        output_file = output_dir / f"{comparison_name}_DRT.png"
        create_drt_comparison(drt_data, output_file)
    
    if eis_data:
        output_file = output_dir / f"{comparison_name}_Nyquist.png"
        create_nyquist_comparison(eis_data, output_file)
    
    print("\n[SUCCESS] Comparison plots created!")
    print(f"Location: {output_dir}")


if __name__ == "__main__":
    main()

