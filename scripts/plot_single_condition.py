#!/usr/bin/env python3
"""
Plot a single experiment at a specific condition (e.g., M1 at 4A)
Shows both DRT and Nyquist plots
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def plot_drt_single_condition(csv_file, experiment_name, condition, output_file):
    """Create clean DRT plot for single condition"""
    try:
        # Read CSV (skip first 2 header rows)
        df = pd.read_csv(csv_file, skiprows=2)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        plt.style.use('default')
        
        # Plot data
        plt.plot(df['tau'], df['gamma'], 'o-', 
                linewidth=2.5, markersize=4,
                color='#e41a1c', label=f'{experiment_name} @ {condition}')
        
        # Format
        plt.xlabel('τ (s)', fontsize=13)
        plt.ylabel('Y (Ω)', fontsize=13)
        plt.title(f'{experiment_name} - DRT @ {condition}', fontsize=15, pad=15)
        plt.xscale('log')
        plt.grid(True, alpha=0.4, which='both')
        plt.legend(fontsize=12, loc='upper right', frameon=True, fancybox=True)
        plt.tight_layout()
        
        # Save
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"[+] DRT plot saved: {output_file}")
        return True
        
    except Exception as e:
        print(f"[!] Error creating DRT plot: {e}")
        return False


def plot_nyquist_single_condition(csv_file, experiment_name, condition, output_file):
    """Create clean Nyquist plot for single condition"""
    try:
        # Read CSV
        df = pd.read_csv(csv_file)
        
        # Find Z columns
        z_real_col = None
        z_imag_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if 'z_re' in col_lower or 'mu_z_re' in col_lower:
                z_real_col = col
            if 'z_im' in col_lower or 'mu_z_im' in col_lower:
                z_imag_col = col
        
        if not z_real_col or not z_imag_col:
            print(f"[!] Could not find Z columns in {csv_file}")
            return False
        
        # Create plot
        plt.figure(figsize=(12, 9))
        plt.style.use('default')
        
        # Plot data
        plt.plot(df[z_real_col], -df[z_imag_col], 'o-', 
                linewidth=2.5, markersize=4,
                color='#e41a1c', label=f'{experiment_name} @ {condition}')
        
        # Format
        plt.xlabel('Z_re (Ω)', fontsize=13)
        plt.ylabel('-Z_im (Ω)', fontsize=13)
        plt.title(f'{experiment_name} - Nyquist @ {condition}', fontsize=15, pad=15)
        plt.grid(True, alpha=0.4)
        plt.axis('equal')
        plt.legend(fontsize=12, loc='upper right', frameon=True, fancybox=True)
        plt.tight_layout()
        
        # Save
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"[+] Nyquist plot saved: {output_file}")
        return True
        
    except Exception as e:
        print(f"[!] Error creating Nyquist plot: {e}")
        return False


def main():
    """
    Usage: python plot_single_condition.py <experiment_name> <condition>
    Example: python plot_single_condition.py M1_M1-5psi-07162025 4A
    """
    if len(sys.argv) < 3:
        print("Usage: python plot_single_condition.py <experiment_name> <condition>")
        print("\nExample:")
        print('  python plot_single_condition.py "M1_M1-5psi-07162025" "4A"')
        return
    
    experiment_name = sys.argv[1]
    condition = sys.argv[2]
    
    print("="*70)
    print(f"Plotting {experiment_name} @ {condition}")
    print("="*70)
    
    # Find CSV files
    base_dir = Path('separated_by_current') / experiment_name
    
    if not base_dir.exists():
        print(f"[!] Experiment folder not found: {base_dir}")
        return
    
    # Get short name for file output
    exp_short = experiment_name.split('_')[0]
    
    # Find DRT CSV
    drt_dir = base_dir / 'drt'
    drt_csv = None
    if drt_dir.exists():
        for csv_file in drt_dir.glob(f'*{condition}*DRT*.csv'):
            drt_csv = csv_file
            break
    
    # Find EIS CSV
    eis_dir = base_dir / 'eis'
    eis_csv = None
    if eis_dir.exists():
        for csv_file in eis_dir.glob(f'*{condition}*EIS*.csv'):
            eis_csv = csv_file
            break
    
    # Create plots
    if drt_csv:
        output_drt = f"{exp_short}_{condition}_DRT_clean.png"
        plot_drt_single_condition(drt_csv, exp_short, condition, output_drt)
    else:
        print(f"[!] DRT CSV not found for {condition}")
    
    if eis_csv:
        output_eis = f"{exp_short}_{condition}_Nyquist_clean.png"
        plot_nyquist_single_condition(eis_csv, exp_short, condition, output_eis)
    else:
        print(f"[!] EIS CSV not found for {condition}")
    
    print("\n" + "="*70)
    print("[COMPLETE] Check output files!")
    print("="*70)


if __name__ == "__main__":
    main()



