#!/usr/bin/env python3
"""
Interactive Plot Generator for Eleryc Experiment Viewer
Converts experiment data to interactive Plotly HTML plots with hover tooltips
"""

import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from pathlib import Path

def create_interactive_drt_plot(data_file, output_file, experiment_name):
    """
    Create interactive DRT plot from data file
    
    Parameters:
    - data_file: Path to CSV/Excel file with DRT data
    - output_file: Path to save HTML plot
    - experiment_name: Name of experiment for title
    
    Expected data format:
    - Column 1: Frequency or Time (x-axis)
    - Column 2+: DRT values (y-axis) for different conditions
    """
    try:
        # Read data (supports CSV, Excel)
        if data_file.endswith('.csv'):
            df = pd.read_csv(data_file)
        elif data_file.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(data_file)
        else:
            print(f"Unsupported file format: {data_file}")
            return False
        
        # Create interactive plot
        fig = go.Figure()
        
        # Assuming first column is x-axis, rest are y-values
        x_col = df.columns[0]
        
        # Add traces for each data series
        for col in df.columns[1:]:
            fig.add_trace(go.Scatter(
                x=df[x_col],
                y=df[col],
                mode='lines+markers',
                name=col,
                hovertemplate='<b>%{fullData.name}</b><br>' +
                              f'{x_col}: %{{x}}<br>' +
                              'Value: %{y}<br>' +
                              '<extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title=f'{experiment_name} - DRT Analysis',
            xaxis_title=x_col,
            yaxis_title='DRT Value',
            hovermode='closest',
            template='plotly_white',
            height=600,
            showlegend=True,
            font=dict(size=12)
        )
        
        # Save as HTML
        fig.write_html(output_file, include_plotlyjs='cdn')
        print(f"✓ Created: {output_file}")
        return True
        
    except Exception as e:
        print(f"✗ Error processing {data_file}: {e}")
        return False


def create_interactive_nyquist_plot(data_file, output_file, experiment_name):
    """
    Create interactive Nyquist plot (Nyquist + Bode) from data file
    
    Expected data format:
    - Frequency, Z_real, Z_imag, |Z|, Phase
    """
    try:
        # Read data
        if data_file.endswith('.csv'):
            df = pd.read_csv(data_file)
        elif data_file.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(data_file)
        else:
            print(f"Unsupported file format: {data_file}")
            return False
        
        # Create subplots: Nyquist and Bode
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Nyquist Plot', 'Bode Magnitude', 
                          'Bode Phase', 'Complex Impedance'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # Nyquist Plot (Z_real vs -Z_imag)
        fig.add_trace(
            go.Scatter(
                x=df['Z_real'] if 'Z_real' in df.columns else df.iloc[:, 1],
                y=-df['Z_imag'] if 'Z_imag' in df.columns else -df.iloc[:, 2],
                mode='lines+markers',
                name='Nyquist',
                hovertemplate='Z_real: %{x:.3f}<br>-Z_imag: %{y:.3f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Bode Magnitude
        fig.add_trace(
            go.Scatter(
                x=df['Frequency'] if 'Frequency' in df.columns else df.iloc[:, 0],
                y=df['|Z|'] if '|Z|' in df.columns else df.iloc[:, 3],
                mode='lines+markers',
                name='|Z|',
                hovertemplate='Freq: %{x:.2e} Hz<br>|Z|: %{y:.3f}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # Bode Phase
        fig.add_trace(
            go.Scatter(
                x=df['Frequency'] if 'Frequency' in df.columns else df.iloc[:, 0],
                y=df['Phase'] if 'Phase' in df.columns else df.iloc[:, 4],
                mode='lines+markers',
                name='Phase',
                hovertemplate='Freq: %{x:.2e} Hz<br>Phase: %{y:.2f}°<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Update axes
        fig.update_xaxes(title_text="Z_real (Ω)", row=1, col=1)
        fig.update_yaxes(title_text="-Z_imag (Ω)", row=1, col=1)
        fig.update_xaxes(title_text="Frequency (Hz)", type="log", row=1, col=2)
        fig.update_yaxes(title_text="|Z| (Ω)", type="log", row=1, col=2)
        fig.update_xaxes(title_text="Frequency (Hz)", type="log", row=2, col=1)
        fig.update_yaxes(title_text="Phase (°)", row=2, col=1)
        
        # Update layout
        fig.update_layout(
            title_text=f'{experiment_name} - Nyquist Analysis',
            height=800,
            showlegend=False,
            template='plotly_white',
            hovermode='closest'
        )
        
        # Save as HTML
        fig.write_html(output_file, include_plotlyjs='cdn')
        print(f"✓ Created: {output_file}")
        return True
        
    except Exception as e:
        print(f"✗ Error processing {data_file}: {e}")
        return False


def process_all_experiments(base_path='outputs'):
    """
    Process all experiments and create interactive plots
    """
    print("=" * 60)
    print("Interactive Plot Generator - Eleryc Experiment Viewer")
    print("=" * 60)
    print()
    
    base_dir = Path(base_path)
    
    if not base_dir.exists():
        print(f"Error: Directory '{base_path}' not found!")
        return
    
    experiments = [d for d in base_dir.iterdir() if d.is_dir()]
    total = len(experiments)
    success = 0
    
    print(f"Found {total} experiments to process...")
    print()
    
    for exp_dir in experiments:
        exp_name = exp_dir.name
        print(f"Processing: {exp_name}")
        
        # Look for data files (CSV or Excel)
        drt_data = None
        nyquist_data = None
        
        # Search for DRT data file
        for pattern in ['*DRT*.csv', '*drt*.csv', '*DRT*.xlsx', '*drt*.xlsx']:
            files = list(exp_dir.glob(pattern))
            if files:
                drt_data = files[0]
                break
        
        # Search for Nyquist/EIS data file
        for pattern in ['*EIS*.csv', '*eis*.csv', '*Nyquist*.csv', '*nyquist*.csv', '*EIS*.xlsx', '*eis*.xlsx', '*Nyquist*.xlsx', '*nyquist*.xlsx']:
            files = list(exp_dir.glob(pattern))
            if files:
                nyquist_data = files[0]
                break
        
        # Create interactive plots
        if drt_data:
            output_file = exp_dir / f"{exp_name}_DRT_interactive.html"
            create_interactive_drt_plot(str(drt_data), str(output_file), exp_name)
            success += 1
        else:
            print(f"  ⚠ No DRT data file found")
        
        if nyquist_data:
            output_file = exp_dir / f"{exp_name}_EIS_interactive.html"
            create_interactive_nyquist_plot(str(nyquist_data), str(output_file), exp_name)
            success += 1
        else:
            print(f"  ⚠ No Nyquist data file found")
        
        print()
    
    print("=" * 60)
    print(f"Completed! Successfully processed {success}/{total*2} plots")
    print("=" * 60)


def main():
    """Main function"""
    print()
    print("📊 INTERACTIVE PLOT GENERATOR")
    print()
    print("This script converts your experiment data to interactive")
    print("Plotly plots with hover tooltips showing x/y values.")
    print()
    print("Requirements:")
    print("  - Data files (CSV or Excel) in each experiment folder")
    print("  - File names should contain 'DRT' or 'EIS'/'Nyquist'")
    print()
    
    response = input("Process all experiments? (y/n): ")
    
    if response.lower() == 'y':
        process_all_experiments()
    else:
        print("Cancelled.")


if __name__ == "__main__":
    main()

