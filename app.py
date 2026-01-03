"""
Flask backend for automated comparison plot generation
"""
from flask import Flask, jsonify, request, send_from_directory, send_file
from flask_cors import CORS
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from datetime import datetime

# Optional imports for logo support
try:
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    from PIL import Image
    LOGO_SUPPORT = True
except ImportError:
    LOGO_SUPPORT = False
import sys
import traceback
import tempfile
import os
from io import BytesIO
import base64
import numpy as np
from datetime import datetime
import json

# Import configuration
from config import (
    BASE_DIR, DATA_DIR, M_SERIES_DIR, DURATION_TESTS_DIR, OUTPUTS_DIR,
    STATIC_DIR, TEMP_DIR, WATERMARK_CONFIG, COLORS, PLOT_DPI,
    SERIES_DESCRIPTIONS, EXISTING_DURATION_FOLDERS,
    PORT, DEBUG, HOST, IS_PRODUCTION, get_config_summary,
    SECRET_KEY, SESSION_LIFETIME_HOURS, VALID_USERS, LOGIN_DISABLED, REMEMBER_ME_DAYS
)

# Authentication imports
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import check_password_hash, generate_password_hash
from datetime import timedelta
from functools import wraps

# Import DRT analysis tools - EXACT same code as GUI
try:
    import sys
    import os
    drt_path = os.path.join(os.path.dirname(__file__), 'drt_tools')
    if drt_path not in sys.path:
        sys.path.insert(0, drt_path)
    
    # Import the exact same modules we used in GUI
    from runs import EIS_object, simple_run
    DRT_AVAILABLE = True
    print("DRT tools imported successfully - using exact GUI code")
except ImportError as e:
    print(f"WARNING: DRT tools not available: {e}")
    DRT_AVAILABLE = False

# Use EXACT same DRT calculation as GUI - no modifications
def calculate_drt_exact_gui_code(freq, zre, zim, rbf_type='Gaussian', data_used='Combined Re-Im Data', 
                                inductance='Fitting w/o Inductance', der_used='1st order', cv_type='GCV', lambda_value=None):
    """
    EXACT same DRT calculation as used in GUI - using pyDRTtools
    """
    if not DRT_AVAILABLE:
        raise Exception("DRT tools not available")
    
    try:
        # Create EIS object - EXACT same as GUI
        eis_data = EIS_object(freq, zre, zim)
        
        # Set inductance parameter
        if inductance == 'Fitting w/o Inductance':
            induct_used = 0
        elif inductance == 'Fitting w/ Inductance':
            induct_used = 1
        elif inductance == 'Fitting w/ Negative Inductance':
            induct_used = 2
        else:
            induct_used = 0
        
        # Set regularization parameter
        if lambda_value is not None and lambda_value != "":
            reg_param = float(lambda_value)
            cv_type_used = 'custom'
        else:
            reg_param = 1E-3
            cv_type_used = cv_type
        
        # Call simple_run - EXACT same as GUI
        result = simple_run(eis_data, 
                          rbf_type=rbf_type, 
                          data_used=data_used, 
                          induct_used=induct_used, 
                          der_used=der_used, 
                          cv_type=cv_type_used, 
                          reg_param=reg_param)
        
        return result
        
    except Exception as e:
        raise Exception(f"DRT calculation failed: {str(e)}")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# =============================================================================
#                         AUTHENTICATION SETUP
# =============================================================================
app.secret_key = SECRET_KEY
app.config['REMEMBER_COOKIE_DURATION'] = timedelta(days=REMEMBER_ME_DAYS)
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=SESSION_LIFETIME_HOURS)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'info'

class User(UserMixin):
    """Simple user class for Flask-Login"""
    def __init__(self, username):
        self.id = username
        self.username = username

@login_manager.user_loader
def load_user(username):
    """Load user by username"""
    if username in VALID_USERS:
        return User(username)
    return None

def auth_required(f):
    """
    Custom decorator that skips auth if LOGIN_DISABLED is True
    Useful for development/testing
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if LOGIN_DISABLED:
            return f(*args, **kwargs)
        return login_required(f)(*args, **kwargs)
    return decorated_function

# TEMP_DIR, COLORS, and WATERMARK_CONFIG are now imported from config.py

def add_watermark_to_plot(fig=None, ax=None, custom_text=None):
    """
    Add professional watermark with logo to matplotlib plots
    
    Args:
        fig: matplotlib figure object (if None, uses current figure)
        ax: matplotlib axes object (if None, uses current axes)
        custom_text: Custom watermark text (if None, uses default)
    """
    if fig is None:
        fig = plt.gcf()
    if ax is None:
        ax = plt.gca()
    
    # Get current timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M UTC')
    
    # Calculate position based on configuration
    if WATERMARK_CONFIG['position'] == 'bottom_right':
        text_x, text_y = 0.98, 0.02
        logo_x, logo_y = 0.95, 0.08
        ha, va = 'right', 'bottom'
    elif WATERMARK_CONFIG['position'] == 'top_right':
        text_x, text_y = 0.98, 0.98
        logo_x, logo_y = 0.95, 0.92
        ha, va = 'right', 'top'
    elif WATERMARK_CONFIG['position'] == 'bottom_left':
        text_x, text_y = 0.02, 0.02
        logo_x, logo_y = 0.05, 0.08
        ha, va = 'left', 'bottom'
    else:  # top_left
        text_x, text_y = 0.02, 0.98
        logo_x, logo_y = 0.05, 0.92
        ha, va = 'left', 'top'
    
    # Try to add logo first
    logo_added = False
    if LOGO_SUPPORT and os.path.exists(WATERMARK_CONFIG['logo_path']):
        try:
            # Load and resize logo
            logo_img = Image.open(WATERMARK_CONFIG['logo_path'])
            
            # Convert to RGBA if needed
            if logo_img.mode != 'RGBA':
                logo_img = logo_img.convert('RGBA')
            
            # Apply transparency
            logo_array = np.array(logo_img)
            logo_array[:, :, 3] = logo_array[:, :, 3] * WATERMARK_CONFIG['logo_alpha']
            logo_img = Image.fromarray(logo_array.astype(np.uint8))
            
            # Calculate logo size in pixels
            fig_width = fig.get_figwidth()
            logo_size_inches = fig_width * WATERMARK_CONFIG['logo_size']
            
            # Add logo to plot
            imagebox = OffsetImage(logo_img, zoom=logo_size_inches/2)
            ab = AnnotationBbox(imagebox, (logo_x, logo_y), 
                              xycoords='figure fraction',
                              frameon=False, 
                              boxcoords='figure fraction')
            fig.add_artist(ab)
            logo_added = True
            
            # Adjust text position if logo was added
            if WATERMARK_CONFIG['position'] == 'bottom_right':
                text_y = 0.12  # Move text up to avoid logo
            elif WATERMARK_CONFIG['position'] == 'top_right':
                text_y = 0.88  # Move text down to avoid logo
            elif WATERMARK_CONFIG['position'] == 'bottom_left':
                text_y = 0.12  # Move text up to avoid logo
            else:  # top_left
                text_y = 0.88  # Move text down to avoid logo
                
        except Exception as e:
            print(f"Warning: Could not load logo ({e}). Using text-only watermark.")
    
    # Watermark text
    if custom_text:
        if logo_added:
            watermark_text = f"{custom_text}\n{WATERMARK_CONFIG['company']} | {timestamp}"
        else:
            watermark_text = f"{custom_text}\n{WATERMARK_CONFIG['company']} | Generated: {timestamp}"
    else:
        watermark_text = f"{WATERMARK_CONFIG['text']}\n{WATERMARK_CONFIG['company']} | Generated: {timestamp}"
    
    # Add watermark text
    fig.text(text_x, text_y, watermark_text,
             fontsize=WATERMARK_CONFIG['font_size'],
             alpha=WATERMARK_CONFIG['alpha'],
             color=WATERMARK_CONFIG['color'],
             ha=ha, va=va,
             weight='normal',
             style='italic',
             transform=fig.transFigure)
    
    # Add subtle border watermark
    rect = Rectangle((0, 0), 1, 1, transform=fig.transFigure, 
                    fill=False, edgecolor=WATERMARK_CONFIG['color'], 
                    linewidth=1, alpha=WATERMARK_CONFIG['alpha']/2)
    fig.patches.append(rect)

def create_drt_comparison(experiments_data):
    """
    Create DRT comparison overlay plot - returns base64 encoded image
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
    
    # Add watermark before saving
    add_watermark_to_plot(custom_text="DRT Comparison Analysis")
    
    # Save to BytesIO object (in memory)
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=PLOT_DPI, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    print(f"[+] Created DRT comparison (in memory)")
    return img_base64


def create_nyquist_comparison(experiments_data):
    """
    Create Nyquist comparison overlay plot - returns base64 encoded image
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
    
    # Add watermark before saving
    add_watermark_to_plot(custom_text="Nyquist Comparison Analysis")
    
    # Save to BytesIO object (in memory)
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=PLOT_DPI, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    print(f"[+] Created Nyquist comparison (in memory)")
    return img_base64


@app.route('/api/data-series', methods=['GET'])
def get_data_series():
    """
    API endpoint to get available data series (M-Series, CV-Series, etc.)
    """
    try:
        data_dir = DATA_DIR
        series = []
        
        if data_dir.exists():
            for series_dir in data_dir.iterdir():
                if series_dir.is_dir():
                    # Count experiments in this series
                    exp_count = 0
                    separated_dir = series_dir / 'separated_by_current'
                    if separated_dir.exists():
                        exp_count = len([d for d in separated_dir.iterdir() if d.is_dir()])
                    
                    series.append({
                        'name': series_dir.name,
                        'path': str(series_dir),
                        'experiment_count': exp_count,
                        'description': get_series_description(series_dir.name)
                    })
        
        return jsonify({'success': True, 'series': series})
        
    except Exception as e:
        print(f"Error getting data series: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

def get_series_description(series_name):
    """Get description for data series"""
    return SERIES_DESCRIPTIONS.get(series_name, 'Experimental Data Analysis')

def map_experiment_name_to_folder(exp_name):
    """
    Map UI experiment names (M75-M93) to actual folder names
    Some experiments don't have data folders, so they map to None
    """
    if exp_name in EXISTING_DURATION_FOLDERS:
        return exp_name
    else:
        return None

def process_duration_test_data(csv_file):
    """
    Process Duration Test CSV data for electrochemical analysis
    Returns structured data for various plot types
    """
    try:
        df = pd.read_csv(csv_file)
        
        # Extract key columns
        time_col = 'Elapsed Time (s)'
        voltage_col = 'Potential (V)'
        current_col = 'Current (A)'
        
        # Convert time to hours for better readability
        if time_col in df.columns:
            df['Time (h)'] = df[time_col] / 3600
        
        # Calculate power if both V and I are available
        if voltage_col in df.columns and current_col in df.columns:
            df['Power (W)'] = df[voltage_col] * df[current_col]
        
        # Identify if this is EIS data
        is_eis = 'Frequency (Hz)' in df.columns and df['Frequency (Hz)'].notna().any()
        
        return {
            'data': df,
            'is_eis': is_eis,
            'has_time_series': time_col in df.columns,
            'columns': df.columns.tolist()
        }
    except Exception as e:
        print(f"Error processing duration test data: {e}")
        return None

def create_performance_vs_time_plot(experiments_data):
    """
    Create performance vs time plot for duration tests
    """
    plt.figure(figsize=(14, 10))
    
    # Create subplots for different metrics
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    for idx, (exp_name, csv_file) in enumerate(experiments_data):
        try:
            processed = process_duration_test_data(csv_file)
            if not processed or not processed['has_time_series']:
                continue
                
            df = processed['data']
            color = COLORS[idx % len(COLORS)]
            
            # Plot 1: Voltage vs Time
            if 'Time (h)' in df.columns and 'Potential (V)' in df.columns:
                ax1.plot(df['Time (h)'], df['Potential (V)'], 
                        color=color, linewidth=2, label=exp_name, alpha=0.8)
            
            # Plot 2: Current vs Time
            if 'Time (h)' in df.columns and 'Current (A)' in df.columns:
                ax2.plot(df['Time (h)'], df['Current (A)'], 
                        color=color, linewidth=2, label=exp_name, alpha=0.8)
            
            # Plot 3: Power vs Time
            if 'Time (h)' in df.columns and 'Power (W)' in df.columns:
                ax3.plot(df['Time (h)'], df['Power (W)'], 
                        color=color, linewidth=2, label=exp_name, alpha=0.8)
            
            # Plot 4: I-V Curve (if voltage and current available)
            if 'Potential (V)' in df.columns and 'Current (A)' in df.columns:
                ax4.scatter(df['Potential (V)'], df['Current (A)'], 
                           color=color, alpha=0.6, s=1, label=exp_name)
                
        except Exception as e:
            print(f"Error plotting {exp_name}: {e}")
            continue
    
    # Customize subplots
    ax1.set_xlabel('Time (h)', fontsize=12)
    ax1.set_ylabel('Potential (V)', fontsize=12)
    ax1.set_title('Voltage vs Time', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    ax2.set_xlabel('Time (h)', fontsize=12)
    ax2.set_ylabel('Current (A)', fontsize=12)
    ax2.set_title('Current vs Time', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    ax3.set_xlabel('Time (h)', fontsize=12)
    ax3.set_ylabel('Power (W)', fontsize=12)
    ax3.set_title('Power vs Time', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)
    
    ax4.set_xlabel('Potential (V)', fontsize=12)
    ax4.set_ylabel('Current (A)', fontsize=12)
    ax4.set_title('I-V Characteristics', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=10)
    
    plt.tight_layout()
    
    # Add watermark before saving
    add_watermark_to_plot(custom_text="Performance vs Time Analysis")
    
    # Save to BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=PLOT_DPI, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return img_base64

def create_eis_evolution_plot(experiments_data):
    """
    Create EIS evolution plot (Nyquist over time)
    """
    plt.figure(figsize=(14, 10))
    
    for idx, (exp_name, csv_files) in enumerate(experiments_data):
        try:
            color = COLORS[idx % len(COLORS)]
            
            # Process multiple EIS files for time evolution
            for file_idx, csv_file in enumerate(csv_files):
                processed = process_duration_test_data(csv_file)
                if not processed or not processed['is_eis']:
                    continue
                    
                df = processed['data']
                
                if 'Zre (ohms)' in df.columns and 'Zim (ohms)' in df.columns:
                    alpha = 0.8 - (file_idx * 0.15)  # Fade over time
                    alpha = max(alpha, 0.3)
                    
                    plt.plot(df['Zre (ohms)'], -df['Zim (ohms)'], 
                            'o-', color=color, alpha=alpha, 
                            markersize=3, linewidth=1.5,
                            label=f'{exp_name} - T{file_idx+1}')
                    
        except Exception as e:
            print(f"Error plotting EIS evolution for {exp_name}: {e}")
            continue
    
    plt.xlabel('Z_re (Ω)', fontsize=13)
    plt.ylabel('-Z_im (Ω)', fontsize=13)
    plt.title('EIS Evolution Over Time', fontsize=15, fontweight='bold')
    plt.grid(True, alpha=0.4)
    plt.axis('equal')
    plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Add watermark before saving
    add_watermark_to_plot(custom_text="EIS Evolution Analysis")
    
    # Save to BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=PLOT_DPI, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return img_base64

def create_degradation_analysis_plot(experiments_data):
    """
    Create degradation analysis comparing different experiments
    """
    plt.figure(figsize=(16, 8))
    
    # Create subplots for degradation metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    degradation_data = []
    
    for idx, (exp_name, csv_file) in enumerate(experiments_data):
        try:
            processed = process_duration_test_data(csv_file)
            if not processed or not processed['has_time_series']:
                continue
                
            df = processed['data']
            color = COLORS[idx % len(COLORS)]
            
            # Calculate degradation metrics
            if 'Time (h)' in df.columns and 'Potential (V)' in df.columns:
                time_hours = df['Time (h)']
                voltage = df['Potential (V)']
                
                # Calculate voltage degradation rate (V/h)
                if len(time_hours) > 10:
                    # Use linear regression for degradation rate
                    coeffs = np.polyfit(time_hours, voltage, 1)
                    degradation_rate = coeffs[0] * 1000  # mV/h
                    
                    # Initial and final voltage
                    initial_voltage = voltage.iloc[0]
                    final_voltage = voltage.iloc[-1]
                    total_degradation = (initial_voltage - final_voltage) * 1000  # mV
                    
                    degradation_data.append({
                        'experiment': exp_name,
                        'degradation_rate': degradation_rate,
                        'total_degradation': total_degradation,
                        'duration': time_hours.iloc[-1]
                    })
                    
                    # Plot voltage degradation
                    ax1.plot(time_hours, voltage, color=color, linewidth=2, 
                            label=f'{exp_name} ({degradation_rate:.2f} mV/h)')
                    
        except Exception as e:
            print(f"Error analyzing degradation for {exp_name}: {e}")
            continue
    
    # Plot degradation rates comparison
    if degradation_data:
        experiments = [d['experiment'] for d in degradation_data]
        rates = [d['degradation_rate'] for d in degradation_data]
        
        bars = ax2.bar(range(len(experiments)), rates, 
                      color=[COLORS[i % len(COLORS)] for i in range(len(experiments))])
        ax2.set_xticks(range(len(experiments)))
        ax2.set_xticklabels(experiments, rotation=45, ha='right')
        ax2.set_ylabel('Degradation Rate (mV/h)', fontsize=12)
        ax2.set_title('Voltage Degradation Rate Comparison', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, rate in zip(bars, rates):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{rate:.2f}', ha='center', va='bottom', fontsize=10)
    
    # Customize voltage plot
    ax1.set_xlabel('Time (h)', fontsize=12)
    ax1.set_ylabel('Potential (V)', fontsize=12)
    ax1.set_title('Voltage Degradation Over Time', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    plt.tight_layout()
    
    # Add watermark before saving
    add_watermark_to_plot(custom_text="Degradation Analysis")
    
    # Save to BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=PLOT_DPI, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return img_base64

@app.route('/api/experiment/plot', methods=['POST'])
def generate_single_experiment_plot():
    """
    API endpoint to generate a single experiment plot (DRT or Nyquist)
    First tries to serve existing PNG files, then falls back to CSV generation
    """
    import io
    import base64
    
    try:
        data = request.get_json()
        experiment_name = data.get('experiment')
        plot_type = data.get('plot_type', 'drt').lower()
        series = data.get('series', 'M-Series')
        
        if not experiment_name:
            return jsonify({'success': False, 'error': 'No experiment name provided'}), 400
        
        # Determine base directory based on series
        if series == 'M-Series':
            base_dir = M_SERIES_DIR
        else:
            base_dir = M_SERIES_DIR  # Default to M-Series
        
        exp_dir = base_dir / experiment_name
        
        if not exp_dir.exists():
            return jsonify({'success': False, 'error': f'Experiment directory not found: {experiment_name}'}), 404
        
        # FIRST: Try to serve existing PNG file
        if plot_type == 'drt':
            png_file = exp_dir / f'{experiment_name}_DRT_overlay.png'
        else:
            png_file = exp_dir / f'{experiment_name}_EIS_overlay.png'
        
        if png_file.exists():
            print(f"Serving existing PNG: {png_file}")
            with open(png_file, 'rb') as f:
                image_base64 = base64.b64encode(f.read()).decode('utf-8')
            return jsonify({
                'success': True,
                'plot': image_base64,
                'experiment': experiment_name,
                'plot_type': plot_type,
                'source': 'existing_png'
            })
        
        # FALLBACK: Generate from CSV data
        print(f"PNG not found, generating from CSV for {experiment_name} - {plot_type}")
        
        # Find all conditions (4A, 8A, 12A, 16A)
        conditions_data = []
        
        if plot_type == 'drt':
            drt_dir = exp_dir / 'drt'
            if drt_dir.exists():
                for csv_file in drt_dir.glob('*DRT*.csv'):
                    # Extract condition from filename (e.g., M1_4A_DRT.csv)
                    filename = csv_file.name
                    for cond in ['4A', '8A', '12A', '16A']:
                        if cond in filename:
                            conditions_data.append((cond, csv_file))
                            break
        elif plot_type == 'nyquist':
            eis_dir = exp_dir / 'eis'
            if eis_dir.exists():
                for csv_file in eis_dir.glob('*EIS*.csv'):
                    filename = csv_file.name
                    for cond in ['4A', '8A', '12A', '16A']:
                        if cond in filename:
                            conditions_data.append((cond, csv_file))
                            break
        
        if not conditions_data:
            return jsonify({'success': False, 'error': f'No {plot_type.upper()} CSV files found for {experiment_name}'}), 404
        
        # Sort by condition
        conditions_data.sort(key=lambda x: int(x[0].replace('A', '')))
        
        # Generate plot
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899']
        
        if plot_type == 'drt':
            # DRT Plot - columns are lowercase: tau, gamma
            for idx, (condition, csv_file) in enumerate(conditions_data):
                try:
                    # Skip first 2 header rows (L,0 and R,value), row 3 has column names
                    df = pd.read_csv(csv_file, skiprows=2)
                    
                    # Column names are lowercase: tau, gamma
                    tau_col = None
                    gamma_col = None
                    for col in df.columns:
                        if col.lower() == 'tau':
                            tau_col = col
                        elif col.lower() == 'gamma':
                            gamma_col = col
                    
                    if tau_col and gamma_col:
                        tau = df[tau_col].values
                        gamma = df[gamma_col].values
                        color = colors[idx % len(colors)]
                        ax.semilogx(tau, gamma, linewidth=2, color=color, label=condition, alpha=0.9)
                        print(f"  Added DRT data for {condition}: {len(tau)} points")
                except Exception as e:
                    print(f"Error reading DRT file {csv_file}: {e}")
                    continue
            
            ax.set_xlabel('τ (s)', fontsize=12, fontweight='bold', color='#1e3a8a')
            ax.set_ylabel('γ (Ω)', fontsize=12, fontweight='bold', color='#1e3a8a')
            ax.set_title(f'Distribution of Relaxation Times - {experiment_name}', fontsize=14, fontweight='bold', color='#1e3a8a')
            
        elif plot_type == 'nyquist':
            # Nyquist/EIS Plot - columns: mu_Z_re, mu_Z_im
            for idx, (condition, csv_file) in enumerate(conditions_data):
                try:
                    df = pd.read_csv(csv_file)
                    
                    # Try different column names
                    zre_col = None
                    zim_col = None
                    
                    for col in df.columns:
                        col_lower = col.lower()
                        # Check for mu_Z_re first (specific to this data format)
                        if col == 'mu_Z_re' or 'mu_z_re' in col_lower:
                            zre_col = col
                        elif col == 'mu_Z_im' or 'mu_z_im' in col_lower:
                            zim_col = col
                        # Fallback to generic names
                        elif zre_col is None and ('zre' in col_lower or 'z_re' in col_lower or "z'" == col_lower):
                            zre_col = col
                        elif zim_col is None and ('zim' in col_lower or 'z_im' in col_lower or "z''" == col_lower):
                            zim_col = col
                    
                    if zre_col and zim_col:
                        zre = df[zre_col].values
                        zim = df[zim_col].values
                        # For Nyquist, we plot -Zim vs Zre (positive imaginary up)
                        zim_plot = -zim if np.mean(zim) < 0 else zim
                        color = colors[idx % len(colors)]
                        ax.scatter(zre, zim_plot, s=30, color=color, label=condition, alpha=0.7, edgecolors='none')
                        print(f"  Added EIS data for {condition}: {len(zre)} points (cols: {zre_col}, {zim_col})")
                except Exception as e:
                    print(f"Error reading EIS file {csv_file}: {e}")
                    continue
            
            ax.set_xlabel('Z_re (Ω)', fontsize=12, fontweight='bold', color='#1e3a8a')
            ax.set_ylabel('-Z_im (Ω)', fontsize=12, fontweight='bold', color='#1e3a8a')
            ax.set_title(f'Nyquist Plot - {experiment_name}', fontsize=14, fontweight='bold', color='#1e3a8a')
            ax.set_aspect('equal', adjustable='box')
        
        # Common styling
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_facecolor('#fafafa')
        ax.legend(loc='best', fontsize=10, framealpha=0.9)
        
        for spine in ax.spines.values():
            spine.set_color('#e5e7eb')
        
        plt.tight_layout()
        
        # Add watermark
        try:
            add_watermark(fig, ax)
        except:
            pass
        
        # Convert to base64
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close(fig)
        
        return jsonify({
            'success': True,
            'plot': image_base64,
            'experiment': experiment_name,
            'plot_type': plot_type,
            'conditions_found': [c[0] for c in conditions_data],
            'source': 'generated_from_csv'
        })
        
    except Exception as e:
        print(f"Error generating experiment plot: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/generate-comparison', methods=['POST'])
def generate_comparison():
    """
    API endpoint to generate comparison plots (returns base64 images, no files saved)
    Expects JSON: {"selections": [{"experiment": "M1_...", "condition": "4A"}, ...]}
    """
    try:
        data = request.get_json()
        selections = data.get('selections', [])
        
        if not selections:
            return jsonify({'success': False, 'error': 'No selections provided'}), 400
        
        base_dir = M_SERIES_DIR
        
        drt_data = []
        eis_data = []
        
        # Collect CSV file paths
        for selection in selections:
            exp_name = selection['experiment']
            condition = selection['condition']
            exp_dir = base_dir / exp_name
            
            # Find DRT CSV
            drt_dir = exp_dir / 'drt'
            if drt_dir.exists():
                for csv_file in drt_dir.glob(f'*{condition}*.csv'):
                    if 'DRT' in csv_file.name.upper():
                        drt_data.append((exp_name, condition, csv_file))
                        break
                else:
                    # Try alternate pattern
                    for csv_file in drt_dir.glob(f'*DRT*{condition}*.csv'):
                        drt_data.append((exp_name, condition, csv_file))
                        break
            
            # Find EIS CSV
            eis_dir = exp_dir / 'eis'
            if eis_dir.exists():
                for csv_file in eis_dir.glob(f'*{condition}*.csv'):
                    if 'EIS' in csv_file.name.upper():
                        eis_data.append((exp_name, condition, csv_file))
                        break
                else:
                    # Try alternate pattern
                    for csv_file in eis_dir.glob(f'*EIS*{condition}*.csv'):
                        eis_data.append((exp_name, condition, csv_file))
                        break
        
        if not drt_data and not eis_data:
            return jsonify({'success': False, 'error': 'No CSV files found for selected experiments'}), 404
        
        # Generate comparison filename for display
        comparison_parts = [f"{exp}_{cond}" for exp, cond, _ in (drt_data if drt_data else eis_data)]
        comparison_name = '_vs_'.join(comparison_parts)
        if len(comparison_name) > 150:
            comparison_name = f"comparison_{len(selections)}experiments"
        
        # Create plots (in memory, base64 encoded)
        plots_base64 = {}
        
        if drt_data:
            drt_img = create_drt_comparison(drt_data)
            plots_base64['drt'] = drt_img
        
        if eis_data:
            nyquist_img = create_nyquist_comparison(eis_data)
            plots_base64['nyquist'] = nyquist_img
        
        return jsonify({
            'success': True,
            'plots': plots_base64,
            'comparison_name': comparison_name
        })
        
    except Exception as e:
        print(f"Error generating comparison: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# =============================================================================
#                         AUTHENTICATION ROUTES
# =============================================================================

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page and authentication"""
    from flask import render_template_string, redirect, url_for
    
    # If already logged in, redirect to home
    if current_user.is_authenticated:
        return redirect('/')
    
    error = None
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        remember = request.form.get('remember', False)
        
        # Check credentials
        if username in VALID_USERS and VALID_USERS[username] == password:
            user = User(username)
            login_user(user, remember=bool(remember))
            
            # Redirect to requested page or home
            next_page = request.args.get('next')
            if next_page and next_page.startswith('/'):
                return redirect(next_page)
            return redirect('/')
        else:
            error = 'Invalid username or password'
    
    # Read and render login template
    try:
        with open('login.html', 'r', encoding='utf-8') as f:
            template = f.read()
        return render_template_string(template, error=error)
    except FileNotFoundError:
        return "Login page not found", 500

@app.route('/logout')
@login_required
def logout():
    """Logout and redirect to login page"""
    from flask import redirect
    logout_user()
    return redirect('/login')

@app.route('/api/user')
@auth_required
def get_current_user():
    """Get current logged-in user info"""
    if current_user.is_authenticated:
        return jsonify({
            'authenticated': True,
            'username': current_user.username
        })
    return jsonify({'authenticated': False})

# =============================================================================
#                         PROTECTED ROUTES
# =============================================================================

# Serve static files
@app.route('/')
@auth_required
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    # Allow access to login page and static assets without auth
    public_paths = ['login', 'login.html', 'assets/', 'static/']
    if any(path.startswith(p) or path == p for p in public_paths):
        return send_from_directory('.', path)
    
    # Require auth for everything else
    if not LOGIN_DISABLED and not current_user.is_authenticated:
        from flask import redirect
        return redirect(f'/login?next=/{path}')
    
    return send_from_directory('.', path)

# Duration Tests API Endpoints
@app.route('/api/duration-tests/performance', methods=['POST'])
def generate_duration_performance():
    """
    API endpoint for Duration Tests performance analysis
    """
    try:
        data = request.get_json()
        selections = data.get('selections', [])
        
        if not selections:
            return jsonify({'success': False, 'error': 'No selections provided'}), 400
        
        base_dir = DURATION_TESTS_DIR
        experiments_data = []
        
        # Collect CSV file paths
        for selection in selections:
            exp_name = selection['experiment']
            folder_name = map_experiment_name_to_folder(exp_name)
            
            if folder_name is None:
                continue  # Skip experiments without data folders
                
            exp_dir = base_dir / folder_name
            
            # Find the main data CSV file
            if exp_dir.exists():
                for csv_file in exp_dir.glob('*.csv'):
                    # Skip EIS files, look for main time-series data
                    if 'EIS' not in csv_file.name.upper():
                        experiments_data.append((exp_name, csv_file))  # Use original exp_name for display
                        break
        
        if not experiments_data:
            return jsonify({'success': False, 'error': 'No valid data files found'}), 400
        
        # Generate performance vs time plot
        plot_base64 = create_performance_vs_time_plot(experiments_data)
        
        return jsonify({
            'success': True,
            'plot': plot_base64,
            'plot_type': 'performance_vs_time'
        })
        
    except Exception as e:
        print(f"Error generating duration performance plot: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/duration-tests/degradation', methods=['POST'])
def generate_duration_degradation():
    """
    API endpoint for Duration Tests degradation analysis
    """
    try:
        data = request.get_json()
        selections = data.get('selections', [])
        
        if not selections:
            return jsonify({'success': False, 'error': 'No selections provided'}), 400
        
        base_dir = DURATION_TESTS_DIR
        experiments_data = []
        
        # Collect CSV file paths
        for selection in selections:
            exp_name = selection['experiment']
            folder_name = map_experiment_name_to_folder(exp_name)
            
            if folder_name is None:
                continue  # Skip experiments without data folders
                
            exp_dir = base_dir / folder_name
            
            # Find the main data CSV file
            if exp_dir.exists():
                for csv_file in exp_dir.glob('*.csv'):
                    # Skip EIS files, look for main time-series data
                    if 'EIS' not in csv_file.name.upper():
                        experiments_data.append((exp_name, csv_file))  # Use original exp_name for display
                        break
        
        if not experiments_data:
            return jsonify({'success': False, 'error': 'No valid data files found'}), 400
        
        # Generate degradation analysis plot
        plot_base64 = create_degradation_analysis_plot(experiments_data)
        
        return jsonify({
            'success': True,
            'plot': plot_base64,
            'plot_type': 'degradation_analysis'
        })
        
    except Exception as e:
        print(f"Error generating degradation analysis plot: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/duration-tests/eis-evolution', methods=['POST'])
def generate_eis_evolution_api():
    """
    API endpoint for EIS evolution analysis
    """
    try:
        data = request.get_json()
        selections = data.get('selections', [])
        
        if not selections:
            return jsonify({'success': False, 'error': 'No selections provided'}), 400
        
        base_dir = DURATION_TESTS_DIR
        experiments_data = []
        
        # Collect EIS CSV file paths
        for selection in selections:
            exp_name = selection['experiment']
            folder_name = map_experiment_name_to_folder(exp_name)
            
            if folder_name is None:
                continue  # Skip experiments without data folders
                
            exp_dir = base_dir / folder_name
            
            # Find EIS files
            eis_files = []
            if exp_dir.exists():
                for csv_file in sorted(exp_dir.glob('*EIS*.csv')):
                    eis_files.append(csv_file)
                
                if eis_files:
                    experiments_data.append((exp_name, eis_files))  # Use original exp_name for display
        
        if not experiments_data:
            return jsonify({'success': False, 'error': 'No EIS data files found'}), 400
        
        # Generate EIS evolution plot
        plot_base64 = create_eis_evolution_plot(experiments_data)
        
        return jsonify({
            'success': True,
            'plot': plot_base64,
            'plot_type': 'eis_evolution'
        })
        
    except Exception as e:
        print(f"Error generating EIS evolution plot: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

def create_iv_curves_timeline_plot(experiments_data):
    """
    Create I-V curves at different timestamps for Duration Tests
    Based on actual data structure analysis - most tests are 8-24 hours with discrete time points
    """
    # Analyze available data to understand the actual testing pattern
    experiment_analysis = {}
    
    for exp_name, csv_file in experiments_data:
        try:
            # Get the experiment directory to find all related files
            exp_dir = csv_file.parent
            
            # Find all time-based files for this experiment
            time_files = []
            main_file_duration = 0
            
            # Check main file duration
            processed = process_duration_test_data(csv_file)
            if processed and processed['has_time_series']:
                df = processed['data']
                if 'Time (h)' in df.columns:
                    main_file_duration = df['Time (h)'].max()
            
            # Look for time-specific files (like M90_120min.csv, M77_EIS_2hour.csv)
            for file in exp_dir.glob('*.csv'):
                filename = file.name.lower()
                
                # Extract time information from filenames
                if 'min' in filename:
                    # Files like M90_120min.csv
                    import re
                    match = re.search(r'(\d+)min', filename)
                    if match:
                        minutes = int(match.group(1))
                        time_files.append((minutes/60, file))  # Convert to hours
                
                elif 'hour' in filename:
                    # Files like M77_EIS_2hour.csv
                    match = re.search(r'(\d+)\s*hour', filename)
                    if match:
                        hours = int(match.group(1))
                        time_files.append((hours, file))
                
                elif 'h_' in filename or 'h.' in filename:
                    # Files like M81_2h_EIS.csv
                    match = re.search(r'(\d+)h[_.]', filename)
                    if match:
                        hours = int(match.group(1))
                        time_files.append((hours, file))
            
            # Sort time files by time
            time_files.sort(key=lambda x: x[0])
            
            experiment_analysis[exp_name] = {
                'main_duration': main_file_duration,
                'time_files': time_files,
                'has_discrete_times': len(time_files) > 0
            }
            
        except Exception as e:
            print(f"Error analyzing experiment {exp_name}: {e}")
            continue
    
    # Determine visualization approach based on actual data
    has_discrete_time_files = any(exp['has_discrete_times'] for exp in experiment_analysis.values())
    
    if has_discrete_time_files:
        # Use discrete time files for I-V analysis
        return create_iv_discrete_time_plot(experiments_data, experiment_analysis)
    else:
        # Fall back to continuous time analysis
        return create_iv_continuous_time_plot(experiments_data, experiment_analysis)

def create_iv_discrete_time_plot(experiments_data, experiment_analysis):
    """
    Create I-V curves using discrete time files (e.g., M90_120min.csv, M77_EIS_2hour.csv)
    This matches the actual data structure where specific time points are saved as separate files
    """
    # Collect all unique time points across all experiments
    all_time_points = set()
    for exp_name, analysis in experiment_analysis.items():
        for time_hours, _ in analysis['time_files']:
            all_time_points.add(time_hours)
    
    # Sort time points and limit to reasonable number for visualization
    sorted_times = sorted(all_time_points)[:8]  # Max 8 time points for clarity
    
    if len(sorted_times) == 0:
        return create_iv_continuous_time_plot(experiments_data, experiment_analysis)
    
    # Create subplots for each time point
    n_times = len(sorted_times)
    n_cols = min(3, n_times)
    n_rows = (n_times + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_times == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if n_times > 1 else [axes]
    else:
        axes = axes.flatten()
    
    for time_idx, time_hours in enumerate(sorted_times):
        if time_idx >= len(axes):
            break
            
        ax = axes[time_idx]
        
        for exp_idx, (exp_name, _) in enumerate(experiments_data):
            analysis = experiment_analysis.get(exp_name, {})
            time_files = analysis.get('time_files', [])
            
            # Find the file closest to this time point
            best_file = None
            min_time_diff = float('inf')
            
            for file_time, file_path in time_files:
                time_diff = abs(file_time - time_hours)
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    best_file = file_path
            
            if best_file and min_time_diff < 1.0:  # Within 1 hour tolerance
                try:
                    # Check if this is EIS data or I-V data
                    processed = process_duration_test_data(best_file)
                    if not processed:
                        continue
                    
                    df = processed['data']
                    color = COLORS[exp_idx % len(COLORS)]
                    
                    if processed['is_eis']:
                        # For EIS files, we need to extract I-V characteristics differently
                        # EIS files typically have constant current/voltage during measurement
                        if 'Potential (V)' in df.columns and 'Current (A)' in df.columns:
                            # Use the operating point from EIS measurement
                            voltage = df['Potential (V)'].iloc[0]  # Constant during EIS
                            current = df['Current (A)'].iloc[0]    # Constant during EIS
                            
                            ax.scatter(voltage, current, color=color, s=100, 
                                     alpha=0.8, label=f'{exp_name}', marker='o')
                    else:
                        # Regular I-V data - look for voltage sweep or polarization data
                        if 'Potential (V)' in df.columns and 'Current (A)' in df.columns:
                            voltage = df['Potential (V)']
                            current = df['Current (A)']
                            
                            # Check if this looks like an I-V sweep (varying voltage)
                            voltage_range = voltage.max() - voltage.min()
                            if voltage_range > 0.1:  # Significant voltage variation
                                # Sort by voltage for proper I-V curve
                                sorted_indices = voltage.argsort()
                                voltage_sorted = voltage.iloc[sorted_indices]
                                current_sorted = current.iloc[sorted_indices]
                                
                                ax.plot(voltage_sorted, current_sorted, 'o-', 
                                       color=color, alpha=0.7, linewidth=2, 
                                       markersize=3, label=exp_name)
                            else:
                                # Single operating point
                                ax.scatter(voltage.mean(), current.mean(), 
                                         color=color, s=100, alpha=0.8, 
                                         label=f'{exp_name}', marker='s')
                        
                except Exception as e:
                    print(f"Error processing {best_file}: {e}")
                    continue
        
        # Format time label
        if time_hours < 1:
            time_label = f'{int(time_hours * 60)} min'
        elif time_hours == int(time_hours):
            time_label = f'{int(time_hours)} hour{"s" if time_hours != 1 else ""}'
        else:
            time_label = f'{time_hours:.1f} hours'
        
        ax.set_xlabel('Potential (V)', fontsize=11)
        ax.set_ylabel('Current (A)', fontsize=11)
        ax.set_title(f'I-V at {time_label}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if time_idx == 0:  # Show legend on first subplot
            ax.legend(fontsize=9, loc='best')
    
    # Hide unused subplots
    for idx in range(len(sorted_times), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('I-V Characteristics at Discrete Time Points', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Add watermark before saving
    add_watermark_to_plot(custom_text="I-V Discrete Time Analysis")
    
    # Save to BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=PLOT_DPI, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return img_base64

def create_iv_continuous_time_plot(experiments_data, experiment_analysis):
    """
    Create I-V curves from continuous time-series data (fallback method)
    Used when discrete time files are not available
    """
    # This is similar to the original timeline approach but more conservative
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Use realistic time points based on actual data (most tests are 8-24 hours)
    time_points = [0.5, 4, 8, 12]  # 30min, 4h, 8h, 12h
    time_labels = ['30 min', '4 hours', '8 hours', '12 hours']
    
    for time_idx, (time_hours, time_label) in enumerate(zip(time_points, time_labels)):
        ax = axes[time_idx]
        
        for exp_idx, (exp_name, csv_file) in enumerate(experiments_data):
            analysis = experiment_analysis.get(exp_name, {})
            main_duration = analysis.get('main_duration', 0)
            
            # Skip if experiment doesn't run long enough for this time point
            if main_duration < time_hours * 0.8:
                continue
            
            try:
                processed = process_duration_test_data(csv_file)
                if not processed or not processed['has_time_series']:
                    continue
                    
                df = processed['data']
                color = COLORS[exp_idx % len(COLORS)]
                
                if 'Time (h)' in df.columns and 'Potential (V)' in df.columns and 'Current (A)' in df.columns:
                    time_col = df['Time (h)']
                    
                    # Get a window around this time point (±30 minutes)
                    window_mask = abs(time_col - time_hours) <= 0.5
                    window_data = df[window_mask]
                    
                    if len(window_data) > 10:  # Need enough points
                        voltage = window_data['Potential (V)']
                        current = window_data['Current (A)']
                        
                        # Check if there's voltage variation (I-V sweep)
                        voltage_range = voltage.max() - voltage.min()
                        if voltage_range > 0.05:  # Significant voltage variation
                            # Sort by voltage for proper I-V curve
                            sorted_indices = voltage.argsort()
                            voltage_sorted = voltage.iloc[sorted_indices]
                            current_sorted = current.iloc[sorted_indices]
                            
                            ax.plot(voltage_sorted, current_sorted, 'o-', 
                                   color=color, alpha=0.7, linewidth=2, 
                                   markersize=2, label=exp_name)
                        else:
                            # Single operating point
                            ax.scatter(voltage.mean(), current.mean(), 
                                     color=color, s=80, alpha=0.8, 
                                     label=f'{exp_name}', marker='o')
                        
            except Exception as e:
                print(f"Error plotting continuous I-V for {exp_name} at {time_label}: {e}")
                continue
        
        ax.set_xlabel('Potential (V)', fontsize=11)
        ax.set_ylabel('Current (A)', fontsize=11)
        ax.set_title(f'I-V at {time_label}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if time_idx == 0:
            ax.legend(fontsize=9, loc='best')
    
    plt.suptitle('I-V Characteristics Over Time', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Add watermark before saving
    add_watermark_to_plot(custom_text="I-V Continuous Time Analysis")
    
    # Save to BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=PLOT_DPI, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return img_base64

def create_iv_overlay_plot(experiments_data, time_data_availability):
    """
    Create overlay I-V curves showing evolution over time for each experiment
    """
    fig, axes = plt.subplots(1, len(experiments_data), figsize=(6*len(experiments_data), 8))
    if len(experiments_data) == 1:
        axes = [axes]
    
    # Color map for time progression (blue to red)
    time_colors = ['#0066cc', '#3399ff', '#66ccff', '#99ff99', '#ffcc66', '#ff9933', '#ff3300']
    
    for exp_idx, (exp_name, csv_file) in enumerate(experiments_data):
        ax = axes[exp_idx]
        
        try:
            processed = process_duration_test_data(csv_file)
            if not processed or not processed['has_time_series']:
                continue
                
            df = processed['data']
            
            if 'Time (h)' in df.columns and 'Potential (V)' in df.columns and 'Current (A)' in df.columns:
                max_time = time_data_availability.get(exp_name, 0)
                
                # Define adaptive time points based on experiment duration
                if max_time > 336:  # > 2 weeks
                    time_points = [0, 24, 72, 168, 336, 504, max_time*0.9]
                    time_labels = ['Start', '1d', '3d', '1w', '2w', '3w', 'End']
                elif max_time > 168:  # > 1 week
                    time_points = [0, 12, 48, 120, 168, max_time*0.9]
                    time_labels = ['Start', '12h', '2d', '5d', '1w', 'End']
                elif max_time > 48:  # > 2 days
                    time_points = [0, 6, 24, 48, max_time*0.9]
                    time_labels = ['Start', '6h', '1d', '2d', 'End']
                else:  # Short duration
                    time_points = [0, max_time*0.25, max_time*0.5, max_time*0.75, max_time*0.9]
                    time_labels = ['Start', '25%', '50%', '75%', 'End']
                
                time_col = df['Time (h)']
                
                for time_idx, (time_hours, time_label) in enumerate(zip(time_points, time_labels)):
                    # Get a window around this time point
                    window_size = max(1.0, max_time * 0.02)  # Adaptive window size
                    window_mask = abs(time_col - time_hours) <= window_size
                    window_data = df[window_mask]
                    
                    if len(window_data) > 5:  # Need enough points for I-V curve
                        voltage = window_data['Potential (V)']
                        current = window_data['Current (A)']
                        
                        # Sort by voltage for proper I-V curve
                        sorted_indices = voltage.argsort()
                        voltage_sorted = voltage.iloc[sorted_indices]
                        current_sorted = current.iloc[sorted_indices]
                        
                        # Use color progression to show time evolution
                        color = time_colors[time_idx % len(time_colors)]
                        alpha = 0.8 - (time_idx * 0.1)  # Fade older curves slightly
                        alpha = max(alpha, 0.4)
                        
                        ax.plot(voltage_sorted, current_sorted, 'o-', 
                               color=color, alpha=alpha, linewidth=2, 
                               markersize=4, label=f'{time_label}')
                        
        except Exception as e:
            print(f"Error creating overlay plot for {exp_name}: {e}")
            continue
        
        ax.set_xlabel('Potential (V)', fontsize=12)
        ax.set_ylabel('Current (A)', fontsize=12)
        ax.set_title(f'{exp_name} - I-V Evolution', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='best')
    
    plt.suptitle('I-V Curves Evolution Over Time (Overlay View)', fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    
    # Add watermark before saving
    add_watermark_to_plot(custom_text="I-V Overlay Analysis")
    
    # Save to BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=PLOT_DPI, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return img_base64

def create_iv_timeline_subplots(experiments_data):
    """
    Create traditional timeline subplots for I-V curves
    """
    plt.figure(figsize=(16, 10))
    
    # Create subplots for different time points
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Define time points to analyze (in hours)
    time_points = [0, 24, 48, 96, 168, 336]  # 0h, 1d, 2d, 4d, 1w, 2w
    time_labels = ['Start', '1 Day', '2 Days', '4 Days', '1 Week', '2 Weeks']
    
    for time_idx, (time_hours, time_label) in enumerate(zip(time_points, time_labels)):
        if time_idx >= len(axes):
            break
            
        ax = axes[time_idx]
        
        for exp_idx, (exp_name, csv_file) in enumerate(experiments_data):
            try:
                processed = process_duration_test_data(csv_file)
                if not processed or not processed['has_time_series']:
                    continue
                    
                df = processed['data']
                color = COLORS[exp_idx % len(COLORS)]
                
                # Find data points closest to the target time
                if 'Time (h)' in df.columns and 'Potential (V)' in df.columns and 'Current (A)' in df.columns:
                    time_col = df['Time (h)']
                    
                    # Skip if experiment doesn't have data at this time point
                    if time_col.max() < time_hours * 0.8:
                        continue
                    
                    # Find the closest time point
                    time_diff = abs(time_col - time_hours)
                    closest_idx = time_diff.idxmin()
                    
                    # Get a window around this time point (±1 hour)
                    window_mask = abs(time_col - time_hours) <= 1.0
                    window_data = df[window_mask]
                    
                    if len(window_data) > 5:  # Need enough points for I-V curve
                        voltage = window_data['Potential (V)']
                        current = window_data['Current (A)']
                        
                        # Sort by voltage for proper I-V curve
                        sorted_indices = voltage.argsort()
                        voltage_sorted = voltage.iloc[sorted_indices]
                        current_sorted = current.iloc[sorted_indices]
                        
                        ax.plot(voltage_sorted, current_sorted, 'o-', 
                               color=color, alpha=0.7, linewidth=2, 
                               markersize=3, label=exp_name)
                        
            except Exception as e:
                print(f"Error plotting I-V curve for {exp_name} at {time_label}: {e}")
                continue
        
        ax.set_xlabel('Potential (V)', fontsize=11)
        ax.set_ylabel('Current (A)', fontsize=11)
        ax.set_title(f'I-V Curves at {time_label}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if time_idx == 0:  # Only show legend on first subplot
            ax.legend(fontsize=9, loc='best')
    
    # Hide unused subplots
    for idx in range(len(time_points), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('I-V Curves Evolution Over Time', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Add watermark before saving
    add_watermark_to_plot(custom_text="I-V Timeline Analysis")
    
    # Save to BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=PLOT_DPI, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return img_base64

@app.route('/api/duration-tests/iv-curves', methods=['POST'])
def generate_iv_curves_timeline():
    """
    API endpoint for I-V curves timeline analysis
    """
    try:
        data = request.get_json()
        selections = data.get('selections', [])
        
        if not selections:
            return jsonify({'success': False, 'error': 'No selections provided'}), 400
        
        base_dir = DURATION_TESTS_DIR
        experiments_data = []
        
        # Collect CSV file paths
        for selection in selections:
            exp_name = selection['experiment']
            folder_name = map_experiment_name_to_folder(exp_name)
            
            if folder_name is None:
                continue  # Skip experiments without data folders
                
            exp_dir = base_dir / folder_name
            
            # Find the main data CSV file
            if exp_dir.exists():
                for csv_file in exp_dir.glob('*.csv'):
                    # Skip EIS files, look for main time-series data
                    if 'EIS' not in csv_file.name.upper():
                        experiments_data.append((exp_name, csv_file))  # Use original exp_name for display
                        break
        
        if not experiments_data:
            return jsonify({'success': False, 'error': 'No valid data files found'}), 400
        
        # Generate I-V curves timeline plot
        plot_base64 = create_iv_curves_timeline_plot(experiments_data)
        
        return jsonify({
            'success': True,
            'plot': plot_base64,
            'plot_type': 'iv_curves_timeline'
        })
        
    except Exception as e:
        print(f"Error generating I-V curves timeline plot: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/drt/export/<format_type>', methods=['POST'])
def export_drt_results(format_type):
    """
    Export DRT results in different formats (CSV, XLSX, JSON)
    """
    try:
        data = request.get_json()
        results = data.get('results')
        parameters = data.get('parameters', {})
        
        if not results:
            return jsonify({'success': False, 'error': 'No results data provided'}), 400
        
        # Prepare export data
        export_data = {
            'tau': results.get('tau', []),
            'gamma': results.get('gamma', []),
            'freq': results.get('freq', []),
            'zre_fit': results.get('zre_fit', []),
            'zim_fit': results.get('zim_fit', []),
            'residuals': results.get('residuals', {}),
            'lambda_value': results.get('lambda_value'),
            'method': results.get('method'),
            'parameters': parameters
        }
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format_type.lower() == 'xlsx':
            # Create Excel file using pandas
            output = BytesIO()
            
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # DRT Results sheet
                drt_df = pd.DataFrame({
                    'Tau (s)': export_data['tau'],
                    'Gamma (Ohm)': export_data['gamma']
                })
                drt_df.to_excel(writer, sheet_name='DRT_Results', index=False)
                
                # Fitted Data sheet
                if export_data['freq'] and export_data['zre_fit']:
                    fitted_df = pd.DataFrame({
                        'Frequency (Hz)': export_data['freq'],
                        'Zre_fit (Ohm)': export_data['zre_fit'],
                        'Zim_fit (Ohm)': export_data['zim_fit']
                    })
                    
                    if export_data['residuals'].get('zre') and export_data['residuals'].get('zim'):
                        fitted_df['Zre_residual (Ohm)'] = export_data['residuals']['zre']
                        fitted_df['Zim_residual (Ohm)'] = export_data['residuals']['zim']
                    
                    fitted_df.to_excel(writer, sheet_name='Fitted_Data', index=False)
                
                # Parameters sheet
                params_df = pd.DataFrame(list(export_data['parameters'].items()), 
                                       columns=['Parameter', 'Value'])
                params_df = pd.concat([
                    pd.DataFrame([
                        ['Method', export_data['method']],
                        ['Lambda', export_data['lambda_value']]
                    ], columns=['Parameter', 'Value']),
                    params_df
                ], ignore_index=True)
                params_df.to_excel(writer, sheet_name='Parameters', index=False)
            
            output.seek(0)
            
            return send_file(
                output,
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                as_attachment=True,
                download_name=f'drt_results_{timestamp}.xlsx'
            )
            
        elif format_type.lower() == 'csv':
            # Create CSV file
            output = BytesIO()
            
            # Write parameters section
            csv_content = "DRT Analysis Results\n"
            csv_content += f"Method,{export_data['method']}\n"
            csv_content += f"Lambda,{export_data['lambda_value']}\n"
            for key, value in export_data['parameters'].items():
                csv_content += f"{key},{value}\n"
            csv_content += "\n"
            
            # Write DRT results
            csv_content += "Tau (s),Gamma (Ohm)\n"
            for tau, gamma in zip(export_data['tau'], export_data['gamma']):
                csv_content += f"{tau},{gamma}\n"
            
            output.write(csv_content.encode('utf-8'))
            output.seek(0)
            
            return send_file(
                output,
                mimetype='text/csv',
                as_attachment=True,
                download_name=f'drt_results_{timestamp}.csv'
            )
        
        else:
            return jsonify({'success': False, 'error': f'Unsupported format: {format_type}'}), 400
            
    except Exception as e:
        print(f"Error exporting DRT results: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

# =============================================================================
#                            DRT ANALYSIS ROUTES
# =============================================================================

@app.route('/drt-analysis')
@auth_required
def drt_analysis_page():
    """Interactive DRT Analysis Page"""
    return send_from_directory('.', 'drt_analysis.html')

@app.route('/api/drt/upload', methods=['POST'])
def upload_drt_data():
    """
    Upload and validate CSV file for DRT analysis
    Supports both:
    1. Simple format: Frequency(Hz), Zre(ohms), Zim(ohms)
    2. Raw experiment format: Multi-current data with EIS measurements
    """
    if not DRT_AVAILABLE:
        return jsonify({'success': False, 'error': 'DRT analysis not available'}), 503
        
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
            
        # Try to read CSV file - handle both with and without headers
        try:
            # First, try reading with headers
            df = pd.read_csv(file)
            
            # Enhanced column detection for various raw data formats
            col_mapping = {}
            headers_found = True
            
            # Define comprehensive patterns for each required column
            frequency_patterns = [
                'frequency', 'freq', 'f(hz)', 'f', 'hz', 'f_hz', 'frequency(hz)', 
                'frequency_hz', 'freqhz', 'freq_hz'
            ]
            
            zre_patterns = [
                'zre', 'z_re', 'z re', 'zreal', 'z_real', 'real', 're', 'real_z', 
                'realz', 'z\'', 'zprime', 'z_prime', 'real_impedance', 'zre(ohms)', 
                'zre(ohm)', 'zre (ohms)', 'zre (ohm)', 'real_part', 'real part'
            ]
            
            zim_patterns = [
                'zim', 'z_im', 'z im', 'zimag', 'z_imag', 'imag', 'im', 'imag_z', 
                'imagz', 'z\'\'', 'zdoubleprime', 'z_doubleprime', 'imag_impedance', 
                'zim(ohms)', 'zim(ohm)', 'zim (ohms)', 'zim (ohm)', 'imag_part', 
                'imag part', 'imaginary', 'imaginary_part'
            ]
            
            # Current column patterns (for raw experiment files)
            current_patterns = [
                'current', 'current (a)', 'current(a)', 'i', 'i (a)', 'i(a)',
                'current_a', 'amps', 'amperes'
            ]
            
            # Function to find best column match
            def find_column_match(patterns, available_columns):
                """Find the best matching column for given patterns"""
                available_lower = [(col, col.lower().strip()) for col in available_columns]
                
                # First, try exact matches
                for pattern in patterns:
                    for col, col_lower in available_lower:
                        if pattern.lower() == col_lower:
                            return col
                
                # Then, try substring matches (prefer shorter matches to avoid false positives)
                for pattern in patterns:
                    matches = []
                    for col, col_lower in available_lower:
                        if pattern.lower() in col_lower:
                            matches.append((col, len(col)))  # Store column and its length
                    
                    if matches:
                        # Return the shortest matching column name (most specific)
                        return min(matches, key=lambda x: x[1])[0]
                
                return None
            
            # Find matching columns
            freq_col = find_column_match(frequency_patterns, df.columns)
            zre_col = find_column_match(zre_patterns, df.columns)
            zim_col = find_column_match(zim_patterns, df.columns)
            current_col = find_column_match(current_patterns, df.columns)
            
            if freq_col and zre_col and zim_col:
                col_mapping = {
                    'freq': freq_col,
                    'zre': zre_col,
                    'zim': zim_col
                }
                if current_col:
                    col_mapping['current'] = current_col
                headers_found = True
            else:
                headers_found = False
                missing = []
                if not freq_col: missing.append('Frequency')
                if not zre_col: missing.append('Zre (Real impedance)')
                if not zim_col: missing.append('Zim (Imaginary impedance)')
            
            if headers_found:
                # Check if this is a raw experiment file with multiple currents
                # Raw files have Frequency=0 for non-EIS data points
                is_raw_experiment = False
                detected_currents = []
                
                if current_col and freq_col:
                    # Filter for EIS data only (Frequency > 0)
                    eis_mask = df[freq_col] > 0
                    eis_data = df[eis_mask]
                    
                    if len(eis_data) > 0:
                        # Get unique currents from EIS measurements, rounded to nearest integer
                        currents = eis_data[current_col].values
                        rounded_currents = np.round(currents).astype(int)
                        unique_currents = sorted(set(rounded_currents))
                        
                        # If we have multiple distinct currents, this is a raw experiment file
                        if len(unique_currents) > 1 or (len(unique_currents) == 1 and len(df[~eis_mask]) > 0):
                            is_raw_experiment = True
                            detected_currents = [int(c) for c in unique_currents]
                
                if is_raw_experiment:
                    # Return info about detected currents - user will select one
                    return jsonify({
                        'success': True,
                        'message': f'Raw experiment file detected with {len(detected_currents)} current levels',
                        'is_raw_experiment': True,
                        'detected_currents': detected_currents,
                        'total_rows': len(df),
                        'eis_rows': int(eis_mask.sum()),
                        'columns_detected': col_mapping,
                        'raw_data': {
                            'all_freq': df[freq_col].tolist(),
                            'all_zre': df[zre_col].tolist(),
                            'all_zim': df[zim_col].tolist(),
                            'all_current': df[current_col].tolist() if current_col else []
                        }
                    })
                else:
                    # Simple DRT file - extract data directly
                    freq = df[col_mapping['freq']].values
                    zre = df[col_mapping['zre']].values
                    zim = df[col_mapping['zim']].values
                    data_source = f"detected columns: {col_mapping['freq']}, {col_mapping['zre']}, {col_mapping['zim']}"
                    
                    # Validate data
                    if len(freq) < 5:
                        return jsonify({'success': False, 'error': 'Insufficient data points (minimum 5 required)'}), 400
                    
                    return jsonify({
                        'success': True,
                        'message': f'File uploaded successfully ({data_source})',
                        'is_raw_experiment': False,
                        'data_points': len(freq),
                        'freq_range': [float(freq.min()), float(freq.max())],
                        'impedance_range': {
                            'zre': [float(zre.min()), float(zre.max())],
                            'zim': [float(zim.min()), float(zim.max())]
                        },
                        'file_data': {
                            'freq': freq.tolist(),
                            'zre': zre.tolist(), 
                            'zim': zim.tolist()
                        }
                    })
            else:
                # No proper headers found, provide detailed error
                available_columns = list(df.columns)
                error_msg = f"Could not detect required columns. Missing: {', '.join(missing)}.\n"
                error_msg += f"Available columns: {', '.join(available_columns)}\n\n"
                error_msg += "Expected column patterns:\n"
                error_msg += "• Frequency: frequency, freq, f(hz), f, hz\n"
                error_msg += "• Real impedance: zre, z_re, real, re, zre(ohms)\n"
                error_msg += "• Imaginary impedance: zim, z_im, imag, im, zim(ohms)\n\n"
                
                # Try fallback only if we have at least 3 numeric columns
                if df.shape[1] >= 3:
                    try:
                        # Check if first 3 columns are numeric
                        test_cols = df.iloc[:, :3]
                        numeric_cols = test_cols.select_dtypes(include=[np.number]).shape[1]
                        
                        if numeric_cols >= 3:
                            error_msg += "Fallback: Using first 3 numeric columns as Frequency, Zre, Zim"
                            freq = df.iloc[:, 0].values  # First column = frequency
                            zre = df.iloc[:, 1].values   # Second column = Zre
                            zim = df.iloc[:, 2].values   # Third column = Zim
                            data_source = f"fallback mode (columns: {df.columns[0]}, {df.columns[1]}, {df.columns[2]})"
                            headers_found = True  # Allow processing to continue
                        else:
                            error_msg += f"First 3 columns are not all numeric ({numeric_cols}/3 are numeric)"
                    except:
                        error_msg += "Could not process first 3 columns as numeric data"
                
                if not headers_found:
                    return jsonify({
                        'success': False, 
                        'error': error_msg
                    }), 400
                
                # Validate data
                if len(freq) < 5:
                    return jsonify({'success': False, 'error': 'Insufficient data points (minimum 5 required)'}), 400
                    
                return jsonify({
                    'success': True,
                    'message': f'File uploaded successfully ({data_source})',
                    'is_raw_experiment': False,
                    'data_points': len(freq),
                    'freq_range': [float(freq.min()), float(freq.max())],
                    'impedance_range': {
                        'zre': [float(zre.min()), float(zre.max())],
                        'zim': [float(zim.min()), float(zim.max())]
                    },
                    'file_data': {
                        'freq': freq.tolist(),
                        'zre': zre.tolist(), 
                        'zim': zim.tolist()
                    }
                })
                
        except Exception as e:
            return jsonify({
                'success': False, 
                'error': f'Error reading CSV file: {str(e)}'
            }), 400
        
    except Exception as e:
        print(f"Error uploading DRT data: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/drt/extract-current', methods=['POST'])
def extract_current_data():
    """
    Extract EIS data for a specific current from raw experiment file
    """
    if not DRT_AVAILABLE:
        return jsonify({'success': False, 'error': 'DRT analysis not available'}), 503
        
    try:
        data = request.get_json()
        
        selected_current = data.get('selected_current')
        raw_data = data.get('raw_data')
        
        if selected_current is None:
            return jsonify({'success': False, 'error': 'No current selected'}), 400
        
        if not raw_data:
            return jsonify({'success': False, 'error': 'No raw data provided'}), 400
        
        # Convert to numpy arrays
        all_freq = np.array(raw_data['all_freq'])
        all_zre = np.array(raw_data['all_zre'])
        all_zim = np.array(raw_data['all_zim'])
        all_current = np.array(raw_data['all_current'])
        
        # Filter for EIS data (Frequency > 0) at selected current
        # Round currents and match to selected current
        rounded_currents = np.round(all_current).astype(int)
        
        # Create mask: Frequency > 0 AND current matches selected
        mask = (all_freq > 0) & (rounded_currents == selected_current)
        
        # Extract filtered data
        freq = all_freq[mask]
        zre = all_zre[mask]
        zim = all_zim[mask]
        
        if len(freq) < 5:
            return jsonify({
                'success': False, 
                'error': f'Insufficient EIS data points at {selected_current}A (found {len(freq)}, minimum 5 required)'
            }), 400
        
        return jsonify({
            'success': True,
            'message': f'Extracted {len(freq)} EIS data points at {selected_current}A',
            'selected_current': selected_current,
            'data_points': len(freq),
            'freq_range': [float(freq.min()), float(freq.max())],
            'impedance_range': {
                'zre': [float(zre.min()), float(zre.max())],
                'zim': [float(zim.min()), float(zim.max())]
            },
            'file_data': {
                'freq': freq.tolist(),
                'zre': zre.tolist(),
                'zim': zim.tolist()
            }
        })
        
    except Exception as e:
        print(f"Error extracting current data: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/drt/compare-currents', methods=['POST'])
def compare_currents_drt():
    """
    Calculate DRT for multiple currents and generate comparison plots
    """
    if not DRT_AVAILABLE:
        return jsonify({'success': False, 'error': 'DRT analysis not available'}), 503
        
    try:
        data = request.get_json()
        
        selected_currents = data.get('selected_currents', [])
        raw_data = data.get('raw_data')
        parameters = data.get('parameters', {})
        
        if not selected_currents or len(selected_currents) < 1:
            return jsonify({'success': False, 'error': 'No currents selected'}), 400
        
        if not raw_data:
            return jsonify({'success': False, 'error': 'No raw data provided'}), 400
        
        # Convert to numpy arrays
        all_freq = np.array(raw_data['all_freq'])
        all_zre = np.array(raw_data['all_zre'])
        all_zim = np.array(raw_data['all_zim'])
        all_current = np.array(raw_data['all_current'])
        rounded_currents = np.round(all_current).astype(int)
        
        # Extract DRT parameters
        rbf_type = parameters.get('rbf_type', 'Gaussian')
        data_used = parameters.get('data_used', 'Combined Re-Im Data')
        inductance = parameters.get('inductance', 'Fitting w/o Inductance')
        der_used = parameters.get('der_used', '1st order')
        cv_type = parameters.get('cv_type', 'GCV')
        lambda_value = parameters.get('lambda_value', None)
        
        # Process each selected current
        results_by_current = {}
        eis_data_by_current = {}
        
        for current in selected_currents:
            current = int(current)
            
            # Extract EIS data for this current
            mask = (all_freq > 0) & (rounded_currents == current)
            freq = all_freq[mask]
            zre = all_zre[mask]
            zim = all_zim[mask]
            
            if len(freq) < 5:
                continue  # Skip currents with insufficient data
            
            # Store EIS data for plotting
            eis_data_by_current[current] = {
                'freq': freq,
                'zre': zre,
                'zim': zim
            }
            
            # Calculate DRT
            try:
                result = calculate_drt_exact_gui_code(
                    freq, zre, zim,
                    rbf_type=rbf_type,
                    data_used=data_used,
                    inductance=inductance,
                    der_used=der_used,
                    cv_type=cv_type,
                    lambda_value=lambda_value
                )
                
                results_by_current[current] = {
                    'tau': result.out_tau_vec.tolist(),
                    'gamma': result.gamma.tolist(),
                    'lambda_value': float(result.lambda_value) if hasattr(result, 'lambda_value') else None,
                    'freq': result.freq.tolist(),
                    'zre_fit': result.mu_Z_re.tolist() if hasattr(result, 'mu_Z_re') else None,
                    'zim_fit': result.mu_Z_im.tolist() if hasattr(result, 'mu_Z_im') else None
                }
            except Exception as calc_error:
                print(f"DRT calculation failed for {current}A: {calc_error}")
                continue
        
        if not results_by_current:
            return jsonify({'success': False, 'error': 'DRT calculation failed for all selected currents'}), 500
        
        # Generate comparison plots
        plots = {}
        
        # DRT Comparison Plot
        plots['drt'] = create_drt_comparison_plot(results_by_current)
        
        # Nyquist Comparison Plot
        plots['nyquist'] = create_nyquist_comparison_plot(eis_data_by_current, results_by_current)
        
        return jsonify({
            'success': True,
            'message': f'DRT comparison completed for {len(results_by_current)} currents',
            'currents_analyzed': list(results_by_current.keys()),
            'results_by_current': results_by_current,
            'plots': plots
        })
        
    except Exception as e:
        print(f"Error in DRT comparison: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


def create_drt_comparison_plot(results_by_current):
    """
    Create DRT comparison overlay plot for multiple currents
    """
    plt.figure(figsize=(12, 8))
    plt.style.use('default')
    
    for idx, (current, results) in enumerate(sorted(results_by_current.items())):
        tau = np.array(results['tau'])
        gamma = np.array(results['gamma'])
        color = COLORS[idx % len(COLORS)]
        
        plt.semilogx(tau, gamma, '-', linewidth=2.5, color=color, label=f'{current}A')
    
    plt.xlabel('τ [s]', fontsize=13)
    plt.ylabel('γ [Ω]', fontsize=13)
    plt.title('DRT Comparison - Multiple Currents', fontsize=15, fontweight='bold')
    plt.grid(True, alpha=0.4, which='both')
    plt.legend(fontsize=11, loc='upper right', frameon=True, fancybox=True, shadow=False, title='Current')
    plt.ylim(bottom=0)
    plt.tight_layout()
    
    # Add watermark
    add_watermark_to_plot(custom_text="DRT Comparison Analysis")
    
    # Save to BytesIO
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=PLOT_DPI, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return img_base64


def create_nyquist_comparison_plot(eis_data_by_current, results_by_current):
    """
    Create Nyquist comparison overlay plot for multiple currents
    Shows only fitted data (lines) for clean comparison
    """
    plt.figure(figsize=(12, 9))
    plt.style.use('default')
    
    for idx, (current, results) in enumerate(sorted(results_by_current.items())):
        color = COLORS[idx % len(COLORS)]
        
        # Plot only fitted data (lines)
        if results.get('zre_fit') and results.get('zim_fit'):
            zre_fit = np.array(results['zre_fit'])
            zim_fit = np.array(results['zim_fit'])
            plt.plot(zre_fit, -zim_fit, '-', linewidth=2.5, color=color, label=f'{current}A')
    
    plt.xlabel('Z_re [Ω]', fontsize=13)
    plt.ylabel('-Z_im [Ω]', fontsize=13)
    plt.title('Nyquist Comparison - Multiple Currents', fontsize=15, fontweight='bold')
    plt.grid(True, alpha=0.4)
    plt.axis('equal')
    plt.legend(fontsize=11, loc='upper right', frameon=True, fancybox=True, shadow=False, title='Current')
    plt.tight_layout()
    
    # Add watermark
    add_watermark_to_plot(custom_text="Nyquist Comparison Analysis")
    
    # Save to BytesIO
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=PLOT_DPI, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return img_base64

@app.route('/api/drt/calculate', methods=['POST'])
def calculate_drt():
    """
    Perform DRT calculation with specified parameters
    """
    if not DRT_AVAILABLE:
        return jsonify({'success': False, 'error': 'DRT analysis not available'}), 503
        
    try:
        data = request.get_json()
        
        # Extract file data
        freq = np.array(data['freq'])
        zre = np.array(data['zre'])
        zim = np.array(data['zim'])
        
        # Extract parameters
        rbf_type = data.get('rbf_type', 'Gaussian')
        data_used = data.get('data_used', 'Combined Re-Im Data')
        inductance = data.get('inductance', 'Fitting w/o Inductance')
        der_used = data.get('der_used', '1st order')
        cv_type = data.get('cv_type', 'GCV')
        lambda_value = data.get('lambda_value', None)
        
        # Use EXACT same DRT calculation as GUI
        try:
            result = calculate_drt_exact_gui_code(
                freq, zre, zim,
                rbf_type=rbf_type,
                data_used=data_used,
                inductance=inductance,
                der_used=der_used,
                cv_type=cv_type,
                lambda_value=lambda_value
            )
            
            # Extract results EXACTLY as GUI does
            return jsonify({
                'success': True,
                'message': 'DRT calculation completed',
                'results': {
                    'tau': result.out_tau_vec.tolist(),
                    'gamma': result.gamma.tolist(),
                    'lambda_value': float(result.lambda_value) if hasattr(result, 'lambda_value') else None,
                    'method': getattr(result, 'method', 'simple'),
                    'freq': result.freq.tolist(),
                    'zre_fit': result.mu_Z_re.tolist() if hasattr(result, 'mu_Z_re') else None,
                    'zim_fit': result.mu_Z_im.tolist() if hasattr(result, 'mu_Z_im') else None,
                    'residuals': {
                        'zre': result.res_re.tolist() if hasattr(result, 'res_re') else None,
                        'zim': result.res_im.tolist() if hasattr(result, 'res_im') else None
                    }
                }
            })
            
        except Exception as calc_error:
            print(f"DRT calculation error: {calc_error}")
            import traceback
            traceback.print_exc()
            return jsonify({
                'success': False, 
                'error': f'DRT calculation failed: {str(calc_error)}'
            }), 500
        
    except Exception as e:
        print(f"Error calculating DRT: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/drt/plot', methods=['POST'])
def generate_drt_plot():
    """
    Generate DRT plot (returns base64 encoded image)
    """
    if not DRT_AVAILABLE:
        return jsonify({'success': False, 'error': 'DRT analysis not available'}), 503
        
    try:
        data = request.get_json()
        plot_type = data.get('plot_type', 'drt')
        results = data.get('results')
        
        if not results:
            return jsonify({'success': False, 'error': 'No results data provided'}), 400
            
        # Create plot based on type
        plt.figure(figsize=(10, 6))
        
        if plot_type == 'drt':
            tau = np.array(results['tau'])
            gamma = np.array(results['gamma'])
            plt.semilogx(tau, gamma, '-', linewidth=2, color='blue')
            plt.xlabel('τ [s]', fontsize=12)
            plt.ylabel('γ [Ω]', fontsize=12)
            plt.title('Distribution of Relaxation Times', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.ylim(bottom=0)
            
        elif plot_type == 'nyquist':
            freq = np.array(results['freq'])
            zre_orig = np.array(data['zre'])
            zim_orig = np.array(data['zim'])
            plt.plot(zre_orig, -zim_orig, 'o', markersize=4, label='Original', alpha=0.7)
            
            if results.get('zre_fit') and results.get('zim_fit'):
                zre_fit = np.array(results['zre_fit'])
                zim_fit = np.array(results['zim_fit'])
                plt.plot(zre_fit, -zim_fit, '-', linewidth=2, label='Fitted', color='red')
                plt.legend()
                
            plt.xlabel('Z_re [Ω]', fontsize=12)
            plt.ylabel('-Z_im [Ω]', fontsize=12)
            plt.title('Nyquist Plot', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.axis('equal')
            
        elif plot_type == 'magnitude':
            freq = np.array(results['freq'])
            zre_orig = np.array(data['zre'])
            zim_orig = np.array(data['zim'])
            mag_orig = np.sqrt(zre_orig**2 + zim_orig**2)
            plt.loglog(freq, mag_orig, 'o', markersize=4, label='Original', alpha=0.7)
            
            if results.get('zre_fit') and results.get('zim_fit'):
                zre_fit = np.array(results['zre_fit'])
                zim_fit = np.array(results['zim_fit'])
                mag_fit = np.sqrt(zre_fit**2 + zim_fit**2)
                plt.loglog(freq, mag_fit, '-', linewidth=2, label='Fitted', color='red')
                plt.legend()
                
            plt.xlabel('Frequency [Hz]', fontsize=12)
            plt.ylabel('|Z| [Ω]', fontsize=12)
            plt.title('Magnitude vs Frequency', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            
        elif plot_type == 'phase':
            freq = np.array(results['freq'])
            zre_orig = np.array(data['zre'])
            zim_orig = np.array(data['zim'])
            phase_orig = np.angle(zre_orig + 1j * zim_orig, deg=True)
            plt.semilogx(freq, phase_orig, 'o', markersize=4, label='Original', alpha=0.7)
            
            if results.get('zre_fit') and results.get('zim_fit'):
                zre_fit = np.array(results['zre_fit'])
                zim_fit = np.array(results['zim_fit'])
                phase_fit = np.angle(zre_fit + 1j * zim_fit, deg=True)
                plt.semilogx(freq, phase_fit, '-', linewidth=2, label='Fitted', color='red')
                plt.legend()
                
            plt.xlabel('Frequency [Hz]', fontsize=12)
            plt.ylabel('Phase [°]', fontsize=12)
            plt.title('Phase vs Frequency', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            
        else:
            return jsonify({'success': False, 'error': f'Unknown plot type: {plot_type}'}), 400
            
        plt.tight_layout()
        
        # Add watermark before saving
        add_watermark_to_plot(custom_text=f"DRT Analysis - {plot_type.upper()}")
        
        # Convert to base64
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return jsonify({
            'success': True,
            'plot': img_base64,
            'plot_type': plot_type
        })
        
    except Exception as e:
        print(f"Error generating DRT plot: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

# =============================================================================
#                         RAW DATA READER ROUTES
# =============================================================================

@app.route('/raw-data-reader')
@auth_required
def raw_data_reader_page():
    """Raw Data Reader Page"""
    return send_from_directory('.', 'raw_data_reader.html')


@app.route('/doe-planner')
@auth_required
def doe_planner_page():
    """DOE Planner Page"""
    return send_from_directory('.', 'doe_planner.html')


# =============================================================================
#                         DOE PLANNER API ROUTES
# =============================================================================

# Import DOE database functions
try:
    from doe_database import (
        init_database, get_all_experiments, get_experiment, add_experiment,
        update_experiment, delete_experiment, get_all_outcomes, add_outcome,
        update_outcome, delete_outcome, get_all_intake, add_intake,
        promote_intake_to_experiment, get_dropdown_options, get_standard_conditions,
        get_overview, get_statistics, get_db_connection, populate_dropdown_options,
        populate_standard_conditions, populate_overview, DB_PATH, add_dropdown_option,
        # New imports for enhanced features
        duplicate_experiment, toggle_favorite, bulk_update_experiments, bulk_delete_experiments,
        get_all_templates, get_template, add_template, delete_template, create_experiment_from_template,
        get_comments, add_comment, delete_comment, get_activity_log, log_activity,
        get_user_preferences, update_user_preferences, get_notifications, add_notification,
        mark_notification_read, mark_all_notifications_read, get_calendar_data, get_kanban_data,
        update_experiment_status, populate_default_templates,
        # Backup functions
        create_backup, get_all_backups, get_backup_path, restore_backup, delete_backup, BACKUP_DIR
    )
    DOE_DB_AVAILABLE = True
    # Auto-initialize database if it doesn't exist
    if not DB_PATH.exists():
        print("Initializing DOE database...")
        init_database()
        populate_dropdown_options()
        populate_standard_conditions()
        populate_overview()
        populate_default_templates()
        print("DOE database initialized!")
except ImportError as e:
    DOE_DB_AVAILABLE = False
    print(f"Warning: DOE database module not available: {e}")


@app.route('/api/doe/stats')
@auth_required
def doe_stats():
    """Get DOE dashboard statistics"""
    if not DOE_DB_AVAILABLE:
        return jsonify({'success': False, 'error': 'DOE database not available'}), 500
    try:
        stats = get_statistics()
        return jsonify({'success': True, 'stats': stats})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/doe/overview')
@auth_required
def doe_overview():
    """Get DOE overview information"""
    if not DOE_DB_AVAILABLE:
        return jsonify({'success': False, 'error': 'DOE database not available'}), 500
    try:
        overview = get_overview()
        return jsonify({'success': True, 'overview': overview})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/doe/dropdowns')
@auth_required
def doe_dropdowns():
    """Get all dropdown options"""
    if not DOE_DB_AVAILABLE:
        return jsonify({'success': False, 'error': 'DOE database not available'}), 500
    try:
        dropdowns = {
            'test_classification': get_dropdown_options('test_classification'),
            'anode_current_collector': get_dropdown_options('anode_current_collector'),
            'cathode_current_collector': get_dropdown_options('cathode_current_collector'),
            'priority': get_dropdown_options('priority'),
            'separator': get_dropdown_options('separator'),
            'cathode': get_dropdown_options('cathode'),
            'test_station': get_dropdown_options('test_station'),
            'time_slot': get_dropdown_options('time_slot'),
        }
        return jsonify({'success': True, 'dropdowns': dropdowns})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/doe/defaults')
@auth_required
def doe_defaults():
    """Get standard condition defaults"""
    if not DOE_DB_AVAILABLE:
        return jsonify({'success': False, 'error': 'DOE database not available'}), 500
    try:
        defaults = get_standard_conditions()
        return jsonify({'success': True, 'defaults': defaults})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/doe/dropdowns/add', methods=['POST'])
@auth_required
def doe_add_dropdown_option():
    """Add a new custom option to a dropdown category"""
    if not DOE_DB_AVAILABLE:
        return jsonify({'success': False, 'error': 'DOE database not available'}), 500
    try:
        data = request.get_json()
        category = data.get('category')
        value = data.get('value')
        
        if not category or not value:
            return jsonify({'success': False, 'error': 'Category and value are required'}), 400
        
        add_dropdown_option(category, value)
        return jsonify({'success': True, 'message': f'Added "{value}" to {category}'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# EC Experiments CRUD
@app.route('/api/doe/experiments', methods=['GET'])
@auth_required
def doe_get_experiments():
    """Get all experiments with optional pagination and filtering"""
    if not DOE_DB_AVAILABLE:
        return jsonify({'success': False, 'error': 'DOE database not available'}), 500
    try:
        # Get pagination params
        page = request.args.get('page', type=int)
        per_page = request.args.get('per_page', 50, type=int)
        
        # Get filter params
        filters = {}
        if request.args.get('search'):
            filters['search'] = request.args.get('search')
        if request.args.get('priority'):
            filters['priority'] = request.args.get('priority')
        if request.args.get('status'):
            filters['status'] = request.args.get('status')
        if request.args.get('assigned_to'):
            filters['assigned_to'] = request.args.get('assigned_to')
        if request.args.get('classification'):
            filters['classification'] = request.args.get('classification')
        if request.args.get('favorites_only'):
            filters['favorites_only'] = True
        
        # Get experiments (paginated or all)
        result = get_all_experiments(filters=filters if filters else None, page=page, per_page=per_page)
        
        # Handle paginated vs non-paginated response
        if isinstance(result, dict):
            return jsonify({
                'success': True,
                'experiments': result['experiments'],
                'count': len(result['experiments']),
                'total': result['total'],
                'page': result['page'],
                'per_page': result['per_page'],
                'total_pages': result['total_pages']
            })
        else:
            return jsonify({'success': True, 'experiments': result, 'count': len(result)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/doe/experiments/<ec_id>', methods=['GET'])
@auth_required
def doe_get_experiment(ec_id):
    """Get single experiment"""
    if not DOE_DB_AVAILABLE:
        return jsonify({'success': False, 'error': 'DOE database not available'}), 500
    try:
        experiment = get_experiment(ec_id)
        if experiment:
            return jsonify({'success': True, 'experiment': experiment})
        return jsonify({'success': False, 'error': 'Experiment not found'}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/doe/experiments', methods=['POST'])
@auth_required
def doe_add_experiment():
    """Add new experiment"""
    if not DOE_DB_AVAILABLE:
        return jsonify({'success': False, 'error': 'DOE database not available'}), 500
    try:
        data = request.get_json()
        ec_id = add_experiment(data)
        return jsonify({'success': True, 'ec_id': ec_id, 'message': 'Experiment added successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/doe/experiments/<ec_id>', methods=['PUT'])
@auth_required
def doe_update_experiment(ec_id):
    """Update experiment"""
    if not DOE_DB_AVAILABLE:
        return jsonify({'success': False, 'error': 'DOE database not available'}), 500
    try:
        data = request.get_json()
        update_experiment(ec_id, data)
        return jsonify({'success': True, 'message': 'Experiment updated successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/doe/experiments/<ec_id>', methods=['DELETE'])
@auth_required
def doe_delete_experiment(ec_id):
    """Delete experiment"""
    if not DOE_DB_AVAILABLE:
        return jsonify({'success': False, 'error': 'DOE database not available'}), 500
    try:
        delete_experiment(ec_id)
        return jsonify({'success': True, 'message': 'Experiment deleted successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# EC Outcomes CRUD
@app.route('/api/doe/outcomes', methods=['GET'])
@auth_required
def doe_get_outcomes():
    """Get all outcomes"""
    if not DOE_DB_AVAILABLE:
        return jsonify({'success': False, 'error': 'DOE database not available'}), 500
    try:
        outcomes = get_all_outcomes()
        return jsonify({'success': True, 'outcomes': outcomes, 'count': len(outcomes)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/doe/outcomes', methods=['POST'])
@auth_required
def doe_add_outcome():
    """Add new outcome"""
    if not DOE_DB_AVAILABLE:
        return jsonify({'success': False, 'error': 'DOE database not available'}), 500
    try:
        data = request.get_json()
        outcome_id = add_outcome(data)
        return jsonify({'success': True, 'id': outcome_id, 'message': 'Outcome added successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/doe/outcomes/<int:outcome_id>', methods=['PUT'])
@auth_required
def doe_update_outcome(outcome_id):
    """Update outcome"""
    if not DOE_DB_AVAILABLE:
        return jsonify({'success': False, 'error': 'DOE database not available'}), 500
    try:
        data = request.get_json()
        update_outcome(outcome_id, data)
        return jsonify({'success': True, 'message': 'Outcome updated successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/doe/outcomes/<int:outcome_id>', methods=['DELETE'])
@auth_required
def doe_delete_outcome(outcome_id):
    """Delete outcome"""
    if not DOE_DB_AVAILABLE:
        return jsonify({'success': False, 'error': 'DOE database not available'}), 500
    try:
        delete_outcome(outcome_id)
        return jsonify({'success': True, 'message': 'Outcome deleted successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# Intake Queue CRUD
@app.route('/api/doe/intake', methods=['GET'])
@auth_required
def doe_get_intake():
    """Get all intake queue items"""
    if not DOE_DB_AVAILABLE:
        return jsonify({'success': False, 'error': 'DOE database not available'}), 500
    try:
        intake = get_all_intake()
        return jsonify({'success': True, 'intake': intake, 'count': len(intake)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/doe/intake', methods=['POST'])
@auth_required
def doe_add_intake():
    """Add new intake queue item"""
    if not DOE_DB_AVAILABLE:
        return jsonify({'success': False, 'error': 'DOE database not available'}), 500
    try:
        data = request.get_json()
        intake_id = add_intake(data)
        return jsonify({'success': True, 'id': intake_id, 'message': 'Intake item added successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/doe/intake/<int:intake_id>/promote', methods=['POST'])
@auth_required
def doe_promote_intake(intake_id):
    """Promote intake item to EC Experiment"""
    if not DOE_DB_AVAILABLE:
        return jsonify({'success': False, 'error': 'DOE database not available'}), 500
    try:
        ec_id = promote_intake_to_experiment(intake_id)
        if ec_id:
            return jsonify({'success': True, 'ec_id': ec_id, 'message': f'Promoted to experiment {ec_id}'})
        return jsonify({'success': False, 'error': 'Intake item not found'}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/doe/intake/<int:intake_id>', methods=['DELETE'])
@auth_required
def doe_delete_intake(intake_id):
    """Delete intake queue item"""
    if not DOE_DB_AVAILABLE:
        return jsonify({'success': False, 'error': 'DOE database not available'}), 500
    try:
        conn = get_db_connection()
        conn.execute('DELETE FROM intake_queue WHERE id = ?', (intake_id,))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': 'Intake item deleted successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# Export to Excel
@app.route('/api/doe/export', methods=['GET'])
@auth_required
def doe_export():
    """Export all DOE data to Excel"""
    if not DOE_DB_AVAILABLE:
        return jsonify({'success': False, 'error': 'DOE database not available'}), 500
    try:
        import pandas as pd
        from io import BytesIO
        
        conn = get_db_connection()
        
        # Create Excel writer
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Export each table
            pd.read_sql_query('SELECT * FROM ec_experiments', conn).to_excel(writer, sheet_name='EC Experiments', index=False)
            pd.read_sql_query('SELECT * FROM ec_outcomes', conn).to_excel(writer, sheet_name='EC Outcomes', index=False)
            pd.read_sql_query('SELECT * FROM intake_queue', conn).to_excel(writer, sheet_name='Intake Queue', index=False)
            pd.read_sql_query('SELECT * FROM overview', conn).to_excel(writer, sheet_name='Overview', index=False)
        
        conn.close()
        output.seek(0)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=f'DOE_Planner_Export_{timestamp}.xlsx'
        )
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ============= BACKUP ENDPOINTS =============

@app.route('/api/doe/backup', methods=['POST'])
@auth_required
def doe_create_backup():
    """Create a backup of the database"""
    if not DOE_DB_AVAILABLE:
        return jsonify({'success': False, 'error': 'DOE database not available'}), 500
    try:
        reason = request.json.get('reason', 'manual') if request.json else 'manual'
        result = create_backup(reason=reason)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/doe/backups', methods=['GET'])
@auth_required
def doe_list_backups():
    """Get list of all backups"""
    if not DOE_DB_AVAILABLE:
        return jsonify({'success': False, 'error': 'DOE database not available'}), 500
    try:
        backups = get_all_backups()
        return jsonify({'success': True, 'backups': backups, 'count': len(backups)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/doe/backup/download/<filename>', methods=['GET'])
@auth_required
def doe_download_backup(filename):
    """Download a backup file"""
    if not DOE_DB_AVAILABLE:
        return jsonify({'success': False, 'error': 'DOE database not available'}), 500
    try:
        backup_path = get_backup_path(filename)
        if not backup_path:
            return jsonify({'success': False, 'error': 'Backup not found'}), 404
        
        return send_file(
            backup_path,
            mimetype='application/octet-stream',
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/doe/backup/restore/<filename>', methods=['POST'])
@auth_required
def doe_restore_backup(filename):
    """Restore database from a backup"""
    if not DOE_DB_AVAILABLE:
        return jsonify({'success': False, 'error': 'DOE database not available'}), 500
    try:
        result = restore_backup(filename)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/doe/backup/<filename>', methods=['DELETE'])
@auth_required
def doe_delete_backup(filename):
    """Delete a backup file"""
    if not DOE_DB_AVAILABLE:
        return jsonify({'success': False, 'error': 'DOE database not available'}), 500
    try:
        result = delete_backup(filename)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# =============================================================================
#                    ENHANCED DOE FEATURES (20 Features)
# =============================================================================

# Feature 2: Duplicate/Clone Experiment
@app.route('/api/doe/experiments/<ec_id>/duplicate', methods=['POST'])
@auth_required
def doe_duplicate_experiment(ec_id):
    """Duplicate an existing experiment"""
    if not DOE_DB_AVAILABLE:
        return jsonify({'success': False, 'error': 'DOE database not available'}), 500
    try:
        new_ec_id = duplicate_experiment(ec_id)
        if new_ec_id:
            return jsonify({'success': True, 'ec_id': new_ec_id, 'message': f'Duplicated to {new_ec_id}'})
        return jsonify({'success': False, 'error': 'Experiment not found'}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# Feature 9: Bulk Actions
@app.route('/api/doe/experiments/bulk-update', methods=['POST'])
@auth_required
def doe_bulk_update():
    """Bulk update multiple experiments"""
    if not DOE_DB_AVAILABLE:
        return jsonify({'success': False, 'error': 'DOE database not available'}), 500
    try:
        data = request.get_json()
        ec_ids = data.get('ec_ids', [])
        updates = data.get('updates', {})
        bulk_update_experiments(ec_ids, updates)
        return jsonify({'success': True, 'message': f'Updated {len(ec_ids)} experiments'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/doe/experiments/bulk-delete', methods=['POST'])
@auth_required
def doe_bulk_delete():
    """Bulk delete multiple experiments"""
    if not DOE_DB_AVAILABLE:
        return jsonify({'success': False, 'error': 'DOE database not available'}), 500
    try:
        data = request.get_json()
        ec_ids = data.get('ec_ids', [])
        bulk_delete_experiments(ec_ids)
        return jsonify({'success': True, 'message': f'Deleted {len(ec_ids)} experiments'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# Feature 10: Templates
@app.route('/api/doe/templates', methods=['GET'])
@auth_required
def doe_get_templates():
    """Get all templates"""
    if not DOE_DB_AVAILABLE:
        return jsonify({'success': False, 'error': 'DOE database not available'}), 500
    try:
        templates = get_all_templates()
        return jsonify({'success': True, 'templates': templates})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/doe/templates', methods=['POST'])
@auth_required
def doe_add_template():
    """Add a new template"""
    if not DOE_DB_AVAILABLE:
        return jsonify({'success': False, 'error': 'DOE database not available'}), 500
    try:
        data = request.get_json()
        template_id = add_template(data)
        return jsonify({'success': True, 'id': template_id, 'message': 'Template created'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/doe/templates/<int:template_id>', methods=['DELETE'])
@auth_required
def doe_delete_template(template_id):
    """Delete a template"""
    if not DOE_DB_AVAILABLE:
        return jsonify({'success': False, 'error': 'DOE database not available'}), 500
    try:
        delete_template(template_id)
        return jsonify({'success': True, 'message': 'Template deleted'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/doe/templates/<int:template_id>/create-experiment', methods=['POST'])
@auth_required
def doe_create_from_template(template_id):
    """Create experiment from template"""
    if not DOE_DB_AVAILABLE:
        return jsonify({'success': False, 'error': 'DOE database not available'}), 500
    try:
        data = request.get_json() or {}
        ec_id = create_experiment_from_template(template_id, data)
        if ec_id:
            return jsonify({'success': True, 'ec_id': ec_id, 'message': f'Created {ec_id} from template'})
        return jsonify({'success': False, 'error': 'Template not found'}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# Feature 12: Activity Log
@app.route('/api/doe/activity', methods=['GET'])
@auth_required
def doe_get_activity():
    """Get activity log"""
    if not DOE_DB_AVAILABLE:
        return jsonify({'success': False, 'error': 'DOE database not available'}), 500
    try:
        limit = request.args.get('limit', 50, type=int)
        entity_type = request.args.get('entity_type')
        entity_id = request.args.get('entity_id')
        activities = get_activity_log(limit, entity_type, entity_id)
        return jsonify({'success': True, 'activities': activities})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# Feature 13: Comments
@app.route('/api/doe/experiments/<ec_id>/comments', methods=['GET'])
@auth_required
def doe_get_comments(ec_id):
    """Get comments for an experiment"""
    if not DOE_DB_AVAILABLE:
        return jsonify({'success': False, 'error': 'DOE database not available'}), 500
    try:
        comments = get_comments(ec_id)
        return jsonify({'success': True, 'comments': comments})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/doe/experiments/<ec_id>/comments', methods=['POST'])
@auth_required
def doe_add_comment(ec_id):
    """Add a comment to an experiment"""
    if not DOE_DB_AVAILABLE:
        return jsonify({'success': False, 'error': 'DOE database not available'}), 500
    try:
        data = request.get_json()
        user_name = current_user.name if hasattr(current_user, 'name') else 'User'
        comment_id = add_comment(ec_id, user_name, data.get('comment', ''))
        return jsonify({'success': True, 'id': comment_id, 'message': 'Comment added'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/doe/comments/<int:comment_id>', methods=['DELETE'])
@auth_required
def doe_delete_comment(comment_id):
    """Delete a comment"""
    if not DOE_DB_AVAILABLE:
        return jsonify({'success': False, 'error': 'DOE database not available'}), 500
    try:
        delete_comment(comment_id)
        return jsonify({'success': True, 'message': 'Comment deleted'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# Feature 15: Notifications
@app.route('/api/doe/notifications', methods=['GET'])
@auth_required
def doe_get_notifications():
    """Get notifications for current user"""
    if not DOE_DB_AVAILABLE:
        return jsonify({'success': False, 'error': 'DOE database not available'}), 500
    try:
        user_name = current_user.name if hasattr(current_user, 'name') else 'User'
        unread_only = request.args.get('unread_only', 'false').lower() == 'true'
        notifications = get_notifications(user_name, unread_only)
        return jsonify({'success': True, 'notifications': notifications})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/doe/notifications/<int:notif_id>/read', methods=['POST'])
@auth_required
def doe_mark_notification_read(notif_id):
    """Mark a notification as read"""
    if not DOE_DB_AVAILABLE:
        return jsonify({'success': False, 'error': 'DOE database not available'}), 500
    try:
        mark_notification_read(notif_id)
        return jsonify({'success': True, 'message': 'Notification marked as read'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/doe/notifications/read-all', methods=['POST'])
@auth_required
def doe_mark_all_read():
    """Mark all notifications as read"""
    if not DOE_DB_AVAILABLE:
        return jsonify({'success': False, 'error': 'DOE database not available'}), 500
    try:
        user_name = current_user.name if hasattr(current_user, 'name') else 'User'
        mark_all_notifications_read(user_name)
        return jsonify({'success': True, 'message': 'All notifications marked as read'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# Feature 17: Favorites
@app.route('/api/doe/experiments/<ec_id>/favorite', methods=['POST'])
@auth_required
def doe_toggle_favorite(ec_id):
    """Toggle favorite status"""
    if not DOE_DB_AVAILABLE:
        return jsonify({'success': False, 'error': 'DOE database not available'}), 500
    try:
        is_favorite = toggle_favorite(ec_id)
        return jsonify({'success': True, 'is_favorite': is_favorite})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# Feature 18: User Preferences (Column Customization, Theme)
@app.route('/api/doe/preferences', methods=['GET'])
@auth_required
def doe_get_preferences():
    """Get user preferences"""
    if not DOE_DB_AVAILABLE:
        return jsonify({'success': False, 'error': 'DOE database not available'}), 500
    try:
        user_name = current_user.name if hasattr(current_user, 'name') else 'User'
        preferences = get_user_preferences(user_name)
        return jsonify({'success': True, 'preferences': preferences})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/doe/preferences', methods=['POST'])
@auth_required
def doe_update_preferences():
    """Update user preferences"""
    if not DOE_DB_AVAILABLE:
        return jsonify({'success': False, 'error': 'DOE database not available'}), 500
    try:
        user_name = current_user.name if hasattr(current_user, 'name') else 'User'
        data = request.get_json()
        update_user_preferences(user_name, data)
        return jsonify({'success': True, 'message': 'Preferences saved'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# Feature 7: Kanban Board View
@app.route('/api/doe/kanban', methods=['GET'])
@auth_required
def doe_get_kanban():
    """Get kanban board data"""
    if not DOE_DB_AVAILABLE:
        return jsonify({'success': False, 'error': 'DOE database not available'}), 500
    try:
        kanban_data = get_kanban_data()
        return jsonify({'success': True, 'kanban': kanban_data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/doe/experiments/<ec_id>/status', methods=['POST'])
@auth_required
def doe_update_status(ec_id):
    """Update experiment status (for kanban drag-drop)"""
    if not DOE_DB_AVAILABLE:
        return jsonify({'success': False, 'error': 'DOE database not available'}), 500
    try:
        data = request.get_json()
        new_status = data.get('status', 'pending')
        update_experiment_status(ec_id, new_status)
        return jsonify({'success': True, 'message': f'Status updated to {new_status}'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# Feature 8: Calendar View
@app.route('/api/doe/calendar', methods=['GET'])
@auth_required
def doe_get_calendar():
    """Get calendar data for a specific month"""
    if not DOE_DB_AVAILABLE:
        return jsonify({'success': False, 'error': 'DOE database not available'}), 500
    try:
        year = request.args.get('year', datetime.now().year, type=int)
        month = request.args.get('month', datetime.now().month, type=int)
        calendar_data = get_calendar_data(year, month)
        return jsonify({'success': True, 'experiments': calendar_data, 'year': year, 'month': month})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# Feature 11: Import from Excel
@app.route('/api/doe/import', methods=['POST'])
@auth_required
def doe_import_excel():
    """Import experiments from Excel file"""
    if not DOE_DB_AVAILABLE:
        return jsonify({'success': False, 'error': 'DOE database not available'}), 500
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        # Read Excel file
        filename_lower = file.filename.lower()
        if filename_lower.endswith('.xlsx'):
            df = pd.read_excel(file, engine='openpyxl')
        elif filename_lower.endswith('.xls'):
            df = pd.read_excel(file, engine='xlrd')
        elif filename_lower.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            return jsonify({'success': False, 'error': 'Unsupported file format'}), 400
        
        # Import rows as experiments
        imported = 0
        for _, row in df.iterrows():
            try:
                exp_data = {
                    'proposer': str(row.get('Proposer', '')) if not pd.isna(row.get('Proposer', '')) else None,
                    'test_purpose': str(row.get('Test Purpose', '')) if not pd.isna(row.get('Test Purpose', '')) else None,
                    'test_classification': str(row.get('Test Classification', '')) if not pd.isna(row.get('Test Classification', '')) else None,
                    'anode': str(row.get('Anode', '')) if not pd.isna(row.get('Anode', '')) else None,
                    'separator': str(row.get('Separator', '')) if not pd.isna(row.get('Separator', '')) else None,
                    'cathode': str(row.get('Cathode', '')) if not pd.isna(row.get('Cathode', '')) else None,
                    'temperature': float(row.get('Temperature', 80)) if not pd.isna(row.get('Temperature', 80)) else 80,
                    'dp_psi': float(row.get('dP', 5)) if not pd.isna(row.get('dP', 5)) else 5,
                    'priority_level': str(row.get('Priority', '')) if not pd.isna(row.get('Priority', '')) else None,
                }
                add_experiment(exp_data)
                imported += 1
            except Exception as e:
                print(f"Error importing row: {e}")
                continue
        
        return jsonify({'success': True, 'message': f'Imported {imported} experiments', 'count': imported})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/raw-data/analyze', methods=['POST'])
def analyze_raw_data():
    """
    Analyze raw data files (CSV, XLS, XLSX) and return statistics, detected columns, and available analysis options
    Supports multi-file upload for comparison
    """
    try:
        files = request.files.getlist('file')
        if not files or len(files) == 0:
            # Try single file
            if 'file' in request.files:
                files = [request.files['file']]
            else:
                return jsonify({'success': False, 'error': 'No file(s) provided'}), 400
        
        results = []
        
        for file in files:
            if file.filename == '':
                continue
            
            filename_lower = file.filename.lower()
            
            try:
                # Read file based on extension
                if filename_lower.endswith('.csv'):
                    df = pd.read_csv(file)
                elif filename_lower.endswith('.xlsx'):
                    df = pd.read_excel(file, engine='openpyxl')
                elif filename_lower.endswith('.xls'):
                    df = pd.read_excel(file, engine='xlrd')
                else:
                    results.append({
                        'filename': file.filename,
                        'success': False,
                        'error': f'Unsupported file format. Use CSV, XLS, or XLSX.'
                    })
                    continue
            except Exception as e:
                results.append({
                    'filename': file.filename,
                    'success': False,
                    'error': f'Failed to read file: {str(e)}'
                })
                continue
            
            # Auto-detect columns
            detected_columns = auto_detect_columns(df)
            
            # Calculate statistics for all columns
            column_stats = calculate_column_statistics(df)
            
            # Identify available analysis options
            analysis_options = identify_analysis_options(df, detected_columns)
            
            # Get data preview (first 10 rows)
            preview = df.head(10).replace({np.nan: None}).to_dict('records')
            
            # Store raw data for later use (as lists for JSON serialization)
            raw_data = {}
            for col in df.columns:
                raw_data[col] = df[col].replace({np.nan: None}).tolist()
            
            results.append({
                'filename': file.filename,
                'success': True,
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'columns': df.columns.tolist(),
                'detected_columns': detected_columns,
                'column_stats': column_stats,
                'analysis_options': analysis_options,
                'preview': preview,
                'raw_data': raw_data
            })
        
        if len(results) == 1:
            return jsonify(results[0])
        else:
            return jsonify({
                'success': True,
                'multi_file': True,
                'file_count': len(results),
                'files': results
            })
        
    except Exception as e:
        print(f"Error analyzing raw data: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


def auto_detect_columns(df):
    """
    Auto-detect column types based on column names and data patterns
    Returns a mapping of detected data types to column names
    """
    detected = {
        'time': None,
        'potential': None,
        'current': None,
        'frequency': None,
        'zre': None,
        'zim': None,
        'power': None,
        'temperature': None,
        'other_numeric': [],
        'non_numeric': []
    }
    
    # Column name patterns for detection
    patterns = {
        'time': ['time', 'elapsed', 't(s)', 't (s)', 'seconds', 'timestamp', 'time(s)', 'time (s)', 'elapsed time'],
        'potential': ['potential', 'voltage', 'volt', 'v(v)', 'v (v)', 'e(v)', 'e (v)', 'potential (v)', 'voltage (v)', 'ecell', 'ewe'],
        'current': ['current', 'amp', 'i(a)', 'i (a)', 'current (a)', 'amps', 'i/ma', 'current(a)'],
        'frequency': ['frequency', 'freq', 'f(hz)', 'f (hz)', 'hz', 'frequency (hz)', 'frequency(hz)'],
        'zre': ['zre', 'z_re', 'z re', 'zreal', 'z_real', 'real', 're', 'zre(ohms)', 'zre (ohms)', 'z\''],
        'zim': ['zim', 'z_im', 'z im', 'zimag', 'z_imag', 'imag', 'im', 'zim(ohms)', 'zim (ohms)', 'z\'\''],
        'power': ['power', 'p(w)', 'p (w)', 'watt', 'power (w)', 'power(w)'],
        'temperature': ['temperature', 'temp', 't(c)', 't (c)', 'celsius', 'temperature (c)', 'temp (c)']
    }
    
    for col in df.columns:
        col_lower = col.lower().strip()
        matched = False
        
        for data_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                if pattern in col_lower or col_lower == pattern:
                    if detected[data_type] is None:
                        detected[data_type] = col
                        matched = True
                        break
            if matched:
                break
        
        if not matched:
            # Check if numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                detected['other_numeric'].append(col)
            else:
                detected['non_numeric'].append(col)
    
    return detected


def calculate_column_statistics(df):
    """
    Calculate comprehensive statistics for all columns
    """
    stats = {}
    
    for col in df.columns:
        col_stats = {
            'dtype': str(df[col].dtype),
            'non_null_count': int(df[col].notna().sum()),
            'null_count': int(df[col].isna().sum())
        }
        
        if pd.api.types.is_numeric_dtype(df[col]):
            # Numeric column statistics
            col_data = df[col].dropna()
            if len(col_data) > 0:
                col_stats.update({
                    'min': float(col_data.min()),
                    'max': float(col_data.max()),
                    'mean': float(col_data.mean()),
                    'std': float(col_data.std()) if len(col_data) > 1 else 0,
                    'median': float(col_data.median()),
                    'q25': float(col_data.quantile(0.25)),
                    'q75': float(col_data.quantile(0.75)),
                    'unique_count': int(col_data.nunique())
                })
        else:
            # Non-numeric column statistics
            col_stats['unique_count'] = int(df[col].nunique())
            top_values = df[col].value_counts().head(5).to_dict()
            col_stats['top_values'] = {str(k): int(v) for k, v in top_values.items()}
        
        stats[col] = col_stats
    
    return stats


def identify_analysis_options(df, detected_columns):
    """
    Identify which analysis options are available based on detected columns
    """
    options = {
        'time_plots': [],
        'performance_curves': [],
        'eis_analysis': [],
        'statistics': ['basic_stats', 'correlation_matrix']
    }
    
    has_time = detected_columns.get('time') is not None
    has_potential = detected_columns.get('potential') is not None
    has_current = detected_columns.get('current') is not None
    has_frequency = detected_columns.get('frequency') is not None
    has_zre = detected_columns.get('zre') is not None
    has_zim = detected_columns.get('zim') is not None
    
    # Time-based plots
    if has_time:
        if has_potential:
            options['time_plots'].append({
                'id': 'potential_vs_time',
                'name': 'Potential vs Time',
                'description': 'Plot potential changes over time',
                'x_col': detected_columns['time'],
                'y_col': detected_columns['potential']
            })
        if has_current:
            options['time_plots'].append({
                'id': 'current_vs_time',
                'name': 'Current vs Time',
                'description': 'Plot current changes over time',
                'x_col': detected_columns['time'],
                'y_col': detected_columns['current']
            })
        if has_potential and has_current:
            options['time_plots'].append({
                'id': 'power_vs_time',
                'name': 'Power vs Time',
                'description': 'Plot power (V×I) changes over time',
                'x_col': detected_columns['time'],
                'y_cols': [detected_columns['potential'], detected_columns['current']],
                'computed': True
            })
    
    # Performance curves (Polarization)
    if has_potential and has_current:
        options['performance_curves'].append({
            'id': 'potential_vs_current',
            'name': 'Potential vs Current',
            'description': 'V-I characteristic curve',
            'x_col': detected_columns['current'],
            'y_col': detected_columns['potential']
        })
        options['performance_curves'].append({
            'id': 'current_vs_potential',
            'name': 'Current vs Potential (Polarization)',
            'description': 'Polarization curve (I-V)',
            'x_col': detected_columns['potential'],
            'y_col': detected_columns['current']
        })
    
    # EIS Analysis
    if has_frequency and has_zre and has_zim:
        options['eis_analysis'].append({
            'id': 'nyquist',
            'name': 'Nyquist Plot',
            'description': 'Zre vs -Zim impedance plot',
            'x_col': detected_columns['zre'],
            'y_col': detected_columns['zim']
        })
        options['eis_analysis'].append({
            'id': 'bode_magnitude',
            'name': 'Bode Magnitude',
            'description': '|Z| vs Frequency',
            'x_col': detected_columns['frequency'],
            'y_cols': [detected_columns['zre'], detected_columns['zim']]
        })
        options['eis_analysis'].append({
            'id': 'bode_phase',
            'name': 'Bode Phase',
            'description': 'Phase angle vs Frequency',
            'x_col': detected_columns['frequency'],
            'y_cols': [detected_columns['zre'], detected_columns['zim']]
        })
        
        # Check for multi-current EIS
        if has_current:
            freq_col = detected_columns['frequency']
            current_col = detected_columns['current']
            
            # Filter for EIS data (frequency > 0)
            eis_mask = df[freq_col] > 0
            if eis_mask.sum() > 0:
                eis_currents = df.loc[eis_mask, current_col].dropna()
                rounded_currents = np.round(eis_currents).astype(int)
                unique_currents = sorted(set(rounded_currents))
                
                if len(unique_currents) > 1:
                    options['eis_analysis'].append({
                        'id': 'eis_by_current',
                        'name': 'EIS by Current Level',
                        'description': f'EIS data available at {len(unique_currents)} current levels',
                        'current_levels': unique_currents
                    })
    
    return options


@app.route('/api/raw-data/generate-plot', methods=['POST'])
def generate_raw_data_plot():
    """
    Generate interactive Plotly plots for raw data analysis
    Returns Plotly JSON for client-side rendering
    """
    try:
        data = request.get_json()
        plot_type = data.get('plot_type')
        raw_data = data.get('raw_data')
        options = data.get('options', {})
        file_info = data.get('file_info', {})
        
        if not plot_type or not raw_data:
            return jsonify({'success': False, 'error': 'Missing plot_type or raw_data'}), 400
        
        # Convert raw_data dict to DataFrame
        df = pd.DataFrame(raw_data)
        
        # Generate plot based on type
        if plot_type == 'potential_vs_time':
            plot_data = generate_time_plot(df, options.get('x_col'), options.get('y_col'), 
                                          'Time', 'Potential (V)', 'Potential vs Time', file_info)
        
        elif plot_type == 'current_vs_time':
            plot_data = generate_time_plot(df, options.get('x_col'), options.get('y_col'),
                                          'Time', 'Current (A)', 'Current vs Time', file_info)
        
        elif plot_type == 'power_vs_time':
            # Compute power
            v_col = options.get('y_cols', [None, None])[0]
            i_col = options.get('y_cols', [None, None])[1]
            x_col = options.get('x_col')
            if v_col and i_col and x_col:
                df['_Power'] = df[v_col] * df[i_col]
                plot_data = generate_time_plot(df, x_col, '_Power',
                                              'Time', 'Power (W)', 'Power vs Time', file_info)
            else:
                return jsonify({'success': False, 'error': 'Missing columns for power calculation'}), 400
        
        elif plot_type == 'potential_vs_current':
            plot_data = generate_scatter_plot(df, options.get('x_col'), options.get('y_col'),
                                             'Current (A)', 'Potential (V)', 'V-I Characteristic', file_info)
        
        elif plot_type == 'current_vs_potential':
            plot_data = generate_scatter_plot(df, options.get('x_col'), options.get('y_col'),
                                             'Potential (V)', 'Current (A)', 'Polarization Curve', file_info)
        
        elif plot_type == 'nyquist':
            plot_data = generate_nyquist_plot(df, options.get('x_col'), options.get('y_col'),
                                             file_info, options.get('current_filter'))
        
        elif plot_type == 'bode_magnitude':
            plot_data = generate_bode_magnitude(df, options.get('x_col'), options.get('y_cols'),
                                               file_info, options.get('current_filter'))
        
        elif plot_type == 'bode_phase':
            plot_data = generate_bode_phase(df, options.get('x_col'), options.get('y_cols'),
                                           file_info, options.get('current_filter'))
        
        elif plot_type == 'eis_by_current':
            plot_data = generate_eis_by_current(df, options, file_info)
        
        else:
            return jsonify({'success': False, 'error': f'Unknown plot type: {plot_type}'}), 400
        
        return jsonify({
            'success': True,
            'plot_data': plot_data
        })
        
    except Exception as e:
        print(f"Error generating plot: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


def generate_matplotlib_plot(df, x_col, y_col, x_label, y_label, title, plot_type='line', file_info=None):
    """Generate plot using Matplotlib and return as base64 image"""
    import io
    import base64
    
    if not x_col or not y_col:
        return None
    
    # Clean data
    plot_df = df[[x_col, y_col]].dropna()
    
    # Downsample if too many points
    max_points = 5000
    if len(plot_df) > max_points:
        step = len(plot_df) // max_points
        plot_df = plot_df.iloc[::step]
    
    if len(plot_df) == 0:
        return None
    
    x_data = plot_df[x_col].values
    y_data = plot_df[y_col].values
    
    # Convert time to hours if needed
    if 'time' in x_label.lower() and ('(s)' in x_col.lower() or 'second' in x_col.lower()):
        x_data = x_data / 3600
        x_label = 'Time (hours)'
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot based on type
    if plot_type == 'scatter':
        ax.scatter(x_data, y_data, s=15, alpha=0.7, color='#3b82f6', edgecolors='none')
    else:
        ax.plot(x_data, y_data, linewidth=1.5, color='#3b82f6')
    
    # Styling
    ax.set_xlabel(x_label, fontsize=12, fontweight='bold', color='#1e3a8a')
    ax.set_ylabel(y_label, fontsize=12, fontweight='bold', color='#1e3a8a')
    ax.set_title(title, fontsize=14, fontweight='bold', color='#1e3a8a', pad=15)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_facecolor('#fafafa')
    
    # Add tick styling
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Spine styling
    for spine in ax.spines.values():
        spine.set_color('#e5e7eb')
    
    plt.tight_layout()
    
    # Add watermark if available
    try:
        add_watermark(fig, ax)
    except:
        pass
    
    # Convert to base64
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close(fig)
    
    return image_base64


def generate_matplotlib_nyquist(df, zre_col, zim_col, title='Nyquist Plot', current_filter=None, current_col=None):
    """Generate Nyquist plot using Matplotlib"""
    import io
    import base64
    
    if not zre_col or not zim_col:
        return None
    
    plot_df = df.copy()
    
    # Apply current filter if specified
    if current_filter is not None and current_col:
        rounded_currents = np.round(plot_df[current_col]).astype(int)
        plot_df = plot_df[rounded_currents == current_filter]
    
    # Drop NaN
    plot_df = plot_df[[zre_col, zim_col]].dropna()
    
    if len(plot_df) == 0:
        return None
    
    zre = plot_df[zre_col].values
    zim = -plot_df[zim_col].values  # Negative for Nyquist
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.scatter(zre, zim, s=30, alpha=0.8, color='#3b82f6', edgecolors='#1e40af', linewidths=0.5)
    
    # Styling
    ax.set_xlabel('Z_re (Ω)', fontsize=12, fontweight='bold', color='#1e3a8a')
    ax.set_ylabel('-Z_im (Ω)', fontsize=12, fontweight='bold', color='#1e3a8a')
    ax.set_title(title, fontsize=14, fontweight='bold', color='#1e3a8a', pad=15)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_facecolor('#fafafa')
    ax.set_aspect('equal', adjustable='box')
    
    for spine in ax.spines.values():
        spine.set_color('#e5e7eb')
    
    plt.tight_layout()
    
    try:
        add_watermark(fig, ax)
    except:
        pass
    
    # Convert to base64
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close(fig)
    
    return image_base64


def generate_time_plot(df, x_col, y_col, x_label, y_label, title, file_info):
    """Generate time-series plot data for Plotly"""
    # Drop NaN values and get clean data
    plot_df = df[[x_col, y_col]].dropna()
    
    # Downsample if too many points (for performance)
    max_points = 5000
    if len(plot_df) > max_points:
        step = len(plot_df) // max_points
        plot_df = plot_df.iloc[::step]
    
    x_data = plot_df[x_col].tolist()
    y_data = plot_df[y_col].tolist()
    
    # Convert time to hours if in seconds
    x_converted = x_data
    if x_label == 'Time' and x_col:
        x_col_lower = x_col.lower()
        if 'second' in x_col_lower or '(s)' in x_col_lower:
            x_converted = [x / 3600 if x is not None else None for x in x_data]
            x_label = 'Time (hours)'
    
    filename = file_info.get('filename', 'Data')
    
    return {
        'data': [{
            'x': x_converted,
            'y': y_data,
            'type': 'scatter',
            'mode': 'lines',
            'name': filename,
            'line': {'width': 1.5, 'color': '#3b82f6'}
        }],
        'layout': {
            'title': {'text': title, 'font': {'size': 16, 'color': '#1e3a8a'}},
            'xaxis': {'title': x_label, 'gridcolor': '#e5e7eb'},
            'yaxis': {'title': y_label, 'gridcolor': '#e5e7eb'},
            'plot_bgcolor': 'white',
            'paper_bgcolor': 'white',
            'hovermode': 'closest',
            'showlegend': True
        }
    }


def generate_scatter_plot(df, x_col, y_col, x_label, y_label, title, file_info):
    """Generate scatter plot data for Plotly"""
    # Drop NaN values and get clean data
    plot_df = df[[x_col, y_col]].dropna()
    
    # Downsample if too many points (for performance)
    max_points = 5000
    if len(plot_df) > max_points:
        step = len(plot_df) // max_points
        plot_df = plot_df.iloc[::step]
    
    x_data = plot_df[x_col].tolist()
    y_data = plot_df[y_col].tolist()
    filename = file_info.get('filename', 'Data')
    
    return {
        'data': [{
            'x': x_data,
            'y': y_data,
            'type': 'scatter',
            'mode': 'markers',
            'name': filename,
            'marker': {'size': 4, 'color': '#8b5cf6', 'opacity': 0.7}
        }],
        'layout': {
            'title': {'text': title, 'font': {'size': 16, 'color': '#1e3a8a'}},
            'xaxis': {'title': x_label, 'gridcolor': '#e5e7eb'},
            'yaxis': {'title': y_label, 'gridcolor': '#e5e7eb'},
            'plot_bgcolor': 'white',
            'paper_bgcolor': 'white',
            'hovermode': 'closest'
        }
    }


def generate_nyquist_plot(df, zre_col, zim_col, file_info, current_filter=None):
    """Generate Nyquist plot data for Plotly"""
    if not zre_col or not zim_col:
        return {'data': [], 'layout': {'title': 'No impedance data available'}}
    
    plot_df = df.copy()
    
    # Apply current filter if specified
    if current_filter is not None and 'current_col' in file_info and file_info.get('current_col'):
        current_col = file_info['current_col']
        rounded_currents = np.round(plot_df[current_col]).astype(int)
        plot_df = plot_df[rounded_currents == current_filter]
    
    # Drop NaN values
    plot_df = plot_df[[zre_col, zim_col]].dropna()
    
    if len(plot_df) == 0:
        return {'data': [], 'layout': {'title': 'No valid impedance data'}}
    
    zre = plot_df[zre_col].tolist()
    zim = [-z for z in plot_df[zim_col].tolist()]  # -Zim for Nyquist
    
    filename = file_info.get('filename', 'Data')
    suffix = f' @ {current_filter}A' if current_filter else ''
    
    return {
        'data': [{
            'x': zre,
            'y': zim,
            'type': 'scatter',
            'mode': 'markers',
            'name': filename + suffix,
            'marker': {'size': 6, 'color': '#3b82f6'}
        }],
        'layout': {
            'title': {'text': f'Nyquist Plot{suffix}', 'font': {'size': 16, 'color': '#1e3a8a'}},
            'xaxis': {'title': 'Z_re (Ω)', 'gridcolor': '#e5e7eb', 'scaleanchor': 'y'},
            'yaxis': {'title': '-Z_im (Ω)', 'gridcolor': '#e5e7eb'},
            'plot_bgcolor': 'white',
            'paper_bgcolor': 'white',
            'hovermode': 'closest'
        }
    }


def generate_bode_magnitude(df, freq_col, z_cols, file_info, current_filter=None):
    """Generate Bode magnitude plot data for Plotly"""
    if not freq_col or not z_cols or len(z_cols) < 2:
        return {'data': [], 'layout': {'title': 'No frequency/impedance data available'}}
    
    plot_df = df.copy()
    
    if current_filter is not None and 'current_col' in file_info and file_info.get('current_col'):
        current_col = file_info['current_col']
        rounded_currents = np.round(plot_df[current_col]).astype(int)
        plot_df = plot_df[rounded_currents == current_filter]
    
    # Drop NaN values and filter for positive frequencies (EIS data)
    plot_df = plot_df[[freq_col, z_cols[0], z_cols[1]]].dropna()
    plot_df = plot_df[plot_df[freq_col] > 0]
    
    if len(plot_df) == 0:
        return {'data': [], 'layout': {'title': 'No valid EIS data'}}
    
    freq = plot_df[freq_col].tolist()
    zre = np.array(plot_df[z_cols[0]])
    zim = np.array(plot_df[z_cols[1]])
    magnitude = np.sqrt(zre**2 + zim**2).tolist()
    
    filename = file_info.get('filename', 'Data')
    suffix = f' @ {current_filter}A' if current_filter else ''
    
    return {
        'data': [{
            'x': freq,
            'y': magnitude,
            'type': 'scatter',
            'mode': 'lines+markers',
            'name': filename + suffix,
            'line': {'width': 2, 'color': '#10b981'},
            'marker': {'size': 4}
        }],
        'layout': {
            'title': {'text': f'Bode Magnitude{suffix}', 'font': {'size': 16, 'color': '#1e3a8a'}},
            'xaxis': {'title': 'Frequency (Hz)', 'type': 'log', 'gridcolor': '#e5e7eb'},
            'yaxis': {'title': '|Z| (Ω)', 'type': 'log', 'gridcolor': '#e5e7eb'},
            'plot_bgcolor': 'white',
            'paper_bgcolor': 'white',
            'hovermode': 'closest'
        }
    }


def generate_bode_phase(df, freq_col, z_cols, file_info, current_filter=None):
    """Generate Bode phase plot data for Plotly"""
    if not freq_col or not z_cols or len(z_cols) < 2:
        return {'data': [], 'layout': {'title': 'No frequency/impedance data available'}}
    
    plot_df = df.copy()
    
    if current_filter is not None and 'current_col' in file_info and file_info.get('current_col'):
        current_col = file_info['current_col']
        rounded_currents = np.round(plot_df[current_col]).astype(int)
        plot_df = plot_df[rounded_currents == current_filter]
    
    # Drop NaN values and filter for positive frequencies (EIS data)
    plot_df = plot_df[[freq_col, z_cols[0], z_cols[1]]].dropna()
    plot_df = plot_df[plot_df[freq_col] > 0]
    
    if len(plot_df) == 0:
        return {'data': [], 'layout': {'title': 'No valid EIS data'}}
    
    freq = plot_df[freq_col].tolist()
    zre = np.array(plot_df[z_cols[0]])
    zim = np.array(plot_df[z_cols[1]])
    phase = np.degrees(np.arctan2(zim, zre)).tolist()
    
    filename = file_info.get('filename', 'Data')
    suffix = f' @ {current_filter}A' if current_filter else ''
    
    return {
        'data': [{
            'x': freq,
            'y': phase,
            'type': 'scatter',
            'mode': 'lines+markers',
            'name': filename + suffix,
            'line': {'width': 2, 'color': '#f59e0b'},
            'marker': {'size': 4}
        }],
        'layout': {
            'title': {'text': f'Bode Phase{suffix}', 'font': {'size': 16, 'color': '#1e3a8a'}},
            'xaxis': {'title': 'Frequency (Hz)', 'type': 'log', 'gridcolor': '#e5e7eb'},
            'yaxis': {'title': 'Phase (°)', 'gridcolor': '#e5e7eb'},
            'plot_bgcolor': 'white',
            'paper_bgcolor': 'white',
            'hovermode': 'closest'
        }
    }


def generate_eis_by_current(df, options, file_info):
    """Generate EIS comparison plot for multiple current levels"""
    current_levels = options.get('current_levels', [])
    selected_currents = options.get('selected_currents', current_levels)
    zre_col = options.get('zre_col')
    zim_col = options.get('zim_col')
    freq_col = options.get('freq_col')
    current_col = options.get('current_col')
    
    if not all([zre_col, zim_col, freq_col, current_col]):
        return {'error': 'Missing required columns'}
    
    traces = []
    colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899', '#06b6d4', '#84cc16']
    
    for idx, current in enumerate(selected_currents):
        # Filter for this current level (frequency > 0 for EIS data)
        mask = (df[freq_col] > 0) & (np.round(df[current_col]).astype(int) == current)
        filtered = df[mask]
        
        if len(filtered) == 0:
            continue
        
        zre = filtered[zre_col].tolist()
        zim = [-z for z in filtered[zim_col].tolist()]
        color = colors[idx % len(colors)]
        
        traces.append({
            'x': zre,
            'y': zim,
            'type': 'scatter',
            'mode': 'markers+lines',
            'name': f'{current}A',
            'marker': {'size': 6, 'color': color},
            'line': {'width': 1.5, 'color': color}
        })
    
    return {
        'data': traces,
        'layout': {
            'title': {'text': 'EIS Comparison by Current Level', 'font': {'size': 16, 'color': '#1e3a8a'}},
            'xaxis': {'title': 'Z_re (Ω)', 'gridcolor': '#e5e7eb', 'scaleanchor': 'y'},
            'yaxis': {'title': '-Z_im (Ω)', 'gridcolor': '#e5e7eb'},
            'plot_bgcolor': 'white',
            'paper_bgcolor': 'white',
            'hovermode': 'closest',
            'showlegend': True,
            'legend': {'title': {'text': 'Current'}}
        }
    }


@app.route('/api/raw-data/compare', methods=['POST'])
def compare_raw_data():
    """
    Generate comparison plots for multiple files
    """
    try:
        data = request.get_json()
        files_data = data.get('files', [])
        plot_type = data.get('plot_type')
        options = data.get('options', {})
        
        if not files_data or len(files_data) < 2:
            return jsonify({'success': False, 'error': 'At least 2 files required for comparison'}), 400
        
        colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899', '#06b6d4', '#84cc16',
                 '#6366f1', '#14b8a6', '#f97316', '#a855f7', '#0ea5e9', '#22c55e', '#eab308', '#d946ef']
        
        traces = []
        
        for idx, file_data in enumerate(files_data):
            df = pd.DataFrame(file_data['raw_data'])
            filename = file_data.get('filename', f'File {idx + 1}')
            color = colors[idx % len(colors)]
            detected = file_data.get('detected_columns', {})
            
            if plot_type == 'potential_vs_time':
                x_col = detected.get('time')
                y_col = detected.get('potential')
                if x_col and y_col:
                    x_data = df[x_col].tolist()
                    y_data = df[y_col].tolist()
                    # Convert to hours if in seconds
                    if '(s)' in x_col.lower() or 'second' in x_col.lower():
                        x_data = [x / 3600 if x else None for x in x_data]
                    traces.append({
                        'x': x_data, 'y': y_data,
                        'type': 'scatter', 'mode': 'lines',
                        'name': filename, 'line': {'color': color, 'width': 1.5}
                    })
            
            elif plot_type == 'current_vs_time':
                x_col = detected.get('time')
                y_col = detected.get('current')
                if x_col and y_col:
                    x_data = df[x_col].tolist()
                    y_data = df[y_col].tolist()
                    if '(s)' in x_col.lower() or 'second' in x_col.lower():
                        x_data = [x / 3600 if x else None for x in x_data]
                    traces.append({
                        'x': x_data, 'y': y_data,
                        'type': 'scatter', 'mode': 'lines',
                        'name': filename, 'line': {'color': color, 'width': 1.5}
                    })
            
            elif plot_type == 'power_vs_time':
                x_col = detected.get('time')
                v_col = detected.get('potential')
                i_col = detected.get('current')
                if x_col and v_col and i_col:
                    x_data = df[x_col].tolist()
                    power = (df[v_col] * df[i_col]).tolist()
                    if '(s)' in x_col.lower() or 'second' in x_col.lower():
                        x_data = [x / 3600 if x else None for x in x_data]
                    traces.append({
                        'x': x_data, 'y': power,
                        'type': 'scatter', 'mode': 'lines',
                        'name': filename, 'line': {'color': color, 'width': 1.5}
                    })
            
            elif plot_type == 'polarization':
                v_col = detected.get('potential')
                i_col = detected.get('current')
                if v_col and i_col:
                    traces.append({
                        'x': df[v_col].tolist(), 'y': df[i_col].tolist(),
                        'type': 'scatter', 'mode': 'markers',
                        'name': filename, 'marker': {'color': color, 'size': 4, 'opacity': 0.7}
                    })
            
            elif plot_type == 'nyquist':
                zre_col = detected.get('zre')
                zim_col = detected.get('zim')
                freq_col = detected.get('frequency')
                if zre_col and zim_col and freq_col:
                    # Filter for EIS data (freq > 0)
                    mask = df[freq_col] > 0
                    zre = df.loc[mask, zre_col].tolist()
                    zim = [-z for z in df.loc[mask, zim_col].tolist()]
                    traces.append({
                        'x': zre, 'y': zim,
                        'type': 'scatter', 'mode': 'markers',
                        'name': filename, 'marker': {'color': color, 'size': 5}
                    })
        
        # Build layout based on plot type
        layouts = {
            'potential_vs_time': {
                'title': 'Potential vs Time - Comparison',
                'xaxis': {'title': 'Time (hours)'},
                'yaxis': {'title': 'Potential (V)'}
            },
            'current_vs_time': {
                'title': 'Current vs Time - Comparison',
                'xaxis': {'title': 'Time (hours)'},
                'yaxis': {'title': 'Current (A)'}
            },
            'power_vs_time': {
                'title': 'Power vs Time - Comparison',
                'xaxis': {'title': 'Time (hours)'},
                'yaxis': {'title': 'Power (W)'}
            },
            'polarization': {
                'title': 'Polarization Curve - Comparison',
                'xaxis': {'title': 'Potential (V)'},
                'yaxis': {'title': 'Current (A)'}
            },
            'nyquist': {
                'title': 'Nyquist Plot - Comparison',
                'xaxis': {'title': 'Z_re (Ω)', 'scaleanchor': 'y'},
                'yaxis': {'title': '-Z_im (Ω)'}
            }
        }
        
        layout_config = layouts.get(plot_type, {'title': 'Comparison Plot'})
        
        plot_data = {
            'data': traces,
            'layout': {
                'title': {'text': layout_config['title'], 'font': {'size': 16, 'color': '#1e3a8a'}},
                'xaxis': {**layout_config.get('xaxis', {}), 'gridcolor': '#e5e7eb'},
                'yaxis': {**layout_config.get('yaxis', {}), 'gridcolor': '#e5e7eb'},
                'plot_bgcolor': 'white',
                'paper_bgcolor': 'white',
                'hovermode': 'closest',
                'showlegend': True,
                'legend': {'orientation': 'h', 'y': -0.15}
            }
        }
        
        return jsonify({'success': True, 'plot_data': plot_data})
        
    except Exception as e:
        print(f"Error in comparison: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/raw-data/compare-matplotlib', methods=['POST'])
def compare_raw_data_matplotlib():
    """
    Generate comparison plots for multiple files using Matplotlib
    Same approach as DRT Analysis for consistency
    """
    import io
    import base64
    
    try:
        data = request.get_json()
        files_data = data.get('files', [])
        plot_type = data.get('plot_type')
        options = data.get('options', {})
        
        if not files_data or len(files_data) < 2:
            return jsonify({'success': False, 'error': 'At least 2 files required for comparison'}), 400
        
        colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899', '#06b6d4', '#84cc16',
                 '#6366f1', '#14b8a6', '#f97316', '#a855f7', '#0ea5e9', '#22c55e', '#eab308', '#d946ef']
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        title = 'Comparison Plot'
        x_label = 'X'
        y_label = 'Y'
        
        for idx, file_data in enumerate(files_data):
            df = pd.DataFrame(file_data['raw_data'])
            filename = file_data.get('filename', f'File {idx + 1}')
            # Shorten filename for legend
            short_name = filename[:20] + '...' if len(filename) > 20 else filename
            color = colors[idx % len(colors)]
            detected = file_data.get('detected_columns', {})
            
            if plot_type == 'potential_vs_time':
                x_col = detected.get('time')
                y_col = detected.get('potential')
                title = 'Potential vs Time - Comparison'
                x_label = 'Time (hours)'
                y_label = 'Potential (V)'
                
                if x_col and y_col:
                    plot_df = df[[x_col, y_col]].dropna()
                    # Downsample
                    if len(plot_df) > 3000:
                        step = len(plot_df) // 3000
                        plot_df = plot_df.iloc[::step]
                    
                    x_data = plot_df[x_col].values
                    y_data = plot_df[y_col].values
                    
                    # Convert to hours if in seconds
                    if '(s)' in x_col.lower() or 'second' in x_col.lower():
                        x_data = x_data / 3600
                    
                    ax.plot(x_data, y_data, linewidth=1.2, color=color, label=short_name, alpha=0.8)
            
            elif plot_type == 'current_vs_time':
                x_col = detected.get('time')
                y_col = detected.get('current')
                title = 'Current vs Time - Comparison'
                x_label = 'Time (hours)'
                y_label = 'Current (A)'
                
                if x_col and y_col:
                    plot_df = df[[x_col, y_col]].dropna()
                    if len(plot_df) > 3000:
                        step = len(plot_df) // 3000
                        plot_df = plot_df.iloc[::step]
                    
                    x_data = plot_df[x_col].values
                    y_data = plot_df[y_col].values
                    
                    if '(s)' in x_col.lower() or 'second' in x_col.lower():
                        x_data = x_data / 3600
                    
                    ax.plot(x_data, y_data, linewidth=1.2, color=color, label=short_name, alpha=0.8)
            
            elif plot_type == 'potential_vs_current' or plot_type == 'polarization':
                v_col = detected.get('potential')
                i_col = detected.get('current')
                title = 'Polarization Curve - Comparison'
                x_label = 'Current (A)'
                y_label = 'Potential (V)'
                
                if v_col and i_col:
                    plot_df = df[[i_col, v_col]].dropna()
                    if len(plot_df) > 3000:
                        step = len(plot_df) // 3000
                        plot_df = plot_df.iloc[::step]
                    
                    ax.scatter(plot_df[i_col].values, plot_df[v_col].values, 
                              s=10, color=color, label=short_name, alpha=0.6, edgecolors='none')
            
            elif plot_type == 'current_vs_potential':
                v_col = detected.get('potential')
                i_col = detected.get('current')
                title = 'Current vs Potential - Comparison'
                x_label = 'Potential (V)'
                y_label = 'Current (A)'
                
                if v_col and i_col:
                    plot_df = df[[v_col, i_col]].dropna()
                    if len(plot_df) > 3000:
                        step = len(plot_df) // 3000
                        plot_df = plot_df.iloc[::step]
                    
                    ax.scatter(plot_df[v_col].values, plot_df[i_col].values, 
                              s=10, color=color, label=short_name, alpha=0.6, edgecolors='none')
            
            elif plot_type == 'nyquist':
                zre_col = detected.get('zre')
                zim_col = detected.get('zim')
                freq_col = detected.get('frequency')
                title = 'Nyquist Plot - Comparison'
                x_label = 'Z_re (Ω)'
                y_label = '-Z_im (Ω)'
                
                if zre_col and zim_col:
                    plot_df = df.copy()
                    if freq_col:
                        plot_df = plot_df[plot_df[freq_col] > 0]
                    plot_df = plot_df[[zre_col, zim_col]].dropna()
                    
                    zre = plot_df[zre_col].values
                    zim = -plot_df[zim_col].values
                    
                    ax.scatter(zre, zim, s=20, color=color, label=short_name, alpha=0.7, edgecolors='none')
        
        # Styling
        ax.set_xlabel(x_label, fontsize=12, fontweight='bold', color='#1e3a8a')
        ax.set_ylabel(y_label, fontsize=12, fontweight='bold', color='#1e3a8a')
        ax.set_title(title, fontsize=14, fontweight='bold', color='#1e3a8a', pad=15)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_facecolor('#fafafa')
        ax.legend(loc='best', fontsize=9, framealpha=0.9)
        
        for spine in ax.spines.values():
            spine.set_color('#e5e7eb')
        
        # Equal aspect for Nyquist
        if plot_type == 'nyquist':
            ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        
        try:
            add_watermark(fig, ax)
        except:
            pass
        
        # Convert to base64
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close(fig)
        
        return jsonify({
            'success': True,
            'image': image_base64,
            'title': title
        })
        
    except Exception as e:
        print(f"Error in matplotlib comparison: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/raw-data/export', methods=['POST'])
def export_raw_data_analysis():
    """
    Export analysis results as CSV
    """
    try:
        data = request.get_json()
        export_type = data.get('export_type', 'statistics')
        raw_data = data.get('raw_data')
        filename_base = data.get('filename', 'analysis')
        
        if not raw_data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        df = pd.DataFrame(raw_data)
        output = BytesIO()
        
        if export_type == 'statistics':
            # Export column statistics
            stats = calculate_column_statistics(df)
            stats_df = pd.DataFrame(stats).T
            stats_df.to_csv(output)
        
        elif export_type == 'filtered_data':
            # Export filtered data
            filters = data.get('filters', {})
            if 'time_range' in filters:
                time_col = filters.get('time_col')
                if time_col:
                    t_min, t_max = filters['time_range']
                    df = df[(df[time_col] >= t_min) & (df[time_col] <= t_max)]
            if 'current_filter' in filters:
                current_col = filters.get('current_col')
                if current_col:
                    df = df[np.round(df[current_col]).astype(int) == filters['current_filter']]
            df.to_csv(output, index=False)
        
        else:
            df.to_csv(output, index=False)
        
        output.seek(0)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        return send_file(
            output,
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'{filename_base}_{export_type}_{timestamp}.csv'
        )
        
    except Exception as e:
        print(f"Error exporting data: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/raw-data/generate-matplotlib-plot', methods=['POST'])
def generate_matplotlib_plot_api():
    """
    Generate plot using Matplotlib and return as base64 image
    Same approach as DRT Analysis for consistency
    """
    try:
        data = request.get_json()
        plot_type = data.get('plot_type')
        raw_data = data.get('raw_data')
        options = data.get('options', {})
        file_info = data.get('file_info', {})
        
        if not raw_data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        df = pd.DataFrame(raw_data)
        
        x_col = options.get('x_col')
        y_col = options.get('y_col')
        
        image_base64 = None
        title = ''
        
        if plot_type == 'potential_vs_time':
            title = 'Potential vs Time'
            image_base64 = generate_matplotlib_plot(
                df, x_col, y_col, 
                'Time', 'Potential (V)', title, 
                plot_type='line', file_info=file_info
            )
        
        elif plot_type == 'current_vs_time':
            title = 'Current vs Time'
            image_base64 = generate_matplotlib_plot(
                df, x_col, y_col,
                'Time', 'Current (A)', title,
                plot_type='line', file_info=file_info
            )
        
        elif plot_type == 'potential_vs_current':
            title = 'Potential vs Current'
            image_base64 = generate_matplotlib_plot(
                df, x_col, y_col,
                'Current (A)', 'Potential (V)', title,
                plot_type='scatter', file_info=file_info
            )
        
        elif plot_type == 'current_vs_potential':
            title = 'Current vs Potential (Polarization)'
            image_base64 = generate_matplotlib_plot(
                df, x_col, y_col,
                'Potential (V)', 'Current (A)', title,
                plot_type='scatter', file_info=file_info
            )
        
        elif plot_type == 'nyquist':
            zre_col = options.get('zre_col')
            zim_col = options.get('zim_col')
            current_filter = options.get('current_filter')
            current_col = options.get('current_col')
            
            suffix = f' @ {current_filter}A' if current_filter else ''
            title = f'Nyquist Plot{suffix}'
            
            image_base64 = generate_matplotlib_nyquist(
                df, zre_col, zim_col, title,
                current_filter=current_filter, current_col=current_col
            )
        
        elif plot_type == 'bode_magnitude':
            freq_col = options.get('freq_col')
            zre_col = options.get('zre_col')
            zim_col = options.get('zim_col')
            current_filter = options.get('current_filter')
            current_col = options.get('current_col')
            
            if freq_col and zre_col and zim_col:
                plot_df = df.copy()
                
                if current_filter is not None and current_col:
                    rounded_currents = np.round(plot_df[current_col]).astype(int)
                    plot_df = plot_df[rounded_currents == current_filter]
                
                # Filter for EIS data
                plot_df = plot_df[plot_df[freq_col] > 0]
                plot_df = plot_df[[freq_col, zre_col, zim_col]].dropna()
                
                if len(plot_df) > 0:
                    freq = plot_df[freq_col].values
                    zre = plot_df[zre_col].values
                    zim = plot_df[zim_col].values
                    magnitude = np.sqrt(zre**2 + zim**2)
                    
                    suffix = f' @ {current_filter}A' if current_filter else ''
                    title = f'Bode Magnitude{suffix}'
                    
                    import io
                    import base64
                    
                    fig, ax = plt.subplots(figsize=(12, 7))
                    ax.loglog(freq, magnitude, 'o-', linewidth=1.5, markersize=4, color='#10b981')
                    ax.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold', color='#1e3a8a')
                    ax.set_ylabel('|Z| (Ω)', fontsize=12, fontweight='bold', color='#1e3a8a')
                    ax.set_title(title, fontsize=14, fontweight='bold', color='#1e3a8a', pad=15)
                    ax.grid(True, alpha=0.3, which='both')
                    ax.set_facecolor('#fafafa')
                    
                    try:
                        add_watermark(fig, ax)
                    except:
                        pass
                    
                    plt.tight_layout()
                    buffer = io.BytesIO()
                    fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
                    buffer.seek(0)
                    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    plt.close(fig)
        
        elif plot_type == 'bode_phase':
            freq_col = options.get('freq_col')
            zre_col = options.get('zre_col')
            zim_col = options.get('zim_col')
            current_filter = options.get('current_filter')
            current_col = options.get('current_col')
            
            if freq_col and zre_col and zim_col:
                plot_df = df.copy()
                
                if current_filter is not None and current_col:
                    rounded_currents = np.round(plot_df[current_col]).astype(int)
                    plot_df = plot_df[rounded_currents == current_filter]
                
                plot_df = plot_df[plot_df[freq_col] > 0]
                plot_df = plot_df[[freq_col, zre_col, zim_col]].dropna()
                
                if len(plot_df) > 0:
                    freq = plot_df[freq_col].values
                    zre = plot_df[zre_col].values
                    zim = plot_df[zim_col].values
                    phase = np.degrees(np.arctan2(zim, zre))
                    
                    suffix = f' @ {current_filter}A' if current_filter else ''
                    title = f'Bode Phase{suffix}'
                    
                    import io
                    import base64
                    
                    fig, ax = plt.subplots(figsize=(12, 7))
                    ax.semilogx(freq, phase, 'o-', linewidth=1.5, markersize=4, color='#f59e0b')
                    ax.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold', color='#1e3a8a')
                    ax.set_ylabel('Phase (°)', fontsize=12, fontweight='bold', color='#1e3a8a')
                    ax.set_title(title, fontsize=14, fontweight='bold', color='#1e3a8a', pad=15)
                    ax.grid(True, alpha=0.3, which='both')
                    ax.set_facecolor('#fafafa')
                    
                    try:
                        add_watermark(fig, ax)
                    except:
                        pass
                    
                    plt.tight_layout()
                    buffer = io.BytesIO()
                    fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
                    buffer.seek(0)
                    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    plt.close(fig)
        
        if image_base64:
            return jsonify({
                'success': True,
                'image': image_base64,
                'title': title
            })
        else:
            return jsonify({'success': False, 'error': 'Could not generate plot - missing or invalid data'}), 400
    
    except Exception as e:
        print(f"Error generating matplotlib plot: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    # Print configuration summary
    print(get_config_summary())
    
    print("=" * 80)
    print("  ELERYC EXPERIMENT VIEWER - Backend Server")
    print("=" * 80)
    print(f"\n  Server running at: http://localhost:{PORT}")
    print(f"  API endpoint: http://localhost:{PORT}/api/generate-comparison")
    print("\n  Press Ctrl+C to stop the server\n")
    print("=" * 80)
    
    app.run(debug=DEBUG, port=PORT, host=HOST)

