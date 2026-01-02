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
    Expected format: Frequency(Hz), Zre(ohms), Zim(ohms)
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
            
            if freq_col and zre_col and zim_col:
                col_mapping = {
                    'freq': freq_col,
                    'zre': zre_col,
                    'zim': zim_col
                }
                headers_found = True
            else:
                headers_found = False
                missing = []
                if not freq_col: missing.append('Frequency')
                if not zre_col: missing.append('Zre (Real impedance)')
                if not zim_col: missing.append('Zim (Imaginary impedance)')
            
            if headers_found:
                # Extract data using detected column names
                freq = df[col_mapping['freq']].values
                zre = df[col_mapping['zre']].values
                zim = df[col_mapping['zim']].values
                data_source = f"detected columns: {col_mapping['freq']}, {col_mapping['zre']}, {col_mapping['zim']}"
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
                
        except Exception as e:
            return jsonify({
                'success': False, 
                'error': f'Error reading CSV file: {str(e)}'
            }), 400
        
        # Validate data
        if len(freq) < 5:
            return jsonify({'success': False, 'error': 'Insufficient data points (minimum 5 required)'}), 400
        return jsonify({
            'success': True,
            'message': f'File uploaded successfully ({data_source})',
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
        print(f"Error uploading DRT data: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

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
#                    DYNAMIC PLOT GENERATION FOR EXPERIMENTS
# =============================================================================

@app.route('/api/experiment/plot', methods=['POST'])
def generate_experiment_plot():
    """
    Generate DRT or Nyquist plot dynamically from experiment CSV data
    Handles both M-Series (has drt/, eis/ subfolders) and Duration-Tests (flat structure)
    """
    try:
        data = request.get_json()
        experiment = data.get('experiment')
        plot_type = data.get('plot_type', 'drt')  # 'drt' or 'nyquist'
        series = data.get('series', 'M-Series')
        
        if not experiment:
            return jsonify({'success': False, 'error': 'No experiment specified'}), 400
        
        # Determine base directory
        if series == 'M-Series':
            base_dir = M_SERIES_DIR
        else:
            base_dir = DURATION_TESTS_DIR
        
        exp_dir = base_dir / experiment
        
        if not exp_dir.exists():
            return jsonify({'success': False, 'error': f'Experiment directory not found: {experiment}'}), 404
        
        plt.figure(figsize=(12, 8))
        plt.style.use('default')
        
        # Define condition colors to match the example
        CONDITION_COLORS = {
            '4A': '#e41a1c',   # Red
            '8A': '#377eb8',   # Blue  
            '12A': '#4daf4a',  # Green
            '16A': '#984ea3',  # Purple
            '20A': '#ff7f00',  # Orange
            '24A': '#ffff33',  # Yellow
        }
        
        # Extract short experiment name (e.g., "M1" from "M1_M1-5psi-07162025")
        exp_short = experiment.split('_')[0]
        
        if series == 'M-Series':
            # M-Series has drt/ and eis/ subfolders
            if plot_type == 'drt':
                drt_dir = exp_dir / 'drt'
                if not drt_dir.exists():
                    return jsonify({'success': False, 'error': 'DRT data folder not found'}), 404
                
                csv_files = list(drt_dir.glob('*.csv'))
                if not csv_files:
                    return jsonify({'success': False, 'error': 'No DRT CSV files found'}), 404
                
                # Sort files by condition (4A, 8A, 12A, 16A)
                def get_condition(f):
                    cond = f.stem.split('_')[-1]
                    try:
                        return int(cond.replace('A', ''))
                    except:
                        return 999
                csv_files = sorted(csv_files, key=get_condition)
                
                for idx, csv_file in enumerate(csv_files):
                    try:
                        df = pd.read_csv(csv_file, skiprows=2)
                        tau_col = None
                        gamma_col = None
                        for col in df.columns:
                            col_lower = col.lower()
                            if 'tau' in col_lower:
                                tau_col = col
                            if 'gamma' in col_lower:
                                gamma_col = col
                        
                        if not tau_col or not gamma_col:
                            tau_col = df.columns[0]
                            gamma_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
                        
                        condition = csv_file.stem.split('_')[-1]
                        color = CONDITION_COLORS.get(condition, COLORS[idx % len(COLORS)])
                        plt.plot(df[tau_col], df[gamma_col], 'o-', 
                                linewidth=2, markersize=5,
                                color=color, label=condition)
                    except Exception as e:
                        print(f"Error processing DRT file {csv_file}: {e}")
                        continue
                
                plt.xlabel(r'$\tau$ (s)', fontsize=12)
                plt.ylabel(r'$\gamma$ (Ω)', fontsize=12)
                plt.title(f'{exp_short} - DRT Plot (All Currents)', fontsize=14, fontweight='bold')
                plt.xscale('log')
                plt.grid(True, alpha=0.3, which='both')
                plt.legend(fontsize=10, loc='upper right', frameon=True)
                
            elif plot_type == 'nyquist':
                eis_dir = exp_dir / 'eis'
                if not eis_dir.exists():
                    return jsonify({'success': False, 'error': 'EIS data folder not found'}), 404
                
                csv_files = list(eis_dir.glob('*.csv'))
                if not csv_files:
                    return jsonify({'success': False, 'error': 'No EIS CSV files found'}), 404
                
                # Sort files by condition (4A, 8A, 12A, 16A)
                def get_condition(f):
                    cond = f.stem.split('_')[-1]
                    try:
                        return int(cond.replace('A', ''))
                    except:
                        return 999
                csv_files = sorted(csv_files, key=get_condition)
                
                for idx, csv_file in enumerate(csv_files):
                    try:
                        df = pd.read_csv(csv_file)
                        z_real_col = None
                        z_imag_col = None
                        for col in df.columns:
                            col_lower = col.lower()
                            if 'mu_z_re' in col_lower or 'zre' in col_lower or 'z_re' in col_lower:
                                z_real_col = col
                            if 'mu_z_im' in col_lower or 'zim' in col_lower or 'z_im' in col_lower:
                                z_imag_col = col
                        
                        if not z_real_col or not z_imag_col:
                            continue
                        
                        condition = csv_file.stem.split('_')[-1]
                        color = CONDITION_COLORS.get(condition, COLORS[idx % len(COLORS)])
                        plt.plot(df[z_real_col], -df[z_imag_col], 'o-', 
                                linewidth=2, markersize=5,
                                color=color, label=condition)
                    except Exception as e:
                        print(f"Error processing EIS file {csv_file}: {e}")
                        continue
                
                plt.xlabel(r'$Z_{re}$ (Ω)', fontsize=12)
                plt.ylabel(r'$-Z_{im}$ (Ω)', fontsize=12)
                plt.title(f'{exp_short} - Nyquist Plot (All Currents)', fontsize=14, fontweight='bold')
                plt.grid(True, alpha=0.3)
                plt.axis('equal')
                plt.legend(fontsize=10, loc='upper right', frameon=True)
        
        else:
            # Duration-Tests have flat structure with EIS files containing 'EIS' in name
            csv_files = list(exp_dir.glob('*.csv'))
            if not csv_files:
                return jsonify({'success': False, 'error': 'No CSV files found'}), 404
            
            if plot_type == 'nyquist' or plot_type == 'eis':
                # Find EIS files
                eis_files = [f for f in csv_files if 'EIS' in f.name.upper()]
                if not eis_files:
                    return jsonify({'success': False, 'error': 'No EIS files found for Duration Test'}), 404
                
                for idx, csv_file in enumerate(sorted(eis_files)):
                    try:
                        df = pd.read_csv(csv_file)
                        z_real_col = None
                        z_imag_col = None
                        freq_col = None
                        
                        for col in df.columns:
                            col_lower = col.lower()
                            if 'zre' in col_lower or 'z_re' in col_lower or 'real' in col_lower:
                                z_real_col = col
                            if 'zim' in col_lower or 'z_im' in col_lower or 'imag' in col_lower:
                                z_imag_col = col
                            if 'freq' in col_lower:
                                freq_col = col
                        
                        if not z_real_col or not z_imag_col:
                            continue
                        
                        # Extract time/condition from filename
                        label = csv_file.stem.split('_')[-1]
                        color = COLORS[idx % len(COLORS)]
                        plt.plot(df[z_real_col], -df[z_imag_col], 'o-', 
                                linewidth=2.5, markersize=4,
                                color=color, label=label)
                    except Exception as e:
                        print(f"Error processing Duration EIS file {csv_file}: {e}")
                        continue
                
                plt.xlabel('Z_re (Ω)', fontsize=13)
                plt.ylabel('-Z_im (Ω)', fontsize=13)
                plt.title(f'Nyquist (EIS) - {experiment}', fontsize=15, pad=15)
                plt.grid(True, alpha=0.4)
                plt.axis('equal')
                plt.legend(fontsize=11, loc='upper right')
            
            elif plot_type == 'drt':
                # For Duration Tests, try to plot V-I or time series data
                main_files = [f for f in csv_files if 'EIS' not in f.name.upper() and 'vm' not in f.name.lower()]
                if not main_files:
                    main_files = csv_files[:1]  # Use first file as fallback
                
                for idx, csv_file in enumerate(main_files[:3]):  # Limit to 3 files
                    try:
                        df = pd.read_csv(csv_file)
                        
                        # Try to find voltage/current columns for Duration Tests
                        time_col = None
                        voltage_col = None
                        current_col = None
                        
                        for col in df.columns:
                            col_lower = col.lower()
                            if 'time' in col_lower or 'elapsed' in col_lower:
                                time_col = col
                            if 'potential' in col_lower or 'voltage' in col_lower or 'v' == col_lower:
                                voltage_col = col
                            if 'current' in col_lower or 'i' == col_lower:
                                current_col = col
                        
                        if time_col and voltage_col:
                            # Convert time to hours
                            time_hours = df[time_col] / 3600
                            label = csv_file.stem.split('_')[-1] if '_' in csv_file.stem else csv_file.stem
                            color = COLORS[idx % len(COLORS)]
                            plt.plot(time_hours, df[voltage_col], '-', 
                                    linewidth=2, color=color, label=label, alpha=0.8)
                    except Exception as e:
                        print(f"Error processing Duration file {csv_file}: {e}")
                        continue
                
                plt.xlabel('Time (hours)', fontsize=13)
                plt.ylabel('Potential (V)', fontsize=13)
                plt.title(f'Performance vs Time - {experiment}', fontsize=15, pad=15)
                plt.grid(True, alpha=0.4)
                plt.legend(fontsize=11, loc='upper right')
            
            else:
                return jsonify({'success': False, 'error': f'Unknown plot type: {plot_type}'}), 400
        
        plt.tight_layout()
        add_watermark_to_plot(custom_text=f"{plot_type.upper()} Analysis - {experiment}")
        
        # Convert to base64
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=PLOT_DPI, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return jsonify({
            'success': True,
            'plot': img_base64,
            'plot_type': plot_type,
            'experiment': experiment
        })
        
    except Exception as e:
        print(f"Error generating experiment plot: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/experiment/conditions', methods=['GET'])
def get_experiment_conditions():
    """
    Get available conditions for an experiment
    """
    try:
        experiment = request.args.get('experiment')
        series = request.args.get('series', 'M-Series')
        
        if not experiment:
            return jsonify({'success': False, 'error': 'No experiment specified'}), 400
        
        if series == 'M-Series':
            base_dir = M_SERIES_DIR
        else:
            base_dir = DURATION_TESTS_DIR
        
        exp_dir = base_dir / experiment
        conditions = []
        
        # Check DRT folder for conditions
        drt_dir = exp_dir / 'drt'
        if drt_dir.exists():
            for csv_file in drt_dir.glob('*.csv'):
                condition = csv_file.stem.split('_')[-1]
                if condition not in conditions:
                    conditions.append(condition)
        
        # Check EIS folder for conditions
        eis_dir = exp_dir / 'eis'
        if eis_dir.exists():
            for csv_file in eis_dir.glob('*.csv'):
                condition = csv_file.stem.split('_')[-1]
                if condition not in conditions:
                    conditions.append(condition)
        
        return jsonify({
            'success': True,
            'experiment': experiment,
            'conditions': sorted(conditions)
        })
        
    except Exception as e:
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

