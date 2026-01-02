"""
Configuration settings for Eleryc DRT Analysis App
Supports both local development and cloud deployment
"""
import os
from pathlib import Path

# Base directory (where this config file lives)
BASE_DIR = Path(__file__).parent.resolve()

# =============================================================================
#                         ENVIRONMENT DETECTION
# =============================================================================
# Detect if running in production (cloud) or development (local)
IS_PRODUCTION = os.environ.get('RENDER', False) or os.environ.get('PRODUCTION', False)

# =============================================================================
#                         SERVER CONFIGURATION
# =============================================================================
PORT = int(os.environ.get('PORT', 8080))
DEBUG = os.environ.get('FLASK_DEBUG', 'true').lower() == 'true' if not IS_PRODUCTION else False
HOST = '0.0.0.0'

# =============================================================================
#                         DATA PATHS
# =============================================================================
# These can be overridden with environment variables for different deployments
DATA_DIR = Path(os.environ.get('DATA_DIR', BASE_DIR / 'data'))
M_SERIES_DIR = DATA_DIR / 'M-Series' / 'separated_by_current'
DURATION_TESTS_DIR = DATA_DIR / 'Duration-Tests' / 'separated_by_current'
OUTPUTS_DIR = Path(os.environ.get('OUTPUTS_DIR', BASE_DIR / 'outputs'))

# =============================================================================
#                         STATIC FILES
# =============================================================================
STATIC_DIR = Path(os.environ.get('STATIC_DIR', BASE_DIR / 'static'))
ASSETS_DIR = Path(os.environ.get('ASSETS_DIR', BASE_DIR / 'assets'))

# =============================================================================
#                         WATERMARK / BRANDING
# =============================================================================
WATERMARK_CONFIG = {
    'text': os.environ.get('WATERMARK_TEXT', 'Confidential - DRT Analysis Tool'),
    'company': os.environ.get('COMPANY_NAME', 'Eleryc Inc.'),
    'font_size': 10,
    'alpha': 0.6,
    'color': '#888888',
    'position': 'bottom_right',
    'logo_path': str(STATIC_DIR / 'eleryc_logo.png'),
    'logo_size': 0.08,
    'logo_alpha': 0.7
}

# =============================================================================
#                         PLOT SETTINGS
# =============================================================================
COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']
PLOT_DPI = int(os.environ.get('PLOT_DPI', 300))
PLOT_FORMAT = os.environ.get('PLOT_FORMAT', 'png')

# =============================================================================
#                         TEMP DIRECTORY
# =============================================================================
import tempfile
TEMP_DIR = Path(os.environ.get('TEMP_DIR', tempfile.gettempdir())) / 'eleryc_plots'
TEMP_DIR.mkdir(exist_ok=True)

# =============================================================================
#                         SERIES DESCRIPTIONS
# =============================================================================
SERIES_DESCRIPTIONS = {
    'M-Series': 'Distribution of Relaxation Times (DRT) & Nyquist Plots',
    'Duration-Tests': 'Long-term Performance & Stability Analysis',
    'CV-Series': 'Cyclic Voltammetry Data',
    'Polarization-Series': 'Polarization Curve Analysis'
}

# =============================================================================
#                         EXPERIMENT FOLDER MAPPING
# =============================================================================
# Actual folders that exist for Duration Tests
EXISTING_DURATION_FOLDERS = ['M75', 'M77', 'M78', 'M81', 'M84', 'M85', 'M86', 'M87', 'M88', 'M89', 'M90', 'M91', 'M92', 'M93']

# =============================================================================
#                         AUTHENTICATION SETTINGS
# =============================================================================
# Secret key for session encryption (CHANGE THIS IN PRODUCTION!)
SECRET_KEY = os.environ.get('SECRET_KEY', 'eleryc-drt-secret-key-change-me-in-production')

# Session settings
SESSION_LIFETIME_HOURS = int(os.environ.get('SESSION_LIFETIME_HOURS', 24))

# User credentials (can be overridden with environment variables)
# Format: comma-separated "username:password" pairs
# Example: "admin:password123,analyst:secure456"
USERS_ENV = os.environ.get('APP_USERS', 'admin:eleryc2025')

def parse_users(users_string):
    """Parse users from environment variable or default"""
    users = {}
    for user_pass in users_string.split(','):
        if ':' in user_pass:
            username, password = user_pass.strip().split(':', 1)
            users[username.strip()] = password.strip()
    return users

# Dictionary of valid users {username: password}
VALID_USERS = parse_users(USERS_ENV)

# Login settings
LOGIN_DISABLED = os.environ.get('LOGIN_DISABLED', 'false').lower() == 'true'
REMEMBER_ME_DAYS = int(os.environ.get('REMEMBER_ME_DAYS', 30))


def get_config_summary():
    """Print configuration summary for debugging"""
    return f"""
    ============================================
    ELERYC DRT ANALYSIS - CONFIGURATION
    ============================================
    Environment: {'PRODUCTION' if IS_PRODUCTION else 'DEVELOPMENT'}
    Base Dir: {BASE_DIR}
    Data Dir: {DATA_DIR}
    Port: {PORT}
    Debug: {DEBUG}
    ============================================
    """

