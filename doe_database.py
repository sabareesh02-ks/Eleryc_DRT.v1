"""
DOE Planner Database Models and Utilities
Electrochemical Testing Workflow Database
Supports both SQLite (local) and PostgreSQL (production on Render)
"""

import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
import json
import time
import shutil
import glob

# Database configuration
DATABASE_URL = os.environ.get('DATABASE_URL')
IS_POSTGRES = DATABASE_URL is not None

# For SQLite (local development)
DB_PATH = Path(__file__).parent / 'doe_planner.db'
BACKUP_DIR = Path(__file__).parent / 'backups'

# PostgreSQL support
if IS_POSTGRES:
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
        # Fix Render's postgres:// vs postgresql://
        if DATABASE_URL.startswith('postgres://'):
            DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)
        print(f"Using PostgreSQL database")
    except ImportError:
        print("psycopg2 not installed, falling back to SQLite")
        IS_POSTGRES = False


# ============= DATABASE CONNECTION =============

def get_db_connection():
    """Get database connection - PostgreSQL or SQLite"""
    if IS_POSTGRES:
        conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
        conn.autocommit = True
        return conn
    else:
        conn = sqlite3.connect(
            str(DB_PATH),
            timeout=30.0,
            isolation_level=None,
            check_same_thread=False
        )
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('PRAGMA busy_timeout=30000')
        conn.execute('PRAGMA synchronous=NORMAL')
        return conn


def execute_query(conn, query, params=None):
    """Execute query with proper parameter placeholder based on database type"""
    if IS_POSTGRES:
        # Convert ? to %s for PostgreSQL
        query = query.replace('?', '%s')
    cursor = conn.cursor()
    if params:
        cursor.execute(query, params)
    else:
        cursor.execute(query)
    return cursor


def fetchone(cursor):
    """Fetch one row and convert to dict"""
    row = cursor.fetchone()
    if row is None:
        return None
    if IS_POSTGRES:
        return dict(row)
    else:
        return dict(row)


def fetchall(cursor):
    """Fetch all rows and convert to list of dicts"""
    rows = cursor.fetchall()
    if IS_POSTGRES:
        return [dict(row) for row in rows]
    else:
        return [dict(row) for row in rows]


# ============= BACKUP FUNCTIONS (SQLite only) =============

def create_backup(reason='manual'):
    """Create timestamped backup of database (SQLite only)"""
    if IS_POSTGRES:
        return {'success': True, 'message': 'Backups managed by Render for PostgreSQL'}
    
    try:
        BACKUP_DIR.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        backup_filename = f'doe_planner_{timestamp}_{reason}.db'
        backup_path = BACKUP_DIR / backup_filename
        
        source = sqlite3.connect(DB_PATH)
        dest = sqlite3.connect(backup_path)
        source.backup(dest)
        source.close()
        dest.close()
        
        cleanup_old_backups(keep=7)
        file_size = os.path.getsize(backup_path)
        
        return {
            'success': True,
            'filename': backup_filename,
            'path': str(backup_path),
            'size': file_size,
            'timestamp': timestamp,
            'reason': reason
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


def cleanup_old_backups(keep=7):
    """Delete old backups, keeping only the most recent 'keep' number"""
    if IS_POSTGRES:
        return
    try:
        backups = sorted(BACKUP_DIR.glob('doe_planner_*.db'), key=os.path.getmtime, reverse=True)
        for old_backup in backups[keep:]:
            os.remove(old_backup)
    except Exception as e:
        print(f"Cleanup error: {e}")


def get_all_backups():
    """Get list of all available backups"""
    if IS_POSTGRES:
        return []
    try:
        if not BACKUP_DIR.exists():
            return []
        
        backups = []
        for backup_file in sorted(BACKUP_DIR.glob('doe_planner_*.db'), key=os.path.getmtime, reverse=True):
            stat = os.stat(backup_file)
            parts = backup_file.stem.split('_')
            reason = parts[-1] if len(parts) > 3 else 'unknown'
            
            backups.append({
                'filename': backup_file.name,
                'size': stat.st_size,
                'size_mb': round(stat.st_size / (1024 * 1024), 2),
                'created': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                'reason': reason
            })
        return backups
    except Exception as e:
        return []


def get_backup_path(filename):
    """Get full path to a backup file"""
    if IS_POSTGRES:
        return None
    backup_path = BACKUP_DIR / filename
    if backup_path.exists() and backup_path.suffix == '.db':
        return str(backup_path)
    return None


def restore_backup(filename):
    """Restore database from a backup file"""
    if IS_POSTGRES:
        return {'success': False, 'error': 'Restore not available for PostgreSQL'}
    try:
        backup_path = BACKUP_DIR / filename
        if not backup_path.exists():
            return {'success': False, 'error': 'Backup file not found'}
        
        create_backup(reason='before-restore')
        
        source = sqlite3.connect(backup_path)
        dest = sqlite3.connect(DB_PATH)
        source.backup(dest)
        source.close()
        dest.close()
        
        return {'success': True, 'message': f'Restored from {filename}'}
    except Exception as e:
        return {'success': False, 'error': str(e)}


def delete_backup(filename):
    """Delete a specific backup file"""
    if IS_POSTGRES:
        return {'success': False, 'error': 'Not available for PostgreSQL'}
    try:
        backup_path = BACKUP_DIR / filename
        if backup_path.exists() and backup_path.suffix == '.db':
            os.remove(backup_path)
            return {'success': True}
        return {'success': False, 'error': 'Backup not found'}
    except Exception as e:
        return {'success': False, 'error': str(e)}


# ============= DATABASE INITIALIZATION =============

def init_database():
    """Initialize the database with all required tables"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    if IS_POSTGRES:
        # PostgreSQL schema
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS overview (
                id SERIAL PRIMARY KEY,
                field_name TEXT NOT NULL,
                field_value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS intake_queue (
                id SERIAL PRIMARY KEY,
                proposer TEXT,
                test_purpose TEXT,
                test_classification TEXT,
                anode TEXT,
                separator TEXT,
                cathode TEXT,
                anode_current_collector TEXT,
                cathode_current_collector TEXT,
                anolyte TEXT,
                catholyte TEXT,
                temperature REAL,
                dp_psi REAL,
                proposed_priority TEXT,
                due_date TEXT,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ec_experiments (
                id SERIAL PRIMARY KEY,
                ec_id TEXT UNIQUE,
                input_into_jira BOOLEAN DEFAULT FALSE,
                performed BOOLEAN DEFAULT FALSE,
                test_id TEXT,
                proposer TEXT,
                test_purpose TEXT,
                test_classification TEXT,
                anode TEXT,
                separator TEXT,
                cathode TEXT,
                anode_current_collector TEXT,
                cathode_current_collector TEXT,
                anolyte TEXT,
                catholyte TEXT,
                temperature REAL,
                dp_psi REAL,
                priority_level TEXT,
                due_date TEXT,
                notes TEXT,
                assigned_to TEXT,
                is_favorite BOOLEAN DEFAULT FALSE,
                status TEXT DEFAULT 'pending',
                template_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Indexes for PostgreSQL
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_exp_status ON ec_experiments(status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_exp_priority ON ec_experiments(priority_level)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_exp_due_date ON ec_experiments(due_date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_exp_assigned ON ec_experiments(assigned_to)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_exp_classification ON ec_experiments(test_classification)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_exp_favorite ON ec_experiments(is_favorite)')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS weekly_schedule (
                id SERIAL PRIMARY KEY,
                week_start_date TEXT,
                day_of_week TEXT,
                time_slot TEXT,
                test_station TEXT,
                ec_id TEXT,
                experiment_name TEXT,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ec_outcomes (
                id SERIAL PRIMARY KEY,
                test_id TEXT,
                ec_id TEXT,
                test_purpose TEXT,
                operator TEXT,
                date_started TEXT,
                date_finished TEXT,
                outcome TEXT,
                implications TEXT,
                data_location TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS testing_parameters (
                id SERIAL PRIMARY KEY,
                test_id TEXT,
                ec_id TEXT,
                section TEXT,
                cell_config TEXT,
                cem TEXT,
                gde TEXT,
                process_params TEXT,
                conductivity_temp TEXT,
                flooding TEXT,
                key_results TEXT,
                detailed_data TEXT,
                resistance_model TEXT,
                raw_data JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dropdown_options (
                id SERIAL PRIMARY KEY,
                category TEXT NOT NULL,
                option_value TEXT NOT NULL,
                display_order INTEGER DEFAULT 0,
                is_default BOOLEAN DEFAULT FALSE
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS standard_conditions (
                id SERIAL PRIMARY KEY,
                field_name TEXT NOT NULL,
                default_value TEXT,
                description TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS templates (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                test_classification TEXT,
                anode TEXT,
                separator TEXT,
                cathode TEXT,
                anode_current_collector TEXT,
                cathode_current_collector TEXT,
                anolyte TEXT,
                catholyte TEXT,
                temperature REAL,
                dp_psi REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS activity_log (
                id SERIAL PRIMARY KEY,
                user_name TEXT,
                action TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                entity_id TEXT,
                entity_name TEXT,
                details TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS comments (
                id SERIAL PRIMARY KEY,
                ec_id TEXT NOT NULL,
                user_name TEXT NOT NULL,
                comment_text TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_preferences (
                id SERIAL PRIMARY KEY,
                user_name TEXT NOT NULL UNIQUE,
                theme TEXT DEFAULT 'dark',
                visible_columns TEXT,
                column_order TEXT,
                default_view TEXT DEFAULT 'table',
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS notifications (
                id SERIAL PRIMARY KEY,
                user_name TEXT,
                title TEXT NOT NULL,
                message TEXT NOT NULL,
                type TEXT DEFAULT 'info',
                is_read BOOLEAN DEFAULT FALSE,
                ec_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
    else:
        # SQLite schema
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS overview (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                field_name TEXT NOT NULL,
                field_value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS intake_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                proposer TEXT,
                test_purpose TEXT,
                test_classification TEXT,
                anode TEXT,
                separator TEXT,
                cathode TEXT,
                anode_current_collector TEXT,
                cathode_current_collector TEXT,
                anolyte TEXT,
                catholyte TEXT,
                temperature REAL,
                dp_psi REAL,
                proposed_priority TEXT,
                due_date TEXT,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ec_experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ec_id TEXT UNIQUE,
                input_into_jira BOOLEAN DEFAULT FALSE,
                performed BOOLEAN DEFAULT FALSE,
                test_id TEXT,
                proposer TEXT,
                test_purpose TEXT,
                test_classification TEXT,
                anode TEXT,
                separator TEXT,
                cathode TEXT,
                anode_current_collector TEXT,
                cathode_current_collector TEXT,
                anolyte TEXT,
                catholyte TEXT,
                temperature REAL,
                dp_psi REAL,
                priority_level TEXT,
                due_date TEXT,
                notes TEXT,
                assigned_to TEXT,
                is_favorite BOOLEAN DEFAULT FALSE,
                status TEXT DEFAULT 'pending',
                template_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_exp_status ON ec_experiments(status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_exp_priority ON ec_experiments(priority_level)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_exp_due_date ON ec_experiments(due_date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_exp_assigned ON ec_experiments(assigned_to)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_exp_classification ON ec_experiments(test_classification)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_exp_favorite ON ec_experiments(is_favorite)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_exp_created ON ec_experiments(created_at)')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS weekly_schedule (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                week_start_date TEXT,
                day_of_week TEXT,
                time_slot TEXT,
                test_station TEXT,
                ec_id TEXT,
                experiment_name TEXT,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ec_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_id TEXT,
                ec_id TEXT,
                test_purpose TEXT,
                operator TEXT,
                date_started TEXT,
                date_finished TEXT,
                outcome TEXT,
                implications TEXT,
                data_location TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS testing_parameters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_id TEXT,
                ec_id TEXT,
                section TEXT,
                cell_config TEXT,
                cem TEXT,
                gde TEXT,
                process_params TEXT,
                conductivity_temp TEXT,
                flooding TEXT,
                key_results TEXT,
                detailed_data TEXT,
                resistance_model TEXT,
                raw_data JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dropdown_options (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                option_value TEXT NOT NULL,
                display_order INTEGER DEFAULT 0,
                is_default BOOLEAN DEFAULT FALSE
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS standard_conditions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                field_name TEXT NOT NULL,
                default_value TEXT,
                description TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS templates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                test_classification TEXT,
                anode TEXT,
                separator TEXT,
                cathode TEXT,
                anode_current_collector TEXT,
                cathode_current_collector TEXT,
                anolyte TEXT,
                catholyte TEXT,
                temperature REAL,
                dp_psi REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS activity_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_name TEXT,
                action TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                entity_id TEXT,
                entity_name TEXT,
                details TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS comments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ec_id TEXT NOT NULL,
                user_name TEXT NOT NULL,
                comment_text TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_name TEXT NOT NULL UNIQUE,
                theme TEXT DEFAULT 'dark',
                visible_columns TEXT,
                column_order TEXT,
                default_view TEXT DEFAULT 'table',
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS notifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_name TEXT,
                title TEXT NOT NULL,
                message TEXT NOT NULL,
                type TEXT DEFAULT 'info',
                is_read BOOLEAN DEFAULT FALSE,
                ec_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # SQLite column additions
        try:
            cursor.execute('ALTER TABLE ec_experiments ADD COLUMN assigned_to TEXT')
        except:
            pass
        try:
            cursor.execute('ALTER TABLE ec_experiments ADD COLUMN is_favorite BOOLEAN DEFAULT FALSE')
        except:
            pass
        try:
            cursor.execute('ALTER TABLE ec_experiments ADD COLUMN status TEXT DEFAULT "pending"')
        except:
            pass
        try:
            cursor.execute('ALTER TABLE ec_experiments ADD COLUMN template_id INTEGER')
        except:
            pass
    
    conn.close()
    print("Database initialized successfully!")


def populate_dropdown_options():
    """Populate dropdown options based on Excel data"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Clear existing options
    execute_query(conn, 'DELETE FROM dropdown_options')
    
    # Helper to insert options
    def insert_option(category, option_value, order, is_default=False):
        execute_query(conn, 
            'INSERT INTO dropdown_options (category, option_value, display_order, is_default) VALUES (?, ?, ?, ?)',
            (category, option_value, order, is_default))
    
    # Test Classification options
    for i, opt in enumerate(['Anode', 'Cathode', 'Separator', 'Process', 'Electrolyte', 'System']):
        insert_option('test_classification', opt, i)
    
    # Anode Current Collector options
    for i, opt in enumerate(['Ag SS316L', 'Ni', 'Ti', 'Cu']):
        insert_option('anode_current_collector', opt, i, i == 0)
    
    # Cathode Current Collector options
    for i, opt in enumerate(['Ni', 'Ag SS316L', 'Ti', 'Cu']):
        insert_option('cathode_current_collector', opt, i, i == 0)
    
    # Priority options
    for i, opt in enumerate(['H', 'M', 'L']):
        insert_option('priority', opt, i)
    
    # Separator options
    for i, opt in enumerate(['N2060', 'N4800', 'Zirfon']):
        insert_option('separator', opt, i, i == 0)
    
    # Cathode options
    for i, opt in enumerate(['Raney Ni', 'PGM', 'Pd/C']):
        insert_option('cathode', opt, i, i == 0)
    
    # Test Station options
    for i, opt in enumerate(['TS1', 'TS2']):
        insert_option('test_station', opt, i)
    
    # Time Slot options
    for i, opt in enumerate(['AM', 'PM']):
        insert_option('time_slot', opt, i)
    
    # Status options
    for i, opt in enumerate(['pending', 'in_progress', 'completed', 'on_hold']):
        insert_option('status', opt, i)
    
    # Team members
    for i, opt in enumerate(['Unassigned', 'KMC', 'Sai', 'Team Member 1', 'Team Member 2']):
        insert_option('team_member', opt, i)
    
    # Anode options
    for i, opt in enumerate(['Standard', 'Type A', 'Type B', 'Custom']):
        insert_option('anode', opt, i)
    
    conn.close()
    print("Dropdown options populated!")


def populate_standard_conditions():
    """Populate standard conditions"""
    conn = get_db_connection()
    execute_query(conn, 'DELETE FROM standard_conditions')
    
    standards = [
        ('separator', 'N2060', 'Default separator membrane'),
        ('cathode', 'Raney Ni', 'Default cathode material'),
        ('anode_current_collector', 'Ag SS316L', 'Default anode current collector'),
        ('cathode_current_collector', 'Ni', 'Default cathode current collector'),
        ('anolyte', '6% K1, 17% K2', 'Default anolyte composition'),
        ('catholyte', '10 wt.% KOH', 'Default catholyte composition'),
        ('temperature', '80', 'Default temperature in °C'),
        ('dp_psi', '5', 'Default differential pressure in psi'),
    ]
    
    for field, value, desc in standards:
        execute_query(conn, 
            'INSERT INTO standard_conditions (field_name, default_value, description) VALUES (?, ?, ?)',
            (field, value, desc))
    
    conn.close()
    print("Standard conditions populated!")


def populate_overview():
    """Populate overview information"""
    conn = get_db_connection()
    execute_query(conn, 'DELETE FROM overview')
    
    overview_data = [
        ('Description', 'This workbook is to intake, prioritize, plan, and document status of electrochemical testing experiments.'),
        ('Status', 'Alpha'),
        ('Version', 'A.1'),
        ('Date', '2025-11-21'),
        ('Revision Date', str(datetime.now().date())),
        ('Signatures', 'KMC'),
    ]
    
    for field, value in overview_data:
        execute_query(conn, 'INSERT INTO overview (field_name, field_value) VALUES (?, ?)', (field, value))
    
    conn.close()
    print("Overview populated!")


def populate_default_templates():
    """Create default experiment templates"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    templates = [
        ('Standard Anode Test', 'Standard anode material testing configuration', 'Anode', None, 'N2060', 'Raney Ni', 'Ag SS316L', 'Ni', '6% K1, 17% K2', '10 wt.% KOH', 80, 5),
        ('Standard Cathode Test', 'Standard cathode material testing configuration', 'Cathode', None, 'N2060', None, 'Ag SS316L', 'Ni', '6% K1, 17% K2', '10 wt.% KOH', 80, 5),
        ('Separator Comparison', 'Compare different separator materials', 'Separator', None, None, 'Raney Ni', 'Ag SS316L', 'Ni', '6% K1, 17% K2', '10 wt.% KOH', 80, 5),
    ]
    
    for t in templates:
        try:
            execute_query(conn, '''
                INSERT INTO templates 
                (name, description, test_classification, anode, separator, cathode, 
                 anode_current_collector, cathode_current_collector, anolyte, catholyte, temperature, dp_psi)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', t)
        except:
            pass  # Ignore duplicates
    
    conn.close()
    print("Default templates created!")


# ============= HELPER FUNCTIONS =============

def get_next_ec_id():
    """Generate next EC ID (EC0001, EC0002, etc.)"""
    conn = None
    try:
        conn = get_db_connection()
        cursor = execute_query(conn, "SELECT ec_id FROM ec_experiments WHERE ec_id LIKE 'EC%'")
        results = fetchall(cursor)
        
        max_num = 0
        for row in results:
            ec_id = row.get('ec_id')
            if ec_id:
                try:
                    num = int(ec_id.replace('EC', '').replace('ec', ''))
                    if num > max_num:
                        max_num = num
                except:
                    pass
        
        new_id = f'EC{max_num + 1:04d}'
        
        # Verify uniqueness
        cursor = execute_query(conn, 'SELECT COUNT(*) as cnt FROM ec_experiments WHERE ec_id = ?', (new_id,))
        result = fetchone(cursor)
        count = result.get('cnt', 0) if result else 0
        
        if count > 0:
            max_num += 1
            while True:
                new_id = f'EC{max_num:04d}'
                cursor = execute_query(conn, 'SELECT COUNT(*) as cnt FROM ec_experiments WHERE ec_id = ?', (new_id,))
                result = fetchone(cursor)
                if (result.get('cnt', 0) if result else 0) == 0:
                    break
                max_num += 1
        
        return new_id
    finally:
        if conn:
            conn.close()


def log_activity(user_name, action, entity_type, entity_id, entity_name=None, details=None):
    """Log an activity"""
    try:
        conn = get_db_connection()
        execute_query(conn, '''
            INSERT INTO activity_log (user_name, action, entity_type, entity_id, entity_name, details)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_name, action, entity_type, entity_id, entity_name, details))
        conn.close()
    except:
        pass


def convert_excel_date(val):
    """Convert Excel date to string format"""
    import pandas as pd
    if pd.isna(val):
        return None
    if isinstance(val, (pd.Timestamp, datetime)):
        return val.strftime('%Y-%m-%d')
    if isinstance(val, (int, float)):
        try:
            excel_epoch = datetime(1899, 12, 30)
            date = excel_epoch + timedelta(days=float(val))
            return date.strftime('%Y-%m-%d')
        except:
            return str(val)
    return str(val) if val else None


# ============= IMPORT FROM EXCEL =============

def import_excel_data(excel_path):
    """Import data from Excel file into database"""
    import pandas as pd
    
    conn = get_db_connection()
    print(f"Importing data from: {excel_path}")
    
    # Import EC Experiments
    print("Importing EC Experiments...")
    df = pd.read_excel(excel_path, sheet_name='EC Experiments')
    df.columns = [str(c).strip() for c in df.columns]
    
    for _, row in df.iterrows():
        ec_id = row.get('EC ID', '')
        if pd.isna(ec_id) or not ec_id:
            continue
        
        try:
            # Check if exists
            cursor = execute_query(conn, 'SELECT id FROM ec_experiments WHERE ec_id = ?', (str(ec_id),))
            existing = fetchone(cursor)
            
            params = (
                str(ec_id),
                bool(row.get('Input into Jira?', False)),
                bool(row.get('Performed?', False)),
                str(row.get('Test ID', '')) if not pd.isna(row.get('Test ID', '')) else None,
                str(row.get('Proposer', '')) if not pd.isna(row.get('Proposer', '')) else None,
                str(row.get('Test Purpose', '')) if not pd.isna(row.get('Test Purpose', '')) else None,
                str(row.get('Test Classification \n[dropdown]', '')) if not pd.isna(row.get('Test Classification \n[dropdown]', '')) else None,
                str(row.get('Anode', '')) if not pd.isna(row.get('Anode', '')) else None,
                str(row.get('Separator', '')) if not pd.isna(row.get('Separator', '')) else None,
                str(row.get('Cathode', '')) if not pd.isna(row.get('Cathode', '')) else None,
                str(row.get('Anode Current Collector\n[dropdown]', '')) if not pd.isna(row.get('Anode Current Collector\n[dropdown]', '')) else None,
                str(row.get('Cathode Current Collector\n[dropdown]', '')) if not pd.isna(row.get('Cathode Current Collector\n[dropdown]', '')) else None,
                str(row.get('Anolyte', '')) if not pd.isna(row.get('Anolyte', '')) else None,
                str(row.get('Catholyte', '')) if not pd.isna(row.get('Catholyte', '')) else None,
                float(row.get('Temperature \n(°C)', 80)) if not pd.isna(row.get('Temperature \n(°C)', 80)) else 80,
                float(row.get('dP \n(Psi)', 5)) if not pd.isna(row.get('dP \n(Psi)', 5)) else 5,
                str(row.get('Priority Level\n [H / M / L]', '')) if not pd.isna(row.get('Priority Level\n [H / M / L]', '')) else None,
                str(row.get('Due Date', '')) if not pd.isna(row.get('Due Date', '')) else None,
                str(row.get('Notes', '')) if not pd.isna(row.get('Notes', '')) else None,
                'completed' if bool(row.get('Performed?', False)) else 'pending'
            )
            
            if existing:
                # Update
                execute_query(conn, '''
                    UPDATE ec_experiments SET
                    input_into_jira=?, performed=?, test_id=?, proposer=?, test_purpose=?,
                    test_classification=?, anode=?, separator=?, cathode=?, anode_current_collector=?,
                    cathode_current_collector=?, anolyte=?, catholyte=?, temperature=?, dp_psi=?,
                    priority_level=?, due_date=?, notes=?, status=?
                    WHERE ec_id=?
                ''', params[1:] + (params[0],))
            else:
                # Insert
                execute_query(conn, '''
                    INSERT INTO ec_experiments 
                    (ec_id, input_into_jira, performed, test_id, proposer, test_purpose, 
                     test_classification, anode, separator, cathode, anode_current_collector,
                     cathode_current_collector, anolyte, catholyte, temperature, dp_psi,
                     priority_level, due_date, notes, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', params)
        except Exception as e:
            print(f"  Error importing {ec_id}: {e}")
    
    # Import EC Outcomes
    print("Importing EC Outcomes...")
    try:
        df_outcomes = pd.read_excel(excel_path, sheet_name='EC Outcomes')
        for _, row in df_outcomes.iterrows():
            test_id = row.get('Test ID', '')
            if pd.isna(test_id) or not test_id:
                continue
            
            try:
                date_started = row.get('Date Started\n[mm:dd:yy]', row.get('Date Started', None))
                date_finished = row.get('Date Finished\n[mm:dd:yy]', row.get('Date Finished', None))
                
                execute_query(conn, '''
                    INSERT INTO ec_outcomes 
                    (test_id, ec_id, test_purpose, operator, date_started, date_finished, 
                     outcome, implications, data_location)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    str(test_id) if not pd.isna(test_id) else None,
                    str(row.get('EC ID', '')) if not pd.isna(row.get('EC ID', '')) else None,
                    str(row.get('Test Purpose', '')) if not pd.isna(row.get('Test Purpose', '')) else None,
                    str(row.get('Opperator', row.get('Operator', ''))) if not pd.isna(row.get('Opperator', row.get('Operator', ''))) else None,
                    convert_excel_date(date_started),
                    convert_excel_date(date_finished),
                    str(row.get('Outcome', '')) if not pd.isna(row.get('Outcome', '')) else None,
                    str(row.get('Implications and Extrapolations', '')) if not pd.isna(row.get('Implications and Extrapolations', '')) else None,
                    str(row.get('Data Location', '')) if not pd.isna(row.get('Data Location', '')) else None,
                ))
            except Exception as e:
                print(f"  Error importing outcome {test_id}: {e}")
    except Exception as e:
        print(f"  Error reading EC Outcomes sheet: {e}")
    
    # Import Intake Queue
    print("Importing Intake Queue...")
    try:
        df_intake = pd.read_excel(excel_path, sheet_name='Intake Queue for EC Testing')
        for _, row in df_intake.iterrows():
            proposer = row.get('Proposer', '')
            if pd.isna(proposer) or not proposer or proposer == '[Standard Conitions]':
                continue
            
            try:
                execute_query(conn, '''
                    INSERT INTO intake_queue 
                    (proposer, test_purpose, test_classification, anode, separator, cathode,
                     anode_current_collector, cathode_current_collector, anolyte, catholyte,
                     temperature, dp_psi, proposed_priority, due_date, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    str(proposer) if not pd.isna(proposer) else None,
                    str(row.get('Test Purpose', '')) if not pd.isna(row.get('Test Purpose', '')) else None,
                    str(row.get('Test Classification\n[dropdown]', '')) if not pd.isna(row.get('Test Classification\n[dropdown]', '')) else None,
                    str(row.get('Anode', '')) if not pd.isna(row.get('Anode', '')) else None,
                    str(row.get('Separator', '')) if not pd.isna(row.get('Separator', '')) else None,
                    str(row.get('Cathode', '')) if not pd.isna(row.get('Cathode', '')) else None,
                    str(row.get('Anode Current Collector\n[dropdown]', '')) if not pd.isna(row.get('Anode Current Collector\n[dropdown]', '')) else None,
                    str(row.get('Cathode Current Collector\n[dropdown]', '')) if not pd.isna(row.get('Cathode Current Collector\n[dropdown]', '')) else None,
                    str(row.get('Anolyte', '')) if not pd.isna(row.get('Anolyte', '')) else None,
                    str(row.get('Catholyte', '')) if not pd.isna(row.get('Catholyte', '')) else None,
                    float(row.get('Temperature \n(°C)', 80)) if not pd.isna(row.get('Temperature \n(°C)', 80)) else 80,
                    float(row.get('dP \n(Psi)', 5)) if not pd.isna(row.get('dP \n(Psi)', 5)) else 5,
                    str(row.get('Proposed Priority \n[H / M / L]', '')) if not pd.isna(row.get('Proposed Priority \n[H / M / L]', '')) else None,
                    str(row.get('Due Date', '')) if not pd.isna(row.get('Due Date', '')) else None,
                    str(row.get('Notes', '')) if not pd.isna(row.get('Notes', '')) else None,
                ))
            except Exception as e:
                print(f"  Error importing intake: {e}")
    except Exception as e:
        print(f"  Error reading Intake Queue sheet: {e}")
    
    conn.close()
    print("Data import complete!")


# ============= CRUD OPERATIONS - EXPERIMENTS =============

def get_all_experiments(filters=None, page=None, per_page=50):
    """Get all experiments with pagination and filtering"""
    conn = None
    try:
        conn = get_db_connection()
        query = 'SELECT * FROM ec_experiments WHERE 1=1'
        count_query = 'SELECT COUNT(*) as cnt FROM ec_experiments WHERE 1=1'
        params = []
        
        if filters:
            if filters.get('priority'):
                query += ' AND priority_level = ?'
                count_query += ' AND priority_level = ?'
                params.append(filters['priority'])
            if filters.get('status'):
                s = filters['status']
                if s in ('performed', 'completed'):
                    query += ' AND (performed = TRUE OR status = ?)'
                    count_query += ' AND (performed = TRUE OR status = ?)'
                    params.append('completed')
                else:
                    query += ' AND status = ?'
                    count_query += ' AND status = ?'
                    params.append(s)
            if filters.get('assigned_to'):
                query += ' AND assigned_to = ?'
                count_query += ' AND assigned_to = ?'
                params.append(filters['assigned_to'])
            if filters.get('favorites_only'):
                query += ' AND is_favorite = TRUE'
                count_query += ' AND is_favorite = TRUE'
            if filters.get('classification'):
                query += ' AND test_classification = ?'
                count_query += ' AND test_classification = ?'
                params.append(filters['classification'])
            if filters.get('search'):
                search_clause = ' AND (ec_id LIKE ? OR test_purpose LIKE ? OR proposer LIKE ? OR anode LIKE ? OR test_classification LIKE ?)'
                query += search_clause
                count_query += search_clause
                search_term = f'%{filters["search"]}%'
                params.extend([search_term] * 5)
        
        # Get total count
        cursor = execute_query(conn, count_query, params)
        total_result = fetchone(cursor)
        total = total_result.get('cnt', 0) if total_result else 0
        
        query += ' ORDER BY is_favorite DESC, ec_id DESC'
        
        if page is not None and per_page:
            offset = (page - 1) * per_page
            query += f' LIMIT {per_page} OFFSET {offset}'
        
        cursor = execute_query(conn, query, params)
        experiments = fetchall(cursor)
        
        if page is not None:
            return {
                'experiments': experiments,
                'total': total,
                'page': page,
                'per_page': per_page,
                'total_pages': (total + per_page - 1) // per_page
            }
        
        return experiments
    finally:
        if conn:
            conn.close()


def get_experiment(ec_id):
    """Get single experiment by EC ID"""
    conn = get_db_connection()
    cursor = execute_query(conn, 'SELECT * FROM ec_experiments WHERE ec_id = ?', (ec_id,))
    experiment = fetchone(cursor)
    conn.close()
    return experiment


def add_experiment(data):
    """Add a new experiment"""
    conn = None
    try:
        conn = get_db_connection()
        ec_id = get_next_ec_id()
        
        execute_query(conn, '''
            INSERT INTO ec_experiments 
            (ec_id, input_into_jira, performed, test_id, proposer, test_purpose, 
             test_classification, anode, separator, cathode, anode_current_collector,
             cathode_current_collector, anolyte, catholyte, temperature, dp_psi,
             priority_level, due_date, notes, assigned_to, is_favorite, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            ec_id, data.get('input_into_jira', False), data.get('performed', False),
            data.get('test_id'), data.get('proposer'), data.get('test_purpose'),
            data.get('test_classification'), data.get('anode'), data.get('separator'),
            data.get('cathode'), data.get('anode_current_collector'),
            data.get('cathode_current_collector'), data.get('anolyte'), data.get('catholyte'),
            data.get('temperature', 80), data.get('dp_psi', 5),
            data.get('priority_level'), data.get('due_date'), data.get('notes'),
            data.get('assigned_to'), data.get('is_favorite', False), 
            data.get('status', 'pending')
        ))
        
        execute_query(conn, '''
            INSERT INTO activity_log (user_name, action, entity_type, entity_id, entity_name, details)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', ('System', 'created', 'experiment', ec_id, f"Experiment {ec_id}", None))
        
        return ec_id
    finally:
        if conn:
            conn.close()


def update_experiment(ec_id, data):
    """Update an existing experiment"""
    conn = None
    try:
        conn = get_db_connection()
        
        status = data.get('status', 'pending')
        if data.get('performed'):
            status = 'completed'
        
        execute_query(conn, '''
            UPDATE ec_experiments SET
                input_into_jira = ?, performed = ?, test_id = ?, proposer = ?, test_purpose = ?,
                test_classification = ?, anode = ?, separator = ?, cathode = ?,
                anode_current_collector = ?, cathode_current_collector = ?, anolyte = ?,
                catholyte = ?, temperature = ?, dp_psi = ?, priority_level = ?,
                due_date = ?, notes = ?, assigned_to = ?, is_favorite = ?, status = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE ec_id = ?
        ''', (
            data.get('input_into_jira', False), data.get('performed', False),
            data.get('test_id'), data.get('proposer'), data.get('test_purpose'),
            data.get('test_classification'), data.get('anode'), data.get('separator'),
            data.get('cathode'), data.get('anode_current_collector'),
            data.get('cathode_current_collector'), data.get('anolyte'), data.get('catholyte'),
            data.get('temperature', 80), data.get('dp_psi', 5),
            data.get('priority_level'), data.get('due_date'), data.get('notes'),
            data.get('assigned_to'), data.get('is_favorite', False), status, ec_id
        ))
        
        execute_query(conn, '''
            INSERT INTO activity_log (user_name, action, entity_type, entity_id, entity_name, details)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', ('System', 'updated', 'experiment', ec_id, f"Experiment {ec_id}", None))
        
        return True
    finally:
        if conn:
            conn.close()


def delete_experiment(ec_id):
    """Delete an experiment"""
    conn = None
    try:
        conn = get_db_connection()
        execute_query(conn, 'DELETE FROM ec_experiments WHERE ec_id = ?', (ec_id,))
        execute_query(conn, '''
            INSERT INTO activity_log (user_name, action, entity_type, entity_id, entity_name, details)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', ('System', 'deleted', 'experiment', ec_id, f"Experiment {ec_id}", None))
        return True
    finally:
        if conn:
            conn.close()


def toggle_favorite(ec_id):
    """Toggle favorite status"""
    conn = None
    try:
        conn = get_db_connection()
        
        if IS_POSTGRES:
            execute_query(conn, 'UPDATE ec_experiments SET is_favorite = NOT is_favorite WHERE ec_id = ?', (ec_id,))
        else:
            execute_query(conn, 'UPDATE ec_experiments SET is_favorite = NOT is_favorite WHERE ec_id = ?', (ec_id,))
        
        cursor = execute_query(conn, 'SELECT is_favorite FROM ec_experiments WHERE ec_id = ?', (ec_id,))
        result = fetchone(cursor)
        return result.get('is_favorite', False) if result else False
    finally:
        if conn:
            conn.close()


def bulk_update_experiments(ec_ids, updates):
    """Bulk update multiple experiments"""
    conn = get_db_connection()
    
    for ec_id in ec_ids:
        set_clauses = []
        params = []
        for key, value in updates.items():
            set_clauses.append(f'{key} = ?')
            params.append(value)
        
        if set_clauses:
            params.append(ec_id)
            query = f'UPDATE ec_experiments SET {", ".join(set_clauses)}, updated_at = CURRENT_TIMESTAMP WHERE ec_id = ?'
            execute_query(conn, query, params)
    
    conn.close()
    log_activity('System', 'bulk_updated', 'experiments', ','.join(ec_ids), f"Updated {len(ec_ids)} experiments")
    return True


def bulk_delete_experiments(ec_ids):
    """Bulk delete multiple experiments"""
    conn = get_db_connection()
    for ec_id in ec_ids:
        execute_query(conn, 'DELETE FROM ec_experiments WHERE ec_id = ?', (ec_id,))
    conn.close()
    log_activity('System', 'bulk_deleted', 'experiments', ','.join(ec_ids), f"Deleted {len(ec_ids)} experiments")
    return True


# ============= CRUD OPERATIONS - OUTCOMES =============

def get_all_outcomes():
    """Get all outcomes"""
    conn = get_db_connection()
    cursor = execute_query(conn, 'SELECT * FROM ec_outcomes ORDER BY id DESC')
    outcomes = fetchall(cursor)
    conn.close()
    return outcomes


def add_outcome(data):
    """Add a new outcome"""
    conn = None
    try:
        conn = get_db_connection()
        cursor = execute_query(conn, '''
            INSERT INTO ec_outcomes 
            (test_id, ec_id, test_purpose, operator, date_started, date_finished,
             outcome, implications, data_location)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data.get('test_id'), data.get('ec_id'), data.get('test_purpose'),
            data.get('operator'), data.get('date_started'), data.get('date_finished'),
            data.get('outcome'), data.get('implications'), data.get('data_location')
        ))
        
        if IS_POSTGRES:
            cursor.execute('SELECT lastval()')
            outcome_id = cursor.fetchone()['lastval']
        else:
            outcome_id = cursor.lastrowid
        
        log_activity('System', 'created', 'outcome', str(outcome_id), f"Outcome for {data.get('ec_id', 'N/A')}")
        return outcome_id
    finally:
        if conn:
            conn.close()


def update_outcome(outcome_id, data):
    """Update an outcome"""
    conn = None
    try:
        conn = get_db_connection()
        execute_query(conn, '''
            UPDATE ec_outcomes SET
                test_id = ?, ec_id = ?, test_purpose = ?, operator = ?,
                date_started = ?, date_finished = ?, outcome = ?,
                implications = ?, data_location = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (
            data.get('test_id'), data.get('ec_id'), data.get('test_purpose'),
            data.get('operator'), data.get('date_started'), data.get('date_finished'),
            data.get('outcome'), data.get('implications'), data.get('data_location'), outcome_id
        ))
        return True
    finally:
        if conn:
            conn.close()


def delete_outcome(outcome_id):
    """Delete an outcome"""
    conn = None
    try:
        conn = get_db_connection()
        execute_query(conn, 'DELETE FROM ec_outcomes WHERE id = ?', (outcome_id,))
        return True
    finally:
        if conn:
            conn.close()


# ============= CRUD OPERATIONS - INTAKE QUEUE =============

def get_all_intake():
    """Get all intake items"""
    conn = get_db_connection()
    cursor = execute_query(conn, 'SELECT * FROM intake_queue ORDER BY id DESC')
    intake = fetchall(cursor)
    conn.close()
    return intake


def add_intake(data):
    """Add a new intake item"""
    conn = get_db_connection()
    cursor = execute_query(conn, '''
        INSERT INTO intake_queue 
        (proposer, test_purpose, test_classification, anode, separator, cathode,
         anode_current_collector, cathode_current_collector, anolyte, catholyte,
         temperature, dp_psi, proposed_priority, due_date, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        data.get('proposer'), data.get('test_purpose'), data.get('test_classification'),
        data.get('anode'), data.get('separator'), data.get('cathode'),
        data.get('anode_current_collector'), data.get('cathode_current_collector'),
        data.get('anolyte'), data.get('catholyte'),
        data.get('temperature', 80), data.get('dp_psi', 5),
        data.get('proposed_priority'), data.get('due_date'), data.get('notes')
    ))
    
    if IS_POSTGRES:
        cursor.execute('SELECT lastval()')
        intake_id = cursor.fetchone()['lastval']
    else:
        intake_id = cursor.lastrowid
    
    conn.close()
    log_activity('System', 'created', 'intake', str(intake_id), f"Intake request from {data.get('proposer', 'N/A')}")
    return intake_id


def delete_intake(intake_id):
    """Delete an intake item"""
    conn = get_db_connection()
    execute_query(conn, 'DELETE FROM intake_queue WHERE id = ?', (intake_id,))
    conn.close()
    return True


def promote_intake_to_experiment(intake_id):
    """Promote an intake item to experiment"""
    conn = get_db_connection()
    cursor = execute_query(conn, 'SELECT * FROM intake_queue WHERE id = ?', (intake_id,))
    intake = fetchone(cursor)
    
    if not intake:
        conn.close()
        return None
    
    ec_id = add_experiment({
        'proposer': intake.get('proposer'),
        'test_purpose': intake.get('test_purpose'),
        'test_classification': intake.get('test_classification'),
        'anode': intake.get('anode'),
        'separator': intake.get('separator'),
        'cathode': intake.get('cathode'),
        'anode_current_collector': intake.get('anode_current_collector'),
        'cathode_current_collector': intake.get('cathode_current_collector'),
        'anolyte': intake.get('anolyte'),
        'catholyte': intake.get('catholyte'),
        'temperature': intake.get('temperature'),
        'dp_psi': intake.get('dp_psi'),
        'priority_level': intake.get('proposed_priority'),
        'due_date': intake.get('due_date'),
        'notes': intake.get('notes'),
    })
    
    execute_query(conn, 'DELETE FROM intake_queue WHERE id = ?', (intake_id,))
    conn.close()
    
    log_activity('System', 'promoted', 'intake', str(intake_id), f"Promoted to {ec_id}")
    return ec_id


# ============= CRUD OPERATIONS - TEMPLATES =============

def get_all_templates():
    """Get all templates"""
    conn = get_db_connection()
    cursor = execute_query(conn, 'SELECT * FROM templates ORDER BY name')
    templates = fetchall(cursor)
    conn.close()
    return templates


def get_template(template_id):
    """Get single template"""
    conn = get_db_connection()
    cursor = execute_query(conn, 'SELECT * FROM templates WHERE id = ?', (template_id,))
    template = fetchone(cursor)
    conn.close()
    return template


def add_template(data):
    """Add a new template"""
    conn = get_db_connection()
    cursor = execute_query(conn, '''
        INSERT INTO templates 
        (name, description, test_classification, anode, separator, cathode,
         anode_current_collector, cathode_current_collector, anolyte, catholyte, temperature, dp_psi)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        data.get('name'), data.get('description'), data.get('test_classification'),
        data.get('anode'), data.get('separator'), data.get('cathode'),
        data.get('anode_current_collector'), data.get('cathode_current_collector'),
        data.get('anolyte'), data.get('catholyte'), data.get('temperature', 80), data.get('dp_psi', 5)
    ))
    
    if IS_POSTGRES:
        cursor.execute('SELECT lastval()')
        template_id = cursor.fetchone()['lastval']
    else:
        template_id = cursor.lastrowid
    
    conn.close()
    return template_id


def delete_template(template_id):
    """Delete a template"""
    conn = get_db_connection()
    execute_query(conn, 'DELETE FROM templates WHERE id = ?', (template_id,))
    conn.close()
    return True


def create_experiment_from_template(template_id, additional_data=None):
    """Create experiment from template"""
    template = get_template(template_id)
    if not template:
        return None
    
    exp_data = {
        'test_classification': template.get('test_classification'),
        'anode': template.get('anode'),
        'separator': template.get('separator'),
        'cathode': template.get('cathode'),
        'anode_current_collector': template.get('anode_current_collector'),
        'cathode_current_collector': template.get('cathode_current_collector'),
        'anolyte': template.get('anolyte'),
        'catholyte': template.get('catholyte'),
        'temperature': template.get('temperature'),
        'dp_psi': template.get('dp_psi'),
        'test_purpose': f"[From Template: {template.get('name')}]",
    }
    
    if additional_data:
        exp_data.update(additional_data)
    
    ec_id = add_experiment(exp_data)
    log_activity('System', 'created_from_template', 'experiment', ec_id, f"From template: {template.get('name')}")
    return ec_id


# ============= CRUD OPERATIONS - COMMENTS =============

def get_comments(ec_id):
    """Get comments for an experiment"""
    conn = get_db_connection()
    cursor = execute_query(conn, 'SELECT * FROM comments WHERE ec_id = ? ORDER BY created_at DESC', (ec_id,))
    comments = fetchall(cursor)
    conn.close()
    return comments


def add_comment(ec_id, user_name, comment_text):
    """Add a comment"""
    conn = get_db_connection()
    cursor = execute_query(conn, 
        'INSERT INTO comments (ec_id, user_name, comment_text) VALUES (?, ?, ?)',
        (ec_id, user_name, comment_text))
    
    if IS_POSTGRES:
        cursor.execute('SELECT lastval()')
        comment_id = cursor.fetchone()['lastval']
    else:
        comment_id = cursor.lastrowid
    
    conn.close()
    return comment_id


# ============= DROPDOWN & STATISTICS =============

def get_dropdown_options():
    """Get all dropdown options"""
    conn = get_db_connection()
    cursor = execute_query(conn, 'SELECT * FROM dropdown_options ORDER BY category, display_order')
    results = fetchall(cursor)
    conn.close()
    
    options = {}
    for row in results:
        category = row.get('category')
        if category not in options:
            options[category] = []
        options[category].append({
            'value': row.get('option_value'),
            'is_default': row.get('is_default', False)
        })
    
    return options


def add_dropdown_option(category, value):
    """Add a new dropdown option"""
    conn = get_db_connection()
    cursor = execute_query(conn, 'SELECT MAX(display_order) as max_order FROM dropdown_options WHERE category = ?', (category,))
    result = fetchone(cursor)
    max_order = (result.get('max_order') or 0) + 1 if result else 0
    
    execute_query(conn, 
        'INSERT INTO dropdown_options (category, option_value, display_order) VALUES (?, ?, ?)',
        (category, value, max_order))
    conn.close()
    return True


def get_standard_conditions():
    """Get standard conditions"""
    conn = get_db_connection()
    cursor = execute_query(conn, 'SELECT * FROM standard_conditions')
    results = fetchall(cursor)
    conn.close()
    
    return {row['field_name']: row['default_value'] for row in results}


def get_statistics():
    """Get dashboard statistics"""
    conn = get_db_connection()
    
    try:
        # Total experiments
        cursor = execute_query(conn, 'SELECT COUNT(*) as cnt FROM ec_experiments')
        result = fetchone(cursor)
        total = result.get('cnt', 0) if result else 0
        
        # Pending
        cursor = execute_query(conn, "SELECT COUNT(*) as cnt FROM ec_experiments WHERE status = 'pending'")
        result = fetchone(cursor)
        pending = result.get('cnt', 0) if result else 0
        
        # In Progress
        cursor = execute_query(conn, "SELECT COUNT(*) as cnt FROM ec_experiments WHERE status = 'in_progress'")
        result = fetchone(cursor)
        in_progress = result.get('cnt', 0) if result else 0
        
        # Completed - handle both boolean representations (1/true and 0/false)
        if IS_POSTGRES:
            cursor = execute_query(conn, "SELECT COUNT(*) as cnt FROM ec_experiments WHERE performed = true OR status = 'completed'")
        else:
            cursor = execute_query(conn, "SELECT COUNT(*) as cnt FROM ec_experiments WHERE performed = 1 OR status = 'completed'")
        result = fetchone(cursor)
        completed = result.get('cnt', 0) if result else 0
        
        # High Priority
        cursor = execute_query(conn, "SELECT COUNT(*) as cnt FROM ec_experiments WHERE priority_level = 'H'")
        result = fetchone(cursor)
        high_priority = result.get('cnt', 0) if result else 0
        
        # Favorites
        if IS_POSTGRES:
            cursor = execute_query(conn, "SELECT COUNT(*) as cnt FROM ec_experiments WHERE is_favorite = true")
        else:
            cursor = execute_query(conn, "SELECT COUNT(*) as cnt FROM ec_experiments WHERE is_favorite = 1")
        result = fetchone(cursor)
        favorites = result.get('cnt', 0) if result else 0
        
        # Overdue (due_date < today and not completed)
        today = datetime.now().strftime('%Y-%m-%d')
        if IS_POSTGRES:
            cursor = execute_query(conn, f"SELECT COUNT(*) as cnt FROM ec_experiments WHERE due_date < '{today}' AND due_date IS NOT NULL AND due_date != '' AND status != 'completed' AND performed != true")
        else:
            cursor = execute_query(conn, f"SELECT COUNT(*) as cnt FROM ec_experiments WHERE due_date < '{today}' AND due_date IS NOT NULL AND due_date != '' AND status != 'completed' AND performed != 1")
        result = fetchone(cursor)
        overdue = result.get('cnt', 0) if result else 0
        
        conn.close()
        
        # Return with field names that frontend expects
        return {
            'total': total,
            'total_experiments': total,
            'pending': pending,
            'pending_experiments': pending,
            'in_progress': in_progress,
            'completed': completed,
            'performed_experiments': completed,
            'high_priority': high_priority,
            'favorites': favorites,
            'overdue': overdue
        }
    except Exception as e:
        print(f"Error getting statistics: {e}")
        if conn:
            conn.close()
        return {
            'total': 0,
            'total_experiments': 0,
            'pending': 0,
            'pending_experiments': 0,
            'in_progress': 0,
            'completed': 0,
            'performed_experiments': 0,
            'high_priority': 0,
            'favorites': 0,
            'overdue': 0
        }


def get_activity_log(limit=50):
    """Get recent activity log"""
    conn = get_db_connection()
    cursor = execute_query(conn, f'SELECT * FROM activity_log ORDER BY created_at DESC LIMIT {limit}')
    activities = fetchall(cursor)
    conn.close()
    return activities


def get_calendar_data(year, month):
    """Get experiments for calendar view"""
    conn = get_db_connection()
    
    start_date = f'{year}-{month:02d}-01'
    if month == 12:
        end_date = f'{year + 1}-01-01'
    else:
        end_date = f'{year}-{month + 1:02d}-01'
    
    cursor = execute_query(conn, '''
        SELECT * FROM ec_experiments 
        WHERE due_date >= ? AND due_date < ?
        ORDER BY due_date
    ''', (start_date, end_date))
    experiments = fetchall(cursor)
    conn.close()
    
    return experiments


def get_kanban_data():
    """Get experiments organized by status for kanban view"""
    conn = get_db_connection()
    
    result = {
        'pending': [],
        'in_progress': [],
        'completed': [],
        'on_hold': []
    }
    
    for status in result.keys():
        if status == 'completed':
            cursor = execute_query(conn, '''
                SELECT * FROM ec_experiments 
                WHERE status = ? OR performed = TRUE
                ORDER BY updated_at DESC LIMIT 50
            ''', (status,))
        else:
            cursor = execute_query(conn, '''
                SELECT * FROM ec_experiments 
                WHERE status = ?
                ORDER BY updated_at DESC LIMIT 50
            ''', (status,))
        result[status] = fetchall(cursor)
    
    conn.close()
    return result


def update_experiment_status(ec_id, new_status):
    """Update experiment status"""
    conn = get_db_connection()
    
    performed = new_status == 'completed'
    execute_query(conn, '''
        UPDATE ec_experiments 
        SET status = ?, performed = ?, updated_at = CURRENT_TIMESTAMP
        WHERE ec_id = ?
    ''', (new_status, performed, ec_id))
    
    conn.close()
    log_activity('System', 'status_changed', 'experiment', ec_id, f"Status: {new_status}")
    return True


# ============= OVERVIEW =============

def get_overview():
    """Get overview information"""
    conn = get_db_connection()
    cursor = execute_query(conn, 'SELECT * FROM overview')
    results = fetchall(cursor)
    conn.close()
    return {row['field_name']: row['field_value'] for row in results}


# ============= USER PREFERENCES =============

def get_user_preferences(user_name):
    """Get user preferences"""
    conn = get_db_connection()
    cursor = execute_query(conn, 'SELECT * FROM user_preferences WHERE user_name = ?', (user_name,))
    prefs = fetchone(cursor)
    conn.close()
    
    if prefs:
        return prefs
    else:
        return {
            'theme': 'dark',
            'visible_columns': None,
            'column_order': None,
            'default_view': 'table'
        }


def update_user_preferences(user_name, prefs):
    """Update user preferences"""
    conn = get_db_connection()
    
    # Check if exists
    cursor = execute_query(conn, 'SELECT id FROM user_preferences WHERE user_name = ?', (user_name,))
    existing = fetchone(cursor)
    
    if existing:
        execute_query(conn, '''
            UPDATE user_preferences SET
                theme = ?, visible_columns = ?, column_order = ?, default_view = ?, 
                updated_at = CURRENT_TIMESTAMP
            WHERE user_name = ?
        ''', (
            prefs.get('theme', 'dark'),
            prefs.get('visible_columns'),
            prefs.get('column_order'),
            prefs.get('default_view', 'table'),
            user_name
        ))
    else:
        execute_query(conn, '''
            INSERT INTO user_preferences (user_name, theme, visible_columns, column_order, default_view)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            user_name,
            prefs.get('theme', 'dark'),
            prefs.get('visible_columns'),
            prefs.get('column_order'),
            prefs.get('default_view', 'table')
        ))
    
    conn.close()
    return True


# ============= NOTIFICATIONS =============

def get_notifications(user_name=None, unread_only=False):
    """Get notifications"""
    conn = get_db_connection()
    
    query = 'SELECT * FROM notifications WHERE 1=1'
    params = []
    
    if user_name:
        query += ' AND (user_name = ? OR user_name IS NULL)'
        params.append(user_name)
    
    if unread_only:
        query += ' AND is_read = FALSE'
    
    query += ' ORDER BY created_at DESC LIMIT 50'
    
    cursor = execute_query(conn, query, params if params else None)
    notifications = fetchall(cursor)
    conn.close()
    return notifications


def add_notification(user_name, title, message, notif_type='info', ec_id=None):
    """Add a notification"""
    conn = get_db_connection()
    cursor = execute_query(conn, '''
        INSERT INTO notifications (user_name, title, message, type, ec_id)
        VALUES (?, ?, ?, ?, ?)
    ''', (user_name, title, message, notif_type, ec_id))
    
    if IS_POSTGRES:
        cursor.execute('SELECT lastval()')
        notif_id = cursor.fetchone()['lastval']
    else:
        notif_id = cursor.lastrowid
    
    conn.close()
    return notif_id


def mark_notification_read(notif_id):
    """Mark a notification as read"""
    conn = get_db_connection()
    execute_query(conn, 'UPDATE notifications SET is_read = TRUE WHERE id = ?', (notif_id,))
    conn.close()
    return True


def mark_all_notifications_read(user_name=None):
    """Mark all notifications as read"""
    conn = get_db_connection()
    
    if user_name:
        execute_query(conn, 'UPDATE notifications SET is_read = TRUE WHERE user_name = ? OR user_name IS NULL', (user_name,))
    else:
        execute_query(conn, 'UPDATE notifications SET is_read = TRUE')
    
    conn.close()
    return True


# ============= COMMENTS - DELETE =============

def delete_comment(comment_id):
    """Delete a comment"""
    conn = get_db_connection()
    execute_query(conn, 'DELETE FROM comments WHERE id = ?', (comment_id,))
    conn.close()
    return True


# ============= DUPLICATE EXPERIMENT =============

def duplicate_experiment(ec_id):
    """Duplicate an existing experiment with new EC ID"""
    original = get_experiment(ec_id)
    if not original:
        return None
    
    # Create new experiment data without id/ec_id
    original.pop('id', None)
    original.pop('ec_id', None)
    original['performed'] = False
    original['input_into_jira'] = False
    original['status'] = 'pending'
    original['is_favorite'] = False
    original['test_purpose'] = f"[COPY] {original.get('test_purpose', '')}"
    
    new_ec_id = add_experiment(original)
    log_activity('System', 'duplicated', 'experiment', new_ec_id, f"Duplicated from {ec_id}")
    return new_ec_id


# ============= DATA MIGRATION =============

def migrate_sqlite_to_postgres():
    """Migrate data from SQLite to PostgreSQL (run this once when setting up production)"""
    if not IS_POSTGRES:
        print("Not running on PostgreSQL, skipping migration")
        return False
    
    import sqlite3 as sqlite
    
    # Connect to local SQLite
    sqlite_conn = sqlite.connect(str(DB_PATH))
    sqlite_conn.row_factory = sqlite.Row
    
    # Connect to PostgreSQL
    pg_conn = get_db_connection()
    
    tables = [
        'overview', 'dropdown_options', 'standard_conditions', 'templates',
        'ec_experiments', 'ec_outcomes', 'intake_queue', 'activity_log', 'comments'
    ]
    
    for table in tables:
        print(f"Migrating {table}...")
        try:
            rows = sqlite_conn.execute(f'SELECT * FROM {table}').fetchall()
            
            if not rows:
                continue
            
            columns = rows[0].keys()
            # Skip 'id' column as PostgreSQL uses SERIAL
            columns = [c for c in columns if c != 'id']
            
            placeholders = ', '.join(['%s'] * len(columns))
            col_names = ', '.join(columns)
            
            cursor = pg_conn.cursor()
            for row in rows:
                values = tuple(row[c] for c in columns)
                try:
                    cursor.execute(f'INSERT INTO {table} ({col_names}) VALUES ({placeholders})', values)
                except Exception as e:
                    print(f"  Error inserting row: {e}")
            
            print(f"  Migrated {len(rows)} rows")
        except Exception as e:
            print(f"  Error migrating {table}: {e}")
    
    sqlite_conn.close()
    pg_conn.close()
    print("Migration complete!")
    return True


# Initialize when module loads
if __name__ == '__main__':
    init_database()
    print(f"Database type: {'PostgreSQL' if IS_POSTGRES else 'SQLite'}")
