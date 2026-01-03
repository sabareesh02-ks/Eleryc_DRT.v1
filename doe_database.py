"""
DOE Planner Database Models and Utilities
Electrochemical Testing Workflow Database
Enhanced with all 20 usability features
"""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
import json
import time
import os
import shutil
import glob

# Database file path
DB_PATH = Path(__file__).parent / 'doe_planner.db'
BACKUP_DIR = Path(__file__).parent / 'backups'


# ============= BACKUP FUNCTIONS =============

def create_backup(reason='manual'):
    """Create timestamped backup of database"""
    try:
        # Create backup directory if not exists
        BACKUP_DIR.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        backup_filename = f'doe_planner_{timestamp}_{reason}.db'
        backup_path = BACKUP_DIR / backup_filename
        
        # Use SQLite backup API (safe even while db is in use)
        source = sqlite3.connect(DB_PATH)
        dest = sqlite3.connect(backup_path)
        source.backup(dest)
        source.close()
        dest.close()
        
        # Clean up old backups (keep last 7)
        cleanup_old_backups(keep=7)
        
        # Get file size
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
    try:
        backups = sorted(BACKUP_DIR.glob('doe_planner_*.db'), key=os.path.getmtime, reverse=True)
        for old_backup in backups[keep:]:
            os.remove(old_backup)
    except Exception as e:
        print(f"Cleanup error: {e}")


def get_all_backups():
    """Get list of all available backups"""
    try:
        if not BACKUP_DIR.exists():
            return []
        
        backups = []
        for backup_file in sorted(BACKUP_DIR.glob('doe_planner_*.db'), key=os.path.getmtime, reverse=True):
            stat = os.stat(backup_file)
            # Parse filename: doe_planner_2026-01-03_14-00-00_manual.db
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
    backup_path = BACKUP_DIR / filename
    if backup_path.exists() and backup_path.suffix == '.db':
        return str(backup_path)
    return None


def restore_backup(filename):
    """Restore database from a backup file"""
    try:
        backup_path = BACKUP_DIR / filename
        if not backup_path.exists():
            return {'success': False, 'error': 'Backup file not found'}
        
        # First create a backup of current state
        create_backup(reason='before-restore')
        
        # Close any existing connections and restore
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
    try:
        backup_path = BACKUP_DIR / filename
        if backup_path.exists() and backup_path.suffix == '.db':
            os.remove(backup_path)
            return {'success': True}
        return {'success': False, 'error': 'Backup not found'}
    except Exception as e:
        return {'success': False, 'error': str(e)}


def get_db_connection():
    """Get database connection with row factory and proper settings for concurrency"""
    conn = sqlite3.connect(
        str(DB_PATH),
        timeout=30.0,  # Wait up to 30 seconds for locks
        isolation_level=None,  # Autocommit mode
        check_same_thread=False
    )
    conn.row_factory = sqlite3.Row
    # Enable WAL mode for better concurrency
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA busy_timeout=30000')  # 30 second timeout
    conn.execute('PRAGMA synchronous=NORMAL')
    return conn


def execute_with_retry(func, max_retries=3, delay=0.5):
    """Execute a database function with retry logic for locked database"""
    for attempt in range(max_retries):
        try:
            return func()
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower() and attempt < max_retries - 1:
                time.sleep(delay * (attempt + 1))  # Exponential backoff
                continue
            raise
    return None


def init_database():
    """Initialize the database with all required tables"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. Overview table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS overview (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            field_name TEXT NOT NULL,
            field_value TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # 2. Intake Queue table
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
    
    # 3. EC Experiments table (main table) - Enhanced
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
    
    # Add indexes for performance with 1000+ experiments
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_exp_status ON ec_experiments(status)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_exp_priority ON ec_experiments(priority_level)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_exp_due_date ON ec_experiments(due_date)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_exp_assigned ON ec_experiments(assigned_to)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_exp_classification ON ec_experiments(test_classification)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_exp_favorite ON ec_experiments(is_favorite)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_exp_created ON ec_experiments(created_at)')
    
    # 4. Weekly Schedule table
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
    
    # 5. EC Outcomes table
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
    
    # 6. Testing Parameters table
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
    
    # 7. Dropdown Options table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS dropdown_options (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL,
            option_value TEXT NOT NULL,
            display_order INTEGER DEFAULT 0,
            is_default BOOLEAN DEFAULT FALSE
        )
    ''')
    
    # 8. Standard Conditions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS standard_conditions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            field_name TEXT NOT NULL,
            default_value TEXT,
            description TEXT
        )
    ''')
    
    # 9. Templates table - NEW
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
    
    # 10. Activity Log table - NEW
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
    
    # 11. Comments table - NEW
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS comments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ec_id TEXT NOT NULL,
            user_name TEXT NOT NULL,
            comment_text TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # 12. User Preferences table - NEW
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
    
    # 13. Notifications table - NEW
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
    
    # Add new columns to existing tables if they don't exist
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
    
    conn.commit()
    conn.close()
    print("Database initialized successfully!")


def populate_dropdown_options():
    """Populate dropdown options based on Excel data"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Clear existing options
    cursor.execute('DELETE FROM dropdown_options')
    
    # Test Classification options
    classifications = ['Anode', 'Cathode', 'Separator', 'Process', 'Electrolyte', 'System']
    for i, opt in enumerate(classifications):
        cursor.execute('INSERT INTO dropdown_options (category, option_value, display_order) VALUES (?, ?, ?)',
                      ('test_classification', opt, i))
    
    # Anode Current Collector options
    anode_cc = ['Ag SS316L', 'Ni', 'Ti', 'Cu']
    for i, opt in enumerate(anode_cc):
        cursor.execute('INSERT INTO dropdown_options (category, option_value, display_order, is_default) VALUES (?, ?, ?, ?)',
                      ('anode_current_collector', opt, i, i == 0))
    
    # Cathode Current Collector options
    cathode_cc = ['Ni', 'Ag SS316L', 'Ti', 'Cu']
    for i, opt in enumerate(cathode_cc):
        cursor.execute('INSERT INTO dropdown_options (category, option_value, display_order, is_default) VALUES (?, ?, ?, ?)',
                      ('cathode_current_collector', opt, i, i == 0))
    
    # Priority options
    priorities = ['H', 'M', 'L']
    for i, opt in enumerate(priorities):
        cursor.execute('INSERT INTO dropdown_options (category, option_value, display_order) VALUES (?, ?, ?)',
                      ('priority', opt, i))
    
    # Separator options
    separators = ['N2060', 'N4800', 'Zirfon']
    for i, opt in enumerate(separators):
        cursor.execute('INSERT INTO dropdown_options (category, option_value, display_order, is_default) VALUES (?, ?, ?, ?)',
                      ('separator', opt, i, i == 0))
    
    # Cathode options
    cathodes = ['Raney Ni', 'PGM', 'Pd/C']
    for i, opt in enumerate(cathodes):
        cursor.execute('INSERT INTO dropdown_options (category, option_value, display_order, is_default) VALUES (?, ?, ?, ?)',
                      ('cathode', opt, i, i == 0))
    
    # Test Station options
    test_stations = ['TS1', 'TS2']
    for i, opt in enumerate(test_stations):
        cursor.execute('INSERT INTO dropdown_options (category, option_value, display_order) VALUES (?, ?, ?)',
                      ('test_station', opt, i))
    
    # Time Slot options
    time_slots = ['AM', 'PM']
    for i, opt in enumerate(time_slots):
        cursor.execute('INSERT INTO dropdown_options (category, option_value, display_order) VALUES (?, ?, ?)',
                      ('time_slot', opt, i))
    
    # Status options
    statuses = ['pending', 'in_progress', 'completed', 'on_hold']
    for i, opt in enumerate(statuses):
        cursor.execute('INSERT INTO dropdown_options (category, option_value, display_order) VALUES (?, ?, ?)',
                      ('status', opt, i))
    
    # Team members (for assignment)
    team_members = ['Unassigned', 'KMC', 'Sai', 'Team Member 1', 'Team Member 2']
    for i, opt in enumerate(team_members):
        cursor.execute('INSERT INTO dropdown_options (category, option_value, display_order) VALUES (?, ?, ?)',
                      ('team_member', opt, i))
    
    # Anode options (for anode material/type)
    anodes = ['Standard', 'Type A', 'Type B', 'Custom']
    for i, opt in enumerate(anodes):
        cursor.execute('INSERT INTO dropdown_options (category, option_value, display_order) VALUES (?, ?, ?)',
                      ('anode', opt, i))
    
    conn.commit()
    conn.close()
    print("Dropdown options populated!")


def populate_standard_conditions():
    """Populate standard conditions based on Excel data"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Clear existing
    cursor.execute('DELETE FROM standard_conditions')
    
    # Standard conditions from Excel
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
        cursor.execute('INSERT INTO standard_conditions (field_name, default_value, description) VALUES (?, ?, ?)',
                      (field, value, desc))
    
    conn.commit()
    conn.close()
    print("Standard conditions populated!")


def populate_overview():
    """Populate overview information"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Clear existing
    cursor.execute('DELETE FROM overview')
    
    overview_data = [
        ('Description', 'This workbook is to intake, prioritize, plan, and document status of electrochemical testing experiments. Experiments of interest are to be placed in here for review on a weekly basis to ensure visibility and the most valuable experiments are prioritized.'),
        ('Status', 'Alpha'),
        ('Version', 'A.1'),
        ('Date', '2025-11-21'),
        ('Revision Date', str(datetime.now().date())),
        ('Signatures', 'KMC'),
    ]
    
    for field, value in overview_data:
        cursor.execute('INSERT INTO overview (field_name, field_value) VALUES (?, ?)', (field, value))
    
    conn.commit()
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
        ('High Temperature Test', 'Elevated temperature testing', 'Process', None, 'N2060', 'Raney Ni', 'Ag SS316L', 'Ni', '6% K1, 17% K2', '10 wt.% KOH', 90, 5),
        ('Electrolyte Study', 'Electrolyte composition study', 'Electrolyte', None, 'N2060', 'Raney Ni', 'Ag SS316L', 'Ni', None, None, 80, 5),
    ]
    
    for t in templates:
        cursor.execute('''
            INSERT OR IGNORE INTO templates 
            (name, description, test_classification, anode, separator, cathode, 
             anode_current_collector, cathode_current_collector, anolyte, catholyte, temperature, dp_psi)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', t)
    
    conn.commit()
    conn.close()
    print("Default templates created!")


def get_next_ec_id():
    """Generate next EC ID (EC0001, EC0002, etc.) - properly handles numeric ordering"""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get all EC IDs and find the maximum number
        cursor.execute("SELECT ec_id FROM ec_experiments WHERE ec_id LIKE 'EC%'")
        results = cursor.fetchall()
        
        max_num = 0
        for row in results:
            ec_id = row['ec_id'] if row else None
            if ec_id:
                try:
                    # Extract number from EC0001 format
                    num = int(ec_id.replace('EC', '').replace('ec', ''))
                    if num > max_num:
                        max_num = num
                except:
                    pass
        
        new_id = f'EC{max_num + 1:04d}'
        
        # Double-check it doesn't exist
        cursor.execute('SELECT COUNT(*) FROM ec_experiments WHERE ec_id = ?', (new_id,))
        if cursor.fetchone()[0] > 0:
            # If exists, find next available
            max_num += 1
            while True:
                new_id = f'EC{max_num:04d}'
                cursor.execute('SELECT COUNT(*) FROM ec_experiments WHERE ec_id = ?', (new_id,))
                if cursor.fetchone()[0] == 0:
                    break
                max_num += 1
        
        return new_id
    finally:
        if conn:
            conn.close()


def import_excel_data(excel_path):
    """Import data from Excel file into database"""
    import pandas as pd
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
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
            cursor.execute('''
                INSERT OR REPLACE INTO ec_experiments 
                (ec_id, input_into_jira, performed, test_id, proposer, test_purpose, 
                 test_classification, anode, separator, cathode, anode_current_collector,
                 cathode_current_collector, anolyte, catholyte, temperature, dp_psi,
                 priority_level, due_date, notes, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
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
            ))
        except Exception as e:
            print(f"  Error importing {ec_id}: {e}")
    
    # Helper function to convert Excel dates
    def convert_excel_date(val):
        if pd.isna(val):
            return None
        # If it's already a datetime object
        if isinstance(val, (pd.Timestamp, datetime)):
            return val.strftime('%Y-%m-%d')
        # If it's a number (Excel serial date)
        if isinstance(val, (int, float)):
            try:
                # Excel serial date: days since 1900-01-01 (with a bug for 1900 leap year)
                from datetime import timedelta
                excel_epoch = datetime(1899, 12, 30)
                date = excel_epoch + timedelta(days=float(val))
                return date.strftime('%Y-%m-%d')
            except:
                return str(val)
        return str(val) if val else None
    
    # Import EC Outcomes
    print("Importing EC Outcomes...")
    try:
        df_outcomes = pd.read_excel(excel_path, sheet_name='EC Outcomes')
        for _, row in df_outcomes.iterrows():
            test_id = row.get('Test ID', '')
            if pd.isna(test_id) or not test_id:
                continue
            
            try:
                # Get date columns - try multiple possible column names
                date_started = row.get('Date Started\n[mm:dd:yy]', row.get('Date Started', None))
                date_finished = row.get('Date Finished\n[mm:dd:yy]', row.get('Date Finished', None))
                
                cursor.execute('''
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
                cursor.execute('''
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
    
    conn.commit()
    conn.close()
    print("Data import complete!")


# =============================================================================
#                         CRUD OPERATIONS
# =============================================================================

# EC Experiments
def get_all_experiments(filters=None, page=None, per_page=50):
    """Get all experiments with proper connection handling, pagination, and filtering"""
    conn = None
    try:
        conn = get_db_connection()
        query = 'SELECT * FROM ec_experiments WHERE 1=1'
        count_query = 'SELECT COUNT(*) FROM ec_experiments WHERE 1=1'
        params = []
        
        if filters:
            if filters.get('priority'):
                query += ' AND priority_level = ?'
                count_query += ' AND priority_level = ?'
                params.append(filters['priority'])
            if filters.get('status'):
                if filters['status'] == 'performed' or filters['status'] == 'completed':
                    query += ' AND (performed = 1 OR status = "completed")'
                    count_query += ' AND (performed = 1 OR status = "completed")'
                elif filters['status'] == 'pending':
                    query += ' AND status = "pending"'
                    count_query += ' AND status = "pending"'
                elif filters['status'] == 'in_progress':
                    query += ' AND status = "in_progress"'
                    count_query += ' AND status = "in_progress"'
                elif filters['status'] == 'on_hold':
                    query += ' AND status = "on_hold"'
                    count_query += ' AND status = "on_hold"'
            if filters.get('assigned_to'):
                query += ' AND assigned_to = ?'
                count_query += ' AND assigned_to = ?'
                params.append(filters['assigned_to'])
            if filters.get('favorites_only'):
                query += ' AND is_favorite = 1'
                count_query += ' AND is_favorite = 1'
            if filters.get('classification'):
                query += ' AND test_classification = ?'
                count_query += ' AND test_classification = ?'
                params.append(filters['classification'])
            if filters.get('search'):
                query += ' AND (ec_id LIKE ? OR test_purpose LIKE ? OR proposer LIKE ? OR anode LIKE ? OR test_classification LIKE ?)'
                count_query += ' AND (ec_id LIKE ? OR test_purpose LIKE ? OR proposer LIKE ? OR anode LIKE ? OR test_classification LIKE ?)'
                search_term = f'%{filters["search"]}%'
                params.extend([search_term, search_term, search_term, search_term, search_term])
        
        # Get total count for pagination
        total = conn.execute(count_query, params).fetchone()[0]
        
        # Order by: favorites first, then by EC ID descending (newest first)
        query += ' ORDER BY is_favorite DESC, ec_id DESC'
        
        # Apply pagination if specified
        if page is not None and per_page:
            offset = (page - 1) * per_page
            query += f' LIMIT {per_page} OFFSET {offset}'
        
        experiments = conn.execute(query, params).fetchall()
        
        result = [dict(row) for row in experiments]
        
        # Return with pagination info if paginated
        if page is not None:
            return {
                'experiments': result,
                'total': total,
                'page': page,
                'per_page': per_page,
                'total_pages': (total + per_page - 1) // per_page
            }
        
        return result
    finally:
        if conn:
            conn.close()


def get_experiment(ec_id):
    conn = get_db_connection()
    experiment = conn.execute('SELECT * FROM ec_experiments WHERE ec_id = ?', (ec_id,)).fetchone()
    conn.close()
    return dict(experiment) if experiment else None


def add_experiment(data):
    """Add a new experiment with proper connection handling"""
    conn = None
    try:
        conn = get_db_connection()
        ec_id = get_next_ec_id()
        cursor = conn.cursor()
        cursor.execute('''
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
        # Log activity (inline to avoid nested connection)
        cursor.execute('''
            INSERT INTO activity_log (user_name, action, entity_type, entity_id, entity_name, details)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', ('System', 'created', 'experiment', ec_id, f"Experiment {ec_id}", None))
        return ec_id
    finally:
        if conn:
            conn.close()


def update_experiment(ec_id, data):
    """Update experiment with proper connection handling"""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Determine status based on performed flag
        status = data.get('status', 'pending')
        if data.get('performed'):
            status = 'completed'
        
        cursor.execute('''
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
        # Log activity inline
        cursor.execute('''
            INSERT INTO activity_log (user_name, action, entity_type, entity_id, entity_name, details)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', ('System', 'updated', 'experiment', ec_id, f"Experiment {ec_id}", None))
        return True
    finally:
        if conn:
            conn.close()


def delete_experiment(ec_id):
    """Delete experiment with proper connection handling"""
    conn = None
    try:
        conn = get_db_connection()
        conn.execute('DELETE FROM ec_experiments WHERE ec_id = ?', (ec_id,))
        # Log activity inline
        conn.execute('''
            INSERT INTO activity_log (user_name, action, entity_type, entity_id, entity_name, details)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', ('System', 'deleted', 'experiment', ec_id, f"Experiment {ec_id}", None))
        return True
    finally:
        if conn:
            conn.close()


def duplicate_experiment(ec_id):
    """Duplicate an existing experiment with new EC ID"""
    original = get_experiment(ec_id)
    if not original:
        return None
    
    # Remove id and ec_id, set new values
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


def toggle_favorite(ec_id):
    """Toggle favorite status with proper connection handling"""
    conn = None
    try:
        conn = get_db_connection()
        conn.execute('UPDATE ec_experiments SET is_favorite = NOT is_favorite WHERE ec_id = ?', (ec_id,))
        result = conn.execute('SELECT is_favorite FROM ec_experiments WHERE ec_id = ?', (ec_id,)).fetchone()
        return result['is_favorite'] if result else False
    finally:
        if conn:
            conn.close()


def bulk_update_experiments(ec_ids, updates):
    """Bulk update multiple experiments"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    for ec_id in ec_ids:
        set_clauses = []
        params = []
        for key, value in updates.items():
            set_clauses.append(f'{key} = ?')
            params.append(value)
        
        if set_clauses:
            params.append(ec_id)
            query = f'UPDATE ec_experiments SET {", ".join(set_clauses)}, updated_at = CURRENT_TIMESTAMP WHERE ec_id = ?'
            cursor.execute(query, params)
    
    conn.commit()
    conn.close()
    log_activity('System', 'bulk_updated', 'experiments', ','.join(ec_ids), f"Updated {len(ec_ids)} experiments")
    return True


def bulk_delete_experiments(ec_ids):
    """Bulk delete multiple experiments"""
    conn = get_db_connection()
    placeholders = ','.join(['?' for _ in ec_ids])
    conn.execute(f'DELETE FROM ec_experiments WHERE ec_id IN ({placeholders})', ec_ids)
    conn.commit()
    conn.close()
    log_activity('System', 'bulk_deleted', 'experiments', ','.join(ec_ids), f"Deleted {len(ec_ids)} experiments")
    return True


# EC Outcomes
def get_all_outcomes():
    conn = get_db_connection()
    outcomes = conn.execute('SELECT * FROM ec_outcomes ORDER BY id DESC').fetchall()
    conn.close()
    return [dict(row) for row in outcomes]


def add_outcome(data):
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO ec_outcomes 
            (test_id, ec_id, test_purpose, operator, date_started, date_finished,
             outcome, implications, data_location)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data.get('test_id'), data.get('ec_id'), data.get('test_purpose'),
            data.get('operator'), data.get('date_started'), data.get('date_finished'),
            data.get('outcome'), data.get('implications'), data.get('data_location')
        ))
        outcome_id = cursor.lastrowid
        log_activity('System', 'created', 'outcome', str(outcome_id), f"Outcome for {data.get('ec_id', 'N/A')}")
        return outcome_id
    finally:
        if conn:
            conn.close()


def update_outcome(outcome_id, data):
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
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
    conn = None
    try:
        conn = get_db_connection()
        conn.execute('DELETE FROM ec_outcomes WHERE id = ?', (outcome_id,))
        return True
    finally:
        if conn:
            conn.close()


# Intake Queue
def get_all_intake():
    conn = get_db_connection()
    intake = conn.execute('SELECT * FROM intake_queue ORDER BY id DESC').fetchall()
    conn.close()
    return [dict(row) for row in intake]


def add_intake(data):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
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
    conn.commit()
    intake_id = cursor.lastrowid
    conn.close()
    log_activity('System', 'created', 'intake', str(intake_id), f"Intake request from {data.get('proposer', 'N/A')}")
    return intake_id


def promote_intake_to_experiment(intake_id):
    """Move an intake queue item to EC Experiments"""
    conn = get_db_connection()
    intake = conn.execute('SELECT * FROM intake_queue WHERE id = ?', (intake_id,)).fetchone()
    
    if not intake:
        conn.close()
        return None
    
    intake_dict = dict(intake)
    ec_id = add_experiment({
        'proposer': intake_dict.get('proposer'),
        'test_purpose': intake_dict.get('test_purpose'),
        'test_classification': intake_dict.get('test_classification'),
        'anode': intake_dict.get('anode'),
        'separator': intake_dict.get('separator'),
        'cathode': intake_dict.get('cathode'),
        'anode_current_collector': intake_dict.get('anode_current_collector'),
        'cathode_current_collector': intake_dict.get('cathode_current_collector'),
        'anolyte': intake_dict.get('anolyte'),
        'catholyte': intake_dict.get('catholyte'),
        'temperature': intake_dict.get('temperature'),
        'dp_psi': intake_dict.get('dp_psi'),
        'priority_level': intake_dict.get('proposed_priority'),
        'due_date': intake_dict.get('due_date'),
        'notes': intake_dict.get('notes'),
    })
    
    conn.execute('DELETE FROM intake_queue WHERE id = ?', (intake_id,))
    conn.commit()
    conn.close()
    
    log_activity('System', 'promoted', 'intake', str(intake_id), f"Promoted to {ec_id}")
    return ec_id


# Templates
def get_all_templates():
    conn = get_db_connection()
    templates = conn.execute('SELECT * FROM templates ORDER BY name').fetchall()
    conn.close()
    return [dict(row) for row in templates]


def get_template(template_id):
    conn = get_db_connection()
    template = conn.execute('SELECT * FROM templates WHERE id = ?', (template_id,)).fetchone()
    conn.close()
    return dict(template) if template else None


def add_template(data):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
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
    conn.commit()
    template_id = cursor.lastrowid
    conn.close()
    return template_id


def delete_template(template_id):
    conn = get_db_connection()
    conn.execute('DELETE FROM templates WHERE id = ?', (template_id,))
    conn.commit()
    conn.close()
    return True


def create_experiment_from_template(template_id, additional_data=None):
    """Create a new experiment from a template"""
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


# Comments
def get_comments(ec_id):
    conn = get_db_connection()
    comments = conn.execute(
        'SELECT * FROM comments WHERE ec_id = ? ORDER BY created_at DESC', (ec_id,)
    ).fetchall()
    conn.close()
    return [dict(row) for row in comments]


def add_comment(ec_id, user_name, comment_text):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO comments (ec_id, user_name, comment_text) VALUES (?, ?, ?)',
        (ec_id, user_name, comment_text)
    )
    conn.commit()
    comment_id = cursor.lastrowid
    conn.close()
    log_activity(user_name, 'commented', 'experiment', ec_id, comment_text[:50])
    return comment_id


def delete_comment(comment_id):
    conn = get_db_connection()
    conn.execute('DELETE FROM comments WHERE id = ?', (comment_id,))
    conn.commit()
    conn.close()
    return True


# Activity Log
def log_activity(user_name, action, entity_type, entity_id, entity_name, details=None):
    """Log activity with proper connection handling"""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO activity_log (user_name, action, entity_type, entity_id, entity_name, details)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_name, action, entity_type, entity_id, entity_name, details))
    finally:
        if conn:
            conn.close()


def get_activity_log(limit=50, entity_type=None, entity_id=None):
    """Get activity log with proper connection handling"""
    conn = None
    try:
        conn = get_db_connection()
        query = 'SELECT * FROM activity_log WHERE 1=1'
        params = []
        
        if entity_type:
            query += ' AND entity_type = ?'
            params.append(entity_type)
        if entity_id:
            query += ' AND entity_id = ?'
            params.append(entity_id)
        
        query += ' ORDER BY created_at DESC LIMIT ?'
        params.append(limit)
        
        activities = conn.execute(query, params).fetchall()
        return [dict(row) for row in activities]
    finally:
        if conn:
            conn.close()


# User Preferences
def get_user_preferences(user_name):
    conn = get_db_connection()
    prefs = conn.execute('SELECT * FROM user_preferences WHERE user_name = ?', (user_name,)).fetchone()
    conn.close()
    
    if prefs:
        return dict(prefs)
    return {
        'theme': 'dark',
        'visible_columns': None,
        'column_order': None,
        'default_view': 'table'
    }


def update_user_preferences(user_name, preferences):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT OR REPLACE INTO user_preferences 
        (user_name, theme, visible_columns, column_order, default_view, updated_at)
        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    ''', (
        user_name, preferences.get('theme', 'dark'),
        preferences.get('visible_columns'), preferences.get('column_order'),
        preferences.get('default_view', 'table')
    ))
    conn.commit()
    conn.close()
    return True


# Notifications
def get_notifications(user_name, unread_only=False):
    conn = get_db_connection()
    query = 'SELECT * FROM notifications WHERE (user_name = ? OR user_name IS NULL)'
    params = [user_name]
    
    if unread_only:
        query += ' AND is_read = 0'
    
    query += ' ORDER BY created_at DESC LIMIT 50'
    notifications = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(row) for row in notifications]


def add_notification(user_name, title, message, notif_type='info', ec_id=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO notifications (user_name, title, message, type, ec_id)
        VALUES (?, ?, ?, ?, ?)
    ''', (user_name, title, message, notif_type, ec_id))
    conn.commit()
    notif_id = cursor.lastrowid
    conn.close()
    return notif_id


def mark_notification_read(notif_id):
    conn = get_db_connection()
    conn.execute('UPDATE notifications SET is_read = 1 WHERE id = ?', (notif_id,))
    conn.commit()
    conn.close()
    return True


def mark_all_notifications_read(user_name):
    conn = get_db_connection()
    conn.execute('UPDATE notifications SET is_read = 1 WHERE user_name = ? OR user_name IS NULL', (user_name,))
    conn.commit()
    conn.close()
    return True


# Utility functions
def get_dropdown_options(category):
    """Get dropdown options for a category"""
    conn = None
    try:
        conn = get_db_connection()
        options = conn.execute(
            'SELECT option_value, is_default FROM dropdown_options WHERE category = ? ORDER BY display_order',
            (category,)
        ).fetchall()
        return [{'value': row['option_value'], 'is_default': row['is_default']} for row in options]
    finally:
        if conn:
            conn.close()


def add_dropdown_option(category, value):
    """Add a new custom option to a dropdown category"""
    if not value or not value.strip():
        return False
    
    conn = None
    try:
        conn = get_db_connection()
        # Check if already exists
        existing = conn.execute(
            'SELECT id FROM dropdown_options WHERE category = ? AND option_value = ?',
            (category, value.strip())
        ).fetchone()
        
        if existing:
            return True  # Already exists, no need to add
        
        # Get max display_order
        max_order = conn.execute(
            'SELECT MAX(display_order) FROM dropdown_options WHERE category = ?',
            (category,)
        ).fetchone()[0] or 0
        
        conn.execute(
            'INSERT INTO dropdown_options (category, option_value, display_order, is_default) VALUES (?, ?, ?, ?)',
            (category, value.strip(), max_order + 1, False)
        )
        return True
    finally:
        if conn:
            conn.close()


def get_standard_conditions():
    conn = get_db_connection()
    conditions = conn.execute('SELECT * FROM standard_conditions').fetchall()
    conn.close()
    return {row['field_name']: row['default_value'] for row in conditions}


def get_overview():
    conn = get_db_connection()
    overview = conn.execute('SELECT * FROM overview').fetchall()
    conn.close()
    return {row['field_name']: row['field_value'] for row in overview}


def get_statistics():
    """Get dashboard statistics with proper connection handling"""
    conn = None
    try:
        conn = get_db_connection()
        
        stats = {
            'total_experiments': conn.execute('SELECT COUNT(*) FROM ec_experiments').fetchone()[0],
            'performed_experiments': conn.execute('SELECT COUNT(*) FROM ec_experiments WHERE performed = 1').fetchone()[0],
            'pending_experiments': conn.execute("SELECT COUNT(*) FROM ec_experiments WHERE status = 'pending' OR (status IS NULL AND performed = 0)").fetchone()[0],
            'in_progress_experiments': conn.execute("SELECT COUNT(*) FROM ec_experiments WHERE status = 'in_progress'").fetchone()[0],
            'completed_experiments': conn.execute("SELECT COUNT(*) FROM ec_experiments WHERE performed = 1 OR status = 'completed'").fetchone()[0],
            'intake_queue_count': conn.execute('SELECT COUNT(*) FROM intake_queue').fetchone()[0],
            'outcomes_count': conn.execute('SELECT COUNT(*) FROM ec_outcomes').fetchone()[0],
            'high_priority': conn.execute("SELECT COUNT(*) FROM ec_experiments WHERE priority_level = 'H'").fetchone()[0],
            'medium_priority': conn.execute("SELECT COUNT(*) FROM ec_experiments WHERE priority_level = 'M'").fetchone()[0],
            'low_priority': conn.execute("SELECT COUNT(*) FROM ec_experiments WHERE priority_level = 'L'").fetchone()[0],
            'favorites_count': conn.execute('SELECT COUNT(*) FROM ec_experiments WHERE is_favorite = 1').fetchone()[0],
            'templates_count': conn.execute('SELECT COUNT(*) FROM templates').fetchone()[0],
        }
        
        # Due date statistics
        today = datetime.now().date()
        week_later = today + timedelta(days=7)
        
        stats['overdue'] = 0
        stats['due_today'] = 0
        stats['due_this_week'] = 0
        
        experiments = conn.execute('SELECT due_date FROM ec_experiments WHERE performed = 0 AND due_date IS NOT NULL').fetchall()
        for exp in experiments:
            try:
                due = datetime.strptime(str(exp['due_date'])[:10], '%Y-%m-%d').date()
                if due < today:
                    stats['overdue'] += 1
                elif due == today:
                    stats['due_today'] += 1
                elif due <= week_later:
                    stats['due_this_week'] += 1
            except:
                pass
        
        # Classification breakdown
        classifications = conn.execute('''
            SELECT test_classification, COUNT(*) as count 
            FROM ec_experiments 
            WHERE test_classification IS NOT NULL 
            GROUP BY test_classification
        ''').fetchall()
        stats['by_classification'] = {row['test_classification']: row['count'] for row in classifications}
        
        # Monthly trend
        monthly = conn.execute('''
            SELECT strftime('%Y-%m', created_at) as month, COUNT(*) as count 
            FROM ec_experiments 
            GROUP BY month 
            ORDER BY month DESC 
            LIMIT 6
        ''').fetchall()
        stats['monthly_trend'] = [{'month': row['month'], 'count': row['count']} for row in monthly]
        
        return stats
    finally:
        if conn:
            conn.close()


def get_calendar_data(year, month):
    """Get experiments for calendar view"""
    conn = get_db_connection()
    
    # Get experiments with due dates in the specified month
    experiments = conn.execute('''
        SELECT ec_id, test_purpose, due_date, priority_level, performed, status
        FROM ec_experiments
        WHERE due_date LIKE ?
    ''', (f'{year}-{month:02d}%',)).fetchall()
    
    conn.close()
    return [dict(row) for row in experiments]


def get_kanban_data():
    """Get experiments organized by status for kanban view with proper connection handling"""
    conn = None
    try:
        conn = get_db_connection()
        
        # Get experiments grouped by status
        pending = conn.execute('''
            SELECT ec_id, test_purpose, priority_level, proposer, due_date, is_favorite
            FROM ec_experiments WHERE status = 'pending' OR (status IS NULL AND performed = 0)
            ORDER BY CASE priority_level WHEN 'H' THEN 1 WHEN 'M' THEN 2 WHEN 'L' THEN 3 ELSE 4 END
        ''').fetchall()
        
        in_progress = conn.execute('''
            SELECT ec_id, test_purpose, priority_level, proposer, due_date, is_favorite
            FROM ec_experiments WHERE status = 'in_progress'
            ORDER BY CASE priority_level WHEN 'H' THEN 1 WHEN 'M' THEN 2 WHEN 'L' THEN 3 ELSE 4 END
        ''').fetchall()
        
        completed = conn.execute('''
            SELECT ec_id, test_purpose, priority_level, proposer, due_date, is_favorite
            FROM ec_experiments WHERE status = 'completed' OR performed = 1
            ORDER BY updated_at DESC
            LIMIT 20
        ''').fetchall()
        
        on_hold = conn.execute('''
            SELECT ec_id, test_purpose, priority_level, proposer, due_date, is_favorite
            FROM ec_experiments WHERE status = 'on_hold'
            ORDER BY CASE priority_level WHEN 'H' THEN 1 WHEN 'M' THEN 2 WHEN 'L' THEN 3 ELSE 4 END
        ''').fetchall()
        
        return {
            'pending': [dict(row) for row in pending],
            'in_progress': [dict(row) for row in in_progress],
            'completed': [dict(row) for row in completed],
            'on_hold': [dict(row) for row in on_hold]
        }
    finally:
        if conn:
            conn.close()


def update_experiment_status(ec_id, new_status):
    """Update experiment status (for kanban drag-drop) with proper connection handling"""
    conn = None
    try:
        conn = get_db_connection()
        
        performed = 1 if new_status == 'completed' else 0
        
        conn.execute('''
            UPDATE ec_experiments 
            SET status = ?, performed = ?, updated_at = CURRENT_TIMESTAMP 
            WHERE ec_id = ?
        ''', (new_status, performed, ec_id))
        
        # Log activity inline
        conn.execute('''
            INSERT INTO activity_log (user_name, action, entity_type, entity_id, entity_name, details)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', ('System', 'status_changed', 'experiment', ec_id, f"Status changed to {new_status}", None))
        
        return True
    finally:
        if conn:
            conn.close()


if __name__ == '__main__':
    # Initialize database
    print("Initializing DOE Planner database...")
    init_database()
    populate_dropdown_options()
    populate_standard_conditions()
    populate_overview()
    populate_default_templates()
    
    # Import Excel data if file exists
    excel_path = r'C:\Users\SaiSarbareesh\OneDrive - eleryc inc\Desktop\Electrochemical Testing Workflow.xlsx'
    if Path(excel_path).exists():
        import_excel_data(excel_path)
    
    print("\nDatabase setup complete!")
    print(f"Statistics: {get_statistics()}")
