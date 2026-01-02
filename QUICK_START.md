# Quick Start Guide - Eleryc Experiment Data Viewer

## How to Start the Server

### Method 1: Using the Batch File (Easiest)
1. Navigate to the `UI` folder
2. Double-click `START_AUTOMATED_SERVER.bat`
3. The server will start automatically on **http://localhost:5000**

### Method 2: Using Command Line
1. Open PowerShell or Command Prompt
2. Navigate to the UI folder:
   ```powershell
   cd "C:\Users\SaiSarbareesh\OneDrive - eleryc inc\Desktop\GUI\UI"
   ```
3. Run the Flask app:
   ```powershell
   python app.py
   ```
4. Open your browser and go to: **http://localhost:5000**

### Method 3: Using Python Directly
```powershell
python "C:\Users\SaiSarbareesh\OneDrive - eleryc inc\Desktop\GUI\UI\app.py"
```

## Troubleshooting

### If the server won't start:

1. **Check if port 5000 is already in use:**
   ```powershell
   netstat -ano | Select-String ":5000"
   ```
   If something is using port 5000, stop that process first.

2. **Check if Python and Flask are installed:**
   ```powershell
   python --version
   pip list | Select-String -Pattern "flask"
   ```

3. **Install dependencies if needed:**
   ```powershell
   cd "C:\Users\SaiSarbareesh\OneDrive - eleryc inc\Desktop\GUI\UI"
   pip install -r requirements.txt
   ```

### If files are missing:
- Make sure you're in the `UI` folder
- The `app.py` file should be in the `UI` directory
- The `templates` folder should contain `index.html` and `drt_analysis.html`
- The `drt_tools` folder should exist in the `UI` directory

## Server Location
- **URL:** http://localhost:5000
- **Main folder:** `C:\Users\SaiSarbareesh\OneDrive - eleryc inc\Desktop\GUI\UI`
- **Main file:** `UI\app.py`

## To Stop the Server
Press `Ctrl+C` in the terminal where the server is running.

