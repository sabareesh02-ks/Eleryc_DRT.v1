# Eleryc Experiment Viewer - Automated Comparison System

<div align="center">

![Status](https://img.shields.io/badge/Status-Ready-brightgreen)
![Version](https://img.shields.io/badge/Version-2.0-blue)
![Automation](https://img.shields.io/badge/Automation-100%25-orange)

**Professional DRT & Nyquist Plot Comparison Tool**

[Quick Start](#-quick-start) • [Features](#-features) • [Architecture](#-architecture) • [Troubleshooting](#-troubleshooting)

</div>

---

## 🚀 Quick Start

### Option 1: First Time (Installs Everything)
```batch
Double-click: FIRST_TIME_SETUP.bat
```

### Option 2: Already Set Up
```batch
Double-click: START_AUTOMATED_SERVER.bat
```

### Option 3: Command Line
```bash
# Install once
pip install -r requirements.txt

# Start server
python app.py

# Open browser
http://localhost:5000
```

**That's it!** You're ready to compare experiments! 🎉

---

## ✨ Features

### 🔄 Fully Automated Comparison
- Select experiments and conditions in the UI
- Click one button
- Plots appear automatically (2-5 seconds)
- No manual commands, no copy-paste!

### 📊 Professional Plots
- **DRT Plots**: τ vs γ with log scale
- **Nyquist Plots**: Z_re vs -Z_im with equal axes
- Multiple experiments overlaid with different colors
- High resolution (300 DPI) PNG output
- Click to zoom in browser

### 🎨 Modern UI
- Eleryc branding and colors
- Responsive design
- Real-time loading indicators
- Clear error messages with troubleshooting
- Search and filter experiments

### 🔧 Flexible Comparison
- Compare any number of experiments
- Same or different conditions
- Mix and match freely
- Results saved automatically

---

## 📁 Project Structure

```
UI/
├── 🚀 STARTUP SCRIPTS
│   ├── FIRST_TIME_SETUP.bat         # Install + Start
│   └── START_AUTOMATED_SERVER.bat   # Quick Start
│
├── 📱 WEB APPLICATION
│   ├── app.py                       # Flask backend (NEW!)
│   ├── index.html                   # Frontend UI
│   └── requirements.txt             # Dependencies
│
├── 📊 DATA & OUTPUT
│   ├── separated_by_current/        # CSV data input
│   │   └── [experiment]/
│   │       ├── drt/                 # DRT CSV files
│   │       └── eis/                 # Nyquist CSV files
│   └── outputs/
│       ├── [experiment]/            # Individual plots
│       └── comparisons/             # Comparison plots (NEW!)
│
├── 🎨 ASSETS
│   └── eleryc-logo.png              # Company logo
│
├── 📚 DOCUMENTATION
│   ├── README_AUTOMATED.md          # This file
│   ├── AUTOMATED_SETUP_GUIDE.txt    # Full guide
│   ├── QUICK_START.txt              # Quick reference
│   └── WHATS_NEW_AUTOMATED.md       # Change log
│
└── 🛠️ UTILITIES
    ├── generate_comparison_overlay.py  # Manual fallback
    ├── copy_and_generate_plots.py      # Batch processor
    └── cleanup_outputs.py              # Cleanup tool
```

---

## 🏗️ Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        USER BROWSER                          │
│  http://localhost:5000                                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  index.html (Frontend)                              │   │
│  │  • Experiment list                                   │   │
│  │  • Comparison tab                                    │   │
│  │  • JavaScript UI logic                               │   │
│  └──────────────────┬──────────────────────────────────┘   │
└────────────────────┼──────────────────────────────────────┘
                     │
                     │ HTTP POST
                     │ /api/generate-comparison
                     │
┌────────────────────▼──────────────────────────────────────┐
│                  FLASK BACKEND SERVER                       │
│  Port: 5000                                                 │
│  ┌─────────────────────────────────────────────────────┐  │
│  │  app.py                                              │  │
│  │  • REST API endpoint                                 │  │
│  │  • CSV file reading                                  │  │
│  │  • Matplotlib plot generation                        │  │
│  │  • Static file serving                               │  │
│  └──────────────────┬──────────────────────────────────┘  │
└────────────────────┼──────────────────────────────────────┘
                     │
                     │ Read/Write
                     │
┌────────────────────▼──────────────────────────────────────┐
│                    FILE SYSTEM                              │
│  ┌────────────────────┐      ┌─────────────────────────┐  │
│  │ separated_by_      │      │ outputs/comparisons/    │  │
│  │ current/           │      │ • Generated PNGs        │  │
│  │ • CSV data input   │─────▶│ • Saved permanently     │  │
│  └────────────────────┘      └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
1. User selects experiments + conditions
2. Clicks "Show Comparison"
         ↓
3. JavaScript sends POST request:
   {
     "selections": [
       {"experiment": "M1r_...", "condition": "4A"},
       {"experiment": "M1-r3_...", "condition": "4A"}
     ]
   }
         ↓
4. Flask app.py:
   • Locates CSV files
   • Reads data (pandas)
   • Creates plots (matplotlib)
   • Saves PNGs
         ↓
5. Returns:
   {
     "success": true,
     "plots": {
       "drt": "outputs/comparisons/..._DRT.png",
       "nyquist": "outputs/comparisons/..._Nyquist.png"
     }
   }
         ↓
6. Browser displays plots automatically!
```

### Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Backend | Flask 3.0 | Web server & API |
| Data Processing | Pandas 2.1 | CSV reading |
| Plotting | Matplotlib 3.8 | Graph generation |
| Frontend | HTML5/CSS3/JS | User interface |
| API | RESTful JSON | Backend communication |

---

## 📖 Usage Guide

### 1. View Mode (Individual Experiments)
1. Default view when opening the app
2. Click any experiment to see its plots
3. Scroll through DRT and Nyquist plots
4. Click plots to zoom

### 2. Compare Mode (Automated Overlay)

#### Step-by-Step:

**A. Switch to Compare Mode**
- Click "🔀 Compare" tab at top of sidebar

**B. Select Experiments**
- ☑️ Check boxes next to experiments you want
- You can select 2, 3, 4, or more!

**C. Choose Conditions**
- For each checked experiment, select a condition:
  - 4A, 8A, 12A, 16A, 20A, 24A
- Can be same condition for all, or different!

**D. Generate Comparison**
- Click "✨ Show Comparison"
- Loading spinner appears (⚙️)
- Wait 2-5 seconds
- Plots appear automatically! ✅

**E. View Results**
- DRT comparison plot (top)
- Nyquist comparison plot (bottom)
- Each experiment has different color
- Legend shows "Experiment @ Condition"
- Click any plot to open full size

#### Example Scenarios:

**Scenario 1: Compare Two Batches**
```
✓ M1r_M1r-5psi-07172025 @ 4A
✓ M1-r2_M1r2-5psi-07172025 @ 4A
→ See difference between batches at same condition
```

**Scenario 2: Track Condition Changes**
```
✓ M1r @ 4A
✓ M1r @ 8A
✓ M1r @ 12A
→ See how one experiment behaves across currents
```

**Scenario 3: Cross Comparison**
```
✓ M1r @ 4A
✓ M5 @ 16A
→ Any experiment, any condition!
```

---

## 🎨 Plot Specifications

### DRT Plots

**Input Data:**
- File: `separated_by_current/[exp]/drt/[exp]_[condition]_DRT.csv`
- Format: First 2 rows are headers (L, R values), skip them
- Columns: `tau`, `gamma`

**Output Plot:**
- X-axis: τ (seconds) - **log scale**
- Y-axis: Y (Ω)
- Style: Lines with circle markers
- Grid: Semi-transparent
- Legend: Upper right corner

**Example CSV:**
```csv
L,0
R,0.018
tau,gamma
0.0001,0.5
0.001,1.2
0.01,0.8
...
```

### Nyquist Plots

**Input Data:**
- File: `separated_by_current/[exp]/eis/[exp]_[condition]_EIS.csv`
- Format: Standard CSV with headers
- Columns: `mu_Z_re`, `mu_Z_im`

**Output Plot:**
- X-axis: Z_re (Ω)
- Y-axis: -Z_im (Ω) - **negative of imaginary**
- Style: Lines with circle markers
- Axes: **Equal scaling** for proper semicircles
- Grid: Semi-transparent
- Legend: Upper right corner

**Example CSV:**
```csv
mu_Z_re,mu_Z_im
0.15,0.05
0.20,0.10
0.25,0.08
...
```

### Color Palette

Experiments are assigned colors in order:

1. 🔴 Red (#e41a1c)
2. 🔵 Blue (#377eb8)
3. 🟢 Green (#4daf4a)
4. 🟣 Purple (#984ea3)
5. 🟠 Orange (#ff7f00)
6. 🟡 Yellow (#ffff33)
7. 🟤 Brown (#a65628)
8. 🎀 Pink (#f781bf)

Colors cycle if more than 8 experiments selected.

---

## ⚙️ Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)
- Modern web browser (Chrome, Firefox, Edge)

### Automated Installation

**Windows:**
```batch
Double-click: FIRST_TIME_SETUP.bat
```

**Or manually:**
```bash
pip install -r requirements.txt
```

### Dependencies

```
flask==3.0.0        # Web framework
pandas==2.1.0       # Data processing
matplotlib==3.8.0   # Plotting
openpyxl==3.1.2     # Excel support
```

---

## 🚦 Running the Application

### Option 1: Batch Script (Easiest)
```batch
Double-click: START_AUTOMATED_SERVER.bat
```

### Option 2: Command Line
```bash
python app.py
```

### Option 3: Custom Port
Edit `app.py` line 213:
```python
app.run(debug=True, port=5001, host='0.0.0.0')  # Changed from 5000
```

### Server Output

You should see:
```
================================================================================
  ELERYC EXPERIMENT VIEWER - Backend Server
================================================================================

  Server running at: http://localhost:5000
  API endpoint: http://localhost:5000/api/generate-comparison

  Press Ctrl+C to stop the server

================================================================================
```

### Accessing the UI

Open your browser and navigate to:
```
http://localhost:5000
```

---

## 🔍 Troubleshooting

### Server Won't Start

**Problem:** Port 5000 already in use

**Solution:**
1. Find what's using port 5000:
   ```powershell
   netstat -ano | findstr :5000
   ```
2. Kill that process or change port in `app.py`

**Problem:** `ModuleNotFoundError: No module named 'flask'`

**Solution:**
```bash
pip install -r requirements.txt
```

### Connection Errors

**Problem:** "Failed to fetch" or "Connection refused"

**Solution:**
- Make sure Flask server is running (`python app.py`)
- Check browser console (F12) for errors
- Verify URL is `http://localhost:5000` (not 8000)

### No Plots Generated

**Problem:** "No CSV files found for selected experiments"

**Solution:**
1. Check folder structure:
   ```
   separated_by_current/
   └── M1r_M1r-5psi-07172025/
       ├── drt/
       │   └── M1r_4A_DRT.csv  ✓
       └── eis/
           └── M1r_4A_EIS.csv  ✓
   ```
2. Verify experiment name matches folder name exactly
3. Verify CSV files contain the condition (e.g., `_4A_`)

### Wrong Plot Data

**Problem:** DRT plot looks exponential instead of peaked

**Solution:**
- Check CSV has 2 header rows to skip (L, R values)
- Verify columns are named `tau` and `gamma`

**Problem:** Nyquist plot axes don't match

**Solution:**
- Check columns are `mu_Z_re` and `mu_Z_im`
- Verify no preprocessing removed inductance

### Browser Issues

**Problem:** Plots don't appear after "Success" message

**Solution:**
- Hard refresh: Ctrl+F5 (Windows) or Cmd+Shift+R (Mac)
- Clear browser cache
- Check browser console (F12) for errors

---

## 📊 API Reference

### Generate Comparison

**Endpoint:** `POST /api/generate-comparison`

**Request:**
```json
{
  "selections": [
    {
      "experiment": "M1r_M1r-5psi-07172025",
      "condition": "4A"
    },
    {
      "experiment": "M1-r3_M1r3-5psi-07182025",
      "condition": "4A"
    }
  ]
}
```

**Response (Success):**
```json
{
  "success": true,
  "plots": {
    "drt": "outputs/comparisons/M1r_..._4A_vs_M1-r3_..._4A_DRT.png",
    "nyquist": "outputs/comparisons/M1r_..._4A_vs_M1-r3_..._4A_Nyquist.png"
  },
  "comparison_name": "M1r_M1r-5psi-07172025_4A_vs_M1-r3_M1r3-5psi-07182025_4A"
}
```

**Response (Error):**
```json
{
  "success": false,
  "error": "No CSV files found for selected experiments"
}
```

**Status Codes:**
- 200: Success
- 400: Bad request (no selections)
- 404: CSV files not found
- 500: Internal server error

---

## 🤝 Contributing

### Adding New Experiments

1. Create folder: `separated_by_current/[Experiment_Name]/`
2. Add subfolders: `drt/` and `eis/`
3. Add CSV files with naming pattern: `[Exp]_[Condition]_DRT.csv`
4. Refresh the UI - experiments appear automatically!

### Customizing Colors

Edit `app.py` line 13:
```python
COLORS = ['#e41a1c', '#377eb8', ...]  # Add your colors
```

### Customizing Conditions

Edit `index.html` line 746:
```javascript
const standardConditions = ['4A', '8A', '12A', ...];  # Add your conditions
```

---

## 📝 File Descriptions

### Core Files

- **`app.py`** - Flask backend server, API endpoints, plot generation
- **`index.html`** - Frontend UI, JavaScript logic, styling
- **`requirements.txt`** - Python package dependencies

### Startup Scripts

- **`FIRST_TIME_SETUP.bat`** - Installs packages and starts server
- **`START_AUTOMATED_SERVER.bat`** - Quick start for daily use

### Documentation

- **`README_AUTOMATED.md`** - This comprehensive guide
- **`AUTOMATED_SETUP_GUIDE.txt`** - Detailed setup instructions
- **`QUICK_START.txt`** - Ultra-quick reference
- **`WHATS_NEW_AUTOMATED.md`** - Change log and new features
- **`COMPARISON_USAGE_GUIDE.txt`** - Manual comparison method

### Utilities

- **`generate_comparison_overlay.py`** - Manual CLI comparison tool
- **`copy_and_generate_plots.py`** - Batch plot generator
- **`cleanup_outputs.py`** - Remove generated plots
- **`server.py`** - Old simple HTTP server (deprecated)

---

## 🎓 Tips & Best Practices

### Performance

- First comparison takes 3-5 seconds
- Subsequent comparisons are faster (cached data)
- Larger CSV files (>1000 rows) may take longer

### Organization

- Name experiments consistently: `[Batch]_[Description]_[Date]`
- Use standard conditions: 4A, 8A, 12A, etc.
- Keep CSV files organized in drt/ and eis/ subfolders

### Workflow

1. Generate individual experiment plots first (copy_and_generate_plots.py)
2. Use View mode to inspect individual experiments
3. Switch to Compare mode for side-by-side analysis
4. Save comparison plots for reports (already saved to outputs/comparisons/)

---

## 📫 Support

### Getting Help

1. Read **`AUTOMATED_SETUP_GUIDE.txt`** for detailed instructions
2. Check **Troubleshooting** section above
3. Review **`QUICK_START.txt`** for quick answers

### Common Questions

**Q: Can I compare more than 2 experiments?**  
A: Yes! Select as many as you want. Each gets a different color.

**Q: Can I mix different conditions?**  
A: Absolutely! Compare M1 @ 4A vs M5 @ 16A if you want.

**Q: Where are plots saved?**  
A: `outputs/comparisons/` folder. They persist until you delete them.

**Q: Can I use the old manual method?**  
A: Yes! `generate_comparison_overlay.py` still works for scripting.

**Q: Does this work offline?**  
A: Yes! Everything runs locally, no internet needed.

---

## 📈 Future Enhancements

Potential features for future versions:

- [ ] Export comparison data to Excel
- [ ] Custom color selection in UI
- [ ] Save favorite comparisons
- [ ] Batch comparison generation
- [ ] PDF report generation
- [ ] Statistical analysis overlay
- [ ] Interactive Plotly comparisons

---

## 📄 License

Internal tool for Eleryc use.

---

## 🙏 Acknowledgments

Made with ❤️ for the Eleryc team.

**Enjoy your fully automated comparison system!** 🎉

---

<div align="center">

**Version 2.0** | **November 2025** | **Eleryc Inc.**

[Back to Top](#eleryc-experiment-viewer---automated-comparison-system)

</div>



