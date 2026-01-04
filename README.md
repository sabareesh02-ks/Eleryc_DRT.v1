# Eleryc Data Analysis Platform

A comprehensive web-based platform for electrochemical data analysis, experiment planning, and visualization.

![Eleryc Logo](assets/eleryc-logo.png)

## 🚀 Features

### 1. **DRT Analysis**
- Distribution of Relaxation Times (DRT) calculation from EIS data
- Interactive Nyquist, Bode Magnitude, and Phase plots
- Multi-current comparison with overlay plots
- CSV export of results

### 2. **M-Series Data Viewer**
- View and compare M-Series experiment data
- DRT and EIS overlay comparisons
- Organized by experiment folders

### 3. **Raw Data Reader**
- Upload and analyze CSV/Excel files
- Auto-detect column types
- Time-based plots, polarization curves
- EIS analysis by current level
- Multi-file comparison

### 4. **DOE Planner**
- Design of Experiments planning and tracking
- EC Experiments management with CRUD operations
- Intake Queue for new experiment requests
- EC Outcomes tracking
- Calendar view for scheduling
- Analytics and charts
- Import/Export to Excel
- Database backup and restore

## 📁 Project Structure

```
Eleryc_DRT.v1/
├── app.py                 # Main Flask application
├── config.py              # Configuration settings
├── doe_database.py        # DOE Planner database module
│
├── index.html             # Landing page
├── login.html             # Login page
├── drt_analysis.html      # DRT Analysis page
├── raw_data_reader.html   # Raw Data Reader page
├── doe_planner.html       # DOE Planner page
│
├── assets/                # Images and logos
│   └── eleryc-logo.png
│
├── data/                  # Experiment data
│   ├── M-Series/          # M-Series experiment data
│   └── Duration-Tests/    # Duration test data
│
├── drt_tools/             # DRT calculation modules
│   ├── basics.py
│   ├── runs.py
│   ├── nearest_PD.py
│   └── parameter_selection.py
│
├── scripts/               # Utility scripts
├── docs/                  # Documentation
│
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker configuration
├── render.yaml            # Render deployment config
└── README.md              # This file
```

## 🛠️ Setup

### Local Development

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sabareesh02-ks/Eleryc_DRT.v1.git
   cd Eleryc_DRT.v1
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   python app.py
   ```

5. **Open in browser:**
   ```
   http://localhost:8080
   ```

### Render Deployment

The app is configured for deployment on Render:
- Push to `main` branch triggers auto-deploy
- Environment variable `DATABASE_URL` for PostgreSQL (optional)


## 📊 Supported File Formats

- CSV files (.csv)
- Excel files (.xls, .xlsx)


## 📝 License

© 2024-2026 Eleryc Inc. All rights reserved.

## 👥 Contributors

- Eleryc Engineering Team

