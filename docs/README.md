# Eleryc Experiment Viewer

**Professional web application for viewing Distribution of Relaxation Times (DRT) and Nyquist plots**

<div align="center">
  <img src="eleryc-logo.svg" alt="Eleryc Logo" width="200"/>
</div>

---

## 📋 Features

- 🔬 View DRT and Nyquist plots for 64+ experiments
- 🔍 Real-time search and filter experiments by name
- 📱 Fully responsive design (works on desktop, tablet, and mobile)
- 🎨 Professional Eleryc-branded interface
- ⚡ Fast, lightweight, and no installation required
- 🎯 Experiments displayed in chronological order
- ✨ Interactive plots with hover tooltips (optional)

---

## 🚀 How to Use

### Method 1: Using Python Server (Recommended)

1. Open terminal/PowerShell in the UI folder
2. Run:
   ```bash
   python server.py
   ```
3. Browser will automatically open to `http://localhost:8000`
4. Press `Ctrl+C` to stop the server when done

### Method 2: Direct Browser Access

Simply **double-click** the `index.html` file - it will open directly in your default browser!

---

## 📤 How to Share with Others

### Option 1: Share as ZIP File (Easiest)

1. **Select these items:**
   - `index.html`
   - `outputs/` folder (with all experiment data)
   - `README.md` (optional)
   - `server.py` (optional, for Python server method)

2. **Create ZIP:**
   - Right-click → "Send to" → "Compressed (zipped) folder"
   - Name it: `Experiment_Viewer.zip`

3. **Share:**
   - Email, USB drive, cloud storage (Google Drive, OneDrive, Dropbox)
   
4. **Recipient Instructions:**
   - Unzip the folder
   - Double-click `index.html` to view

### Option 2: Share via Cloud Storage

1. **Upload to Cloud:**
   - Upload the entire `UI` folder to Google Drive, OneDrive, or Dropbox
   
2. **Share Link:**
   - Get shareable link with "Anyone with link can view" permissions
   
3. **Recipient Instructions:**
   - Download the entire folder
   - Open `index.html` in browser

### Option 3: Share via USB Drive

1. Copy the entire `UI` folder to USB drive
2. Recipients can:
   - Copy folder to their computer
   - Open `index.html`

### Option 4: Deploy as Website (Advanced)

For web hosting, you can use free services like:
- **GitHub Pages** (free, requires Git)
- **Netlify** (free, drag & drop)
- **Vercel** (free, drag & drop)

**Steps for Netlify (simplest):**
1. Go to [netlify.com](https://netlify.com)
2. Drag & drop your `UI` folder
3. Get a public URL to share (e.g., `your-experiments.netlify.app`)
4. Anyone can access without downloading

---

## 📁 Folder Structure

```
UI/
├── index.html          # Main web application
├── server.py           # Python server (optional)
├── README.md           # This file
└── outputs/            # Experiment data
    ├── M1_M1-5psi-07162025/
    │   ├── M1_M1-5psi-07162025_DRT_overlay.png
    │   └── M1_M1-5psi-07162025_EIS_overlay.png
    ├── M2.../
    └── ...
```

---

## 🔧 Requirements

- **No installation required** for basic use!
- Modern web browser (Chrome, Firefox, Safari, Edge)
- Python 3.x (only if using `server.py`)

---

## 📝 Experiment List (Ordered)

The viewer displays experiments in chronological order (oldest to newest):

1. M1_M1-5psi-07162025
2. M1-r2_M1r2-5psi-07172025
3. M1r_M1r-5psi-07172025
4. ...and 44 more experiments
5. M46_M46r_5psi_09102025 (latest)

---

## 🐛 Troubleshooting

### Images not loading?
- Ensure the `outputs/` folder is in the same directory as `index.html`
- Check that PNG files exist in each experiment folder

### Search not working?
- The search is case-insensitive and searches experiment names
- Try typing "M1" or "5psi" to filter results

### Browser compatibility?
- Works on Chrome, Firefox, Safari, Edge (2020+)
- Use `server.py` if direct file access has issues

---

## 💡 Tips

- Use the search box to quickly find specific experiments (e.g., "M35", "gen1", "pressure")
- Experiments are listed in the order they were saved
- Click any experiment to instantly switch views
- Images are high-resolution and can be zoomed in browser

---

## 📧 Support

For questions or issues, contact the experiment team.

---

**Created:** November 2025  
**Version:** 1.0

