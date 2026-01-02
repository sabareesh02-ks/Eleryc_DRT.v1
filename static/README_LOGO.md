# Eleryc Logo Setup

## To enable logo watermarking on all plots:

1. **Save your Eleryc logo** as: `eleryc_logo.png` in this directory
2. **Recommended specifications:**
   - Format: PNG with transparency (RGBA)
   - Size: 200x200 pixels or smaller
   - Background: Transparent
   - Colors: Company brand colors

3. **The logo will automatically appear on:**
   - All DRT analysis plots (DRT, Nyquist, Magnitude, Phase)
   - Comparison plots
   - Performance analysis plots
   - All exported plots

## Current Logo Settings:
- **Position:** Bottom-right corner
- **Size:** 8% of plot width
- **Transparency:** 70%
- **Auto-positioning:** Avoids data overlap

## File Structure:
```
static/
├── eleryc_logo.png          ← Place your logo here
└── README_LOGO.md          ← This file
```

## Testing:
Run `python setup_logo.py` to test logo integration and verify setup.

---
**Note:** If no logo file is found, the system gracefully falls back to text-only watermarks.

