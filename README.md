━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    NOTICE OF CONFIDENTIALITY AND RESTRICTED USE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This software and all associated documentation, source code, and materials 
(collectively, the "Software") constitute confidential and proprietary 
information of Eleryc Inc., a corporation organized under the laws of the 
United States of America.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Eleryc Data Analysis Platform

A comprehensive web-based platform for electrochemical data analysis, experiment planning, and visualization.

![Eleryc Logo](assets/eleryc-logo.png)

> 🚧 **UNDER ACTIVE DEVELOPMENT** - This application is currently under development. Features may change without notice.

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
- Weekly EC Schedule with drag-and-drop colors
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

## 🛠️ Setup & Deployment

Setup and deployment instructions are available to authorized Eleryc Inc. personnel only. Please contact your system administrator for access.

## 🔐 Authentication

Access credentials are provided to authorized personnel only. Contact your system administrator for login credentials.

## 📊 Supported File Formats

- CSV files (.csv)
- Excel files (.xls, .xlsx)

## 🔧 Configuration

Environment configuration details are available to authorized system administrators only.

---

## ⚖️ LEGAL NOTICE AND TERMS OF USE

### PROPRIETARY RIGHTS NOTICE

Copyright © 2024-2026 Eleryc Inc. All Rights Reserved.

This Software is the sole and exclusive property of Eleryc Inc. and is protected under United States copyright law, international treaty provisions, and applicable laws of the jurisdiction in which it is used.

### RESTRICTED ACCESS AND AUTHORIZED USE ONLY

This Software is developed exclusively for the internal business operations of Eleryc Inc. and its authorized personnel. Access to, and use of, this Software is strictly limited to individuals who have received express written authorization from Eleryc Inc.

**UNAUTHORIZED ACCESS IS PROHIBITED.** No license, right, or interest in this Software is granted to any party by implication, estoppel, or otherwise, except as expressly set forth in a written agreement signed by an authorized representative of Eleryc Inc.

### PROHIBITION OF UNAUTHORIZED USE

Any person or entity who accesses, uses, copies, modifies, distributes, reverse engineers, decompiles, or disassembles this Software, or any portion thereof, without the prior express written consent of Eleryc Inc., shall be deemed to be in violation of this Notice and applicable law.

### ENFORCEMENT AND LEGAL REMEDIES

Eleryc Inc. reserves the right to pursue all available legal remedies against any individual or entity that engages in unauthorized access or use of this Software, including but not limited to:

**(a) Civil Remedies:** Eleryc Inc. may seek compensatory damages, consequential damages, statutory damages, disgorgement of profits, and equitable relief, including temporary and permanent injunctive relief;

**(b) Criminal Prosecution:** Unauthorized access to computer systems and theft of trade secrets may constitute criminal offenses under the Computer Fraud and Abuse Act (18 U.S.C. § 1030), the Economic Espionage Act (18 U.S.C. §§ 1831-1839), and applicable state laws, punishable by fines and imprisonment;

**(c) Attorneys' Fees and Costs:** The prevailing party in any action to enforce this Notice shall be entitled to recover reasonable attorneys' fees, costs, and expenses.

### DISCLAIMER OF WARRANTIES

THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED. ELERYC INC. DISCLAIMS ALL WARRANTIES, INCLUDING BUT NOT LIMITED TO THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT.

### LIMITATION OF LIABILITY

IN NO EVENT SHALL ELERYC INC. BE LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL, CONSEQUENTIAL, OR PUNITIVE DAMAGES ARISING OUT OF OR RELATED TO THE USE OR INABILITY TO USE THIS SOFTWARE.

### GOVERNING LAW AND JURISDICTION

This Notice shall be governed by and construed in accordance with the laws of the State of California, United States of America, without regard to its conflict of law principles. Any dispute arising under this Notice shall be subject to the exclusive jurisdiction of the state and federal courts located in California.

### SEVERABILITY

If any provision of this Notice is held to be invalid or unenforceable, such provision shall be struck and the remaining provisions shall remain in full force and effect.

### CONTACT INFORMATION

For inquiries regarding authorized access or licensing, please contact:

**Eleryc Inc.**

---

**BY ACCESSING THIS SOFTWARE, YOU ACKNOWLEDGE THAT YOU HAVE READ, UNDERSTOOD, AND AGREE TO BE BOUND BY THE TERMS SET FORTH IN THIS NOTICE. IF YOU DO NOT AGREE TO THESE TERMS, YOU MUST IMMEDIATELY CEASE ALL ACCESS TO AND USE OF THIS SOFTWARE.**

---

## 👥 Contributors

- Eleryc Engineering Team
