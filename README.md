# E-Commerce Customer Intelligence Platform

Brief: Streamlit app and analysis pipeline for the "Online Retail II" dataset (UCI). Provides data cleaning, RFM segmentation, clustering, visualizations and LTV prediction building blocks.

Features
- Data cleaning & summary
- RFM segmentation and cluster profiling (KMeans + PCA)
- Interactive Streamlit dashboard (pages)
- Notebook-based visualizations and diagnostics

Repository layout (important files/folders)
- app.py — Streamlit main app
- pages/ — Streamlit pages (data_cleaning.py, rfm_Segmentation.py, ...)
- src/ — helper modules (data_processing.py, segmentation.py, visualizations.py, utils.py)
- notebooks/ — exploratory notebooks (visualization.ipynb)
- data/online_retail_II.xlsx — dataset (not committed)

Prerequisites
- Python 3.8+ (Windows recommended for the commands below)
- Git (optional)
- Recommended packages: streamlit, pandas, numpy, scikit-learn, plotly, openpyxl, psutil

Quick setup (Windows PowerShell)
1. Create & activate venv:
   py -3 -m venv .venv
   .\.venv\Scripts\Activate.ps1

   If activation blocked:
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force

2. Install dependencies:
   pip install --upgrade pip
   pip install streamlit pandas numpy scikit-learn plotly openpyxl psutil

   Or (if you add a requirements.txt):
   pip install -r requirements.txt

Run the Streamlit app
- In project root:
  streamlit run app.py

Open notebooks
- Jupyter / VS Code: open files under notebooks/ and run cells after activating the venv.

Data file & path
- Default path referenced in code: `c:/Users/USER/OneDrive/Desktop/learningUtsav/data/online_retail_II.xlsx`
- To change, edit DEFAULT_PATH in `src/utils.py` or pass a different path to loader functions.

Common troubleshooting (PermissionError / OneDrive)
1. Close Excel and any app using the file. Check Task Manager for EXCEL.EXE and end it.
2. Pause OneDrive sync (right-click OneDrive tray icon → Pause syncing) or move the file to a non-OneDrive folder (e.g., `C:\Temp`) and update DEFAULT_PATH.
3. Make a local copy of the file and read the copy.
4. Unblock file (File Properties → Unblock) or use PowerShell:
   Unblock-File -Path "C:\path\to\file.xlsx"
5. Run VS Code / Jupyter as Administrator and retry.
6. Check ACLs (Admin cmd):
   takeown /F "C:\path\to\file.xlsx"
   icacls "C:\path\to\file.xlsx" /grant "%USERNAME%:R"
7. Identify locker process: install Sysinternals Process Explorer or use `handle.exe` to find the process locking the file (then close it).
8. If using the app to diagnose, use the Terminal diagnostics panel in app.py (if present) to collect outputs.

Windows-specific notes
- If `python` opens Microsoft Store, disable app execution aliases: Settings → Apps → Advanced app settings → App execution aliases → turn off python.exe / python3.exe.
- If PowerShell blocks activation, use the ExecutionPolicy command shown above.

If you want, I can also:
- produce a requirements.txt
- add a small README section describing each module (data_processing, segmentation)
- update src/utils.py to accept environment variable or CLI override for the data path

## Demo Video

A short demo walkthrough showing the Streamlit app, RFM pipeline and key visualizations.

- Watch the demo:https://youtu.be/5ajCyfCsvAc

