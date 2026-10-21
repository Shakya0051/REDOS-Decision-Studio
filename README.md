# Decision Studio — cleaned Streamlit dashboard

A healthcare operations analytics demo built with Streamlit, Plotly, NumPy, and pandas. It uses synthetic data to explore how demand, staffing, capacity, and triage assumptions affect simulated waiting time and risk.

## Files
- `app.py` — simulation logic and dashboard UI
- `styles.css` — custom dashboard styling
- `requirements.txt` — required Python packages

## Run on Windows

1. Install Python 3.10 or newer.
2. Open the folder in VS Code.
3. Open **Terminal → New Terminal**.
4. (Recommended) Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

5. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

6. Start the app:

   ```powershell
   streamlit run app.py
   ```

7. Open the local address shown in the terminal, usually `http://localhost:8501`.

## What was removed or simplified
- Removed the compute-heavy sensitivity-analysis and permutation-importance loops.
- Removed unused mode, routing, preference, and penalty-weight controls.
- Removed unused imports and repeated chart setup.
- Kept the Monte Carlo simulation, scenario controls, KPI cards, interactive charts, result table, and CSV export.
- Added custom HTML for the hero/section headings/KPI cards and a separate CSS stylesheet.

## Important note
All data is synthetic, and risk values are outputs of a simplified simulation. This is an educational prototype only, not a medical or clinical decision-making system.
