# IPL Final Score Predictor — Updated

Flask dashboard comparing Random Forest and XGBoost predicted final scores.

## Features
- Batting team, bowling team, venue, current over, wickets and current runs.
- Side-by-side Random Forest and XGBoost predictions plus average.
- R² scores are calculated from the local model's held-out test split and shown only after clicking **Predict Final Score**.
- No MAE or RMSE shown.
- Head-to-head match counts and average recorded total where available.

## Run on Windows
Open this folder in Visual Studio Code, then use the terminal:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000 in your browser.

If PowerShell cannot find `python` after activation, use `py app.py` only after installing packages into the same environment, or select `.venv` as the VS Code Python interpreter.

## Dataset limitation
The supplied `ipl_data.csv` is ball-by-ball data with columns such as `mid`, `date`, `venue`, `bat_team`, `bowl_team`, `runs`, `wickets`, `overs`, and `total`. It does **not** contain a match-winner/result column, so true historical match wins cannot be computed from this CSV alone. The dashboard marks those values unavailable instead of inventing them.

R² is a regression metric, not classification accuracy. Displayed values are computed locally on the current training/test split; they are not copied from Google Colab and can vary with library versions or changes to the training setup.
