import joblib

import os
import warnings
import threading

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

app = Flask(__name__)

DATA_PATH = os.environ.get("IPL_DATA_PATH", "ipl_data.csv")

DATA = None
TARGET_COL = None
FEATURE_COLS, CATEGORICAL_COLS, NUMERIC_COLS = [], [], []
RF_PIPELINE = XGB_PIPELINE = None
METRICS = {}
TRAINING_ERROR = None
TRAINING_STATUS = "Waiting to start model training..."

ALIASES = {
    "batting": [
        "batting_team", "bat_team", "batting team",
        "team_batting", "battingteam", "batting side"
    ],
    "bowling": [
        "bowling_team", "bowl_team", "bowling team",
        "team_bowling", "bowlingteam", "bowling side",
        "bowl_team_name"
    ],
    "venue": ["venue", "stadium", "ground", "city", "match_venue"],
    "over": ["current_over", "overs", "over", "over_number", "overs_completed"],
    "wickets": ["current_wickets", "wickets", "wicket", "wickets_lost", "total_wickets"],
    "runs": ["current_runs", "runs", "score", "total_runs", "total", "current_score", "runs_scored"],
    "date": ["date", "season", "match_date", "year"],
    "winner": ["winner", "winning_team", "match_winner", "winningteam"],
}

TEAM_ALIASES = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
}


def norm(value):
    return "".join(
        ch for ch in str(value).lower().strip() if ch.isalnum()
    )


def find_col(df, candidates):
    normalized = {norm(c): c for c in df.columns}

    for name in candidates:
        if norm(name) in normalized:
            return normalized[norm(name)]

    for c in df.columns:
        nc = norm(c)
        if any(
            norm(name) in nc or nc in norm(name)
            for name in candidates
            if len(norm(name)) > 3
        ):
            return c

    return None


def normalize_team(value):
    if pd.isna(value):
        return value
    return str(value).strip()


def load_saved_models():
    global DATA, TARGET_COL, FEATURE_COLS, CATEGORICAL_COLS, NUMERIC_COLS
    global RF_PIPELINE, XGB_PIPELINE, METRICS
    global TRAINING_ERROR, TRAINING_STATUS

    model_path = os.environ.get(
        "IPL_MODEL_PATH", "saved_models.joblib"
    )

    TRAINING_STATUS = "Loading saved models..."
    TRAINING_ERROR = None

    try:
        print(f"Loading saved models from {model_path}...", flush=True)

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Saved model file not found: {model_path}"
            )

        bundle = joblib.load(model_path)

        DATA = bundle["data"]
        TARGET_COL = bundle["target_col"]
        FEATURE_COLS = bundle["feature_cols"]
        CATEGORICAL_COLS = bundle["categorical_cols"]
        NUMERIC_COLS = bundle["numeric_cols"]
        RF_PIPELINE = bundle["rf_pipeline"]
        XGB_PIPELINE = bundle["xgb_pipeline"]
        METRICS = bundle["metrics"]

        TRAINING_STATUS = "Ready"
        print("Model status: Ready", flush=True)
        print("R² percentages:", METRICS, flush=True)

    except Exception as exc:
        DATA = None
        RF_PIPELINE = None
        XGB_PIPELINE = None
        METRICS = {}
        TRAINING_ERROR = str(exc)
        TRAINING_STATUS = "Failed"
        print("Saved model loading failed:", TRAINING_ERROR, flush=True)


# Load pre-trained models instead of retraining at every startup.
load_saved_models()


def get_ui_columns():
    if DATA is None:
        return {}

    return {
        key: find_col(DATA, candidates)
        for key, candidates in ALIASES.items()
    }


def available_values(col):
    if DATA is None or not col or col not in DATA.columns:
        return []

    return sorted(
        DATA[col].dropna().map(normalize_team).astype(str).unique().tolist()
    )[:300]


def build_model_row(payload):
    row = {}

    for col in FEATURE_COLS:
        series = DATA[col]

        if col in CATEGORICAL_COLS:
            vals = series.dropna().astype(str).mode()
            row[col] = vals.iloc[0] if len(vals) else "Unknown"
        else:
            vals = pd.to_numeric(series, errors="coerce")
            row[col] = (
                float(vals.median()) if vals.notna().any() else 0.0
            )

    cols = get_ui_columns()

    field_keys = {
        "batting": "batting_team",
        "bowling": "bowling_team",
        "venue": "venue",
        "over": "current_over",
        "wickets": "current_wickets",
        "runs": "current_runs",
    }

    for field, key in field_keys.items():
        col = cols.get(field)
        if col in row and key in payload:
            row[col] = payload[key]

    for col in FEATURE_COLS:
        n = norm(col)

        for field, key in field_keys.items():
            if (
                any(norm(a) in n for a in ALIASES[field])
                and key in payload
            ):
                row[col] = payload[key]

    for col in FEATURE_COLS:
        if col in CATEGORICAL_COLS:
            row[col] = str(row[col])
        else:
            try:
                row[col] = float(row[col])
            except (TypeError, ValueError):
                row[col] = 0.0

    return pd.DataFrame([row], columns=FEATURE_COLS)


def historical_stats(payload):
    """Summarize the supplied ball-by-ball CSV at match level."""

    result = {
        "matches_played": 0,
        "batting_team_wins": "N/A",
        "bowling_team_wins": "N/A",
        "average_first_innings_score": "—",
        "recent_matches": [],
        "win_data_note": (
            "The supplied CSV has no match-winner column, "
            "so wins cannot be calculated reliably."
        ),
    }

    if DATA is None:
        return result

    cols = get_ui_columns()
    bat_col = cols.get("batting")
    bowl_col = cols.get("bowling")
    venue_col = cols.get("venue")
    date_col = cols.get("date")

    score_col = find_col(
        DATA, ["total", "final_score", "innings_total", "target_score"]
    )
    match_id_col = find_col(DATA, ["mid", "match_id", "matchid", "id"])

    if not bat_col or not bowl_col or not match_id_col:
        result["win_data_note"] = (
            "Required team or match ID columns were not found in the dataset."
        )
        return result

    bat = str(payload.get("batting_team", ""))
    bowl = str(payload.get("bowling_team", ""))

    bat_values = DATA[bat_col].map(normalize_team).astype(str)
    bowl_values = DATA[bowl_col].map(normalize_team).astype(str)

    matches = DATA[
        ((bat_values == bat) & (bowl_values == bowl))
        | ((bat_values == bowl) & (bowl_values == bat))
    ].copy()

    if venue_col and payload.get("venue"):
        at_venue = matches[
            matches[venue_col].astype(str) == str(payload["venue"])
        ]
        if not at_venue.empty:
            matches = at_venue

    if matches.empty:
        return result

    sort_cols = [
        c for c in [date_col, match_id_col]
        if c and c in matches.columns
    ]

    if sort_cols:
        matches = matches.sort_values(sort_cols)

    aggregations = {
        bat_col: "first",
        bowl_col: "first",
    }

    if venue_col:
        aggregations[venue_col] = "first"
    if date_col:
        aggregations[date_col] = "first"
    if score_col:
        aggregations[score_col] = "max"

    match_rows = (
        matches.groupby(match_id_col, sort=False)
        .agg(aggregations)
        .reset_index()
    )

    result["matches_played"] = int(len(match_rows))

    if score_col:
        scores = pd.to_numeric(
            match_rows[score_col], errors="coerce"
        ).dropna()

        if not scores.empty:
            result["average_first_innings_score"] = round(
                float(scores.mean()), 1
            )

    for _, r in match_rows.tail(8).iloc[::-1].iterrows():
        date_value = r[date_col] if date_col else None
        date_value = (
            str(date_value)[:10]
            if date_value is not None and pd.notna(date_value)
            else "—"
        )

        score_value = (
            r[score_col]
            if score_col and pd.notna(r[score_col])
            else None
        )

        result["recent_matches"].append({
            "date_or_season": date_value,
            "venue": (
                str(r[venue_col])
                if venue_col and pd.notna(r[venue_col])
                else "—"
            ),
            "batting_team_score": (
                str(int(score_value))
                if score_value is not None
                else "—"
            ),
            "bowling_team_score": "Not in dataset",
            "winner": "Not in dataset",
        })

    return result


@app.route("/")
def index():
    return render_template("index.html")


@app.get("/api/options")
def api_options():
    if DATA is None:
        if TRAINING_ERROR:
            message = TRAINING_ERROR
        else:
            message = TRAINING_STATUS

        return jsonify({
            "status": TRAINING_STATUS,
            "error": message,
        }), 503

    cols = get_ui_columns()

    team_cols = [
        c for c in [cols.get("batting"), cols.get("bowling")] if c
    ]

    teams = sorted(
        set(v for c in team_cols for v in available_values(c))
    )
    venues = available_values(cols.get("venue"))

    return jsonify({
        "status": TRAINING_STATUS,
        "teams": teams,
        "venues": venues,
        "columns_detected": {
            k: str(v) if v else None
            for k, v in cols.items()
        },
    })


@app.get("/api/status")
def api_status():
    return jsonify({
        "status": TRAINING_STATUS,
        "ready": (
            DATA is not None
            and RF_PIPELINE is not None
            and XGB_PIPELINE is not None
        ),
        "error": TRAINING_ERROR,
        "metrics": METRICS,
    })


@app.post("/api/predict")
def api_predict():
    if DATA is None or RF_PIPELINE is None or XGB_PIPELINE is None:
        message = TRAINING_ERROR or TRAINING_STATUS
        return jsonify({
            "status": TRAINING_STATUS,
            "error": message,
        }), 503

    payload = request.get_json(silent=True) or {}

    required = [
        "batting_team",
        "bowling_team",
        "venue",
        "current_over",
        "current_wickets",
        "current_runs",
    ]

    if any(payload.get(k) in (None, "") for k in required):
        return jsonify({
            "error": (
                "Please provide all match inputs, including current runs."
            )
        }), 400

    if payload["batting_team"] == payload["bowling_team"]:
        return jsonify({
            "error": "Batting and bowling teams must be different."
        }), 400

    try:
        row = build_model_row(payload)
        current_runs = float(payload["current_runs"])

        rf_score = max(
            current_runs, float(RF_PIPELINE.predict(row)[0])
        )
        xgb_score = max(
            current_runs, float(XGB_PIPELINE.predict(row)[0])
        )

        return jsonify({
            "predictions": {
                "random_forest": round(rf_score),
                "xgboost": round(xgb_score),
            },
            "metrics": METRICS,
            "stats": historical_stats(payload),
        })

    except Exception as exc:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": f"Prediction failed: {exc}"
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5000")),
        debug=False,
        threaded=True,
    )

