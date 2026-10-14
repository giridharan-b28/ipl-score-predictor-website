
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


DATASET_PATH = "ipl_data.csv"
MODEL_PATH = "saved_models.joblib"


def main():
    print("Loading IPL dataset...", flush=True)

    df = pd.read_csv(DATASET_PATH)

    if df.empty or len(df.columns) < 4:
        raise ValueError("The CSV has too few columns to train a model.")

    target_col = df.columns[-1]
    feature_cols = list(df.columns[2:-1])

    if not feature_cols:
        raise ValueError("No feature columns found.")

    # Match the preprocessing in app.py.
    model_df = df[feature_cols + [target_col]].copy()

    model_df[target_col] = pd.to_numeric(
        model_df[target_col], errors="coerce"
    )
    model_df = model_df.dropna(subset=[target_col])

    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(model_df[col]):
            model_df[col] = pd.to_numeric(
                model_df[col], errors="coerce"
            )
            median_value = model_df[col].median()
            model_df[col] = model_df[col].fillna(
                median_value if pd.notna(median_value) else 0
            )
        else:
            model_df[col] = (
                model_df[col].fillna("Unknown").astype(str)
            )

    if len(model_df) < 10:
        raise ValueError("At least 10 valid rows are required.")

    X = model_df[feature_cols]
    y = model_df[target_col]

    cat_cols = [
        col for col in feature_cols
        if not pd.api.types.is_numeric_dtype(X[col])
    ]
    num_cols = [col for col in feature_cols if col not in cat_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=32
    )

    def make_preprocessor():
        try:
            encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True,
            )
        except TypeError:
            encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse=True,
            )

        return ColumnTransformer(
            [
                ("categorical", encoder, cat_cols),
                ("numeric", "passthrough", num_cols),
            ],
            remainder="drop",
        )

    print("Training Random Forest...", flush=True)

    rf = Pipeline([
        ("preprocessor", make_preprocessor()),
        ("model", RandomForestRegressor(
            n_estimators=50,
            random_state=0,
            n_jobs=2,
        )),
    ])

    rf.fit(X_train, y_train)
    rf_r2 = float(r2_score(y_test, rf.predict(X_test)))

    print(f"Random Forest R²: {rf_r2 * 100:.2f}%", flush=True)

    print("Training XGBoost...", flush=True)

    xgb = Pipeline([
        ("preprocessor", make_preprocessor()),
        ("model", XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=2,
        )),
    ])

    xgb.fit(X_train, y_train)
    xgb_r2 = float(r2_score(y_test, xgb.predict(X_test)))

    print(f"XGBoost R²: {xgb_r2 * 100:.2f}%", flush=True)

    bundle = {
        "data": df,
        "target_col": target_col,
        "feature_cols": feature_cols,
        "categorical_cols": cat_cols,
        "numeric_cols": num_cols,
        "rf_pipeline": rf,
        "xgb_pipeline": xgb,
        "metrics": {
            "random_forest": {
                "r2": round(rf_r2 * 100, 2)
            },
            "xgboost": {
                "r2": round(xgb_r2 * 100, 2)
            },
        },
    }

    print("Saving trained models...", flush=True)
    joblib.dump(bundle, MODEL_PATH, compress=3)

    print(f"Models saved successfully to {MODEL_PATH}", flush=True)


if __name__ == "__main__":
    main()
