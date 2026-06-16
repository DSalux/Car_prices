from __future__ import annotations

import json
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    import joblib
except ImportError:
    joblib = None


RANDOM_STATE = 42
CURRENT_YEAR = 2026
DATA_DIR = Path("data")
ARTIFACTS_DIR = Path("artifacts")
AU_DATASET_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "nelgiriyewithana/australian-vehicle-prices"
)
AU_ZIP_PATH = DATA_DIR / "australian-vehicle-prices.zip"
LOCAL_AU_ZIP_FALLBACK = Path("work") / "australian-vehicle-prices.zip"
AU_CSV_PATH = DATA_DIR / "Australian Vehicle Prices.csv"
REPORT_PATH = ARTIFACTS_DIR / "domain_shift_australia_report.json"
METRICS_PATH = ARTIFACTS_DIR / "domain_shift_australia_metrics.csv"
AU_MODEL_PATH = ARTIFACTS_DIR / "car_price_model_australia.joblib"

INR_TO_AUD_FALLBACK = 0.018

INDIA_FEATURES = [
    "brand",
    "fuel_type",
    "transmission",
    "owner_number",
    "manufacture",
    "car_age",
    "kms_driven_num",
    "engine_cc",
    "seats_num",
]

AU_FEATURES = [
    "brand",
    "model",
    "fuel_type",
    "transmission",
    "used_or_new",
    "drive_type",
    "body_type",
    "state",
    "manufacture",
    "car_age",
    "kms_driven_num",
    "engine_cc",
    "seats_num",
    "doors_num",
    "cylinders_num",
]


def parse_first_number(value):
    if pd.isna(value):
        return np.nan
    text = str(value).replace(",", "")
    match = re.search(r"([0-9]*\.?[0-9]+)", text)
    return float(match.group(1)) if match else np.nan


def parse_price_aud(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip().replace(",", "")
    if text.upper() in {"POA", "TBA", ""}:
        return np.nan
    match = re.search(r"([0-9]*\.?[0-9]+)", text)
    return float(match.group(1)) if match else np.nan


def parse_engine_cc(value):
    if pd.isna(value):
        return np.nan
    text = str(value).lower().replace(",", "")
    match_liters = re.search(r"([0-9]*\.?[0-9]+)\s*l", text)
    if match_liters:
        return float(match_liters.group(1)) * 1000
    match_cc = re.search(r"([0-9]*\.?[0-9]+)\s*cc", text)
    if match_cc:
        return float(match_cc.group(1))
    return parse_first_number(text)


def extract_state(value):
    if pd.isna(value):
        return "Unknown"
    parts = str(value).split(",")
    return parts[-1].strip() if len(parts) > 1 else "Unknown"


def normalize_text(value):
    if pd.isna(value):
        return "Unknown"
    text = str(value).strip()
    return text if text else "Unknown"


def ensure_au_dataset() -> Path:
    if AU_CSV_PATH.exists():
        return AU_CSV_PATH

    DATA_DIR.mkdir(exist_ok=True)
    if LOCAL_AU_ZIP_FALLBACK.exists():
        AU_ZIP_PATH.write_bytes(LOCAL_AU_ZIP_FALLBACK.read_bytes())
    elif not AU_ZIP_PATH.exists():
        urllib.request.urlretrieve(AU_DATASET_URL, AU_ZIP_PATH)
    with zipfile.ZipFile(AU_ZIP_PATH) as zf:
        zf.extractall(DATA_DIR)
    if not AU_CSV_PATH.exists():
        raise FileNotFoundError("Australian Vehicle Prices.csv was not extracted")
    return AU_CSV_PATH


def load_au_data() -> pd.DataFrame:
    csv_path = ensure_au_dataset()
    raw = pd.read_csv(csv_path)
    data = raw.copy()

    data["price_aud"] = data["Price"].apply(parse_price_aud)
    data["brand"] = data["Brand"].apply(normalize_text)
    data["model"] = data["Model"].apply(normalize_text)
    data["fuel_type"] = data["FuelType"].apply(normalize_text)
    data["transmission"] = data["Transmission"].apply(normalize_text)
    data["used_or_new"] = data["UsedOrNew"].apply(normalize_text)
    data["drive_type"] = data["DriveType"].apply(normalize_text)
    data["body_type"] = data["BodyType"].apply(normalize_text)
    data["state"] = data["Location"].apply(extract_state)
    data["manufacture"] = pd.to_numeric(data["Year"], errors="coerce")
    data["car_age"] = CURRENT_YEAR - data["manufacture"]
    data["kms_driven_num"] = data["Kilometres"].apply(parse_first_number)
    data["engine_cc"] = data["Engine"].apply(parse_engine_cc)
    data["seats_num"] = data["Seats"].apply(parse_first_number)
    data["doors_num"] = data["Doors"].apply(parse_first_number)
    data["cylinders_num"] = data["CylindersinEngine"].apply(parse_first_number)
    data["owner_number"] = 1

    required = [
        "price_aud",
        "manufacture",
        "car_age",
        "kms_driven_num",
        "engine_cc",
        "seats_num",
    ]
    data = data.dropna(subset=required)
    data = data[data["price_aud"].between(1_000, 500_000)]
    data = data[data["car_age"].between(0, 60)]
    data = data[data["kms_driven_num"].between(0, 1_000_000)]
    data = data[data["engine_cc"].between(500, 10_000)]
    data = data[data["seats_num"].between(1, 12)]
    data = data.drop_duplicates()
    return data


def make_pipeline(X: pd.DataFrame) -> Pipeline:
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=400,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def metric_row(name: str, y_true, y_pred) -> dict[str, float | str]:
    return {
        "model": name,
        "MAE_AUD": mean_absolute_error(y_true, y_pred),
        "RMSE_AUD": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def get_inr_to_aud_rate() -> tuple[float, bool]:
    try:
        with urllib.request.urlopen("https://open.er-api.com/v6/latest/INR", timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        aud_rate = payload.get("rates", {}).get("AUD")
        if payload.get("result") == "success" and aud_rate:
            return float(aud_rate), True
    except Exception:
        pass
    return INR_TO_AUD_FALLBACK, False


def load_india_model():
    if Path("app.py").exists():
        sys.path.insert(0, str(Path.cwd()))
    elif Path("outputs/app.py").exists():
        sys.path.insert(0, str(Path.cwd() / "outputs"))

    import app

    india_data = app.load_clean_data()
    return app.load_or_train_model(india_data)


def run_experiment() -> dict:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    data = load_au_data()
    X_old = data[INDIA_FEATURES].copy()
    X_new = data[AU_FEATURES].copy()
    y = data["price_aud"].copy()

    train_idx, test_idx = train_test_split(
        data.index,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )
    X_old_train, X_old_test = X_old.loc[train_idx], X_old.loc[test_idx]
    X_new_train, X_new_test = X_new.loc[train_idx], X_new.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]

    baseline = DummyRegressor(strategy="median")
    baseline.fit(X_new_train, y_train)
    baseline_pred = baseline.predict(X_new_test)

    inr_to_aud, live_rate = get_inr_to_aud_rate()
    india_model = load_india_model()
    transfer_pred_aud = india_model.predict(X_old_test) * inr_to_aud

    au_model = make_pipeline(X_new_train)
    au_model.fit(X_new_train, y_train)
    au_pred = au_model.predict(X_new_test)

    metrics = pd.DataFrame(
        [
            metric_row("Baseline median on AU", y_test, baseline_pred),
            metric_row("India model transferred to AU", y_test, transfer_pred_aud),
            metric_row("AU RandomForest with extended features", y_test, au_pred),
        ]
    ).sort_values("MAE_AUD")

    if joblib is not None:
        joblib.dump(au_model, AU_MODEL_PATH)

    report = {
        "dataset": "Kaggle: nelgiriyewithana/australian-vehicle-prices",
        "market": "Australia",
        "rows_after_cleaning": int(len(data)),
        "target": "price_aud",
        "inr_to_aud_rate": inr_to_aud,
        "inr_to_aud_rate_live": live_rate,
        "old_india_feature_schema": INDIA_FEATURES,
        "new_au_feature_schema": AU_FEATURES,
        "metrics": metrics.to_dict(orient="records"),
        "saved_au_model": str(AU_MODEL_PATH) if joblib is not None else None,
        "interpretation": (
            "The India model is evaluated as a transfer baseline only. "
            "A dedicated AU model is expected to perform better because prices, "
            "brands, trims and market behavior differ by country."
        ),
    }

    metrics.to_csv(METRICS_PATH, index=False)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run_experiment()
    print(json.dumps(result, ensure_ascii=False, indent=2))
