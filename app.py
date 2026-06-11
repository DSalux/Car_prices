from pathlib import Path
import re
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    import joblib
except ImportError:
    joblib = None


RANDOM_STATE = 42
DATA_URL = "https://www.kaggle.com/api/v1/datasets/download/milanvaddoriya/old-car-price-prediction"
DATA_DIR = Path("data")
DATA_PATH = DATA_DIR / "car_price.csv"
ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "car_price_model.joblib"

FEATURE_COLS = [
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


def ensure_dataset() -> Path:
    if DATA_PATH.exists():
        return DATA_PATH

    DATA_DIR.mkdir(exist_ok=True)
    archive_path = DATA_DIR / "old-car-price-prediction.zip"
    urllib.request.urlretrieve(DATA_URL, archive_path)
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(DATA_DIR)
    if not DATA_PATH.exists():
        raise FileNotFoundError("Не удалось скачать и распаковать car_price.csv")
    return DATA_PATH


def parse_price_rupee(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower().replace(",", "")
    match = re.search(r"([0-9]*\.?[0-9]+)", text)
    if not match:
        return np.nan
    number = float(match.group(1))
    if "crore" in text:
        return number * 10_000_000
    if "lakh" in text:
        return number * 100_000
    return number


def parse_first_number(value):
    if pd.isna(value):
        return np.nan
    text = str(value).replace(",", "")
    match = re.search(r"([0-9]*\.?[0-9]+)", text)
    return float(match.group(1)) if match else np.nan


def parse_ownership(value):
    if pd.isna(value):
        return np.nan
    text = str(value).lower()
    if "first" in text or "1st" in text:
        return 1
    if "second" in text or "2nd" in text:
        return 2
    if "third" in text or "3rd" in text:
        return 3
    if "fourth" in text or "4th" in text:
        return 4
    if "fifth" in text or "5th" in text:
        return 5
    return parse_first_number(text)


def load_clean_data() -> pd.DataFrame:
    dataset_path = ensure_dataset()
    df = pd.read_csv(dataset_path)
    data = df.copy()
    data = data.drop(columns=[col for col in ["Unnamed: 0"] if col in data.columns])
    data = data.drop_duplicates()

    data["price_rupee"] = data["car_prices_in_rupee"].apply(parse_price_rupee)
    data["kms_driven_num"] = data["kms_driven"].apply(parse_first_number)
    data["engine_cc"] = data["engine"].apply(parse_first_number)
    data["seats_num"] = data["Seats"].apply(parse_first_number)
    data["owner_number"] = data["ownership"].apply(parse_ownership)
    data["brand"] = data["car_name"].astype(str).str.split().str[0]
    data["car_age"] = 2026 - data["manufacture"]

    data = data.dropna(
        subset=[
            "price_rupee",
            "kms_driven_num",
            "engine_cc",
            "seats_num",
            "owner_number",
            "manufacture",
        ]
    )
    data = data[data["price_rupee"] > 0]
    data = data[data["kms_driven_num"] >= 0]
    data = data[data["engine_cc"] > 0]
    data = data[data["seats_num"] > 0]
    data = data[data["car_age"] >= 0]
    return data


def train_model(data: pd.DataFrame):
    X = data[FEATURE_COLS].copy()
    y = data["price_rupee"].copy()

    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=500,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(X, y)
    return model


def load_or_train_model(data: pd.DataFrame):
    if joblib is not None and MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

    model = train_model(data)
    if joblib is not None:
        ARTIFACTS_DIR.mkdir(exist_ok=True)
        joblib.dump(model, MODEL_PATH)
    return model


def format_price_rupee(value: float) -> str:
    lakh = value / 100_000
    return f"{value:,.0f} rupees ({lakh:,.2f} Lakh)"


def build_input_row(
    brand,
    manufacture,
    kms_driven_num,
    fuel_type,
    transmission,
    owner_number,
    engine_cc,
    seats_num,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "brand": brand,
                "fuel_type": fuel_type,
                "transmission": transmission,
                "owner_number": owner_number,
                "manufacture": manufacture,
                "car_age": 2026 - manufacture,
                "kms_driven_num": kms_driven_num,
                "engine_cc": engine_cc,
                "seats_num": seats_num,
            }
        ]
    )


def main():
    import streamlit as st

    st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")
    st.title("Прогноз цены автомобиля")

    @st.cache_data
    def cached_data():
        return load_clean_data()

    @st.cache_resource
    def cached_model(_data):
        return load_or_train_model(_data)

    data = cached_data()
    model = cached_model(data)

    brands = sorted(data["brand"].dropna().unique())
    fuels = sorted(data["fuel_type"].dropna().unique())
    transmissions = sorted(data["transmission"].dropna().unique())

    default_year = int(data["manufacture"].median())
    default_kms = int(data["kms_driven_num"].median())
    default_engine = int(data["engine_cc"].median())
    default_seats = int(data["seats_num"].median())

    brand = st.selectbox("Марка", brands, index=brands.index("Maruti") if "Maruti" in brands else 0)
    manufacture = st.slider(
        "Год выпуска",
        int(data["manufacture"].min()),
        int(data["manufacture"].max()),
        default_year,
    )
    kms_driven_num = st.number_input(
        "Пробег, км",
        min_value=0,
        max_value=1_000_000,
        value=default_kms,
        step=1_000,
    )
    fuel_type = st.selectbox("Тип топлива", fuels)
    transmission = st.selectbox("Коробка передач", transmissions)

    col1, col2, col3 = st.columns(3)
    with col1:
        owner_number = st.number_input("Владелец по счету", 1, 5, 1)
    with col2:
        engine_cc = st.number_input("Двигатель, cc", 500, 7000, default_engine, step=50)
    with col3:
        seats_num = st.number_input("Мест", 2, 10, default_seats)

    input_row = build_input_row(
        brand=brand,
        manufacture=manufacture,
        kms_driven_num=kms_driven_num,
        fuel_type=fuel_type,
        transmission=transmission,
        owner_number=owner_number,
        engine_cc=engine_cc,
        seats_num=seats_num,
    )

    prediction = float(model.predict(input_row)[0])
    st.metric("Ориентировочная цена", format_price_rupee(prediction))

    with st.expander("Параметры для модели"):
        st.dataframe(input_row, use_container_width=True)


if __name__ == "__main__":
    main()
