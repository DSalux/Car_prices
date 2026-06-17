# Car Prices

[Русская версия README](README_RU.md)

ML project for predicting used car prices from vehicle attributes. The project started as a notebook experiment and was extended into a small reproducible application with model analysis, Streamlit UI, currency conversion, Docker support, and a domain-shift experiment on a second market.

## What The Project Does

- Cleans raw car listing data where prices, mileage, engine size and seats are stored as text.
- Trains regression models to estimate used car prices.
- Compares a baseline model with RandomForest and GradientBoosting.
- Saves the best model as a reusable scikit-learn `Pipeline`.
- Provides a Streamlit app for interactive predictions.
- Converts the INR prediction into `INR`, `USD`, `EUR`, or `RUB`.
- Tests transfer to a different market with an Australian vehicle dataset.

## Main Dataset

The main model uses Kaggle `Old Car Price Prediction`:

https://www.kaggle.com/datasets/milanvaddoriya/old-car-price-prediction

The original target is `car_prices_in_rupee`, so the base model predicts prices in Indian rupees.

## Results

Best model on the India dataset:

| Model | MAE | RMSE | R2 |
|---|---:|---:|---:|
| RandomForest | 342,919 INR | 837,369 INR | 0.770 |
| GradientBoosting | 365,509 INR | 892,227 INR | 0.739 |
| Baseline median | 771,762 INR | 1,826,970 INR | -0.095 |

Cross-validation MAE for RandomForest:

```text
369,285 +/- 77,750 INR
```

## Streamlit App

Run locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

The app lets a user choose:

- brand;
- manufacture year;
- mileage;
- fuel type;
- transmission;
- owner count;
- engine size;
- number of seats;
- output currency.

## Docker

Build and run:

```bash
docker build -t car-price-app .
docker run --name car-price-app -p 8501:8501 car-price-app
```

Or with Docker Compose:

```bash
docker compose up --build
```

Open on the same machine:

```text
http://localhost:8501
```

Open from another device in the same local network:

```text
http://YOUR_COMPUTER_IP:8501
```

## Domain Shift Experiment

The project includes an extra experiment on Kaggle `Australian Vehicle Prices`:

https://www.kaggle.com/datasets/nelgiriyewithana/australian-vehicle-prices

Run:

```bash
python domain_shift_australia.py
```

The experiment compares:

| Model | MAE AUD | RMSE AUD | R2 |
|---|---:|---:|---:|
| AU RandomForest with extended features | 5,273 | 13,462 | 0.778 |
| Baseline median on AU | 16,737 | 29,264 | -0.050 |
| India model transferred to AU | 18,102 | 28,151 | 0.028 |

This shows that a model trained on one country does not automatically transfer well to another market. Retraining on local data with local features improves quality substantially.

## Important Files

```text
app.py                         Streamlit app
Car_prices_completed.ipynb     completed analysis notebook
domain_shift_australia.py      domain-shift experiment
tests/test_app_helpers.py      unit tests for app helper logic
requirements.txt               Python dependencies
Dockerfile                     Docker image definition
docker-compose.yml             Docker Compose setup
HOW_TO_RUN.md                  detailed run instructions
MODEL_CARD.md                  model limitations and intended use
.github/workflows/ci.yml       GitHub Actions checks
```

Generated files are intentionally not committed:

```text
data/
artifacts/
*.joblib
*.csv
```

They are recreated by running the notebook, app, or experiment scripts.

## Quick Checks

```bash
python -m py_compile app.py domain_shift_australia.py
python -m unittest discover -s tests -p "test_*.py"
python domain_shift_australia.py
streamlit run app.py
```

For full Docker validation:

```bash
docker build -t car-price-app .
```

## Limitations

- The main model is trained on the Indian used-car market.
- It predicts an estimate, not a guaranteed market price.
- Features such as accident history, condition, trim, region and seller type are not fully represented in the main dataset.
- Currency conversion does not make the model global; it only converts the output amount.
- For another country, retraining on local market data is recommended.
