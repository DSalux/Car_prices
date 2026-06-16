# Model Card: Car Price Prediction

## Model Overview

This project contains a regression model for estimating used car prices from listing attributes. The main production-style model is a scikit-learn `Pipeline` with preprocessing and `RandomForestRegressor`.

The model predicts price in Indian rupees because the training target comes from the India-focused Kaggle dataset `Old Car Price Prediction`.

## Intended Use

The model is intended for:

- educational ML portfolio demonstration;
- approximate used-car price estimation;
- experimentation with preprocessing, feature importance, model comparison and deployment;
- demonstrating domain shift between country-specific vehicle markets.

It is not intended for:

- financial decisions without human review;
- official vehicle valuation;
- insurance, lending, legal or tax decisions;
- universal global car pricing without retraining.

## Training Data

Main dataset:

```text
Kaggle: Old Car Price Prediction
```

Source:

```text
https://www.kaggle.com/datasets/milanvaddoriya/old-car-price-prediction
```

Important raw fields:

- `car_name`
- `car_prices_in_rupee`
- `kms_driven`
- `fuel_type`
- `transmission`
- `ownership`
- `manufacture`
- `engine`
- `Seats`

## Target

```text
price_rupee
```

The raw target `car_prices_in_rupee` is parsed into numeric rupees. Values using `Lakh` and `Crore` are converted to absolute INR.

## Features

Main model features:

```text
brand
fuel_type
transmission
owner_number
manufacture
car_age
kms_driven_num
engine_cc
seats_num
```

## Preprocessing

The pipeline:

- parses text prices into numeric rupees;
- parses mileage, engine size and seats into numeric values;
- extracts brand from `car_name`;
- extracts owner number from ownership text;
- imputes missing numeric values with median;
- imputes missing categorical values with most frequent value;
- one-hot encodes categorical values with unknown-category handling.

## Model

Main selected model:

```text
RandomForestRegressor
```

The model is saved as:

```text
artifacts/car_price_model.joblib
```

## Evaluation

Holdout test metrics:

| Model | MAE | RMSE | R2 |
|---|---:|---:|---:|
| RandomForest | 342,919 INR | 837,369 INR | 0.770 |
| GradientBoosting | 365,509 INR | 892,227 INR | 0.739 |
| Baseline median | 771,762 INR | 1,826,970 INR | -0.095 |

Cross-validation MAE for RandomForest:

```text
369,285 +/- 77,750 INR
```

## Feature Importance

Most useful groups of features are typically:

- brand;
- manufacture year / car age;
- engine size;
- mileage;
- fuel type;
- transmission.

Lower-importance features should not be removed automatically. They should be removed only after checking whether they add noise, increase complexity, or create missing-value issues.

## Domain Shift

The repository includes a separate Australia experiment:

```bash
python domain_shift_australia.py
```

Results:

| Model | MAE AUD | RMSE AUD | R2 |
|---|---:|---:|---:|
| AU RandomForest with extended features | 5,273 | 13,462 | 0.778 |
| Baseline median on AU | 16,737 | 29,264 | -0.050 |
| India model transferred to AU | 18,102 | 28,151 | 0.028 |

This indicates that the India model does not transfer well to the Australian market without retraining. The dedicated AU model performs much better.

## Currency Conversion

The Streamlit app can display predictions in:

```text
INR
USD
EUR
RUB
```

The model still predicts in INR. Currency conversion changes only the display currency, not the underlying model logic.

## Limitations

- The main model is trained on one country-specific dataset.
- The model does not know accident history, exact trim, maintenance records, seller type or vehicle condition.
- Some high-end cars may have larger errors due to skewed price distribution.
- Predictions may be unreliable for brands or vehicle types poorly represented in training data.
- Currency conversion does not solve market differences between countries.

## Recommended Improvements

- Add region, trim, body type, seller type and condition fields.
- Train country-specific models.
- Add CatBoost or LightGBM for categorical-heavy data.
- Add prediction intervals.
- Log user predictions for later analysis.
- Add automated tests for preprocessing and model loading.
