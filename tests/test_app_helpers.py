import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import app  # noqa: E402


class AppHelperTests(unittest.TestCase):
    def test_parse_price_rupee_handles_common_units(self):
        self.assertEqual(app.parse_price_rupee("10 Lakh"), 1_000_000)
        self.assertEqual(app.parse_price_rupee("1 Crore"), 10_000_000)
        self.assertEqual(app.parse_price_rupee("12,345"), 12_345)
        self.assertTrue(pd.isna(app.parse_price_rupee(None)))

    def test_parse_first_number_extracts_numeric_part(self):
        self.assertEqual(app.parse_first_number("86,226 kms"), 86_226)
        self.assertEqual(app.parse_first_number("1956 cc"), 1_956)
        self.assertTrue(pd.isna(app.parse_first_number("unknown")))

    def test_parse_ownership_maps_text_to_order_number(self):
        self.assertEqual(app.parse_ownership("First Owner"), 1)
        self.assertEqual(app.parse_ownership("Second Owner"), 2)
        self.assertEqual(app.parse_ownership("4th Owner"), 4)

    def test_build_input_row_matches_model_features(self):
        row = app.build_input_row(
            brand="Maruti",
            manufacture=2018,
            kms_driven_num=50_000,
            fuel_type="Petrol",
            transmission="Manual",
            owner_number=1,
            engine_cc=1197,
            seats_num=5,
        )

        self.assertEqual(row.shape, (1, len(app.FEATURE_COLS)))
        self.assertEqual(list(row.columns), app.FEATURE_COLS)
        self.assertEqual(row.loc[0, "car_age"], 8)

    def test_currency_conversion_and_formatting(self):
        rates = {"INR": 1.0, "USD": 0.012, "EUR": 0.011, "RUB": 1.05}

        self.assertEqual(app.convert_price(100_000, "USD", rates), 1_200)
        self.assertIn("Lakh", app.format_converted_price(100_000, "INR"))
        self.assertEqual(app.format_converted_price(1_200, "USD"), "1,200 USD")

    def test_fetch_exchange_rates_falls_back_when_network_is_unavailable(self):
        with patch("urllib.request.urlopen", side_effect=OSError("offline")):
            rates, is_live = app.fetch_exchange_rates()

        self.assertFalse(is_live)
        self.assertEqual(rates, app.FALLBACK_RATES)


if __name__ == "__main__":
    unittest.main()
