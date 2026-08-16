"""Tests for the leakage-safe modelling workflow."""

import unittest
from pathlib import Path
import sys

import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from flight_delay.modeling import (  # noqa: E402
    CATEGORICAL_FEATURES,
    EXCLUDED_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    build_pipeline,
    load_data,
)


DATA_PATH = Path(__file__).parents[1] / "data/raw/flights_january_2019.csv"


class ModelingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = load_data(DATA_PATH)

    def test_dataset_contract(self) -> None:
        self.assertEqual(self.data.shape, (50_205, 24))
        self.assertEqual(self.data[TARGET].value_counts().to_dict(), {0: 41_498, 1: 8_707})

    def test_target_and_unavailable_fields_are_not_model_features(self) -> None:
        model_features = set(CATEGORICAL_FEATURES + NUMERIC_FEATURES)
        self.assertNotIn(TARGET, model_features)
        self.assertTrue(set(EXCLUDED_FEATURES).isdisjoint(model_features))

    def test_pipeline_accepts_raw_categories_and_unseen_values(self) -> None:
        train = self.data.iloc[:200].copy()
        validation = self.data.iloc[[200]].copy()
        validation.loc[validation.index[0], "PREVIOUS_AIRPORT"] = "UNSEEN AIRPORT"
        features = CATEGORICAL_FEATURES + NUMERIC_FEATURES
        pipeline = build_pipeline(LogisticRegression(max_iter=500))
        pipeline.fit(train[features], train[TARGET])
        prediction = pipeline.predict(validation[features])
        self.assertEqual(prediction.shape, (1,))


if __name__ == "__main__":
    unittest.main()
