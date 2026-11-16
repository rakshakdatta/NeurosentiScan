import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.benchmark import LABELS, classical_model, load_csv, paired_summary, split_rows


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.labels = np.tile(np.arange(3), 30)
        self.features = pd.DataFrame({"eeg_feature": np.arange(90, dtype="float32")})

    def test_split_is_deterministic_and_complete(self):
        first = split_rows(self.features, self.labels)
        second = split_rows(self.features, self.labels)
        for a, b in zip(first[:3], second[:3]):
            np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(np.sort(np.concatenate(first[:3])), np.arange(90))

    def test_duplicate_rows_stay_in_one_partition(self):
        self.features.iloc[3] = self.features.iloc[0]
        train, validation, test, protocol = split_rows(self.features, self.labels)
        self.assertIn("duplicates", protocol)
        self.assertTrue(any({0, 3}.issubset(rows) for rows in (train, validation, test)))

    def test_subject_groups_do_not_cross_partitions(self):
        groups = np.repeat(np.arange(15), 6)
        train, validation, test, protocol = split_rows(self.features, self.labels, groups)
        self.assertEqual(protocol, "participant-held-out")
        self.assertFalse(set(groups[train]) & set(groups[test]))
        self.assertFalse(set(groups[validation]) & set(groups[test]))
        self.assertFalse(set(groups[train]) & set(groups[validation]))

    def test_group_column_is_excluded_from_features(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "eeg.csv"
            pd.DataFrame({"value": [0, 1, 2], "participant": ["A", "A", "B"], "label": LABELS}).to_csv(path, index=False)
            features, labels, groups = load_csv(path, "participant")
        self.assertEqual(list(features.columns), ["value"])
        self.assertEqual(groups.tolist(), ["A", "A", "B"])
        np.testing.assert_array_equal(labels, [0, 1, 2])

    def test_scaler_fits_on_supplied_training_rows(self):
        train, _, test, _ = split_rows(self.features, self.labels)
        model = classical_model("logistic").fit(self.features.iloc[train], self.labels[train])
        scaler = model.named_steps["standardscaler"]
        np.testing.assert_allclose(scaler.mean_, self.features.iloc[train].mean())
        self.assertEqual(len(model.predict(self.features.iloc[test])), len(test))

    def test_paired_comparison_counts_disagreements(self):
        result = paired_summary({"a": np.array([0, 1, 0]), "b": np.array([1, 1, 0])}, np.array([0, 0, 0]))
        self.assertEqual(result.loc[0, "a_only_correct"], 1)
        self.assertEqual(result.loc[0, "b_only_correct"], 0)
        self.assertEqual(result.loc[0, "p_holm"], 1.0)


if __name__ == "__main__":
    unittest.main()
