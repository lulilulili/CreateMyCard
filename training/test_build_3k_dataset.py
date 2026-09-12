"""Regression tests for the deterministic 3B dataset builder."""
import json
import tempfile
import unittest
from pathlib import Path


class DatasetBuilderContractTests(unittest.TestCase):
    def test_cc_builder_emits_both_tasks_and_three_splits(self):
        from training.build_3b_dataset import build_dataset

        with tempfile.TemporaryDirectory() as directory:
            result = build_dataset(Path(directory), variants_per_case=36)
            self.assertGreaterEqual(result["counts"]["first_layer"], 100)
            self.assertGreaterEqual(result["counts"]["second_layer"], 500)
            for split in ("train", "dev", "test"):
                path = Path(directory) / ("cc_" + split + ".jsonl")
                self.assertTrue(path.exists())
                rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
                self.assertTrue(rows)
                self.assertTrue({row["task"] for row in rows} & {"first_layer"})
            combos = {}
            for split in ("train", "dev", "test"):
                for row in [json.loads(line) for line in (Path(directory) / ("cc_" + split + ".jsonl")).read_text(encoding="utf-8").splitlines()]:
                    combos.setdefault(tuple(row["caseId"].split("__")[1:3]), set()).add(split)
            self.assertTrue(all(len(splits) == 1 for splits in combos.values()))

    def test_cc_second_layer_is_a_numbered_choice(self):
        from training.build_3b_dataset import build_dataset

        with tempfile.TemporaryDirectory() as directory:
            build_dataset(Path(directory), variants_per_case=4)
            rows = [json.loads(line) for line in (Path(directory) / "cc_train.jsonl").read_text(encoding="utf-8").splitlines()]
            second = next(row for row in rows if row["task"] == "second_layer")
            self.assertEqual(json.loads(second["messages"][-1]["content"]).keys(), {"plan"})


if __name__ == "__main__":
    unittest.main()
