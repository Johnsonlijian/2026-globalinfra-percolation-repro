"""Independent data-unit, schema, graph-binding and regenerated-table checks."""

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from analyze import run
from common import PACKAGE, Context, sha
from numpy.testing import assert_allclose
from screening import allowance, evaluate_allowance


class DerivedPackageTests(unittest.TestCase):
    def test_manifest_and_no_raw_graphs_or_private_columns(self):
        manifest = json.loads((PACKAGE / "DATA_MANIFEST.json").read_text())
        for item in manifest["datasets"].values():
            path = PACKAGE / item["path"]
            self.assertEqual(sha(path), item["sha256"], item["path"])
            if path.suffix == ".csv":
                fields = set(pd.read_csv(path, nrows=0).columns)
                self.assertFalse(
                    fields
                    & {
                        "private_file",
                        "graph_cache_file",
                        "source_note",
                        "pc_legacy_estimate",
                        "pc_rank",
                        "elapsed_seconds",
                        "stage_edge_snapshot",
                    }
                )
        forbidden = {".pkl", ".pickle", ".npz", ".inp", ".osm", ".pbf", ".zip"}
        self.assertFalse([p for p in PACKAGE.rglob("*") if p.is_file() and p.suffix in forbidden])

    def test_scientific_populations_not_mixed(self):
        ref = pd.read_csv(PACKAGE / "data/observed.csv")
        ext = pd.read_csv(PACKAGE / "data/external.csv")
        water = pd.read_csv(PACKAGE / "data/water_additional.csv")
        self.assertEqual(
            ref.groupby("domain").size().to_dict(), {"power": 10, "rail": 6, "road": 71, "water": 4}
        )
        self.assertEqual(ref[ref.domain == "power"].system_id.nunique(), 6)
        self.assertTrue(ref.pc.between(0, 1, inclusive="neither").all())
        self.assertEqual(float(water.set_index("instance").loc["C-Town", "pc"]), 1.0)
        self.assertEqual(len(water) + int(ref.domain.eq("water").sum()), 7)
        self.assertFalse(set(ref.instance) & set(ext.instance))
        self.assertTrue(ext.groupby("macro_region").size().eq(3).all())
        q = allowance(
            abs(ref[ref.domain == "road"].pc - ref[ref.domain == "road"].pc_dimension).to_numpy()
        )
        decisions = evaluate_allowance(1 - ext.pc_dimension, 1 - ext.pc, q)
        assert_allclose(decisions.retained.mean(), 0.7028996808174638, atol=1e-14)
        self.assertTrue((abs(ext.pc - ext.pc_dimension) <= q).all())
        self.assertEqual(int((decisions.unsafe_retained + decisions.safe_retained > 0).sum()), 0)

    def test_geometry_and_clean_control_binding(self):
        ref = pd.read_csv(PACKAGE / "data/observed.csv")
        rail = pd.read_csv(PACKAGE / "data/rail_geometry.csv")
        roads = pd.read_csv(PACKAGE / "data/road_geometry.csv")
        for row in rail.to_dict("records"):
            target = ref[(ref.domain == "rail") & (ref.instance == row["city"])].iloc[0]
            self.assertEqual(row["input_hash"], target.input_hash)
            self.assertEqual(row["n_edges"], target.n_edges)
        assert_allclose(roads.chi_long + roads.chi_short, roads.chi_poly, atol=1e-14)
        length = pd.read_csv(PACKAGE / "data/length_control.csv")
        self.assertEqual(len(length), 36)
        self.assertTrue(length.zero_selfloops.all() and length.invariants_pass.all())
        names = {"bengaluru_india": "Bengaluru", "osaka_japan": "Osaka", "tehran_iran": "Tehran"}
        for row in length[length.stage_index == 0].to_dict("records"):
            target = ref[(ref.domain == "road") & (ref.instance == names[row["graph"]])].iloc[0]
            self.assertEqual(row["graph_sha256"], target.input_hash)
            self.assertEqual(row["n_edges"], target.n_edges)
        scenarios = pd.read_csv(PACKAGE / "data/scenarios.csv")
        self.assertEqual(len(scenarios), 312)
        self.assertFalse(
            scenarios[
                (scenarios.domain.isin(["power", "water"])) & (scenarios.scenario == "localized")
            ].shape[0]
        )

    def test_regenerated_tables_match_included_results(self):
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(
                data_root=PACKAGE, config=PACKAGE / "config/reproduction.json", output=directory
            )
            run(Context(args))
            for name, keys in [
                ("domain_metrics", ["domain", "predictor"]),
                ("null_summary", ["kind"]),
                ("external_predictors", ["predictor"]),
                ("scenario_instances", ["domain", "instance", "scenario"]),
            ]:
                expected = (
                    pd.read_csv(PACKAGE / "data" / (name + ".csv")).set_index(keys).sort_index()
                )
                actual = pd.read_csv(Path(directory) / (name + ".csv")).set_index(keys).sort_index()
                self.assertEqual(list(expected.index), list(actual.index))
                columns = expected.select_dtypes(include="number").columns.intersection(
                    actual.columns
                )
                assert_allclose(actual[columns], expected[columns], atol=2e-12, rtol=1e-12)
