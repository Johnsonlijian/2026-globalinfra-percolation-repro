"""Optional fixed-count stress tests; snapshots/geography are never downloaded."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from canonical_percolation import count_curve, peak
from common import PACKAGE, safe_path, sha
from graph_helpers import seed_for
from pyproj import CRS, Transformer
from rerun_topology import load_snapshot


def scenarios(n, edges, slug, xy=None):
    grid = np.linspace(0, 1, 401)
    indices = np.rint(grid * len(edges)).astype(int)
    seed = seed_for("scenario:" + slug)
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(edges)
    rows = []

    def record(order, kind, replicate):
        curve = count_curve(n, edges, order)
        sampled = curve[indices]
        pc = peak(grid, np.gradient(sampled, grid))
        rows.append(
            {
                "scenario": kind,
                "replicate": replicate,
                "pc_fractional": pc,
                "point_endpoint": pc in (0, 1),
                "auc_fractional": float(np.trapezoid(curve, dx=1 / len(edges))),
                "lcc_at_half": float(sampled[200]),
            }
        )

    for i in range(3):
        replicate_seed = seed + 1009 * i
        scores = nx.edge_betweenness_centrality(graph, k=min(n, 64), seed=replicate_seed)
        values = np.array([scores[(int(u), int(v))] for u, v in edges])
        order = np.lexsort((np.random.default_rng(replicate_seed).random(len(edges)), values))
        record(order, "targeted", i)
    if xy is not None:
        center = xy.mean(axis=0)
        radius = np.linalg.norm(xy - center, axis=1)
        eligible = np.flatnonzero(radius <= np.quantile(radius, 0.7))
        centers = np.random.default_rng(seed + 7001).choice(eligible, 12, replace=False)
        midpoints = (xy[edges[:, 0]] + xy[edges[:, 1]]) / 2
        for i, center in enumerate(centers):
            distance = np.linalg.norm(midpoints - xy[center], axis=1)
            record(np.argsort(-distance, kind="stable"), "localized", i)
    return rows


def main(args):
    registry = pd.read_csv(args.registry).fillna("")
    frame = registry[(registry.cohort == "reference") & (registry.instance == args.instance)]
    if len(frame) != 1:
        raise ValueError("Choose one primary instance")
    row = frame.iloc[0].to_dict()
    n, edges = load_snapshot(row, args.data_root)
    xy, geographic_crs, coordinate_sha256 = None, "unmeasured", None
    if args.coordinates:
        if row["domain"] not in ("road", "rail"):
            raise ValueError(
                "Water/power coordinates are not authorized as verified hazard geography"
            )
        coordinates = Path(args.coordinates)
        metadata = (
            json.loads(Path(args.coordinate_contract).read_text())
            if args.coordinate_contract
            else {}
        )
        if (
            metadata.get("input_hash") != row["input_hash"]
            or metadata.get("coordinate_sha256") != sha(coordinates)
            or metadata.get("source_crs") != "EPSG:4326"
            or metadata.get("node_order") != "frozen_integer_labels"
        ):
            raise ValueError("Author-held coordinate contract is missing or mismatched")
        with np.load(coordinates, allow_pickle=False) as arrays:
            native = arrays["xy"]
        if native.shape != (n, 2) or not np.isfinite(native).all():
            raise ValueError("Expected n-by-2 longitude/latitude in the frozen node order")
        lon, lat = native.mean(axis=0)
        target = CRS.from_proj4(f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m")
        transformer = Transformer.from_crs("EPSG:4326", target, always_xy=True)
        xy = np.column_stack(transformer.transform(native[:, 0], native[:, 1]))
        geographic_crs, coordinate_sha256 = target.to_string(), sha(coordinates)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    slug = f"{row['domain']}_{''.join(c if c.isalnum() else '_' for c in row['instance'])}"
    path = safe_path(output, slug + ".json")
    if path.exists():
        raise FileExistsError(
            "Use a new output directory; do not overwrite existing scientific results"
        )
    results = scenarios(n, edges, slug, xy)
    for result in results:
        result.update(
            instance=row["instance"],
            domain=row["domain"],
            system_id=row["system_id"],
            input_hash=row["input_hash"],
        )
    payload = {
        "records": results,
        "settings": {
            "grid_points": 401,
            "targeted_replicates": 3,
            "betweenness_sources": min(n, 64),
            "local_centres": 12 if xy is not None else 0,
            "seed": seed_for("scenario:" + slug),
        },
        "source_sha256": row["snapshot_sha256"],
        "coordinate_sha256": coordinate_sha256,
        "geographic_crs": geographic_crs,
        "script_sha256": sha(__file__),
    }
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
    print(f"Computed {len(results)} fixed-count scenarios on one verified snapshot")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data-root", required=True)
    cli.add_argument("--registry", default=str(PACKAGE / "data/source_registry.csv"))
    cli.add_argument("--instance", required=True)
    cli.add_argument("--coordinates", help="Optional author-held NPZ xy array, not included")
    cli.add_argument(
        "--coordinate-contract", help="JSON binds coordinate SHA256, input_hash, CRS and node order"
    )
    cli.add_argument("--output", required=True)
    main(cli.parse_args())
