"""Validate author-held snapshots, or explicitly rerun the declared graph estimator."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from canonical_percolation import METHOD_VERSION, estimate
from common import PACKAGE, safe_path, sha
from graph_helpers import (
    connect_components_by_degree_preserving_switches,
    graph_hash,
    predictors,
    seed_for,
    spectral_threshold,
)


def load_snapshot(row, data_root, allow_trusted_pickle=False):
    path = safe_path(data_root, row["snapshot_name"])
    if not path.is_file():
        raise FileNotFoundError(f"Author-held snapshot required: {row['snapshot_name']}")
    if sha(path) != row["snapshot_sha256"]:
        raise ValueError("Snapshot byte hash differs; a fresh web query is not an exact replay")
    if row["snapshot_format"] == "npz":
        with np.load(path, allow_pickle=False) as arrays:
            n, edges = int(arrays["n_nodes"]), arrays["edges"].copy()
    elif row["snapshot_format"] == "trusted_networkx_pickle":
        if not allow_trusted_pickle:
            raise ValueError(
                "Pickle executes code: pass --allow-trusted-pickle only for a trusted author-held snapshot"
            )
        graph = nx.convert_node_labels_to_integers(pickle.loads(path.read_bytes()))
        if graph.is_directed() or graph.is_multigraph():
            raise ValueError("Expected a simple undirected source graph")
        n, edges = len(graph), np.asarray(list(graph.edges()), dtype=np.int64)
    else:
        raise ValueError("This snapshot format has no verified reader")
    if edges.ndim != 2 or edges.shape[1] != 2 or edges.dtype.kind not in "iu":
        raise ValueError("Edges must be an integer endpoint matrix")
    if (
        n != int(row["n_nodes"])
        or len(edges) != int(row["n_edges"])
        or edges.min() < 0
        or edges.max() >= n
        or np.any(edges[:, 0] == edges[:, 1])
        or len(np.unique(np.sort(edges, axis=1), axis=0)) != len(edges)
    ):
        raise ValueError("Node/edge count, endpoint range or simple-graph contract differs")
    if graph_hash(n, edges) != row["input_hash"]:
        raise ValueError("Canonical topology hash differs")
    ordered = hashlib.sha256(edges.astype("<i8").tobytes()).hexdigest()
    if isinstance(row.get("ordered_edges_sha256"), str) and row["ordered_edges_sha256"]:
        if ordered != row["ordered_edges_sha256"]:
            raise ValueError(
                "Edge array ordering differs; identical topology is insufficient for bitwise RNG replay"
            )
    if row["method_version"] != METHOD_VERSION:
        raise ValueError("Estimator method version differs")
    return n, edges


def run(args):
    registry = pd.read_csv(args.registry).fillna("")
    rows = registry[registry.cohort == args.cohort]
    if args.instance:
        rows = rows[rows.instance == args.instance]
    if args.kind != "observed":
        if args.cohort != "reference":
            raise ValueError("Degree-null protocol applies only to the 71 reference roads")
        rows = rows[rows.domain == "road"]
    if args.limit:
        rows = rows.head(args.limit)
    if rows.empty:
        raise ValueError("No matching graph in the registry")
    config = json.loads(Path(args.config).read_text())
    settings = config["settings"]
    expected_orders = (
        settings["length_orders"]
        if args.cohort == "length_control"
        else settings["reference_orders"]
    )
    checks = []
    for row in rows.to_dict("records"):
        if (
            int(row["n_orders"]) != expected_orders
            or int(row["n_bootstrap"]) != settings["n_bootstrap"]
            or float(row["p_step"]) != settings["step"]
        ):
            raise ValueError("Registered reference estimator settings differ from configuration")
        n, edges = load_snapshot(row, args.data_root, args.allow_trusted_pickle)
        check = {
            "cohort": row["cohort"],
            "domain": row["domain"],
            "instance": row["instance"],
            "n_nodes": n,
            "n_edges": len(edges),
            "input_hash": graph_hash(n, edges),
            "snapshot_sha256": row["snapshot_sha256"],
            "simple_graph_verified": True,
            "settings_verified": True,
            "operation": "snapshot_validation_only",
        }
        if not args.validate_only:
            if args.kind == "observed":
                jobs = [
                    {
                        "replicate": 0,
                        "seed": int(row["seed"]),
                        "n_orders": int(row["n_orders"]),
                        "realization_hash": row["input_hash"],
                    }
                ]
            else:
                nulls = pd.read_csv(PACKAGE / "data" / f"null_{args.kind}.csv")
                jobs = nulls[nulls.instance.eq(row["instance"])].to_dict("records")
            if not jobs:
                raise ValueError("No declared realization jobs for this graph")
            for job in jobs:
                directory = Path(args.output)
                directory.mkdir(parents=True, exist_ok=True)
                name = f"{row['domain']}_{row['instance']}_{args.kind}_{job['replicate']}"
                if (directory / (name + ".json")).exists():
                    raise FileExistsError(
                        "Use a new output directory; existing estimates are immutable"
                    )
                used = edges.copy()
                if args.kind != "observed":
                    graph = nx.Graph()
                    graph.add_nodes_from(range(n))
                    graph.add_edges_from(used)
                    before = np.array([graph.degree(i) for i in range(n)])
                    seed = seed_for(
                        f"{row['domain']}:{row['instance']}:{args.kind}:{job['replicate']}"
                    )
                    nx.double_edge_swap(
                        graph, nswap=2 * len(used), max_tries=40 * len(used), seed=seed
                    )
                    if args.kind == "connected":
                        connect_components_by_degree_preserving_switches(graph, seed + 3001)
                    if not np.array_equal(before, [graph.degree(i) for i in range(n)]):
                        raise ValueError("Node-wise degree changed")
                    used = np.asarray(list(graph.edges()), dtype=np.int64)
                if graph_hash(n, used) != job["realization_hash"]:
                    raise ValueError(
                        "Null realization differs: check byte-identical snapshot and pinned runtime"
                    )
                if args.kind != "observed" and int(job["n_orders"]) != settings["null_orders"]:
                    raise ValueError("Null estimator settings differ")
                result, curves = estimate(
                    n,
                    used,
                    n_orders=int(job["n_orders"]),
                    n_bootstrap=int(row["n_bootstrap"]),
                    seed=int(job["seed"]),
                    step=float(row["p_step"]),
                )
                result.update(
                    instance=row["instance"],
                    domain=row["domain"],
                    kind=args.kind,
                    replicate=int(job["replicate"]),
                    seed=int(job["seed"]),
                    input_hash=row["input_hash"],
                    realization_hash=graph_hash(n, used),
                )
                degrees = np.bincount(used.ravel(), minlength=n)
                pred = predictors(degrees)
                pc_nb, radius, residual = spectral_threshold(n, used)
                result.update(
                    pc_cebh=pred[0],
                    pc_dimension=pred[1],
                    pc_junction_anchor=pred[2],
                    pc_nonbacktracking=pc_nb,
                    nb_spectral_radius=radius,
                    nb_residual=residual,
                )
                for key, value in list(result.items()):
                    if isinstance(value, float) and not np.isfinite(value):
                        result[key] = None
                directory = Path(args.output)
                directory.mkdir(parents=True, exist_ok=True)
                name = f"{row['domain']}_{row['instance']}_{args.kind}_{job['replicate']}"
                with (directory / (name + ".json")).open("x", encoding="utf-8") as handle:
                    json.dump(result, handle, indent=2, allow_nan=False)
                pd.DataFrame(
                    {
                        "p": curves["grid"],
                        "Q": curves["curves"].mean(axis=1),
                        "derivative": curves["derivative"].mean(axis=1),
                        "fixed_count_mean": curves["micro_mean"],
                    }
                ).to_csv(directory / (name + "_response.csv"), index=False)
            check["operation"] = "declared_estimator_rerun"
        checks.append(check)
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(checks).to_csv(directory / "snapshot_validation.csv", index=False)
    print(f"{len(checks)} snapshot contracts verified; mode={checks[0]['operation']}")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument(
        "--data-root",
        required=True,
        help="Directory of author-held snapshots, not supplied in this archive",
    )
    cli.add_argument("--registry", default=str(PACKAGE / "data/source_registry.csv"))
    cli.add_argument("--config", default=str(PACKAGE / "config/reproduction.json"))
    cli.add_argument("--output", default=str(PACKAGE / "generated/snapshot_checks"))
    cli.add_argument(
        "--cohort", choices=["reference", "external", "length_control"], default="reference"
    )
    cli.add_argument("--kind", choices=["observed", "degree", "connected"], default="observed")
    cli.add_argument("--instance")
    cli.add_argument("--limit", type=int, default=0)
    cli.add_argument("--validate-only", action="store_true")
    cli.add_argument("--allow-trusted-pickle", action="store_true")
    run(cli.parse_args())
