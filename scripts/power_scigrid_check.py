"""Mapped SciGRID simple-graph diagnostic using projected endpoint chords.

Bus coordinates come from the released SciGRID model. Geometry is one UTM32N
chord per simple edge, including transformer edges; parallel circuits are
collapsed and node-adjacent pairs are excluded. This is not an operating-data
validation, a line-route crossing measurement, or a power-flow analysis.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import networkx as nx
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely import wkt
from shapely.geometry import LineString
from shapely.strtree import STRtree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from canonical_percolation import estimate  # noqa: E402


def decode(values) -> list[str]:
    return [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values]


def clean_wkt(text: str) -> str:
    """Drop an EWKT SRID prefix so the geometry parses as plain WKT."""
    stripped = text.strip()
    if stripped.upper().startswith("SRID="):
        stripped = stripped.split(";", 1)[-1]
    return stripped


def read_network(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    with h5py.File(path, "r") as handle:
        bus = pd.DataFrame({
            "bus_id": decode(handle["buses_i"][:]),
            "name": decode(handle["buses_osm_name"][:]),
            "x": handle["buses_x"][:],
            "y": handle["buses_y"][:],
            "v_nom": handle["buses_v_nom"][:],
            "wkt": decode(handle["buses_wkt_srid_4326"][:]),
        })
        frames = []
        for name, group in (("line", "lines"), ("transformer", "transformers")):
            wkt_key = f"{group}_wkt_srid_4326"
            frames.append(pd.DataFrame({
                "kind": name,
                "bus0": decode(handle[f"{group}_bus0"][:]),
                "bus1": decode(handle[f"{group}_bus1"][:]),
                "voltage": (handle[f"{group}_voltage"][:]
                            if f"{group}_voltage" in handle else np.full(len(handle[f"{group}_bus0"]), np.nan)),
                "wkt": (decode(handle[wkt_key][:]) if wkt_key in handle
                        else [""] * len(handle[f"{group}_bus0"])),
            }))
    return bus, pd.concat(frames, ignore_index=True)


def predictors(degrees: np.ndarray) -> tuple[float, float]:
    mean_k = float(degrees.mean())
    mean_k2 = float((degrees**2).mean())
    return mean_k / (mean_k2 - mean_k), 2.0 / mean_k


def crossing_pairs(lines: pd.DataFrame) -> tuple[int, int]:
    shapes, kept = [], []
    for _, row in lines.iterrows():
        text = clean_wkt(row.wkt)
        if not text.startswith("LINESTRING"):
            continue
        geometry = wkt.loads(text)
        if len(geometry.coords) < 2:
            continue
        shapes.append(geometry)
        kept.append(row)
    tree = STRtree(shapes)
    pairs = set()
    for position, shape in enumerate(shapes):
        for other in tree.query(shape, predicate="intersects"):
            other = int(other)
            if other <= position:
                continue
            if {kept[position].bus0, kept[position].bus1} & {kept[other].bus0, kept[other].bus1}:
                continue
            if shape.crosses(shapes[other]):
                pairs.add((position, other))
    return len(pairs), len(shapes)


def analyse(bus: pd.DataFrame, edges: pd.DataFrame, label: str, orders: int, seed: int) -> dict:
    graph = nx.Graph()
    graph.add_nodes_from(bus.bus_id.tolist())
    for _, row in edges.iterrows():
        if row.bus0 != row.bus1 and row.bus0 in graph and row.bus1 in graph:
            graph.add_edge(row.bus0, row.bus1)
    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    largest = graph.subgraph(components[0]).copy()
    nodes = sorted(largest.nodes())
    index = {node: position for position, node in enumerate(nodes)}
    array = np.array([(index[u], index[v]) for u, v in largest.edges()], dtype=np.int64)
    degrees = np.array([degree for _, degree in largest.degree()], dtype=float)
    h_deg, h_2d = predictors(degrees)
    # One projected endpoint chord per SIMPLE graph edge, including transformers.
    # Raw parallel circuit rows must not inflate a numerator divided by simple M.
    positions = bus.set_index('bus_id')
    projection = Transformer.from_crs('EPSG:4326','EPSG:32632',always_xy=True)
    xy = {node: projection.transform(float(positions.loc[node,'x']),
                                     float(positions.loc[node,'y'])) for node in nodes}
    in_scope = pd.DataFrame([dict(bus0=u,bus1=v,wkt=LineString([xy[u],xy[v]]).wkt,
                                 voltage=float(positions.loc[u,'v_nom'])*1000)
                             for u,v in largest.edges()])
    chord_pairs, chord_used = crossing_pairs(in_scope)
    record, _ = estimate(len(nodes), array, n_orders=orders, seed=seed)
    n_edges = len(array)
    abstractly_planar, _ = nx.check_planarity(largest, counterexample=False)
    return {
        "scope": label,
        "n_nodes": len(nodes),
        "n_edges": n_edges,
        "mean_degree": float(degrees.mean()),
        "mean_voltage_kv": float(np.nanmean(in_scope.voltage.astype(float)) / 1000.0),
        "abstractly_planar": bool(abstractly_planar),
        "source_components": len(components),
        "largest_component_share": float(len(nodes) / graph.number_of_nodes()),
        "chi_chord": chord_pairs / n_edges,
        "crossing_pairs_chord": int(chord_pairs),
        "chords_used": int(chord_used),
        "pc": record["pc"],
        "pc_low": record["pc_low"],
        "pc_high": record["pc_high"],
        "pc_cebh": h_deg,
        "pc_dimension": h_2d,
        "gap_cebh": record["pc"] - h_deg,
        "gap_2d": record["pc"] - h_2d,
        "orders": orders,
        "crossing_geometry": "One UTM32N endpoint chord per simple edge; node-adjacent pairs excluded",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--orders", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    bus, edges = read_network(args.network)
    print(f"buses {len(bus)} edges {len(edges)} "
          f"({(edges.kind == 'line').sum()} lines, {(edges.kind == 'transformer').sum()} transformers)",
          flush=True)
    scopes = [
        ("germany_full", bus),
        ("germany_north", bus[bus.y >= 51.0]),
        ("germany_south", bus[bus.y < 51.0]),
    ]
    records = []
    for label, subset in scopes:
        if len(subset) < 100:
            print(f"{label}: too few buses, skipped", flush=True)
            continue
        scoped = edges[edges.bus0.isin(set(subset.bus_id)) & edges.bus1.isin(set(subset.bus_id))]
        record = analyse(subset, scoped, label, args.orders, args.seed)
        records.append(record)
        print(json.dumps(record, indent=2), flush=True)

    frame = pd.DataFrame(records)
    frame.to_csv(args.output / "power_scigrid_de.csv", index=False)
    if "germany_north" in set(frame.scope) and "germany_south" in set(frame.scope):
        north = frame[frame.scope.eq("germany_north")].iloc[0]
        south = frame[frame.scope.eq("germany_south")].iloc[0]
        contrast = {
            "north_minus_south_gap": float(north.gap_cebh - south.gap_cebh),
            "north_minus_south_chi": float(north.chi_chord - south.chi_chord),
            "higher_exposure_scope": "germany_north" if north.chi_chord > south.chi_chord
            else "germany_south",
        }
    else:
        contrast = {}
    full = frame[frame.scope.eq("germany_full")].iloc[0]
    summary = {
        "source": "SciGRID reference model of the German transmission network, "
                  "distributed as the PyPSA scigrid_de example network",
        "coordinate_character": (
            "Released WGS84 bus coordinates are projected to UTM32N. Each simple "
            "edge is represented by one endpoint chord. Actual line routes and "
            "utility operating data are not analysed."
        ),
        "scopes": frame.scope.tolist(),
        "full_gap_cebh_positive": bool(full.gap_cebh > 0),
        "full_chi_chord": float(full.chi_chord),
        "full_abstractly_planar": bool(full.abstractly_planar),
        "geometry_limit": (
            "Endpoint chords do not recover mapped line-route crossings. Transformer "
            "edges use bus endpoints, parallel circuits are collapsed, and "
            "node-adjacent edge pairs are excluded."
        ),
        "road_chi_median_for_reference": 0.021,
        "contrast": contrast,
        "scope_note": (
            "Reported as a mapped model of an operating grid, not validation against "
            "utility operating data. It is not added to the six source families for domain "
            "inference and does not inherit the road allowance."
        ),
    }
    (args.output / "power_scigrid_de.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
