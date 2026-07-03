"""R205-S2: matched-move crossing dial + planar-family anchor universality.

Two escape-upgrades in one script.

PART A (anchor universality): the planar-end threshold scale pc*<k>/2 ~ 1 is tested
OUTSIDE road data on independent synthetic planar families (Delaunay, Gabriel,
relative-neighbourhood, Voronoi-edge graphs, diluted square lattices). If the scale
is a property of locally non-crossing embedding rather than of roads, every planar
family should sit near pc*<k>/2 = 1 while CEBH misses low-variance families.

PART B (matched-move dial): the published R107 dial ADDS random edges, which the
cold reads flagged as confounded (edge count and degree change together with the
crossings). Here the dial and the null use the SAME move family: degree-preserving
double-edge swaps in chord space, with opposite geometric acceptance rules:

  arm "noncross": accept a swap only if neither replacement chord properly crosses
                  a non-adjacent existing chord  (planarity-preserving control;
                  the strict-geometry-null move without length bins)
  arm "cross"   : accept a swap only if at least one replacement chord properly
                  crosses a non-adjacent existing chord (crossing-forcing dial)

Both arms preserve degree sequence and edge count EXACTLY, so the pc difference
between arms at matched accepted-swap fractions isolates projected crossings from
density/degree confounds. After each stage the ACTUAL chord-space crossing density
chi_chord is re-measured, converting the abstract swap fraction into the same
measured control parameter used in R205-S1.

Estimator: same rank-order union-find LCC-max-slope family as R106/R107.
Outputs: outputs/R205_S2_planar_family_anchor.csv
         outputs/R205_S2_matched_dial.csv
         outputs/R205_S2_summary.json
"""

from __future__ import annotations

import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from build_R105_finite_size_scaling import _find, cebh_pc_from_edges, load_road_simple  # noqa: E402

CACHE = ROOT / "cache" / "R61_osmnx_city_windows"
OUTDIR = ROOT / "data" / "R205_measured_geometry_dose_response"

DIAL_CITIES = [
    "bengaluru_india__drive_10000m.pkl",
    "osaka_japan__drive_10000m.pkl",
    "tehran_iran__drive_10000m.pkl",
    "cairo_egypt__drive_10000m.pkl",
]
FRACTIONS = [0.005, 0.01, 0.02, 0.04, 0.08]
MAX_ATTEMPT_FACTOR = 600


def pc_estimate(n, edges, n_perm=8, seed=3):
    """Rank-order union-find estimator, LCC max slope (R107 form)."""
    rng = np.random.default_rng(seed)
    grid = np.round(np.arange(0.05, 0.95, 0.025), 4)
    m = len(edges)
    tk = np.clip(np.rint(grid * m).astype(int), 0, m)
    curve = np.zeros(len(grid))
    for _ in range(n_perm):
        parent = np.arange(n)
        size = np.ones(n, int)
        largest = 1
        t = 0
        order = rng.permutation(m)
        samp = np.zeros(len(grid))
        for step, ei in enumerate(order, 1):
            u, v = int(edges[ei, 0]), int(edges[ei, 1])
            ru, rv = _find(parent, u), _find(parent, v)
            if ru != rv:
                if size[ru] < size[rv]:
                    ru, rv = rv, ru
                parent[rv] = ru
                size[ru] += size[rv]
                largest = max(largest, int(size[ru]))
            while t < len(grid) and step >= tk[t]:
                samp[t] = largest / n
                t += 1
        while t < len(grid):
            samp[t] = largest / n
            t += 1
        curve += samp
    curve /= n_perm
    return float(grid[int(np.argmax(np.gradient(curve, grid)))])


# ---------- chord-space segment machinery (matches null semantics, sec 2.5) ----------

def _orient(ax, ay, bx, by, cx, cy):
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def _proper_cross(p1, p2, p3, p4) -> bool:
    """Strict transversal crossing of segments p1p2 and p3p4 (no shared endpoints)."""
    d1 = _orient(*p3, *p4, *p1)
    d2 = _orient(*p3, *p4, *p2)
    d3 = _orient(*p1, *p2, *p3)
    d4 = _orient(*p1, *p2, *p4)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)) and d1 != 0 and d2 != 0 and d3 != 0 and d4 != 0


class ChordGrid:
    """Uniform spatial hash of chords for incremental crossing queries."""

    def __init__(self, pos, edges):
        self.pos = pos
        span = pos.max(0) - pos.min(0)
        lengths = np.linalg.norm(pos[edges[:, 0]] - pos[edges[:, 1]], axis=1)
        self.cell = max(float(np.median(lengths)) * 2.0, float(max(span)) * 1e-4)
        self.origin = pos.min(0)
        self.cells: dict[tuple, set] = {}
        self.edges = {}
        for i, (u, v) in enumerate(edges):
            self.add(i, int(u), int(v))

    def _cover(self, u, v):
        p, q = self.pos[u], self.pos[v]
        c0 = np.floor((np.minimum(p, q) - self.origin) / self.cell).astype(int)
        c1 = np.floor((np.maximum(p, q) - self.origin) / self.cell).astype(int)
        for ix in range(c0[0], c1[0] + 1):
            for iy in range(c0[1], c1[1] + 1):
                yield (ix, iy)

    def add(self, eid, u, v):
        self.edges[eid] = (u, v)
        for c in self._cover(u, v):
            self.cells.setdefault(c, set()).add(eid)

    def remove(self, eid):
        u, v = self.edges.pop(eid)
        for c in self._cover(u, v):
            s = self.cells.get(c)
            if s is not None:
                s.discard(eid)

    def chord_crosses(self, u, v, skip=()):
        """Does chord u-v properly cross any non-adjacent existing chord?"""
        pu, pv = self.pos[u], self.pos[v]
        seen = set(skip)
        for c in self._cover(u, v):
            for eid in self.cells.get(c, ()):
                if eid in seen:
                    continue
                seen.add(eid)
                a, b = self.edges[eid]
                if a in (u, v) or b in (u, v):
                    continue
                if _proper_cross(pu, pv, self.pos[a], self.pos[b]):
                    return True
        return False


def count_chi_chord(pos, edges):
    """Full chord-space crossing recount with shapely STRtree (C-speed)."""
    from shapely.geometry import LineString
    from shapely.strtree import STRtree
    geoms = [LineString([pos[int(u)], pos[int(v)]]) for u, v in edges]
    tree = STRtree(geoms)
    qi, ti = tree.query(geoms, predicate="crosses")
    cnt = 0
    for a, b in zip(qi.tolist(), ti.tolist()):
        if a >= b:
            continue
        ua, va = int(edges[a, 0]), int(edges[a, 1])
        ub, vb = int(edges[b, 0]), int(edges[b, 1])
        if ua in (ub, vb) or va in (ub, vb):
            continue
        cnt += 1
    return cnt / len(edges)


def run_dial_arm(task):
    """One (graph, arm) chain over cumulative accepted-swap fractions."""
    label, n, edges, pos, arm, seed = task
    rng = np.random.default_rng(seed)
    edges = edges.copy()
    grid = ChordGrid(pos, edges)
    edge_set = {tuple(sorted(e)) for e in edges.tolist()}
    m = len(edges)
    rows = []
    accepted = 0
    t0 = time.time()
    for frac in FRACTIONS:
        target = int(round(frac * m))
        attempts = 0
        max_att = MAX_ATTEMPT_FACTOR * max(target - accepted, 1)
        while accepted < target and attempts < max_att:
            attempts += 1
            i, j = rng.integers(0, m, 2)
            if i == j:
                continue
            a, b = int(edges[i, 0]), int(edges[i, 1])
            c, d = int(edges[j, 0]), int(edges[j, 1])
            if len({a, b, c, d}) < 4:
                continue
            if rng.random() < 0.5:
                new1, new2 = (a, c), (b, d)
            else:
                new1, new2 = (a, d), (b, c)
            k1, k2 = tuple(sorted(new1)), tuple(sorted(new2))
            if k1 == k2 or k1 in edge_set or k2 in edge_set:
                continue
            cross1 = grid.chord_crosses(*new1, skip=(i, j))
            cross2 = grid.chord_crosses(*new2, skip=(i, j))
            if arm == "noncross" and (cross1 or cross2):
                continue
            if arm == "cross" and not (cross1 or cross2):
                continue
            edge_set.discard(tuple(sorted((a, b))))
            edge_set.discard(tuple(sorted((c, d))))
            edge_set.add(k1)
            edge_set.add(k2)
            grid.remove(i)
            grid.remove(j)
            edges[i] = new1
            edges[j] = new2
            grid.add(i, *new1)
            grid.add(j, *new2)
            accepted += 1
        chi = count_chi_chord(pos, edges)
        pc = pc_estimate(n, edges, seed=seed + 11)
        lengths = np.linalg.norm(pos[edges[:, 0]] - pos[edges[:, 1]], axis=1)
        rows.append({
            "graph": label, "arm": arm, "target_fraction": frac,
            "achieved_fraction": accepted / m, "accepted_swaps": accepted,
            "chi_chord": chi, "pc": pc,
            "mean_degree": 2.0 * m / n,
            "pc_k_over_2": pc * (2.0 * m / n) / 2.0,
            "mean_chord_len_ratio": float(lengths.mean()),
            "seconds": round(time.time() - t0, 1),
        })
        print(f"[S2-dial] {label} {arm} f={frac} achieved={accepted / m:.4f} "
              f"chi={chi:.5f} pc={pc:.3f}", flush=True)
    return rows


# ---------- Part A: synthetic planar families ----------

def _delaunay_edges(pts):
    from scipy.spatial import Delaunay
    tri = Delaunay(pts)
    es = set()
    for s in tri.simplices:
        for i in range(3):
            a, b = int(s[i]), int(s[(i + 1) % 3])
            es.add((min(a, b), max(a, b)))
    return np.array(sorted(es), dtype=np.int64)


def _prune_gabriel(pts, edges):
    from scipy.spatial import cKDTree
    tree = cKDTree(pts)
    keep = []
    for a, b in edges:
        mid = (pts[a] + pts[b]) / 2.0
        r = np.linalg.norm(pts[a] - pts[b]) / 2.0
        idx = tree.query_ball_point(mid, r * (1 - 1e-9))
        if not any(i not in (a, b) for i in idx):
            keep.append((a, b))
    return np.array(keep, dtype=np.int64)


def _prune_rng(pts, edges):
    from scipy.spatial import cKDTree
    tree = cKDTree(pts)
    keep = []
    for a, b in edges:
        d = np.linalg.norm(pts[a] - pts[b])
        ia = set(tree.query_ball_point(pts[a], d * (1 - 1e-9)))
        ib = set(tree.query_ball_point(pts[b], d * (1 - 1e-9)))
        if not (ia & ib - {a, b}):
            keep.append((a, b))
    return np.array(keep, dtype=np.int64)


def _voronoi_graph(pts):
    from scipy.spatial import Voronoi
    vor = Voronoi(pts)
    lo, hi = pts.min(), pts.max()
    ok = lambda i: i >= 0 and np.all(vor.vertices[i] >= lo) and np.all(vor.vertices[i] <= hi)
    es, remap = set(), {}
    for v1, v2 in vor.ridge_vertices:
        if ok(v1) and ok(v2) and v1 != v2:
            es.add((min(v1, v2), max(v1, v2)))
    nodes = sorted({x for e in es for x in e})
    remap = {x: i for i, x in enumerate(nodes)}
    edges = np.array([(remap[a], remap[b]) for a, b in sorted(es)], dtype=np.int64)
    pos = vor.vertices[nodes]
    return len(nodes), edges, pos


def _square_diluted(L, keep_frac, rng):
    n = L * L
    es = []
    for x in range(L):
        for y in range(L):
            i = x * L + y
            if x + 1 < L:
                es.append((i, (x + 1) * L + y))
            if y + 1 < L:
                es.append((i, x * L + y + 1))
    es = np.array(es, dtype=np.int64)
    if keep_frac < 1.0:
        mask = rng.random(len(es)) < keep_frac
        es = es[mask]
    pos = np.array([(x, y) for x in range(L) for y in range(L)], float)
    return n, es, pos


def planar_family_row(name, n, edges, pos, seed):
    used = np.unique(edges)
    remap = -np.ones(int(used.max()) + 1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    edges = remap[edges]
    n_eff = len(used)
    pos = pos[used] if pos is not None else None
    pc = pc_estimate(n_eff, edges, n_perm=12, seed=seed)
    k = 2.0 * len(edges) / n_eff
    cebh = cebh_pc_from_edges(n_eff, edges)
    return {
        "family": name, "n_nodes": n_eff, "n_edges": len(edges), "mean_degree": k,
        "pc": pc, "pc_k_over_2": pc * k / 2.0,
        "pc_cebh": cebh, "cebh_gap": pc - cebh,
        "chi_chord": count_chi_chord(pos, edges) if pos is not None else 0.0,
    }


def part_a(seed=20260702):
    rng = np.random.default_rng(seed)
    rows = []
    for rep in range(2):
        pts = rng.random((20000, 2))
        dela = _delaunay_edges(pts)
        rows.append(planar_family_row(f"delaunay_rep{rep}", len(pts), dela, pts, seed + rep))
        rows.append(planar_family_row(f"gabriel_rep{rep}", len(pts), _prune_gabriel(pts, dela), pts, seed + rep))
        rows.append(planar_family_row(f"rng_rep{rep}", len(pts), _prune_rng(pts, dela), pts, seed + rep))
        nv, ev, pv = _voronoi_graph(pts)
        rows.append(planar_family_row(f"voronoi_rep{rep}", nv, ev, pv, seed + rep))
        for kf in (1.0, 0.85, 0.7):
            nq, eq, pq = _square_diluted(141, kf, rng)
            rows.append(planar_family_row(f"square_keep{kf}_rep{rep}", nq, eq, pq, seed + rep))
        print(f"[S2-A] rep {rep} done", flush=True)
    return rows


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    rows_a = part_a()
    pd.DataFrame(rows_a).to_csv(OUTDIR / "R205_S2_planar_family_anchor.csv", index=False)
    print("[S2] part A written", flush=True)

    tasks = []
    seed0 = 20260710
    rng = np.random.default_rng(99)
    pts = rng.random((20000, 2))
    dela = _delaunay_edges(pts)
    gab = _prune_gabriel(pts, dela)
    tasks.append(("delaunay20k", len(pts), dela, pts, "cross", seed0))
    tasks.append(("delaunay20k", len(pts), dela, pts, "noncross", seed0 + 1))
    tasks.append(("gabriel20k", len(pts), gab, pts, "cross", seed0 + 2))
    tasks.append(("gabriel20k", len(pts), gab, pts, "noncross", seed0 + 3))
    for i, fn in enumerate(DIAL_CITIES):
        n, edges, pos = load_road_simple(CACHE / fn)
        label = fn.split("__")[0]
        pos = np.asarray(pos, float)
        for j, arm in enumerate(("cross", "noncross")):
            tasks.append((label, n, np.asarray(edges, dtype=np.int64), pos, arm, seed0 + 10 + 2 * i + j))

    all_rows = []
    with Pool(min(12, len(tasks)), maxtasksperchild=1) as pool:
        for rows in pool.imap_unordered(run_dial_arm, tasks):
            all_rows.extend(rows)
            pd.DataFrame(all_rows).to_csv(OUTDIR / "R205_S2_matched_dial.csv", index=False)

    dial = pd.DataFrame(all_rows)
    summary = {"part_a_pc_k_over_2": {
        r["family"]: round(r["pc_k_over_2"], 4) for r in rows_a}}
    for g in dial["graph"].unique():
        sub = dial[dial["graph"] == g]
        summary[f"dial_{g}"] = {
            arm: {str(r["target_fraction"]): [round(r["chi_chord"], 5), round(r["pc"], 3)]
                  for _, r in sub[sub["arm"] == arm].iterrows()}
            for arm in ("cross", "noncross")}
    with open(OUTDIR / "R205_S2_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("[S2] DONE", flush=True)


if __name__ == "__main__":
    main()
