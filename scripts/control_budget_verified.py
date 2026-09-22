"""Rebuild budget diagnostics from identified graphs with complete per-run records."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from canonical_percolation import estimate

BUDGETS = (256, 512, 1024, 2048, 4096)
MULTIPLIERS = (0.02, 0.1, 1.0, 5.0)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def job(spec):
    output = Path(spec['output'])
    key = spec['key']
    cache = output / 'records' / (key + '.json')
    if cache.exists():
        record = json.loads(cache.read_text())
        if record['spec'] != spec:
            raise ValueError('Cached job specification changed: ' + key)
        return record['rows']
    with np.load(spec['graph'], allow_pickle=False) as f:
        n = int(f['n_nodes'])
        original = np.asarray(f['edges'], dtype=np.int64)
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(map(tuple, original))
    degree = np.array([graph.degree(i) for i in range(n)])
    m = graph.number_of_edges()
    swaps = max(1000, int(2 * m * spec['multiplier']))
    nx.double_edge_swap(graph, nswap=swaps, max_tries=swaps * 50, seed=spec['seed'])
    assert graph.number_of_edges() == m and nx.number_of_selfloops(graph) == 0
    assert np.array_equal(degree, [graph.degree(i) for i in range(n)])
    edges = np.array(sorted(tuple(sorted(e)) for e in graph.edges()), dtype=np.int64)
    graph_hash = digest(edges.tobytes())
    np.savez_compressed(output / 'private_controls' / (key + '.npz'),
                        n_nodes=n, edges=edges)
    max_orders = 4096 if spec['ladder'] == 'order' else 1024
    result, _ = estimate(n, edges, n_orders=max_orders, seed=spec['seed'])
    rows = []
    for budget in BUDGETS if spec['ladder'] == 'order' else (1024,):
        pc = result[f'pc_first_{budget}_orders']
        rows.append(dict(ladder=spec['ladder'], instance=spec['instance'],
                         macro_region=spec['macro_region'], replicate=spec['replicate'],
                         swap_multiplier=spec['multiplier'], n_nodes=n, n_edges=m,
                         n_swaps=swaps, swaps_per_edge=swaps/m, orders=budget,
                         seed=spec['seed'], graph_sha256=graph_hash,
                         source_sha256=spec['source_sha256'],
                         pc_control=pc, pc_observed=spec['pc_observed'],
                         pc_cebh=spec['pc_cebh'], null_gap=pc-spec['pc_cebh'],
                         organization_residual=spec['pc_observed']-pc,
                         degree_sequence_exact=True))
    cache.write_text(json.dumps(dict(spec=spec, rows=rows), indent=2)+'\n')
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--graphs', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=6)
    args = ap.parse_args()
    for name in ('records', 'private_controls'):
        (args.output / name).mkdir(parents=True, exist_ok=True)
    observed = pd.read_csv(args.data / 'observed.csv')
    roads = observed[observed.domain.eq('road')].merge(
        pd.read_csv(args.data / 'road_regions.csv'), on='instance', validate='one_to_one')
    pools = {reg: sorted(group.instance) for reg, group in roads.groupby('macro_region')}
    regions = sorted(pools, key=lambda r: (-len(pools[r]), r))
    cities = []
    while len(cities) < 24:
        for region in regions:
            if pools[region] and len(cities) < 24:
                cities.append(pools[region].pop(0))
    tasks = []
    for city in cities:
        row = roads[roads.instance.eq(city)].iloc[0]
        path = args.graphs / ('road_' + city.replace(' ', '_') + '.npz')
        if not path.exists():
            path = args.graphs / ('road_' + city + '.npz')
        source_hash = digest(path.read_bytes())
        common = dict(instance=city, macro_region=row.macro_region,
                      graph=str(path.resolve()), output=str(args.output.resolve()),
                      source_sha256=source_hash, pc_observed=float(row.pc),
                      pc_cebh=float(row.pc_cebh))
        for rep in (1, 2):
            tasks.append(dict(common, ladder='order', replicate=rep, multiplier=1.0,
                              seed=20260905+101*rep,
                              key=f'order_{city.replace(" ", "_")}_{rep}'))
        if city in cities[:8]:
            for multiplier in MULTIPLIERS:
                tasks.append(dict(common, ladder='mixing', replicate=1,
                                  multiplier=multiplier, seed=20260905+7919,
                                  key=f'mixing_{city.replace(" ", "_")}_{multiplier:g}'))
    protocol = dict(status='COMPUTATION_SPECIFICATION', n_tasks=len(tasks),
                    cities=cities, mixing_cities=cities[:8], order_budgets=BUDGETS,
                    multiplier_base='2M successful switches; minimum 1000',
                    selection='Alphabetical within macro-region, round-robin; no outcome selection',
                    purpose='Post hoc numerical sensitivity, not preregistration', tasks=tasks)
    (args.output/'computation_specification.json').write_text(json.dumps(protocol, indent=2)+'\n')
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(job, spec): spec['key'] for spec in tasks}
        for future in as_completed(futures):
            rows.extend(future.result())
            print(f'{len(rows)} rows: {futures[future]} complete', flush=True)
    frame = pd.DataFrame(rows).sort_values(['ladder','instance','replicate','n_swaps','orders'])
    assert len(frame) == 24*2*5 + 8*4
    assert np.allclose(frame.null_gap + frame.pc_cebh, frame.pc_control)
    assert np.allclose(frame.organization_residual + frame.pc_control, frame.pc_observed)
    frame.to_csv(args.output/'control_ladders.csv', index=False)
    order = frame[frame.ladder.eq('order')]
    mix = frame[frame.ladder.eq('mixing')]
    paired = order.pivot(index=['instance','replicate'], columns='orders', values='pc_control')
    summary = dict(status='COMPLETE_VERIFIED', rows=len(frame), order_cities=24,
                   order_replicates=2, mixing_cities=8,
                   order_means=order.groupby('orders')[['null_gap','organization_residual']].mean().to_dict(),
                   order_max_abs_paired_shift=float((paired.sub(paired[256],axis=0)).abs().max().max()),
                   mixing_means=mix.groupby('swap_multiplier')[['null_gap','organization_residual']].mean().to_dict(),
                   min_contrast=float(frame.organization_residual.min()),
                   csv_sha256=digest((args.output/'control_ladders.csv').read_bytes()),
                   scope='Fresh finite controls; no uniform-mixing or causal-geometry claim')
    (args.output/'control_ladders.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    main()
