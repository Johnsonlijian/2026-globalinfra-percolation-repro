"""Alternative finite-graph observables with exact component-size bookkeeping."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit
from canonical_percolation import binomial_kernel, endpoint_slope, peak


@njit(cache=True)
def kth(tree, rank):
    index = 0
    bit = 1
    while bit * 2 < len(tree):
        bit *= 2
    while bit:
        nxt = index + bit
        if nxt < len(tree) and tree[nxt] < rank:
            index = nxt
            rank -= tree[nxt]
        bit //= 2
    return index + 1


@njit(cache=True)
def update(tree, size, delta):
    while size < len(tree):
        tree[size] += delta
        size += size & -size


@njit(cache=True)
def component_curves(n, edges, order):
    parent = np.arange(n)
    sizes = np.ones(n, dtype=np.int64)
    tree = np.zeros(n + 1, dtype=np.int64)
    update(tree, 1, n)
    components = n
    largest = 1
    second = 1 if n > 1 else 0
    sum_squares = float(n)
    out = np.empty((len(order) + 1, 3))
    out[0] = (largest/n, second/n, 1.0 if n > 1 else 0.0)
    for step in range(len(order)):
        e = order[step]
        u, v = edges[e, 0], edges[e, 1]
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        if u != v:
            if sizes[u] < sizes[v]:
                u, v = v, u
            a, b = sizes[u], sizes[v]
            update(tree, a, -1)
            update(tree, b, -1)
            update(tree, a+b, 1)
            parent[v] = u
            sizes[u] += sizes[v]
            components -= 1
            largest = max(largest, a+b)
            second = kth(tree, components-1) if components > 1 else 0
            sum_squares += 2.0*a*b
        susceptibility = ((sum_squares-largest*largest)/(n-largest)
                          if largest < n else 0.0)
        out[step+1] = (largest/n, second/n, susceptibility)
    return out


def compute(spec):
    dest = Path(spec['output'])/'records'/(spec['system_id']+'.json')
    if dest.exists():
        saved = json.loads(dest.read_text())
        if saved['spec'] != spec:
            raise ValueError('Changed specification')
        return saved['result']
    path = Path(spec['graph'])
    with np.load(path, allow_pickle=False) as f:
        n, edges = int(f['n_nodes']), np.asarray(f['edges'], dtype=np.int64)
    m = len(edges)
    grid = np.linspace(0,1,401)
    qk, dk = binomial_kernel(m,grid), binomial_kernel(m-1,grid)
    rng = np.random.default_rng(spec['seed'])
    mean_s2 = np.zeros(401)
    mean_sus = np.zeros(401)
    derivative = np.empty((401,spec['orders']))
    for start in range(0,spec['orders'],32):
        stop = min(start+32,spec['orders'])
        curves = np.array([component_curves(n,edges,rng.permutation(m)) for _ in range(stop-start)])
        derivative[:,start:stop] = m*(dk@np.diff(curves[:,:,0],axis=1).T)
        mean_s2 += (qk@curves[:,:,1].T).sum(axis=1)/spec['orders']
        mean_sus += (qk@curves[:,:,2].T).sum(axis=1)/spec['orders']
    derivative[-1] = endpoint_slope(n,edges)[0]
    per_order = np.array([peak(grid,derivative[:,i]) for i in range(spec['orders'])])
    result = dict(instance=spec['instance'], domain=spec['domain'],
                  orders=spec['orders'], source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  released_p_star=spec['pc'], h_deg=spec['pc_cebh'], h_2d=spec['pc_dimension'],
                  p_star_s1=peak(grid,derivative.mean(axis=1)),
                  p_peak_s2=peak(grid,mean_s2), p_peak_susceptibility=peak(grid,mean_sus),
                  p_star_order_mean=float(per_order.mean()),
                  p_star_order_median=float(np.median(per_order)))
    for key in ('p_star_s1','p_peak_s2','p_peak_susceptibility','p_star_order_mean','p_star_order_median'):
        result['gap_deg_'+key]=result[key]-result['h_deg']
    dest.write_text(json.dumps(dict(spec=spec,result=result),indent=2)+'\n')
    return result


def self_test():
    import networkx as nx
    for n in range(2,10):
        edges=np.array(list(nx.complete_graph(n).edges()),dtype=np.int64)
        for seed in range(3):
            order=np.random.default_rng(seed).permutation(len(edges))
            values=component_curves(n,edges,order)
            g=nx.empty_graph(n)
            for step in range(len(order)+1):
                sizes=sorted((len(c) for c in nx.connected_components(g)),reverse=True)
                expected=(sizes[0]/n, sizes[1]/n if len(sizes)>1 else 0,
                          sum(s*s for s in sizes[1:])/sum(sizes[1:]) if len(sizes)>1 else 0)
                np.testing.assert_allclose(values[step],expected)
                if step<len(order):
                    g.add_edge(*edges[order[step]])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--graphs',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--workers',type=int,default=6)
    ap.add_argument('--orders',type=int,default=512)
    args=ap.parse_args()
    self_test()
    (args.output/'records').mkdir(parents=True,exist_ok=True)
    tasks=[]
    for row in pd.read_csv(args.data/'observed.csv').to_dict('records'):
        stem=row['domain']+'_'+row['instance']
        graph=args.graphs/(stem+'.npz')
        if not graph.exists(): graph=args.graphs/(stem.replace(' ','_')+'.npz')
        if not graph.exists(): raise FileNotFoundError(graph)
        tasks.append({k:row[k] for k in ('instance','domain','pc','pc_cebh','pc_dimension')} |
                     dict(system_id=stem.replace(' ','_'),graph=str(graph.resolve()),
                          output=str(args.output.resolve()),orders=args.orders,seed=20260920))
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(compute,s) for s in tasks]):
            row=future.result(); rows.append(row)
            print(f'{len(rows)}/91 {row["instance"]}',flush=True)
    frame=pd.DataFrame(rows).sort_values(['domain','instance'])
    assert len(frame)==91
    frame.to_csv(args.output/'response_definition_check.csv',index=False)
    summary=dict(status='COMPLETE_VERIFIED',graphs=91,orders=args.orders,
                 algorithm_test='exact NetworkX component comparison at every step; n=2..9, three orders each',
                 positives={c:int((frame[c]>0).sum()) for c in frame if c.startswith('gap_deg_')},
                 min_gaps={c:float(frame[c].min()) for c in frame if c.startswith('gap_deg_')})
    (args.output/'response_definition_check.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__': main()
