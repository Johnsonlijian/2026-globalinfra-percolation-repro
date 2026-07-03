"""R208-A4b: Crossref verification of the second batch of references (reviewer
P0-1/P0-3/P1-11 additions). Output: outputs/R208_A4b_new_refs_verified.csv
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

OUTDIR = Path(__file__).resolve().parents[2] / "data" / "R206_R208_ress_reframe"

CANDIDATES = [
    ("rosas2007", "10.1142/S0218127407018531",
     "Topological vulnerability of the European power grid under errors and attacks"),
    ("sole2008", "10.1103/PhysRevE.77.026102",
     "Robustness of the European power grids under intentional attack"),
    ("pagani2013", "10.1016/j.physa.2013.01.023",
     "The Power Grid as a complex network: A survey"),
    ("melnik2011", "10.1103/PhysRevE.83.036112",
     "Unreasonable effectiveness of tree-based theory for networks with clustering"),
    ("radicchi2015", "10.1038/nphys3374",
     "Predicting percolation thresholds in networks"),
    ("radicchi_castellano2016", "10.1103/PhysRevE.93.030302",
     "Beyond the locally treelike approximation for percolation on real networks"),
    ("li2011dimension", "10.1038/nphys1932",
     "Dimension of spatially embedded networks"),
    ("ouyang2014rail", "10.1016/j.ress.2013.10.003",
     "Comparisons of complex network based models and real train flow model to analyze Chinese railway vulnerability"),
    ("bashan2013", "10.1038/nphys2727",
     "The extreme vulnerability of interdependent spatially embedded networks"),
    ("albert2004", "10.1103/PhysRevE.69.025103",
     "Structural vulnerability of the North American power grid"),
]


def fetch(doi):
    url = f"https://api.crossref.org/works/{urllib.parse.quote(doi)}"
    req = urllib.request.Request(url, headers={"User-Agent": "R208-ref-check/1.0 (mailto:renlijian@imut.edu.cn)"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))["message"]


rows = []
for key, doi, title in CANDIDATES:
    row = {"key": key, "doi": doi, "candidate_title": title}
    try:
        msg = fetch(doi)
        got = (msg.get("title") or [""])[0]
        sim = SequenceMatcher(None, title.lower(), got.lower()).ratio()
        row.update(crossref_title=got, similarity=round(sim, 3),
                   journal=(msg.get("container-title") or [""])[0],
                   year=(msg.get("issued", {}).get("date-parts", [[None]])[0][0]),
                   volume=msg.get("volume", ""), page=msg.get("page", ""),
                   verdict="VERIFIED" if sim >= 0.7 else "CHECK_TITLE")
    except Exception as e:
        row.update(verdict="FAIL", error=str(e)[:200])
    rows.append(row)
    print(f"[A4b] {key}: {row['verdict']} ({row.get('similarity','')}) "
          f"{row.get('journal','')} {row.get('year','')} v{row.get('volume','')} p{row.get('page','')}", flush=True)
    time.sleep(1.0)

pd.DataFrame(rows).to_csv(OUTDIR / "R208_A4b_new_refs_verified.csv", index=False)
print("[A4b] DONE")
