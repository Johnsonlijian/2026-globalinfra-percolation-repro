"""R206-A4: Crossref verification of candidate reliability/vulnerability references
before insertion into the RESS manuscript. DOI-first (api.crossref.org/works/DOI),
title-similarity checked; candidates without a confident DOI use a bibliographic
query and require manual review. Output: outputs/R206_A4_reliability_refs_verified.csv
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
    ("ouyang2014", "10.1016/j.ress.2013.06.040",
     "Review on modeling and simulation of interdependent critical infrastructure systems"),
    ("zio2016", "10.1016/j.ress.2016.02.009",
     "Challenges in the vulnerability and risk analysis of critical infrastructures"),
    ("johansson2010", "10.1016/j.ress.2010.06.010",
     "An approach for modelling interdependent infrastructures in the context of vulnerability analysis"),
    ("larocca2015", "10.1111/risa.12281",
     "Topological performance measures as surrogates for physical flow models for risk and vulnerability analysis for electric power systems"),
    ("hines2010", "10.1063/1.3489887",
     "Do topological models provide good information about electricity infrastructure vulnerability"),
    ("duenasosorio2009", "10.1016/j.strusafe.2008.06.007",
     "Cascading failures in complex infrastructure systems"),
    ("mattsson2015", "10.1016/j.tra.2015.06.002",
     "Vulnerability and resilience of transport systems - a discussion of recent research"),
    ("ganin2017", "10.1126/sciadv.1701079",
     "Resilience and efficiency in transportation networks"),
    ("rocchetta2018", "10.1016/j.ijepes.2017.11.047",
     "Assessment of power grid vulnerabilities accounting for stochastic loads and model imprecision"),
    ("fang2015", "10.1111/risa.12287",
     "Optimization of cascade-resilient electrical infrastructures and its validation by power flow modeling"),
    ("cadini2017", "10.1016/j.apenergy.2016.10.086",
     "A modeling and simulation framework for the reliability/availability assessment of a power transmission grid subject to cascading failures under extreme weather conditions"),
    ("murray2008", "10.1111/j.1468-2257.2008.00447.x",
     "A methodological overview of network vulnerability analysis"),
    ("koc2014", "10.1016/j.physa.2014.01.056",
     "The impact of the topology on cascading failures in a power grid model"),
]


def fetch(doi):
    url = f"https://api.crossref.org/works/{urllib.parse.quote(doi)}"
    req = urllib.request.Request(url, headers={"User-Agent": "R206-ref-check/1.0 (mailto:renlijian@imut.edu.cn)"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))["message"]


rows = []
for key, doi, title in CANDIDATES:
    row = {"key": key, "doi": doi, "candidate_title": title}
    try:
        msg = fetch(doi)
        got = (msg.get("title") or [""])[0]
        sim = SequenceMatcher(None, title.lower(), got.lower()).ratio()
        row.update(
            crossref_title=got,
            similarity=round(sim, 3),
            journal=(msg.get("container-title") or [""])[0],
            year=(msg.get("issued", {}).get("date-parts", [[None]])[0][0]),
            volume=msg.get("volume", ""), page=msg.get("page", ""),
            authors="; ".join(f"{a.get('family','')}, {a.get('given','')}"
                              for a in (msg.get("author") or [])[:12]),
            verdict="VERIFIED" if sim >= 0.75 else "CHECK_TITLE",
        )
    except Exception as e:
        row.update(verdict="FAIL", error=str(e)[:200])
    rows.append(row)
    print(f"[A4] {key}: {row['verdict']} ({row.get('similarity','')}) {row.get('journal','')} {row.get('year','')}",
          flush=True)
    time.sleep(1.0)

OUTDIR.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(OUTDIR / "R206_A4_reliability_refs_verified.csv", index=False)
print("[A4] DONE")
