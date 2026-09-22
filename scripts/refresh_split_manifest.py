"""Bind verified revision inputs; unchanged reference inputs must retain their hashes."""
from pathlib import Path
import csv
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
ALLOWED_CHANGED={
    "external_cohort_split_summary","external_primary_predictors",
    "external_shift_comparator","external_shift_summary","external_split_note",
    "external_predictors","external_decisions",
    "scigrid",
}
NEW_KEYS={"control_budget","response_definitions","scigrid","radial_subsets","external_shift_equal_rank"}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    cfg=json.loads((ROOT/"config/reproduction.json").read_text())
    path=ROOT/"DATA_MANIFEST.json"
    manifest=json.loads(path.read_text())
    changes=[]
    for key,relative in cfg["inputs"].items():
        source=(ROOT/relative).resolve()
        if not source.is_relative_to(ROOT) or not source.is_file():
            raise ValueError(f"Invalid input {key}")
        actual=sha(source)
        prior=manifest["datasets"].get(key)
        if prior and actual!=prior["sha256"] and key not in ALLOWED_CHANGED:
            raise ValueError(f"Unexpected change to frozen input: {key}")
        if not prior and key not in NEW_KEYS:
            raise ValueError(f"Unexpected new input: {key}")
        item=dict(prior or {})
        if prior and actual!=prior["sha256"]:
            item["previous_sha256"]=prior["sha256"]
            changes.append(key)
        item.update(path=relative,sha256=actual,source_sha256=actual)
        if key == "scigrid":
            item["scope"]="Mapped SciGRID model; one projected chord per simple edge; node-adjacent pairs excluded."
        elif key in ALLOWED_CHANGED:
            item["scope"]="Derived retrospective provenance stratification: 19 primary cities, two prior-use cities, all 21 acquired cities retained; allowance unchanged."
        elif key in NEW_KEYS:
            item["scope"]="Verified derived diagnostic data; raw source snapshots not redistributed."
        if source.suffix==".csv":
            with source.open(encoding="utf-8-sig",newline="") as f:
                reader=csv.DictReader(f); rows=list(reader)
                item.update(rows=len(rows),columns=reader.fieldnames)
        manifest["datasets"][key]=item
    path.write_text(json.dumps(manifest,indent=2)+"\n")
    p=ROOT/"CODE_PROVENANCE.json"
    provenance=json.loads(p.read_text())
    provenance["revision_2026_09_20"]={
        "changes":[
            "Retrospective provenance stratification: Stockholm and Cape Town separate from 19 primary cities; full 21 retained",
            "Complete 272-record order/switch diagnostics regenerated; arithmetic and input identities checked",
            "Second-component and finite-cluster susceptibility observables recomputed with exact component-size bookkeeping",
            "SciGRID crossing counts corrected to one chord per simple edge and node-disjoint pairs",
            "Original pre-response protocol records included; not public preregistration",
            "Constant-offset comparisons use shared allowance and separately reported matched-rank allowance",
        ],
        "core_primary_response_estimator_changed":False,
        "new_diagnostic_edge_orders_generated":True,
        "primary_allowance_refitted":False,
        "source_sha256":{f:sha(ROOT/"scripts"/f) for f in
                          ("external_split.py","screening.py","make_paper_assets.py",
                           "control_budget_verified.py","response_definition_verified.py","power_scigrid_check.py")},
    }
    p.write_text(json.dumps(provenance,indent=2)+"\n")
    print(json.dumps({"bound_inputs":len(manifest["datasets"]),"authorized_changed_inputs":changes}))

if __name__=="__main__": main()
