#!/usr/bin/env python3
"""Build the audit's CSV tables of the record from the raw result files in raw/.

Usage: python3 scripts/build_record.py   (no arguments; reads raw/, writes qr/, cholesky/, eigh/)

Reads the B200 results of the QR audit (one JSON per target) and of the
Cholesky and eigendecomposition audit (JSONL per target and stage), and writes
flat CSV tables next to this script's repository root. Nothing here generates
inputs or runs kernels. qr/starter_runs.csv and qr/round2_scores.csv hold the
organizers' measurements and are not built here.
"""
import csv, glob, json, os, sys, collections
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QR_SRC = os.path.join(ROOT, "raw", "qr_b200")
CE_SRC = os.path.join(ROOT, "raw")
MANIFEST = os.path.join(ROOT, "raw", "manifest.json")
MAN = json.load(open(MANIFEST))["problems"] if os.path.exists(MANIFEST) else {}
def sha_for(problem, label):
    for s_ in MAN.get(problem, {}).get("entries", []):
        if s_["target"] == label: return s_
    return {}

def w(path, rows, fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); wr.writeheader()
        for r in rows: wr.writerow(r)
    print(f"{os.path.relpath(path, ROOT):32s} {len(rows):6d} rows")

def fnum(x):
    if x is None: return ""
    if isinstance(x, float) and (x != x): return "nan"
    if x == float("inf"): return "inf"
    return x

# ---------------- QR ----------------
QR_META = {  # label in the raw record -> (round, leaderboard id, handle as on the leaderboard, rank, kind)
 "10billiontokens":   ("qr_v2", 774, "10billiontokens",   1, "competition"),
 "nernst":            ("qr_v2", 774, "gau.nernst",        2, "competition"),
 "dhu.randhar":       ("qr_v2", 774, "dhu.randhar",       3, "competition"),
 "zhongmingee":       ("qr_v2", 774, "zhongmingee",       4, "competition"),
 "michaelmelons":     ("qr_v2", 774, "michaelmelons",     5, "competition"),
 "nikhilbarhate99":   ("qr_v1", 773, "nikhilbarhate99",   1, "competition"),
 "badelsteinlelbach": ("qr_v1", 773, "badelsteinlelbach", 2, "competition"),
 "Sinatras":          ("qr_v1", 773, "Sinatras",          3, "competition"),
 "Maheshram1":        ("qr_v1", 773, "Maheshram1",        4, "competition"),
 "elianaive":         ("qr_v1", 773, "elianaive",         5, "competition"),
 "cusolver":          ("control", None, "cuSOLVER (torch.linalg.qr)", None, "control"),
 "geqrf":             ("control", None, "cuSOLVER geqrf (torch.geqrf)", None, "control"),
 "geqrf_bf16in":      ("control", None, "cuSOLVER geqrf on bfloat16-rounded input", None, "control"),
}
qr_targets, qr_scored, qr_sweep, qr_breaks, qr_bisect, qr_summary = [], [], [], [], [], []
EVAL_FIELDS = ["target","case","n","batch","scored_batch","cond","kappa","seed","pass_v1","pass_v2","broke","f","factor_scaled","orth_scaled","recon_scaled","tri_scaled","status","ran_path","reason","note"]
for path in sorted(glob.glob(os.path.join(QR_SRC, "*.json"))):
    d = json.load(open(path)); lab = d["label"]; rnd, lb, handle, rank, kind = QR_META[lab]
    qr_targets.append({"target": lab, "round": rnd, "leaderboard_id": lb or "", "handle": handle, "rank": rank or "", "kind": kind,
                       "gpu": d.get("gpu"), "torch": d.get("torch"), "submission_sha256": sha_for(rnd, lab).get("sha256","")})
    def ev(e, extra=None):
        r = {k: fnum(e.get(k)) for k in EVAL_FIELDS if k != "target"}; r["target"] = lab
        if extra: r.update(extra)
        return r
    for s in d.get("scored", []): qr_scored.append(ev(s, {"round_of_case": s.get("round")}))
    sw = d.get("sweep", [])
    # randsvd rows record no condition number; it is the bisection's break threshold for that shape
    kap_by_shape = {(b.get("n"), b.get("batch")): b.get("min_break_kappa") for b in d.get("kappa_bisect", [])}
    for e in sw:
        extra = {"kappa": fnum(kap_by_shape.get((e.get("n"), e.get("batch"))))} if e.get("case") == "randsvd" else {"kappa": ""}
        qr_sweep.append(ev(e, extra))
    brk = [e for e in sw if e.get("broke")]
    for e in brk: qr_breaks.append(ev(e, {"kappa": fnum(kap_by_shape.get((e.get("n"), e.get("batch")))) if e.get("case") == "randsvd" else ""}))
    for b in d.get("kappa_bisect", []):
        qr_bisect.append({"target": lab, "n": b.get("n"), "batch": b.get("batch"), "min_break_kappa": fnum(b.get("min_break_kappa")), "realized_kappa": fnum(b.get("realized_kappa")), "f_at_break": fnum(b.get("f_at_break")), "ran_path": b.get("ran_path",""), "note": b.get("note","")})
    sc = d.get("scored", [])
    fam = sorted({e.get("case") for e in brk}); kap = [b.get("min_break_kappa") for b in d.get("kappa_bisect", []) if b.get("min_break_kappa")]
    qr_summary.append({"target": lab, "round": rnd, "leaderboard_id": lb or "", "handle": handle, "rank": rank or "", "kind": kind,
        "scored_cases": len(sc), "scored_pass_v1": sum(1 for s in sc if s.get("pass_v1")), "scored_pass_v2": sum(1 for s in sc if s.get("pass_v2")),
        "sweep_evals": len(sw), "sweep_breaks": len(brk), "families_broken": len(fam), "families": " ".join(fam),
        "min_break_kappa": fnum(min(kap)) if kap else ""})
w(f"{ROOT}/qr/targets.csv", qr_targets, ["target","round","leaderboard_id","handle","rank","kind","submission_sha256","gpu","torch"])
w(f"{ROOT}/qr/scored.csv", qr_scored, ["target","round_of_case"] + EVAL_FIELDS[1:])
w(f"{ROOT}/qr/sweep.csv", qr_sweep, EVAL_FIELDS)
w(f"{ROOT}/qr/breaks.csv", qr_breaks, EVAL_FIELDS)
w(f"{ROOT}/qr/kappa_bisect.csv", qr_bisect, ["target","n","batch","min_break_kappa","realized_kappa","f_at_break","ran_path","note"])
w(f"{ROOT}/qr/summary.csv", qr_summary, ["target","round","leaderboard_id","handle","rank","kind","scored_cases","scored_pass_v1","scored_pass_v2","sweep_evals","sweep_breaks","families_broken","families","min_break_kappa"])

# ---------------- Cholesky and eigh ----------------
CE_FIELDS = ["target","problem","stage","tier","case_id","case_source","confirm_candidate","repeat_id","cold_or_warm","family","difficulty","n","batch","seed",
             "input_valid","input_validity_evidence","vendor_pass","official_pass","official_message","ran_path","error","elapsed_s",
             "diagnostics","env_hash","gpu_model","checker_sha256","submission_sha256","protocol_hash"]
def ce_rows(problem):
    out = collections.defaultdict(list)
    for path in sorted(glob.glob(os.path.join(CE_SRC, problem + "_b200", "*.jsonl"))):
        base = os.path.basename(path)[:-6]
        if "_" in base and base.rsplit("_",1)[1] in ("registered","confirmed","bisect"):
            target, stage = base.rsplit("_",1)
        elif base == "vendor_degraded_registered": target, stage = "vendor_degraded", "registered"
        else: target, stage = base, "official"
        for line in open(path):
            if not line.strip(): continue
            r = json.loads(line)
            row = {k: r.get(k) for k in CE_FIELDS}
            row.update(target=target, problem=problem, stage=stage,
                       difficulty=json.dumps(r.get("difficulty"), sort_keys=True), diagnostics=json.dumps(r.get("diagnostics"), sort_keys=True))
            out[stage].append(row)
    return out
for problem in ("cholesky", "eigh"):
    rows = ce_rows(problem)
    tg = [{"target": x["target"], "problem": problem, "leaderboard_id": MAN[problem]["leaderboard_id"], "handle": x["user_name"], "rank": x["rank"], "submission_id": x.get("submission_id"), "submission_time": x.get("submission_time"), "score": x.get("score"), "submission_sha256": x["sha256"], "lines": x.get("lines")} for x in MAN.get(problem, {}).get("entries", [])]
    routine = {"cholesky": "torch.linalg.cholesky_ex", "eigh": "torch.linalg.eigh"}[problem]
    prec = {"cholesky": "bfloat16", "eigh": "float8 (e4m3)"}[problem]
    tg += [{"target": "vendor", "problem": problem, "handle": f"cuSOLVER via PyTorch ({routine})", "rank": "", "kind": "control"}, {"target": "vendor_degraded", "problem": problem, "handle": f"cuSOLVER via PyTorch ({routine}) on inputs rounded to {prec}", "rank": "", "kind": "control"}]
    for x in tg: x.setdefault("kind", "competition")
    w(f"{ROOT}/{problem}/targets.csv", tg, ["target","problem","leaderboard_id","handle","rank","kind","submission_id","submission_time","score","submission_sha256","lines"])
    for stage in ("official","registered","confirmed"):
        w(f"{ROOT}/{problem}/{stage}.csv", rows.get(stage, []), CE_FIELDS)
    # summary per target
    summ = []
    targets = sorted({r["target"] for s in rows.values() for r in s})
    for t in targets:
        off = [r for r in rows["official"] if r["target"]==t]; reg = [r for r in rows["registered"] if r["target"]==t]; con = [r for r in rows["confirmed"] if r["target"]==t]
        attributable = [r for r in reg if r["official_pass"] is False and r["vendor_pass"] is True and r["input_valid"] is not False]
        vendor_fail_too = [r for r in reg if r["official_pass"] is False and r["vendor_pass"] is False]
        cands = sorted({r["confirm_candidate"] for r in con if r["confirm_candidate"]})
        reproduced = sum(1 for c in cands if all(r["official_pass"] is False and r["vendor_pass"] is True for r in con if r["confirm_candidate"]==c))
        summ.append({"target": t, "problem": problem, "scored_cases": len(off), "scored_pass": sum(1 for r in off if r["official_pass"]),
            "discovery_evals": len(reg), "discovery_valid_inputs": sum(1 for r in reg if r["input_valid"] is not False),
            "discovery_fail_attributable": len(attributable), "discovery_fail_with_vendor": len(vendor_fail_too),
            "confirmation_candidates": len(cands), "confirmation_runs": len(con), "candidates_reproduced_in_all_runs": reproduced,
            "families_attributable": " ".join(sorted({r["family"] for r in attributable}))})
    w(f"{ROOT}/{problem}/summary.csv", summ, ["target","problem","scored_cases","scored_pass","discovery_evals","discovery_valid_inputs","discovery_fail_attributable","discovery_fail_with_vendor","confirmation_candidates","confirmation_runs","candidates_reproduced_in_all_runs","families_attributable"])
