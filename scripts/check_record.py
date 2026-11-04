#!/usr/bin/env python3
"""Recompute every number stated in README.md from the CSV tables and fail on any mismatch."""
import csv, os, sys
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def rows(p): return list(csv.DictReader(open(os.path.join(R, p))))
errs = []
def eq(what, got, exp):
    if str(got) != str(exp): errs.append(f"{what}: got {got}, README says {exp}")
# QR
sweep = rows("qr/sweep.csv"); scored = rows("qr/scored.csv"); summ = {r["target"]: r for r in rows("qr/summary.csv")}
exp_qr = {"10billiontokens": (19,19,157), "nernst": (19,19,46), "dhu.randhar": (19,19,330), "zhongmingee": (19,19,3), "michaelmelons": (19,19,360),
          "nikhilbarhate99": (19,4,755), "badelsteinlelbach": (19,19,92), "Sinatras": (19,19,106), "Maheshram1": (16,4,586), "elianaive": (19,19,3),
          "cusolver": (19,19,0), "geqrf": (19,19,0), "geqrf_bf16in": (8,7,623)}
for t, (v1, v2, br) in exp_qr.items():
    sc = [r for r in scored if r["target"] == t]; sw = [r for r in sweep if r["target"] == t]
    eq(f"{t} scored v1", sum(1 for r in sc if r["pass_v1"] == "True"), v1)
    eq(f"{t} scored v2", sum(1 for r in sc if r["pass_v2"] == "True"), v2)
    eq(f"{t} breaks", sum(1 for r in sw if r["broke"] == "True"), br)
    eq(f"{t} summary breaks", summ[t]["sweep_breaks"], br)
    if not 1050 <= len(sw) <= 1055: errs.append(f"{t}: {len(sw)} sweep rows, README says 1,050 to 1,055")
    for r in sw:
        rule = (r["status"] == "OK" and r["ran_path"] == "kernel" and r["pass_v2"] == "False")
        if (r["broke"] == "True") != rule: errs.append(f"{t}: broke rule violated at {r['case']} n={r['n']} seed={r['seed']}"); break
# Cholesky and eigh
def ce(problem, expected, n_disc, n_valid):
    reg = rows(f"{problem}/registered.csv"); con = rows(f"{problem}/confirmed.csv"); off = rows(f"{problem}/official.csv")
    for t, (sc, attr, rep) in expected.items():
        o = [r for r in off if r["target"] == t]; g = [r for r in reg if r["target"] == t]; c = [r for r in con if r["target"] == t]
        if sc is not None: eq(f"{problem} {t} scored", f"{sum(1 for r in o if r['official_pass']=='True')}/{len(o)}", sc)
        eq(f"{problem} {t} attributable", sum(1 for r in g if r["official_pass"] == "False" and r["vendor_pass"] == "True" and r["input_valid"] != "False"), attr)
        eq(f"{problem} {t} discovery evals", len(g), n_disc)
        eq(f"{problem} {t} valid inputs", sum(1 for r in g if r["input_valid"] != "False"), n_valid)
        if rep is not None:
            cands = sorted({r["confirm_candidate"] for r in c if r["confirm_candidate"]})
            k = sum(1 for cd in cands if all(r["official_pass"] == "False" and r["vendor_pass"] == "True" for r in c if r["confirm_candidate"] == cd))
            eq(f"{problem} {t} reproduced", f"{k} of {len(cands)}", rep)
ce("cholesky", {"zhongmingee": ("32/32",194,"4 of 5"), "Frosty40": ("32/32",129,"5 of 5"), "Olek": ("32/32",42,"5 of 5"), "viridale": ("32/32",233,"5 of 5"),
                "Ravi_Theja": ("32/32",85,"0 of 5"), "vendor": ("32/32",0,None), "vendor_degraded": (None,437,None)}, 602, 567)
ce("eigh", {"teelaitila": ("52/52",11,"5 of 5"), "nikhilbarhate99": ("52/52",0,"0 of 5"), "elianaive": ("52/52",0,"0 of 5"), "zhongmingee": ("52/52",18,"5 of 5"),
            "Olek": ("52/52",61,"5 of 5"), "vendor": ("52/52",0,None), "vendor_degraded": (None,787,None)}, 996, 996)
vend = [r for r in rows("eigh/registered.csv") if r["target"] == "vendor" and r["official_pass"] == "False"]
eq("eigh vendor self-failures", len(vend), 38)
eq("eigh vendor self-failures at cond 12", sum(1 for r in vend if r["difficulty"] == '{"cond": 12}'), 38)
eq("eigh vendor failing families", " ".join(sorted({r["family"] for r in vend})), "dense mixed rowscale")
for t, n in (("elianaive", 439), ("nikhilbarhate99", 186)):
    eq(f"eigh {t} fallback_vendor", sum(1 for r in rows("eigh/registered.csv") if r["target"] == t and r["ran_path"] == "fallback_vendor"), n)
rt = [r for r in rows("cholesky/confirmed.csv") if r["target"] == "Ravi_Theja"]
eq("Ravi_Theja reruns", len(rt), 15); eq("Ravi_Theja reruns passed", sum(1 for r in rt if r["official_pass"] == "True"), 15)
inval = [r for r in rows("cholesky/registered.csv") if r["target"] == "vendor" and r["input_valid"] == "False"]
eq("cholesky invalid inputs", len(inval), 35)
import json, glob
for f in glob.glob(os.path.join(R, "raw", "*", "*.jsonl")):
    for l in open(f):
        if l.strip():
            k = set(json.loads(l)); bad = k & {"gpu_uuid", "process_id", "error_tail"}
            if bad: errs.append(f"{f}: identifier fields present {bad}"); break
for f in glob.glob(os.path.join(R, "scripts", "*.py")) + glob.glob(os.path.join(R, "replay", "*.py")):
    if ("/" + "home/") in open(f).read(): errs.append(f"{f}: host path")
# claims stated in prose
tq = rows("qr/targets.csv"); eq("thirteen QR targets", len(tq), 13)
eq("scored rows 13 x 19", len(scored), 13 * 19)
fam = sorted({r["case"] for r in sweep if r["case"] != "randsvd"})
eq("ten organizer families", " ".join(fam), "band clustered dense diagonal mixed nearcollinear nearrank rankdef rowscale upper")
eq("QR entries that broke", sum(1 for t, (v1, v2, br) in exp_qr.items() if br > 0 and not t.startswith(("cusolver", "geqrf"))), 10)
rep = {}
for problem in ("cholesky", "eigh"):
    con = rows(f"{problem}/confirmed.csv")
    for t in {r["target"] for r in con}:
        cands = sorted({r["confirm_candidate"] for r in con if r["target"] == t and r["confirm_candidate"]})
        rep[(problem, t)] = sum(1 for cd in cands if all(r["official_pass"] == "False" and r["vendor_pass"] == "True" for r in con if r["target"] == t and r["confirm_candidate"] == cd))
        pats = {tuple(sorted((r["case_source"], r["cold_or_warm"]) for r in con if r["target"] == t and r["confirm_candidate"] == cd)) for cd in cands}
        if pats != {(("confirm", "cold"), ("confirm", "warm"), ("confirm_heldout", "cold"))}: errs.append(f"{problem} {t}: rerun structure {pats}")
eq("entries with reproduced failures", sum(1 for v in rep.values() if v > 0), 7)
eq("entries with no reproduced failure", sum(1 for v in rep.values() if v == 0), 3)
for problem in ("cholesky", "eigh"):
    eq(f"{problem} degraded control not run on scored cases", sum(1 for r in rows(f"{problem}/official.csv") if r["target"] == "vendor_degraded"), 0)
import json, glob
for f in glob.glob(os.path.join(R, "raw", "*", "*.json")) + [os.path.join(R, "raw", "manifest.json")]:
    txt = open(f).read()
    if any(k in txt for k in ("gpu_uuid", "process_id", "error_tail", "/" + "home/", "submissions/")): errs.append(f"{f}: identifier, path or private field present")
for f in glob.glob(os.path.join(R, "**", "*"), recursive=True):
    if os.path.isfile(f) and not f.endswith((".csv", ".jsonl", ".json")) and ".git" not in f and os.path.basename(f) != "check_record.py":
        if ("/" + "home/") in open(f, errors="ignore").read(): errs.append(f"{f}: host path")
# every replay example in the README must name a row of sweep.csv
import re
readme = open(os.path.join(R, "README.md")).read()
for m in re.finditer(r"--submission (\S+)\.py \\\n\s+--case (\w+) --n (\d+) --batch (\d+)(?: --cond (\d+))?(?: --kappa (\S+))? --seed (\d+)", readme):
    t, case, n, batch, cond, kappa, seed = m.groups()
    hit = [r for r in sweep if r["target"] == t and r["case"] == case and r["n"] == n and r["batch"] == batch and r["seed"] == seed and (cond is None or r["cond"] == cond) and (kappa is None or r["kappa"] == kappa)]
    if len(hit) != 1: errs.append(f"README replay example {t} {case} n={n} batch={batch} cond={cond} kappa={kappa} seed={seed}: {len(hit)} matching sweep rows")
if errs:
    print("\n".join(errs)); sys.exit(1)
print("check_record: every README number matches the tables")
