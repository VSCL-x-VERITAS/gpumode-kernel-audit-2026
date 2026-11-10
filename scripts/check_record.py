#!/usr/bin/env python3
"""Recompute every table and most counts in README.md from the CSV tables and fail on a mismatch.

Usage (from the repository root):
  python3 scripts/check_record.py            check README.md against the CSV tables
  python3 scripts/check_record.py --tables   print the README's tables as computed from the CSV tables

Every table in README.md (Summary, What was run, Results) is the output of --tables, verbatim,
and the check fails if any of them differs. Most counts stated in the README's prose are rebuilt
from the data below and must appear in the README. The script also checks the counting rules,
the rerun structure, that the record's files carry no identifier fields or host paths, and that
every replay example in the README names a row of qr/sweep.csv that broke. Facts from outside
the record (the organizers' task files and runner commits) are not checked here.
"""
import csv, glob, json, os, re, sys
from collections import Counter

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def rows(p): return list(csv.DictReader(open(os.path.join(R, p), newline="")))
errs = []
def eq(what, got, exp):
    if str(got) != str(exp): errs.append(f"{what}: got {got}, expected {exp}")
def num(k): return f"{k:,}"
def frac(k, n): return f"{k}/{n}"
def table(head, body):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in body]
    return "\n".join(out)

# ---------------------------------------------------------------- QR
qt = rows("qr/targets.csv"); sweep = rows("qr/sweep.csv"); scored = rows("qr/scored.csv")
summ = {r["target"]: r for r in rows("qr/summary.csv")}
ROUND = {"qr_v1": "1", "qr_v2": "2", "control": "control"}
CONTROL_ORDER = ["cusolver", "geqrf", "geqrf_bf16in"]
qorder = sorted((r for r in qt if r["kind"] == "competition"), key=lambda r: (r["round"], int(r["rank"])))
qorder += sorted((r for r in qt if r["kind"] == "control"), key=lambda r: CONTROL_ORDER.index(r["target"]))
key = lambda r: (r["case"], r["n"], r["batch"], r["cond"], r["seed"])
v1_cases = [key(r) for r in scored if r["target"] == "cusolver" and r["round_of_case"] == "v1"]
v2_cases = [key(r) for r in scored if r["target"] == "cusolver" and r["round_of_case"] == "v2"]
qr = {}
for t in qt:
    name = t["target"]; sc = [r for r in scored if r["target"] == name]; sw = [r for r in sweep if r["target"] == name]
    if sorted(key(r) for r in sc if r["round_of_case"] == "v1") != sorted(v1_cases) or sorted(key(r) for r in sc if r["round_of_case"] == "v2") != sorted(v2_cases):
        errs.append(f"{name}: scored cases differ from the controls'")
    distinct = {}
    for r in sc:
        v = (r["pass_v1"], r["pass_v2"])
        if distinct.setdefault(key(r), v) != v: errs.append(f"{name}: the two rows of scored case {key(r)} disagree")
    own = {"qr_v1": ("v1", "pass_v1"), "qr_v2": ("v2", "pass_v2")}.get(t["round"])
    own_cell = ""
    if own:
        o = [r for r in sc if r["round_of_case"] == own[0]]
        own_cell = frac(sum(r[own[1]] == "True" for r in o), len(o))
    for r in sw:
        rule = r["status"] == "OK" and r["ran_path"] == "kernel" and r["pass_v2"] == "False"
        if (r["broke"] == "True") != rule: errs.append(f"{name}: break rule violated at {r['case']} n={r['n']} seed={r['seed']}"); break
    br2 = sum(r["broke"] == "True" for r in sw)
    br1 = sum(r["status"] == "OK" and r["ran_path"] == "kernel" and r["pass_v1"] == "False" for r in sw)
    fams = len({r["case"] for r in sw if r["broke"] == "True"})
    eq(f"{name} summary sweep_evals", summ[name]["sweep_evals"], len(sw))
    eq(f"{name} summary sweep_breaks", summ[name]["sweep_breaks"], br2)
    eq(f"{name} summary families_broken", summ[name]["families_broken"], fams)
    eq(f"{name} summary scored rows", summ[name]["scored_cases"], len(sc))
    qr[name] = dict(row=t, own=own_cell, d1=sum(v[0] == "True" for v in distinct.values()),
                    d2=sum(v[1] == "True" for v in distinct.values()), nd=len(distinct), searched=len(sw),
                    vendor=sum(r["ran_path"] == "fallback_geqrf" for r in sw), br2=br2, br1=br1, fams=fams)

# Controls are shown by the routine they call and their target label. The handle of the
# `cusolver` control in the data reads "cuSOLVER (torch.linalg.qr)", which is wrong (README).
LABEL = {("qr", "cusolver"): "torch.geqrf (`cusolver`)", ("qr", "geqrf"): "torch.geqrf (`geqrf`)",
         ("qr", "geqrf_bf16in"): "torch.geqrf on bfloat16-rounded input (`geqrf_bf16in`)",
         ("cholesky", "vendor"): "torch.linalg.cholesky_ex (`vendor`)",
         ("cholesky", "vendor_degraded"): "torch.linalg.cholesky_ex on bfloat16-rounded input (`vendor_degraded`)",
         ("eigh", "vendor"): "torch.linalg.eigh (`vendor`)",
         ("eigh", "vendor_degraded"): "torch.linalg.eigh on float8-rounded input (`vendor_degraded`)"}
def label(problem, t): return LABEL[(problem, t["target"])] if t["kind"] == "control" else t["handle"]
def qr_name(t): return label("qr", t)
qr_scored_table = table(
    ["Round", "Rank", "Entry", "Own round's cases, own round's checker", "All 12 cases, round-1 checker", "All 12 cases, round-2 checker"],
    [[ROUND[q["row"]["round"]], q["row"]["rank"], qr_name(q["row"]), q["own"], frac(q["d1"], q["nd"]), frac(q["d2"], q["nd"])]
     for q in (qr[t["target"]] for t in qorder)])
qr_search_table = table(
    ["Round", "Rank", "Entry", "Searched", "Vendor-identical", "Breaks, round-2 checker", "Breaks, round-1 checker", "Families with breaks"],
    [[ROUND[q["row"]["round"]], q["row"]["rank"], qr_name(q["row"]), num(q["searched"]), num(q["vendor"]), num(q["br2"]), num(q["br1"]), q["fams"]]
     for q in (qr[t["target"]] for t in qorder)])

# ---------------------------------------------------- Cholesky and eigendecomposition
def ce(problem):
    tg = rows(f"{problem}/targets.csv"); reg = rows(f"{problem}/registered.csv")
    con = rows(f"{problem}/confirmed.csv"); off = rows(f"{problem}/official.csv")
    s = {r["target"]: r for r in rows(f"{problem}/summary.csv")}
    order = sorted((r for r in tg if r["kind"] == "competition"), key=lambda r: int(r["rank"]))
    order += sorted((r for r in tg if r["kind"] == "control"), key=lambda r: r["target"] != "vendor")
    out = {}
    for t in order:
        name = t["target"]; o = [r for r in off if r["target"] == name]; g = [r for r in reg if r["target"] == name]
        c = [r for r in con if r["target"] == name]
        attr = sum(r["official_pass"] == "False" and r["vendor_pass"] == "True" and r["input_valid"] != "False" for r in g)
        cands = sorted({r["confirm_candidate"] for r in c if r["confirm_candidate"]})
        rep = sum(all(r["official_pass"] == "False" and r["vendor_pass"] == "True" for r in c if r["confirm_candidate"] == cd) for cd in cands)
        pats = {tuple(sorted((r["case_source"], r["cold_or_warm"]) for r in c if r["confirm_candidate"] == cd)) for cd in cands}
        if cands and pats != {(("confirm", "cold"), ("confirm", "warm"), ("confirm_heldout", "cold"))}:
            errs.append(f"{problem} {name}: rerun structure {pats}")
        valid = sum(r["input_valid"] != "False" for r in g)
        eq(f"{problem} {name} summary attributable", s[name]["discovery_fail_attributable"], attr)
        eq(f"{problem} {name} summary discovery evals", s[name]["discovery_evals"], len(g))
        eq(f"{problem} {name} summary valid inputs", s[name]["discovery_valid_inputs"], valid)
        if cands: eq(f"{problem} {name} summary reproduced", s[name]["candidates_reproduced_in_all_runs"], rep)
        out[name] = dict(row=t, scored=frac(sum(r["official_pass"] == "True" for r in o), len(o)) if o else "not run",
                         evals=len(g), valid=valid, attr=attr, ncand=len(cands), rep=rep,
                         vendor=sum(r["ran_path"] == "fallback_vendor" for r in g) if t["kind"] == "competition" else "")
    return out
chol, eigh = ce("cholesky"), ce("eigh")
def ce_table(problem, d):
    body = []
    for q in d.values():
        t = q["row"]; comp = t["kind"] == "competition"
        body.append([t["rank"] if comp else "control", label(problem, t), q["scored"], num(q["vendor"]) if comp else "",
                     num(q["attr"]), f"{q['rep']} of {q['ncand']}" if comp else ""])
    return table(["Rank", "Entry", "Scored", "Vendor-identical", "Attributable", "Reproduced"], body)
chol_table, eigh_table = ce_table("cholesky", chol), ce_table("eigh", eigh)

# ---------------------------------------------------------------- overview tables
def entries(d): return [q for q in d.values() if q["row"]["kind"] == "competition"]
def count(qs, f): return f"{sum(1 for q in qs if f(q))} of {len(qs)}"
q1 = [qr[t["target"]] for t in qorder if t["round"] == "qr_v1"]; q2 = [qr[t["target"]] for t in qorder if t["round"] == "qr_v2"]
passed_own = lambda q: q["own"].split("/")[0] == q["own"].split("/")[1]
overview_table = table(
    ["Round", "Leaderboard", "Passed own round's scored cases", "Failed in the search", "Failure reproduced in reruns"],
    [["QR, round 1", "773", count(q1, passed_own), count(q1, lambda q: q["br1"] > 0), "not rerun"],
     ["QR, round 2", "774", count(q2, passed_own), count(q2, lambda q: q["br2"] > 0), "not rerun"],
     ["Cholesky", "776", count(entries(chol), lambda q: q["scored"].split("/")[0] == q["scored"].split("/")[1]), count(entries(chol), lambda q: q["attr"] > 0), count(entries(chol), lambda q: q["rep"] > 0)],
     ["Eigendecomposition", "775", count(entries(eigh), lambda q: q["scored"].split("/")[0] == q["scored"].split("/")[1]), count(entries(eigh), lambda q: q["attr"] > 0), count(entries(eigh), lambda q: q["rep"] > 0)]])
if any(q["br2"] == 0 for q in q1): errs.append("a round-1 entry has no break under the round-2 checker")
def span(qs): v = sorted(q["searched"] for q in qs); return f"{num(v[0])} to {num(v[-1])}"
def one(vals, what):
    vals = sorted(set(vals))
    if len(vals) != 1: errs.append(f"{what}: values differ across entries {vals}")
    return vals[0]
c_scored = one([len([r for r in rows("cholesky/official.csv") if r["target"] == q["row"]["target"]]) for q in entries(chol)], "cholesky scored cases")
e_scored = one([len([r for r in rows("eigh/official.csv") if r["target"] == q["row"]["target"]]) for q in entries(eigh)], "eigh scored cases")
c_evals = one([q["evals"] for q in chol.values()], "cholesky discovery evals"); c_valid = one([q["valid"] for q in chol.values()], "cholesky valid inputs")
e_evals = one([q["evals"] for q in eigh.values()], "eigh discovery evals")
run_table = table(
    ["Round", "Leaderboard", "Scored cases per entry", "Searched evaluations per entry"],
    [["QR, round 1", "773", len(v1_cases), span(q1)],
     ["QR, round 2", "774", len(v2_cases), span(q2)],
     ["Cholesky", "776", c_scored, f"{num(c_evals)} generated, {num(c_valid)} valid"],
     ["Eigendecomposition", "775", e_scored, num(e_evals)]])

TABLES = [("Summary", overview_table), ("What was run", run_table), ("QR scored cases", qr_scored_table),
          ("QR search", qr_search_table), ("Cholesky", chol_table), ("Eigendecomposition", eigh_table)]
if "--tables" in sys.argv[1:]:
    for name, tbl in TABLES: print(f"<!-- {name} -->\n{tbl}\n")
    sys.exit(0)

readme = re.sub(r"<!--.*?-->", "", open(os.path.join(R, "README.md")).read(), flags=re.S)
# Every table in the README, in order, must equal the computed one exactly. The Files table is the only other table.
blocks, cur = [], []
for line in readme.split("\n") + [""]:
    if line.startswith("|"): cur.append(line)
    elif cur: blocks.append("\n".join(cur)); cur = []
blocks = [b for b in blocks if not b.startswith("| Path | Content |")]
if len(blocks) != len(TABLES): errs.append(f"README has {len(blocks)} result tables, expected {len(TABLES)}")
for (name, tbl), b in zip(TABLES, blocks):
    if b != tbl: errs.append(f"README table '{name}' differs from the CSV tables (print it with --tables)")

# ---------------------------------------------------------------- numbers stated in prose
# Each claim is rebuilt from the data and must appear in README.md (whitespace normalized),
# so that editing a number in the README, or a change in the data, fails the check.
flat = " ".join(readme.split())
def says(phrase):
    if " ".join(phrase.split()) not in flat: errs.append(f"README does not say, or no longer matches the data: {phrase!r}")
def need(cond, what):
    if not cond: errs.append(f"data contradicts the README: {what}")
W = {k: w for k, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())}
cfg = lambda f, k: next(l.split(":", 1)[1].strip() for l in open(os.path.join(R, f)) if l.split(":")[0].strip() == k)

# Summary
comp = q1 + q2 + entries(chol) + entries(eigh)
grid = lambda name: [r for r in sweep if r["target"] == name and r["case"] != "randsvd"]
need(all(any(r["broke"] == "True" for r in grid(q["row"]["target"])) for q in q1 + q2), "every QR entry breaks on a grid input")
need(all(r["pass_v1"] == r["pass_v2"] == "True" for c in ("cusolver", "geqrf") for r in grid(c)), "the vendor routine passes every QR grid input")
n_failed = sum(q["br2"] > 0 for q in q1 + q2) + sum(q["attr"] > 0 for d in (chol, eigh) for q in entries(d))
need(all(passed_own(q) for q in q1 + q2) and all(q["scored"].split("/")[0] == q["scored"].split("/")[1] for d in (chol, eigh) for q in entries(d)),
     "every entry passed its own round's scored cases")
says(f"All {W[len(comp)]} entries passed the scored cases of their own round, and {W[n_failed]} failed the same checker on other legal inputs on which the vendor routine")
need(all(q["br1"] > 0 for q in q1) and all(q["br2"] > 0 for q in q1 + q2), "every QR entry broke")
says(f"All {W[len(q1 + q2)]} QR entries failed.")
ncand = one([q["ncand"] for q in entries(chol) + entries(eigh)], "candidates per entry")
nrerun = one([sum(r["target"] == q["row"]["target"] for r in rows(f"{p}/confirmed.csv")) // q["ncand"]
              for p, d in (("cholesky", chol), ("eigh", eigh)) for q in entries(d)], "reruns per candidate")
says(f"{W[ncand]} failing inputs per entry became candidates for reruns.")
reps = [q["rep"] for d in (chol, eigh) for q in entries(d)]
says(f"{W[sum(k > 0 for k in reps)].capitalize()} of those {W[len(reps)]} entries had at least one candidate reproduced: "
     f"all {W[nrerun]} of its reruns failed while the vendor routine passed.")
r5 = next(q for q in entries(chol) if q["row"]["rank"] == "5")
need(r5["rep"] == 0, "the fifth-ranked Cholesky entry reproduced nothing")
rt = [r for r in rows("cholesky/confirmed.csv") if r["target"] == r5["row"]["target"]]
need(rt and all(r["official_pass"] == "True" for r in rt), "the fifth-ranked Cholesky entry passed every rerun")
says(f"the fifth-ranked Cholesky entry, failed {r5['attr']} times in the search and passed every rerun.")
says(f"{r5['row']['handle']}'s {r5['attr']} attributable failures did not reproduce: all {len(rt)} reruns of its {W[r5['ncand']]} candidates passed.")
man = json.load(open(os.path.join(R, "raw", "manifest.json")))
need("July and August 2026" in man["note"], "entries downloaded in July and August 2026")
says("when the entries were downloaded, in July and August 2026.")

# Terms
def tiers(problem):
    o = [r for r in rows(f"{problem}/official.csv") if r["target"] == "vendor"]
    return sum(r["tier"] == "official_test" for r in o), sum(r["tier"] == "official_benchmark" for r in o)
(ct, cb), (et, eb) = tiers("cholesky"), tiers("eigh")
need(ct + cb == c_scored and et + eb == e_scored, "scored cases are the test plus benchmark cases")
says(f"For QR this record runs the benchmark cases ({len(v1_cases)} in round 1, {len(v2_cases)} in round 2)")
says(f"For Cholesky it runs both, {ct} test and {cb} benchmark cases, and for eigendecomposition both, {et} and {eb}.")
inv = [r for r in rows("cholesky/registered.csv") if r["target"] == "vendor" and r["input_valid"] == "False"]
need(c_evals - c_valid == len(inv) and all(r["input_validity_evidence"].startswith("not SPD in fp32: cholesky_ex") for r in inv),
     "Cholesky inputs judged by cholesky_ex")
says(f"`torch.linalg.cholesky_ex`, and the {len(inv)} generated inputs that failed the test are excluded from every verdict.")
alltg = qt + rows("cholesky/targets.csv") + rows("eigh/targets.csv")
says(f"The two differ for {W[sum(t['kind'] == 'competition' and t['target'] != t['handle'] for t in alltg)]} entries")
gpus = {r["gpu"] for r in qt} | {r["gpu_model"] for p in ("cholesky", "eigh") for f in ("official", "registered", "confirmed") for r in rows(f"{p}/{f}.csv")}
need(gpus == {"NVIDIA B200"}, "every evaluation ran on an NVIDIA B200")
says("Every evaluation ran on an NVIDIA B200")

# What was run
shapes = {(c[1], c[2]) for c in v1_cases}
need({(r["n"], r["batch"]) for r in sweep} == shapes, "the QR search keeps the benchmark shapes and batch sizes")
says("The QR search keeps the task's benchmark shapes and batch sizes")
g0 = grid("geqrf"); conds = sorted({int(r["cond"]) for r in g0}); seeds = sorted({int(r["seed"]) for r in g0})
need(all(len(grid(t["target"])) == 10 * len(shapes) * len(conds) * len(seeds) == len({key(r) for r in grid(t["target"])}) for t in qt), "the QR grid")
says(f"For each target it is a grid of {num(len(g0))} evaluations: the organizers' ten families, the {len(shapes)} benchmark shapes, "
     f"the conditioning knob `cond` in {{{', '.join(map(str, conds))}}}, and the seeds {seeds[0]}, {seeds[1]} and {seeds[2]}.")
def eff(r):  # the input the organizers' generator actually builds: clustered ignores cond, rowscale uses max(cond, 4)
    c = None if r["case"] == "clustered" else max(int(r["cond"]), 4) if r["case"] == "rowscale" else int(r["cond"])
    return (r["case"], r["n"], r["batch"], c, r["seed"])
groups = {t["target"]: {} for t in qt}
for r in sweep:
    if r["case"] != "randsvd": groups[r["target"]].setdefault(eff(r), []).append(r)
ndist = one([len(g) for g in groups.values()], "distinct grid inputs")
need(all(len({(r["pass_v1"], r["pass_v2"], r["ran_path"]) for r in v}) == 1 for g in groups.values() for v in g.values()),
     "repeated grid inputs got the same verdicts")
says(f"so the grid holds {ndist} distinct inputs per target, and the repeated inputs always got the same verdicts.")
kbr = rows("qr/kappa_bisect.csv")
need({r["seed"] for r in sweep if r["case"] == "randsvd"} == {"7"} and all(r["min_break_kappa"] or r["note"] == "did not break by kappa=1e12" for r in kbr),
     "randsvd seed 7, searched up to 1e12")
says("a bisection over the condition number up to 10<sup>12</sup>, with seed 7,")
for t in qt:
    rs = [r for r in sweep if r["target"] == t["target"] and r["case"] == "randsvd"]
    th = {(r["n"], r["batch"]): r["min_break_kappa"] for r in kbr if r["target"] == t["target"] and r["min_break_kappa"]}
    need(len(rs) == len(th) and all(r["broke"] == "True" and th.get((r["n"], r["batch"])) == r["kappa"] for r in rs),
         f"{t['target']}: one randsvd row per threshold found")
says(f"so a target has {span(list(qr.values()))} rows. The bisection's other probes are not recorded.")
need({c[0] for c in v1_cases} == {"dense"} and set(v1_cases) <= set(v2_cases), "the round-1 cases are dense and among the round-2 cases")
extra = Counter(c[0] for c in v2_cases if c[0] != "dense")
need(dict(extra) == {"mixed": 2, "rankdef": 1, "clustered": 1, "nearrank": 1}, "the five structured round-2 cases")
says(f"The round-1 benchmark list has {len(v1_cases)} dense cases. The round-2 benchmark list has {len(v2_cases)}: the same {len(v1_cases)} and "
     f"{W[sum(extra.values())]} structured ones (two `mixed`, one `rankdef`, one `clustered`, one `nearrank`).")
says(f"every QR target was graded on all {len(set(v2_cases))} distinct benchmark cases.")
need(sum(r["pass_v1"] == "False" and r["pass_v2"] == "True" for r in sweep) == 0 and sum(r["pass_v2"] == "False" and r["pass_v1"] == "True" for r in sweep) > 0,
     "the round-2 checker rejects every sweep output the round-1 checker rejects, and more")
says("In the search the round-2 checker rejected every output that the round-1 checker rejected, and more.")
for problem in ("cholesky", "eigh"):
    need(sum(r["target"] == "vendor_degraded" for r in rows(f"{problem}/official.csv")) == 0, f"{problem} degraded control not run on scored cases")
deg = {p: next(t["handle"] for t in rows(f"{p}/targets.csv") if t["target"] == "vendor_degraded") for p in ("cholesky", "eigh")}
need("bfloat16" in qr["geqrf_bf16in"]["row"]["handle"] and "bfloat16" in deg["cholesky"] and "float8 (e4m3)" in deg["eigh"], "the degraded controls' precisions")
says("(bfloat16 for QR and Cholesky, float8 e4m3 for eigendecomposition)")
need(all(qr[c]["vendor"] == qr[c]["searched"] and qr[c]["d1"] == qr[c]["d2"] == qr[c]["nd"] for c in ("cusolver", "geqrf")),
     "both QR vendor controls return torch.geqrf's output")
says("QR has two vendor controls, `geqrf` and `cusolver`, and both call `torch.geqrf`")

# How to read a verdict
need(sum(r["ran_path"] != "kernel" and "False" in (r["pass_v1"], r["pass_v2"]) for r in sweep + scored) == 0, "vendor-identical QR outputs passed both checkers")
need({r["status"] for r in sweep + scored} == {"OK"}, "every QR evaluation returned an output")
fam = sorted({r["case"] for r in sweep if r["case"] != "randsvd"})
need(fam == sorted("dense upper diagonal rankdef nearrank clustered band nearcollinear rowscale mixed".split()), "the ten organizer families")
says("For the ten organizer families (`dense`, `upper`, `diagonal`, `rankdef`, `nearrank`, `clustered`, `band`, `nearcollinear`, `rowscale`, `mixed`)")
need(len({r["case"] for r in sweep}) == 11, "eleven input families")
says("out of the 11 families"); says("In the eleventh family, `randsvd`")
says(f"{W[ncand].capitalize()} failing search inputs per entry were promoted to candidates. They are attributable failures wherever the entry had any, "
     f"and otherwise failures on which the vendor routine also failed. Each candidate was rerun {W[nrerun]} times:")
for p, d in (("cholesky", chol), ("eigh", eigh)):
    reg = {(r["target"], r["case_id"]): r for r in rows(f"{p}/registered.csv")}
    for q in entries(d):
        t = q["row"]["target"]; cs = {r["confirm_candidate"] for r in rows(f"{p}/confirmed.csv") if r["target"] == t}
        kinds = {(reg[(t, c)]["official_pass"], reg[(t, c)]["vendor_pass"]) for c in cs}
        need(kinds == ({("False", "True")} if q["attr"] > 0 else {("False", "False")}), f"{p} {t}: candidate selection")
held = {r["seed"] for p in ("cholesky", "eigh") for r in rows(f"{p}/confirmed.csv") if r["case_source"] == "confirm_heldout"}
need(len(held) == 1, "one held-out seed for every candidate")

# QR results
mah = next(q for q in q1 if q["row"]["handle"] == "Maheshram1")
mf = sorted((r["case"], r["n"]) for r in scored if r["target"] == "Maheshram1" and r["round_of_case"] == "v2" and r["pass_v1"] == "False" and key(r) not in set(v1_cases))
need(mah["nd"] - mah["d1"] == len(mf) == 3 and mf == [("clustered", "512"), ("mixed", "512"), ("rankdef", "512")],
     "Maheshram1 fails three round-2 cases under the round-1 checker")
says(f"Maheshram1's {W[len(mf)]} failures under the round-1 checker are the round-2 benchmark cases `mixed` (order 512), `rankdef` and `clustered`, "
     f"which are not in the round-1 benchmark list.")
pairs = {}
for r in scored: pairs.setdefault((r["target"], key(r)), []).append(r)
need(all(len(v) == 2 for (t, k), v in pairs.items() if k in set(v1_cases)), "the shared cases were run twice")
vary = sorted({t for (t, k), v in pairs.items() if len(v) == 2 and v[0]["f"] != v[1]["f"]})
says(f"Each of the {len(v1_cases)} shared cases was run twice, once for each list. The two runs always got the same verdicts, but for "
     f"{' and '.join(qr[t]['row']['handle'] for t in vary)} their residuals differ slightly")
rsall = [r for r in sweep if r["case"] == "randsvd" and r["target"] not in CONTROL_ORDER]
floor = min(float(r["kappa"]) for r in rsall)
need(floor < 1.0001, "the bottom of the searched randsvd range is about 1")
says(f"Of the entries' {len(rsall)} `randsvd` breaks, {sum(float(r['kappa']) == floor for r in rsall)} are at the bottom of the searched range, a condition number of about 1")
win_t = next(q for q in q2 if q["row"]["rank"] == "1")["row"]["target"]
win = [r for r in sweep if r["target"] == win_t and r["broke"] == "True" and r["case"] == "rankdef" and r["n"] == "32"]
need(win and {r["note"] for r in win} == {"Q or R non-finite"}, "the round-2 winner's rank-deficient order-32 breaks are non-finite")
says(f"The first-ranked round-2 entry's breaks include batches of {one([r['batch'] for r in win], 'winner batch')} rank-deficient matrices of order 32 "
     f"that returned non-finite factors.")
kb = [r for r in kbr if r["target"] == win_t and r["min_break_kappa"]]
kmin = min(kb, key=lambda r: float(r["min_break_kappa"]))
says(f"Its lowest `randsvd` threshold was about {round(float(kmin['min_break_kappa']))}, at order {kmin['n']} (`qr/kappa_bisect.csv`).")
need(all(r["pass_v1"] == r["pass_v2"] == "True" for c in ("cusolver", "geqrf") for r in sweep if r["target"] == c), "the geqrf controls passed every searched input")
says("The two `torch.geqrf` controls passed both checkers on every searched input.")
says(f"the `cusolver` control's `handle` in `qr/targets.csv` and `qr/summary.csv` reads \"{qr['cusolver']['row']['handle']}\"")
need(summ["cusolver"]["handle"] == qr["cusolver"]["row"]["handle"] and all("cuSOLVER" in qr[c]["row"]["handle"] for c in CONTROL_ORDER),
     "the QR control labels name cuSOLVER")

# Cholesky and eigendecomposition results
says(f"*Scored* counts the organizers' {ct} test and {cb} benchmark cases passed.")
says(f"search failures attributable to the entry, out of {c_valid} valid inputs.")
says(f"*Reproduced* counts reproduced candidates, out of {ncand}.")
says(f"with {et} test and {eb} benchmark cases and attributable failures out of {e_evals} inputs.")
ev = [r for r in rows("eigh/registered.csv") if r["target"] == "vendor"]
vend = [r for r in ev if r["official_pass"] == "False"]
need({r["difficulty"] for r in vend} == {'{"cond": 12}'} and {r["family"] for r in vend} == {"dense", "rowscale", "mixed"}, "eigh vendor failures")
says(f"The vendor routine itself failed the checker on {len(vend)} of the {e_evals} search inputs, all at the audit's conditioning parameter `cond` = 12, "
     f"in the families `dense`, `rowscale` and `mixed`.")
verr = [r for r in ev if r["official_pass"] == ""]
need(len(verr) == 2 and sorted(r["family"] for r in verr) == ["dense", "rowscale"]
     and all(r["n"] == "512" and r["batch"] == "640" and r["difficulty"] == '{"cond": 12}' and "failed to converge" in r["error"] for r in verr),
     "the eigh vendor convergence errors")
says(f"On {len(verr)} more inputs, `dense` and `rowscale` of order 512 with batch 640 at `cond` = 12, it raised a convergence error.")
zero = [q for q in entries(eigh) if q["attr"] == 0]
econ = rows("eigh/confirmed.csv")
for q in zero:
    need({(r["official_pass"], r["vendor_pass"]) for r in econ if r["target"] == q["row"]["target"]} == {("False", "False")},
         f"eigh {q['row']['target']}: every rerun failed with the vendor failing")
says(f"The {W[len(zero)]} entries with no attributable failure, {' and '.join(q['row']['handle'] for q in zero)}, "
     f"had candidates only from inputs on which the vendor routine also failed.")

# Environment
qenv = os.path.join("environment", "qr_b200.txt"); cenv = os.path.join("environment", "cholesky_eigh_b200.txt")
nvcc = re.search(r"cuda_(\d+\.\d+)", cfg(qenv, "nvcc")).group(1)
says(f"PyTorch {cfg(qenv, 'torch').split('+')[0]} (CUDA {cfg(qenv, 'cuda(rt)')} build), Triton {cfg(qenv, 'triton')}, "
     f"Python {'.'.join(cfg(qenv, 'python').split('.')[:2])} and nvcc {nvcc}")
need(cfg(cenv, "torch") == "2.12.0" and cfg(cenv, "base") == "nvidia/cuda:13.3.0-devel-ubuntu24.04, Python 3.13" and "ae732f7" in cfg(cenv, "runner"),
     "Cholesky and eigendecomposition environment")
says("at kernelbot commit `ae732f7` (2026-07-06): PyTorch 2.12.0, Python 3.13, and the base image `nvidia/cuda:13.3.0-devel-ubuntu24.04`")
envs = {p: Counter((r["target"] == "Olek", r["env_hash"]) for f in ("official", "registered", "confirmed") for r in rows(f"{p}/{f}.csv")) for p in ("cholesky", "eigh")}
olek = {p: one([h for (o, h) in envs[p] if o], f"{p} Olek environment") for p in envs}
other = one([h for p in envs for (o, h) in envs[p] if not o], "other environments")
says(f"so it ran in environments of its own: `{olek['cholesky']}` for Cholesky and `{olek['eigh']}` for eigendecomposition. Every other target ran in `{other}`")
hashes = open(os.path.join(R, "environment", "cholesky_eigh_b200_hashes.txt")).read()
need(all(h in hashes for h in (olek["cholesky"], olek["eigh"], other)), "environment hashes listed in cholesky_eigh_b200_hashes.txt")

# Files, reproduction, citation
says(f"the {W[len(qt)]} QR targets")
need(one([len([r for r in scored if r["target"] == t["target"]]) for t in qt], "scored rows") == len(v1_cases) + len(v2_cases), "scored rows per QR target")
says(f"{len(v1_cases) + len(v2_cases)} rows per target, the {len(v1_cases)} round-1 cases (`round_of_case` = `v1`) and the {len(v2_cases)} round-2 cases (`v2`)")
says(f"the {num(len(g0))} grid evaluations per target and the failing `randsvd` evaluation at each threshold found")
rk = re.search(r"reference-kernels commit at which the file hashes below were verified: (\w+)", open(os.path.join(R, "environment", "checkers.md")).read()).group(1)
says(f"git -C reference-kernels checkout {rk}"); says(f"reference-kernels/blob/{rk}/LICENSE")
hdr = open(os.path.join(R, "qr", "targets.csv")).readline().strip().split(",")
says(f"awk -F, '${hdr.index('handle') + 1}==\"10billiontokens\"{{print ${hdr.index('submission_sha256') + 1}}}' qr/targets.csv")
ex1 = next(r for r in sweep if r["target"] == "10billiontokens" and r["case"] == "rankdef" and r["n"] == "32" and r["cond"] == "2" and r["seed"] == "101")
ex2 = next(r for r in sweep if r["target"] == "10billiontokens" and r["case"] == "randsvd" and r["n"] == "512")
need(ex1["pass_v1"] == ex1["pass_v2"] == "False" and ex2["pass_v1"] == "True" and ex2["pass_v2"] == "False", "the replay examples' verdicts")
says(f"The first example failed both checkers in the audit (\"{ex1['reason']}\"). The second passed the round-1 checker and failed the round-2 checker.")
says(f"(its `f`, the round-2 residual ratio defined in `COLUMNS.md`, is {float(ex2['f']):.2f} against the passing limit of 1)")
req = open(os.path.join(R, "replay", "requirements.txt")).read().split()
need(f"torch=={cfg(qenv, 'torch').split('+')[0]}" in req and f"triton=={cfg(qenv, 'triton')}" in req and any(x.startswith("ninja") for x in req),
     "replay/requirements.txt pins the QR audit's PyTorch and Triton and lists ninja")
cff = open(os.path.join(R, "CITATION.cff")).read()
ver = re.search(r"^version: (\S+)", cff, re.M).group(1); date = re.search(r"^date-released: (\S+)", cff, re.M).group(1)
says(f"This is version {ver} of the record (git tag `v{ver}`, {date}).")
says(f"note = {{Version {ver}, git tag v{ver}}}"); says(f"**{ver} ({date}).**")

# ---------------------------------------------------------------- hygiene
for f in glob.glob(os.path.join(R, "raw", "*", "*.jsonl")):
    for l in open(f):
        if l.strip():
            bad = set(json.loads(l)) & {"gpu_uuid", "process_id", "error_tail"}
            if bad: errs.append(f"{f}: identifier fields present {bad}"); break
for f in glob.glob(os.path.join(R, "raw", "*", "*.json")) + [os.path.join(R, "raw", "manifest.json")]:
    txt = open(f).read()
    if any(k in txt for k in ("gpu_uuid", "process_id", "error_tail", "/" + "home/", "submissions/")): errs.append(f"{f}: identifier, path or private field present")
# Only the record's own files: a reference-kernels clone or a virtual environment next to them is not part of it.
OWN = ["README.md", "COLUMNS.md", "CITATION.cff", "LICENSE", "LICENSE-CODE", ".gitignore", ".gitattributes"]
OWN += [f for d in ("qr", "cholesky", "eigh", "raw", "environment", "replay", "scripts") for f in glob.glob(os.path.join(R, d, "**", "*"), recursive=True)]
for f in OWN:
    f = os.path.join(R, f)
    if os.path.isfile(f) and "__pycache__" not in f and os.path.basename(f) != "check_record.py" and ("/" + "home/") in open(f, errors="ignore").read():
        errs.append(f"{os.path.relpath(f, R)}: host path")

# ---------------------------------------------------------------- replay examples
examples = list(re.finditer(r"--submission (\S+)\.py \\\n\s+--case (\w+) --n (\d+) --batch (\d+)(?: --cond (\d+))?(?: --kappa (\S+))? --seed (\d+)", readme))
for m in examples:
    t, case, n, batch, cond, kappa, seed = m.groups()
    hit = [r for r in sweep if r["target"] == t and r["case"] == case and r["n"] == n and r["batch"] == batch and r["seed"] == seed
           and (cond is None or r["cond"] == cond) and (kappa is None or r["kappa"] == kappa)]
    if len(hit) != 1: errs.append(f"README replay example {t} {case} n={n} batch={batch} cond={cond} kappa={kappa} seed={seed}: {len(hit)} matching sweep rows")
    elif hit[0]["broke"] != "True": errs.append(f"README replay example {t} {case} n={n}: the row did not break")
if len(examples) != len(re.findall(r"python3 replay/replay_qr\.py", readme)) or len(examples) < 2:
    errs.append("README replay examples: every replay_qr.py command must name --submission, --case, --n, --batch and --seed in that order")

if errs:
    print("\n".join(errs)); sys.exit(1)
print("check_record: the README tables and the checked counts match the data")
