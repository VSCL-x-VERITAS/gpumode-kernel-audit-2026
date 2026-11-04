# GPU MODE linear-algebra kernel audit, 2026: results record

**Status: published on GitHub. Zenodo DOI pending.**

This record holds the results of an independent audit of the top-ranked entries in
the GPU MODE community's 2026 competitions for fast GPU linear algebra kernels:
batched QR factorization (two rounds, leaderboards 773 and 774), batched symmetric
eigendecomposition (775), and batched Cholesky factorization (776). The organizers
scored each entry on a fixed set of inputs. We ran the five highest-ranked entries of
each round, unmodified, through the organizers' own checker on further legal inputs.
All ten QR entries broke in that search. Of the ten Cholesky and eigendecomposition
entries, seven had failures that reproduced when rerun in fresh containers, one failed
85 times in the search and never in a rerun, and two had no failure attributable to
them. The audit motivates the paper *Principles for Autoformalizing Numerical Methods*
(Fountoulakis, Melnichenko, Mahoney, 2026), Section 4.4, which cites this record.

The record contains verdicts, the specification of every input, and environment
hashes. It contains no entry's code, which the organizers publish themselves (see
Attribution), and none of the software that searched for the failing inputs.

## Terms

The *vendor routine* is cuSOLVER called through PyTorch (`torch.linalg.qr`,
`torch.geqrf`, `torch.linalg.cholesky_ex`, `torch.linalg.eigh`). A *fresh container* is a
new process on a new cloud instance. A *candidate* is a failing discovery input promoted
to reruns. Entries appear under two names: `target` is the label of the raw result file
and `handle` is the name on the leaderboard. They differ twice: target `nernst` is handle
`gau.nernst`, and target `Ravi_Theja` is handle `Ravi Theja`. Controls have no handle, and
their `handle` column holds a description. The tables below use handles. `COLUMNS.md` defines every column and the file conventions.

## What was run

| Round | Leaderboard | Entries | Scored cases per entry | Searched evaluations per entry |
|---|---|---|---|---|
| QR, round 1 | 773 | ranks 1 to 5 | 19 | 1,050 to 1,055 |
| QR, round 2 | 774 | ranks 1 to 5 | 19 | 1,050 to 1,055 |
| Eigendecomposition | 775 | ranks 1 to 5 | 52 | 996 |
| Cholesky | 776 | ranks 1 to 5 | 32 | 602 generated, 567 valid |

Every evaluation ran on an NVIDIA B200, the GPU the competitions graded on, with the
organizers' checker unmodified. `environment/checkers.md` lists the checker files'
hashes and the `reference-kernels` commit at which they were verified identical to the
public files. Each QR output was graded by both versions of the QR checker, the round-1
version and the round-2 version, which rejects more outputs, as the Scored v1 and v2
columns below show. For Cholesky, 35 of the 602 generated inputs were not positive
definite in float32 and were excluded from every verdict. Each problem had two
controls: the vendor routine, and the vendor routine on inputs first rounded to a
lower precision (bfloat16 for QR and Cholesky, float8 e4m3 for eigendecomposition),
which is expected to fail on many inputs and does.

## How to read a verdict

- **QR.** A break is one searched evaluation in which the entry ran its own code path
  and the output failed the round-2 checker. In `qr/sweep.csv`, `broke` is true exactly
  when `status` is `OK`, `ran_path` is `kernel`, and `pass_v2` is false. Evaluations in
  which the entry dispatched to the vendor fallback, `ran_path` = `fallback_geqrf`, pass
  and are not breaks. The round-1 checker's verdict is in `pass_v1`. QR breaks were not
  rerun in fresh containers.
- **Cholesky and eigendecomposition.** A failure in the discovery sweep is attributable
  to the entry when the organizers' checker fails it, the vendor routine passes the same
  input, and the input passed validation: `official_pass` false, `vendor_pass` true,
  `input_valid` not false. Five failing inputs per entry were promoted to candidates, and
  each candidate was rerun three times: a cold rerun of the input in a fresh container, a
  cold rerun with a held-out seed of the same family and difficulty, and a warm rerun, a
  second call in the same process. A candidate is reproduced when all three reruns fail
  with the vendor passing. The paper's claims about these two problems rest on reproduced
  candidates.

## Results

### QR, leaderboards 773 and 774

Scored = scored cases passed under the round-1 / round-2 checker. Breaks = searched
evaluations that broke. The columns mean the same for the controls.

| Round | Rank | Entry | Scored v1 | Scored v2 | Breaks |
|---|---|---|---|---|---|
| 1 | 1 | nikhilbarhate99 | 19/19 | 4/19 | 755 |
| 1 | 2 | badelsteinlelbach | 19/19 | 19/19 | 92 |
| 1 | 3 | Sinatras | 19/19 | 19/19 | 106 |
| 1 | 4 | Maheshram1 | 16/19 | 4/19 | 586 |
| 1 | 5 | elianaive | 19/19 | 19/19 | 3 |
| 2 | 1 | 10billiontokens | 19/19 | 19/19 | 157 |
| 2 | 2 | gau.nernst | 19/19 | 19/19 | 46 |
| 2 | 3 | dhu.randhar | 19/19 | 19/19 | 330 |
| 2 | 4 | zhongmingee | 19/19 | 19/19 | 3 |
| 2 | 5 | michaelmelons | 19/19 | 19/19 | 360 |
| control | | cuSOLVER, torch.linalg.qr | 19/19 | 19/19 | 0 |
| control | | cuSOLVER, torch.geqrf | 19/19 | 19/19 | 0 |
| control | | cuSOLVER geqrf on bfloat16-rounded input | 8/19 | 7/19 | 623 |

### Cholesky, leaderboard 776

Attributable = discovery failures attributable to the entry, out of 567 valid inputs.
Reproduced = candidates reproduced in all three reruns, out of 5 promoted.

| Rank | Entry | Scored | Attributable | Reproduced |
|---|---|---|---|---|
| 1 | zhongmingee | 32/32 | 194 | 4 of 5 |
| 2 | Frosty40 | 32/32 | 129 | 5 of 5 |
| 3 | Olek | 32/32 | 42 | 5 of 5 |
| 4 | viridale | 32/32 | 233 | 5 of 5 |
| 5 | Ravi Theja | 32/32 | 85 | 0 of 5 |
| control | cuSOLVER, torch.linalg.cholesky_ex | 32/32 | 0 | |
| control | cuSOLVER on bfloat16-rounded input | not run | 437 | |

Ravi Theja's entry failed 85 times in the discovery sweep and passed all 15 reruns of
its five candidates. The record states the observation only.

### Eigendecomposition, leaderboard 775

Attributable = discovery failures attributable to the entry, out of 996 inputs.

| Rank | Entry | Scored | Attributable | Reproduced |
|---|---|---|---|---|
| 1 | teelaitila | 52/52 | 11 | 5 of 5 |
| 2 | nikhilbarhate99 | 52/52 | 0 | 0 of 5 |
| 3 | elianaive | 52/52 | 0 | 0 of 5 |
| 4 | zhongmingee | 52/52 | 18 | 5 of 5 |
| 5 | Olek | 52/52 | 61 | 5 of 5 |
| control | cuSOLVER, torch.linalg.eigh | 52/52 | 0 | |
| control | cuSOLVER on float8-rounded input | not run | 787 | |

The vendor routine itself failed the checker on 38 of the 996 discovery inputs, all at
the audit's conditioning parameter `cond` = 12, in the families `dense`, `rowscale` and
`mixed`. Those inputs count against no entry. The two entries with no attributable
failure dispatch to the vendor routine on many inputs: `ran_path` = `fallback_vendor` on
439 of 996 evaluations for elianaive and 186 for nikhilbarhate99.

## Files

| Path | Content |
|---|---|
| `qr/targets.csv` | the thirteen QR targets: round, leaderboard, handle, rank, SHA-256 of the entry file as downloaded from the leaderboard |
| `qr/scored.csv` | every scored case with both checker verdicts |
| `qr/sweep.csv` | every searched evaluation: family, shape, condition knob, seed, both verdicts, residual diagnostics |
| `qr/breaks.csv` | the rows of `sweep.csv` that broke |
| `qr/kappa_bisect.csv` | per entry and shape, the smallest `randsvd` condition number at which it broke |
| `qr/summary.csv` | one row per target, the numbers in the QR table |
| `qr/randsvd.py` | the one input family that is not the organizers' generator |
| `cholesky/`, `eigh/` | `targets.csv`, `official.csv` (scored cases), `registered.csv` (discovery sweep), `confirmed.csv` (reruns), `summary.csv` |
| `COLUMNS.md` | meaning of every column, and the file conventions |
| `raw/` | the audit's result files, from which `scripts/build_record.py` derives every CSV |
| `raw/manifest.json` | the entries' leaderboard metadata and file hashes |
| `environment/` | software versions, GPU, checker and runner hashes |
| `replay/` | `replay_qr.py`, `fetch_submission.py`, pinned `requirements.txt` |
| `scripts/check_record.py` | recomputes every number in this README from the CSV files |

## Reproducing a QR verdict

Every QR input is determined by its row in `qr/sweep.csv`. For the ten organizer
families (`dense`, `upper`, `diagonal`, `rankdef`, `nearrank`, `clustered`, `band`,
`nearcollinear`, `rowscale`, `mixed`) the input is the organizers' own
`generate_input(batch, n, cond, seed, case)` from `reference-kernels`. For `randsvd` it
is `qr/randsvd.py`. The row's `kappa` is the bisection threshold for that entry and
shape, taken from `qr/kappa_bisect.csv`, and `cond` is empty.

```
git clone https://github.com/gpu-mode/reference-kernels
git -C reference-kernels checkout f3295bb6bd559bae4659d2d8fc45edb1394251b1
pip install -r replay/requirements.txt
python replay/fetch_submission.py --leaderboard 774 --user 10billiontokens \
    --expect-sha256 $(awk -F, '$1=="10billiontokens"{print $7}' qr/targets.csv) --out 10billiontokens.py
python replay/replay_qr.py --reference-kernels reference-kernels --submission 10billiontokens.py \
    --case rankdef --n 32 --batch 20 --cond 2 --seed 101
python replay/replay_qr.py --reference-kernels reference-kernels --submission 10billiontokens.py \
    --case randsvd --n 512 --batch 640 --kappa 794.3258949419277 --seed 7
```

Each replay prints the verdict under both checkers. Compare them with the row's `pass_v1`
and `pass_v2`. Round 2 entries come from the organizers' dataset through
`fetch_submission.py`, which keeps the row whose code hashes to the SHA-256 in
`qr/targets.csv`. Round 1 entries (leaderboard 773) are not in that dataset, and the leaderboard API
(https://www.gpumode.com/api/leaderboard/773) returns rankings without code, so their
verdicts can be checked in the tables but not replayed from public sources unless the
organizers release the round-1 code. The SHA-256 in `qr/targets.csv` identifies the
audited file, and `replay_qr.py --expect-sha256` confirms a file against it. Use the versions in
`environment/qr_b200.txt` and a B200: some entries select a code path by shape, as
`ran_path` records, and other hardware may take a different path.

## Cholesky and eigendecomposition inputs

The discovery inputs for these two problems came from the audit's own generators, which
are not part of this record. Each row of `registered.csv` and `confirmed.csv` names the
family, the difficulty parameters, the shape, the seed, a case id, the validation
evidence for the input, and the checker's diagnostics, so a verdict can be inspected and
a family re-targeted, but the inputs cannot be regenerated from this record alone. The
scored cases in `official.csv` are the organizers' and are reproducible with their code.

## What is not here

- **The search software.** The generators for the Cholesky and eigendecomposition
  families, the sweep and bisection drivers, and the container protocol belong to a
  system described in a separate publication. The raw result files are the audit's own,
  with GPU serial numbers, process ids and local paths removed. The rerun fields that
  remain (`cold_or_warm`, `repeat_id`, `case_source`, `input_validity_evidence`,
  `protocol_hash`) are kept on purpose: they are the evidence behind the reproduction rule.
- **Entry code and per-run logs.** See Attribution for the code. The logs add nothing to
  any verdict.

## Attribution and license

The competitions, the checkers, the scored cases, and the entry dataset are the work of
GPU MODE (https://www.gpumode.com) and its contestants. Entries are named by the handles
the organizers publish on their leaderboards. The organizers release entry code in the
Hugging Face dataset https://huggingface.co/datasets/GPUMODE/kernelbot-data
(configuration `linalg_submissions`, leaderboards 774, 775 and 776) under the June 9
Researcher Reciprocity License in that dataset's `LICENSE` file, which asks users to cite
GPU Mode and link to the dataset. This record does so and copies none of it. The checker
and generator code is at https://github.com/gpu-mode/reference-kernels, under the same
license in that repository's `LICENSE` file. The replay script loads it at run time and
nothing of it is copied here.

Copyright 2026 Maksim Melnichenko. The data in this record, everything outside
`scripts/`, `replay/` and `qr/randsvd.py`, is released under CC BY 4.0 (`LICENSE`). The
scripts are released under the MIT License (`LICENSE-CODE`). Cite as in `CITATION.cff`.
