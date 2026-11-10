# GPU MODE linear-algebra kernel audit, 2026: results record

This is version 1.0.1 of the record (git tag `v1.0.1`, 2026-11-10). The data are under
CC BY 4.0 and the code under the MIT License, with the exceptions stated under Attribution
and license.

This record holds the results of an independent audit of the five top-ranked entries on
each of the four leaderboards of GPU MODE's 2026 competitions for fast GPU linear-algebra
kernels: batched QR factorization (two rounds, leaderboards 773 and 774), batched
Cholesky factorization (776), and batched symmetric eigendecomposition (775). Ranks are
as the leaderboards showed them when the entries were downloaded, in July and August
2026. The organizers checked each entry's accuracy on a fixed set of cases and ranked
the entries that passed by speed. The audit ran each entry, unmodified, through the
organizers' own checker on further legal inputs, on the GPU model the competitions
graded on.

The record contains the verdicts, a description of every recorded evaluation (enough to
regenerate the input of every recorded QR evaluation), and the hashes of the entries,
checkers and environments. It contains no entry's code. The organizers publish the code
of the round-2 QR, Cholesky and eigendecomposition entries (see Attribution and license),
and the round-1 QR code is not public. The record also contains none of the software
that searched for the failing inputs.

## Summary

| Round | Leaderboard | Passed own round's scored cases | Failed in the search | Failure reproduced in reruns |
|---|---|---|---|---|
| QR, round 1 | 773 | 5 of 5 | 5 of 5 | not rerun |
| QR, round 2 | 774 | 5 of 5 | 5 of 5 | not rerun |
| Cholesky | 776 | 5 of 5 | 5 of 5 | 4 of 5 |
| Eigendecomposition | 775 | 5 of 5 | 3 of 5 | 3 of 5 |

All twenty entries passed the scored cases of their own round, and eighteen failed the
same checker on other legal inputs on which the vendor routine (the PyTorch call, backed
by NVIDIA's libraries) passed. All ten QR entries failed. A QR failure is a single
evaluation, and none was rerun. For Cholesky and eigendecomposition, five failing inputs
per entry became candidates for reruns. Seven of those ten entries had at least one
candidate reproduced: all three of its reruns failed while the vendor routine passed. An
eighth, the fifth-ranked Cholesky entry, failed 85 times in the search and passed every
rerun. The next section defines these terms.

The audit is described in the paper *The Need for Formalized Numerical Methods*
(Fountoulakis, Melnichenko, et al., in preparation), which cites this record. This README
will link the paper when it is public.

## Terms

- **Scored cases** are the organizers' own cases that this record regrades, from the task
  files in [`reference-kernels`](https://github.com/gpu-mode/reference-kernels). A
  leaderboard run checks an entry on the task's test cases and then times it on the
  benchmark cases, which are checked too. For QR this record regrades the benchmark cases
  (7 in round 1, 12 in round 2) and not the test cases (19 and 22). For Cholesky it
  regrades both, 17 test and 15 benchmark cases, and for eigendecomposition both, 39 and
  13.
- A **legal input** is one the task allows. For QR it is a batch of n × n float32 matrices
  on the GPU, and the round-2 task statement bounds neither their conditioning nor their
  structure: "Each matrix must be factored correctly on its own merits." A Cholesky
  input must also be positive definite in float32. The audit tested this with
  `torch.linalg.cholesky_ex`, and the 35 generated inputs that failed the test are
  excluded from every verdict.
- The **search** is the audit's own set of further inputs. The files call it the
  discovery sweep (`registered`), and they call the reruns confirmation (`confirmed`).
- The **vendor routine** is the PyTorch call `torch.geqrf`, `torch.linalg.cholesky_ex` or
  `torch.linalg.eigh` on the GPU. In PyTorch 2.12, `torch.geqrf` uses cuBLAS's batched
  geqrf for orders up to 256 in batches of at least max(2, n/16) matrices, which covers
  the QR shapes 32 × 20 and 176 × 40. It uses cuSOLVER otherwise, and the other two calls
  use cuSOLVER. An output is **vendor-identical** when it equals, bit for bit, the vendor
  routine's output on the same input, which indicates that the entry handed that input
  to the vendor routine.
- A **candidate** is a failing search input promoted to reruns, and a **fresh container**
  is a new process on a new cloud instance.

Entries appear under two names: `target` is the label of the raw result file and `handle`
is the name on the leaderboard. The two differ for two entries: target `nernst` is handle
`gau.nernst`, and target `Ravi_Theja` is handle `Ravi Theja`. The tables below use handles
for entries, and the routine called plus the `target` label for controls. `COLUMNS.md`
defines every column and the file conventions.

## What was run

| Round | Leaderboard | Scored cases per entry | Searched evaluations per entry |
|---|---|---|---|
| QR, round 1 | 773 | 7 | 1,050 to 1,055 |
| QR, round 2 | 774 | 12 | 1,050 to 1,053 |
| Cholesky | 776 | 32 | 602 generated, 567 valid |
| Eigendecomposition | 775 | 52 | 996 |

Every evaluation ran on an NVIDIA B200 with the organizers' checker unmodified.
`environment/checkers.md` lists the checker files' hashes and the `reference-kernels`
commit at which they were verified identical to the public files.

The QR search keeps the task's benchmark shapes and batch sizes, so that shape-dependent
fast paths stay engaged. For each target it is a grid of 1,050 evaluations: the
organizers' ten families, the 7 benchmark shapes, the conditioning knob `cond` in
{2, 4, 6, 8, 12}, and the seeds 101, 202 and 303. The generator ignores `cond` for
`clustered` and raises it to at least 4 for `rowscale`, so the grid holds 945 distinct
inputs per target, and the repeated inputs always got the same verdicts. In the eleventh
family, `randsvd`, a bisection over the condition number up to 10<sup>12</sup>, with seed 7,
located for each shape the threshold at which the entry starts to fail. `qr/sweep.csv`
adds the failing evaluation at each threshold found, so a target has 1,050 to 1,055 rows.
The bisection's other probes are not recorded.

The organizers tightened the QR checker between the two rounds. The round-2 checker
judges each matrix's factorization residual against that matrix's own norm, where the
round-1 checker compared the largest residual in the batch with the largest norm, and it
also rejects non-finite Q, R and residuals. The round-1 benchmark list has 7 dense cases.
The round-2 benchmark list has 12: the same 7 and five structured ones (two `mixed`, one
`rankdef`, one `clustered`, one `nearrank`). Every QR output was graded by both checker
versions, and every QR target was graded on all 12 distinct benchmark cases. In the
search the round-2 checker rejected every output that the round-1 checker rejected, and
more.

Each problem had a vendor control and a degraded control. The degraded control calls the
vendor routine on inputs first rounded to a lower precision (bfloat16 for QR and
Cholesky, float8 e4m3 for eigendecomposition). It is expected to fail on many inputs, and
it does. QR has two vendor controls, `geqrf` and `cusolver`, and both call `torch.geqrf`
(see the QR results).

## How to read a verdict

- **QR.** A break is one searched evaluation whose output was not vendor-identical and
  that the round-2 checker rejected. In `qr/sweep.csv`, `broke` is true exactly when
  `status` is `OK`, `ran_path` is `kernel`, and `pass_v2` is false. Vendor-identical
  outputs (`ran_path` = `fallback_geqrf`) passed both checkers on every input and are
  never breaks. The round-1 checker's verdict is in `pass_v1`. QR breaks were not rerun.
- **Cholesky and eigendecomposition.** A failure in the search is attributable to the
  entry when the organizers' checker fails it, the vendor routine passes the same input,
  and the input passed validation: `official_pass` false, `vendor_pass` true,
  `input_valid` not false. Five failing search inputs per entry were promoted to
  candidates. They are attributable failures wherever the entry had any, and otherwise
  failures on which the vendor routine also failed. Each candidate was rerun three times:
  - the same input in a fresh container;
  - a held-out seed of the same family and difficulty, in a fresh container;
  - the same input as a warm call, a later call in an already running process.

  A candidate is reproduced when all three reruns fail with the vendor routine passing.

## Results

### QR, leaderboards 773 and 774

*Own round's cases* means the 7 round-1 benchmark cases under the round-1 checker for
round-1 entries, and the 12 round-2 benchmark cases under the round-2 checker for
round-2 entries. The last two columns grade every target on all 12 distinct benchmark
cases under each checker.

| Round | Rank | Entry | Own round's cases, own round's checker | All 12 cases, round-1 checker | All 12 cases, round-2 checker |
|---|---|---|---|---|---|
| 1 | 1 | nikhilbarhate99 | 7/7 | 12/12 | 2/12 |
| 1 | 2 | badelsteinlelbach | 7/7 | 12/12 | 12/12 |
| 1 | 3 | Sinatras | 7/7 | 12/12 | 12/12 |
| 1 | 4 | Maheshram1 | 7/7 | 9/12 | 3/12 |
| 1 | 5 | elianaive | 7/7 | 12/12 | 12/12 |
| 2 | 1 | 10billiontokens | 12/12 | 12/12 | 12/12 |
| 2 | 2 | gau.nernst | 12/12 | 12/12 | 12/12 |
| 2 | 3 | dhu.randhar | 12/12 | 12/12 | 12/12 |
| 2 | 4 | zhongmingee | 12/12 | 12/12 | 12/12 |
| 2 | 5 | michaelmelons | 12/12 | 12/12 | 12/12 |
| control |  | torch.geqrf (`cusolver`) |  | 12/12 | 12/12 |
| control |  | torch.geqrf (`geqrf`) |  | 12/12 | 12/12 |
| control |  | torch.geqrf on bfloat16-rounded input (`geqrf_bf16in`) |  | 5/12 | 4/12 |

Maheshram1's three failures under the round-1 checker are the round-2 benchmark cases
`mixed` (order 512), `rankdef` and `clustered`, which are not in the round-1 benchmark
list. Each of the 7 shared cases was run twice, once for each list. The two runs always
got the same verdicts, but for dhu.randhar and zhongmingee their residuals differ
slightly, so those entries' outputs vary from call to call.

In the next table, *Searched* counts a target's rows in `qr/sweep.csv`, and
*Vendor-identical* counts how many of them returned `torch.geqrf`'s output. The round-2
break column is the record's break rule (`broke`). The round-1 column applies the same
rule with the checker that round-1 entries competed under. Its `randsvd` rows are those
at the round-2 thresholds, since no bisection used the round-1 checker. *Families with
breaks* counts, out of the 11 families, those with at least one round-2 break.

| Round | Rank | Entry | Searched | Vendor-identical | Breaks, round-2 checker | Breaks, round-1 checker | Families with breaks |
|---|---|---|---|---|---|---|---|
| 1 | 1 | nikhilbarhate99 | 1,055 | 60 | 755 | 194 | 11 |
| 1 | 2 | badelsteinlelbach | 1,051 | 75 | 92 | 74 | 6 |
| 1 | 3 | Sinatras | 1,051 | 315 | 106 | 74 | 5 |
| 1 | 4 | Maheshram1 | 1,051 | 195 | 586 | 246 | 10 |
| 1 | 5 | elianaive | 1,050 | 0 | 3 | 3 | 1 |
| 2 | 1 | 10billiontokens | 1,052 | 12 | 157 | 153 | 10 |
| 2 | 2 | gau.nernst | 1,050 | 210 | 46 | 18 | 2 |
| 2 | 3 | dhu.randhar | 1,053 | 138 | 330 | 304 | 10 |
| 2 | 4 | zhongmingee | 1,050 | 518 | 3 | 3 | 1 |
| 2 | 5 | michaelmelons | 1,053 | 30 | 360 | 336 | 11 |
| control |  | torch.geqrf (`cusolver`) | 1,050 | 1,050 | 0 | 0 | 0 |
| control |  | torch.geqrf (`geqrf`) | 1,050 | 1,050 | 0 | 0 | 0 |
| control |  | torch.geqrf on bfloat16-rounded input (`geqrf_bf16in`) | 1,054 | 0 | 623 | 604 | 11 |

Of the 20 `randsvd` breaks, 15 are at the bottom of the searched range, a condition
number of about 1, so those entries failed on nearly orthogonal matrices. The first-ranked
round-2 entry's breaks include batches of 20 rank-deficient matrices of order 32 that
returned non-finite factors. Its lowest `randsvd` threshold was about 794, at order 512
(`qr/kappa_bisect.csv`). The two replay examples below are these two evaluations.

The two `torch.geqrf` controls passed both checkers on every searched input. The
`cusolver` control is a second run of `torch.geqrf` under another name and agrees with
`geqrf` bit for bit. Two control labels in the data are inaccurate:
- the `cusolver` control's `handle` in `qr/targets.csv` and `qr/summary.csv` reads
  "cuSOLVER (torch.linalg.qr)", but `torch.linalg.qr` does not return the compact
  `(H, tau)` form the task requires;
- the QR control labels name cuSOLVER, but `torch.geqrf` uses cuBLAS at the two smallest
  shapes.

The data files keep these labels so that they stay byte-identical to version 1.0.0.

### Cholesky, leaderboard 776

*Scored* counts the organizers' 17 test and 15 benchmark cases passed. *Vendor-identical*
counts search evaluations whose output was the vendor routine's. *Attributable* counts
search failures attributable to the entry, out of 567 valid inputs. *Reproduced* counts
reproduced candidates, out of 5.

| Rank | Entry | Scored | Vendor-identical | Attributable | Reproduced |
|---|---|---|---|---|---|
| 1 | zhongmingee | 32/32 | 0 | 194 | 4 of 5 |
| 2 | Frosty40 | 32/32 | 66 | 129 | 5 of 5 |
| 3 | Olek | 32/32 | 0 | 42 | 5 of 5 |
| 4 | viridale | 32/32 | 134 | 233 | 5 of 5 |
| 5 | Ravi Theja | 32/32 | 0 | 85 | 0 of 5 |
| control | torch.linalg.cholesky_ex (`vendor`) | 32/32 |  | 0 |  |
| control | torch.linalg.cholesky_ex on bfloat16-rounded input (`vendor_degraded`) | not run |  | 437 |  |

Ravi Theja's 85 attributable failures did not reproduce: all 15 reruns of its five
candidates passed. The record states the observation only.

### Eigendecomposition, leaderboard 775

The columns are as for Cholesky, with 39 test and 13 benchmark cases and attributable
failures out of 996 inputs.

| Rank | Entry | Scored | Vendor-identical | Attributable | Reproduced |
|---|---|---|---|---|---|
| 1 | teelaitila | 52/52 | 286 | 11 | 5 of 5 |
| 2 | nikhilbarhate99 | 52/52 | 186 | 0 | 0 of 5 |
| 3 | elianaive | 52/52 | 439 | 0 | 0 of 5 |
| 4 | zhongmingee | 52/52 | 715 | 18 | 5 of 5 |
| 5 | Olek | 52/52 | 293 | 61 | 5 of 5 |
| control | torch.linalg.eigh (`vendor`) | 52/52 |  | 0 |  |
| control | torch.linalg.eigh on float8-rounded input (`vendor_degraded`) | not run |  | 787 |  |

The vendor routine itself failed the checker on 38 of the 996 search inputs, all at the
audit's conditioning parameter `cond` = 12, in the families `dense`, `rowscale` and
`mixed`. On 2 more inputs, `dense` and `rowscale` of order 512 with batch 640 at
`cond` = 12, it raised a convergence error. These inputs count against no entry. The
two entries with no attributable failure, nikhilbarhate99 and elianaive, had candidates
only from inputs on which the vendor routine also failed. Their reruns failed too, with
the vendor routine failing as well, so none of their candidates is reproduced.

## Environment

- **QR.** PyTorch 2.12.1 (CUDA 13.0 build), Triton 3.7.1, Python 3.12 and nvcc 13.3
  (`environment/qr_b200.txt`). During the QR rounds the organizers' runner was
  `src/runners/modal_runner.py` in https://github.com/gpu-mode/kernelbot, at commits
  `642d72a` (2026-06-12) to `d66f6cd` (2026-06-18), the last change before the QR
  deadline of 2026-06-30. It pinned PyTorch 2.12.0, whose dependencies pin Triton 3.7.0,
  on Python 3.13 and the `nvidia/cuda` 12.9.1 base image. Among these pinned versions,
  the QR audit used later patch releases of PyTorch and Triton, an earlier Python minor
  version, and a newer CUDA compiler for the entries that compile CUDA code. Every
  audited entry built and ran, but this record does not show that each count is the same
  in the organizers' environment.
- **Cholesky and eigendecomposition.** A reconstruction of the organizers' runner recipe
  at kernelbot commit `ae732f7` (2026-07-06): PyTorch 2.12.0, Python 3.13, and the base
  image `nvidia/cuda:13.3.0-devel-ubuntu24.04`, with one fresh container per target
  (`environment/cholesky_eigh_b200.txt`). Every raw record carries the hash of the
  environment it ran in. Olek's entry installs a development build of `fbtriton` with
  pip when it is imported, so it ran in environments of its own: `456e86fdce570529` for
  Cholesky and `ebe84c88c1409029` for eigendecomposition. Every other target ran in
  `791d3d70fbc4357b` (`environment/cholesky_eigh_b200_hashes.txt`).

## Files

| Path | Content |
|---|---|
| `qr/targets.csv` | the thirteen QR targets: round, leaderboard, handle, rank, SHA-256 of the entry file as downloaded from the leaderboard |
| `qr/scored.csv` | every scored case with both checker verdicts: 19 rows per target, the 7 round-1 cases (`round_of_case` = `v1`) and the 12 round-2 cases (`v2`), so the 7 dense cases appear twice |
| `qr/sweep.csv` | the search: the 1,050 grid evaluations per target and the failing `randsvd` evaluation at each threshold found, with family, shape, condition knob, seed, both verdicts and residual diagnostics |
| `qr/breaks.csv` | the rows of `sweep.csv` that broke |
| `qr/kappa_bisect.csv` | per entry and shape, the `randsvd` bisection threshold |
| `qr/summary.csv` | one row per target; its `scored_*` columns count the 19 rows of `scored.csv` |
| `qr/randsvd.py` | the one input family that is not the organizers' generator |
| `cholesky/`, `eigh/` | `targets.csv`, `official.csv` (scored cases), `registered.csv` (search), `confirmed.csv` (reruns), `summary.csv` |
| `COLUMNS.md` | meaning of every column, the file conventions, and the raw files |
| `raw/` | the audit's result files, from which `scripts/build_record.py` derives every CSV |
| `raw/manifest.json` | the entries' leaderboard metadata and file hashes |
| `environment/` | software versions, GPU, checker and runner hashes |
| `replay/` | `replay_qr.py`, `fetch_submission.py`, and `requirements.txt` |
| `scripts/build_record.py` | rebuilds every CSV from `raw/` |
| `scripts/check_record.py` | recomputes every table and most counts in this README from the CSV files |
| `CITATION.cff`, `LICENSE`, `LICENSE-CODE` | citation metadata, CC BY 4.0 (data), MIT (code) |

## Checking this record

Run these from the root of a git clone. Both scripts use only the Python standard
library.

```
python3 scripts/build_record.py && git diff --exit-code -- qr cholesky eigh
python3 scripts/check_record.py
```

The first line rebuilds every CSV from `raw/`, and `git diff` then shows no difference
from the committed files. The second recomputes every table in this README, and most
counts in its prose, from the CSV files, and fails on a mismatch.

## Reproducing a QR verdict

Every recorded QR input is determined by its row in `qr/sweep.csv`. For the ten organizer
families (`dense`, `upper`, `diagonal`, `rankdef`, `nearrank`, `clustered`, `band`,
`nearcollinear`, `rowscale`, `mixed`) the input is the organizers' own
`generate_input(batch, n, cond, seed, case)` from `problems/linalg/qr_v2/reference.py` in
`reference-kernels`. For `randsvd` it is `qr/randsvd.py`. The row's `kappa` is the
bisection threshold for that entry and shape, taken from `qr/kappa_bisect.csv`, and
`cond` is empty.

A replay needs Linux, an NVIDIA B200, a CUDA toolkit with nvcc 13 and a C++ compiler,
and the Python packages in `replay/requirements.txt`, at the versions in
`environment/qr_b200.txt`. Two round-2 entries build CUDA extensions when imported, and
a third compiles CUDA code at run time. Round-2 entries come from the organizers'
dataset through `fetch_submission.py`. It reads through the dataset's
`linalg_submissions` file (about 4.7 GB) and keeps the row whose code hashes to the
SHA-256 in `qr/targets.csv`. Run these from the repository root:

```
python3 -m venv .venv && . .venv/bin/activate
git clone https://github.com/gpu-mode/reference-kernels
git -C reference-kernels checkout f3295bb6bd559bae4659d2d8fc45edb1394251b1
pip install -r replay/requirements.txt
python3 replay/fetch_submission.py --leaderboard 774 --user 10billiontokens \
    --expect-sha256 $(awk -F, '$4=="10billiontokens"{print $7}' qr/targets.csv) --out 10billiontokens.py
python3 replay/replay_qr.py --reference-kernels reference-kernels --submission 10billiontokens.py \
    --case rankdef --n 32 --batch 20 --cond 2 --seed 101
python3 replay/replay_qr.py --reference-kernels reference-kernels --submission 10billiontokens.py \
    --case randsvd --n 512 --batch 640 --kappa 794.3258949419277 --seed 7
```

Each replay prints the verdict under both checkers, to compare with the row's `pass_v1`
and `pass_v2`. The first example failed both checkers in the audit ("H contains NaN or
Inf"). The second passed the round-1 checker and failed the round-2 checker. It sits just
above its bisection threshold (its `f`, the round-2 residual ratio defined in
`COLUMNS.md`, is 1.04 against the passing limit of 1), so a small difference in
environment can flip its verdict. For some entries `ran_path` varies with shape, which
suggests a shape-dependent choice of code path, and other hardware may take a different
path.

Round-1 entries (leaderboard 773) are not in the organizers' dataset, and the leaderboard
API (https://www.gpumode.com/api/leaderboard/773) returns rankings without code. Their
verdicts can be read in the tables, but they cannot be replayed from public sources
unless the organizers release the round-1 code. The SHA-256 in `qr/targets.csv`
identifies the audited file, and `replay_qr.py --expect-sha256` confirms a file against
it.

## Cholesky and eigendecomposition inputs

The search inputs for these two problems came from the audit's own generators, which
are not part of this record. Each row of `registered.csv` and `confirmed.csv` names the
family, the difficulty parameters, the shape, the seed, a case id, the validation
evidence for the input, and the checker's diagnostics. A verdict can therefore be
inspected and a family re-targeted, but the inputs cannot be regenerated from this record
alone. The scored cases in `official.csv` are the organizers' and are reproducible with
their code.

## Attribution and license

The competitions, the checkers, the scored cases, and the entry dataset are the work of
GPU MODE (https://www.gpumode.com) and its contestants. Entries are named by the handles
the organizers publish on their leaderboards.

The organizers release entry code in the Hugging Face dataset
https://huggingface.co/datasets/GPUMODE/kernelbot-data (configuration
`linalg_submissions`, leaderboards 774, 775 and 776). It is under the June 9 Researcher
Reciprocity License adapted from Open RAIL-D
([`LICENSE`](https://huggingface.co/datasets/GPUMODE/kernelbot-data/blob/main/LICENSE)).
The dataset card asks users to cite GPU MODE and link to the dataset, and asks academic
papers to cite KernelBot (see Citing this record). This record does so and copies no
entry code.

The checkers, generators and task files are in https://github.com/gpu-mode/reference-kernels.
They are under the June 9 Researcher Reciprocity License adapted from Open RAIL-S
([`LICENSE` at the audited commit](https://github.com/gpu-mode/reference-kernels/blob/f3295bb6bd559bae4659d2d8fc45edb1394251b1/LICENSE)).
The replay script loads that code at run time and copies none of it.

Some of the data comes from GPU MODE and stays under GPU MODE's terms, wherever it appears
in this record, including `raw/`:
- the specifications of the scored cases (shape, batch, conditioning knob, seed), from
  the `reference-kernels` task files, under that repository's license;
- the leaderboard metadata (ranks, and for Cholesky and eigendecomposition also scores,
  submission ids and times), from GPU MODE's leaderboards.

The rest of the data, the audit's own results, is copyright 2026 Maksim Melnichenko and
released under CC BY 4.0 (`LICENSE`). The code in `scripts/`, `replay/` and
`qr/randsvd.py` is released under the MIT License (`LICENSE-CODE`).

## Citing this record

Cite the version you used (see `CITATION.cff`):

```bibtex
@misc{melnichenko2026auditrecord,
  author       = {Melnichenko, Maksim},
  title        = {{GPU MODE} linear-algebra kernel audit, 2026: results record},
  year         = {2026},
  howpublished = {\url{https://github.com/VSCL-x-VERITAS/gpumode-kernel-audit-2026}},
  note         = {Version 1.0.1, git tag v1.0.1}
}
```

When you use the GPU MODE material, also link the dataset and cite KernelBot, as the
dataset card requests:

```bibtex
@inproceedings{kernelbot2025,
  title     = {{KernelBot}: A Competition Platform for Writing Heterogeneous {GPU} Code},
  author    = {Alex L Zhang and Matej Sirovatka and Erik Schultheis and Benjamin Horowitz and Mark Saroufim},
  booktitle = {Championing Open-source DEvelopment in ML Workshop @ ICML25},
  year      = {2025},
  url       = {https://openreview.net/forum?id=bq9U4dmuyJ}
}
```

To report an error in the record, open an issue on this repository.

## Changes

- **1.0.1 (2026-11-10).** Corrections and additions to the text and the scripts. The data
  files are unchanged from 1.0.0.
  - README: the QR scored cases are given per round, instead of pooled as 19 per entry,
    and for QR they are stated to be the benchmark lists only. New definitions cover
    scored cases, legal inputs, the search and the vendor routine. Each QR entry now has
    its searched and vendor-identical evaluations and its breaks under the round-1
    checker. Further additions: the environment differences from the competition, the
    two inaccurate control labels, the GPU MODE terms for the case specifications and
    leaderboard metadata, and the KernelBot citation. The paper is now named by its
    working title.
  - COLUMNS.md: `ran_path` is defined by bit-identical outputs. New entries cover the
    empty cells, the repeated scored cases, the CSV format and the `raw/` files. The
    residual definitions now state that Q is materialized in float32.
  - CITATION.cff validates against the Citation File Format 1.2.0 schema. It no longer
    gives a preferred citation for the paper and lists the GPU MODE works it draws on.
  - `scripts/check_record.py` checks every table and most counts in this README.
    `replay/replay_qr.py` now requires `--cond` for the organizer families, where it
    silently used 0. `replay/fetch_submission.py` reads a fixed revision of the
    dataset, and `replay/requirements.txt` adds ninja. A new `.gitattributes` keeps the
    data files' line endings.
- **1.0.0 (2026-11-04, commit `595cbc8`).** The first publication of the record.
