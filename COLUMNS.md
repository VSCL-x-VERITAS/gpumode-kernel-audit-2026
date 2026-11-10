# Columns

The CSV files are comma-separated with a header row and CRLF line endings (RFC 4180, as
Python's `csv` module writes them). Booleans are `True` and `False`. `inf` marks a non-finite
diagnostic in the QR files, and the Cholesky and eigendecomposition `diagnostics` JSON writes
`Infinity`. An empty cell means not applicable or not evaluated. In the Cholesky and
eigendecomposition files an empty `official_pass` or `vendor_pass` has one of three causes:
the input was invalid and was not run (both cells empty), the call raised an error (the
entry's error is in `error`, the vendor routine's in `vendor_error` in `raw/`), or the
entry raised, so no vendor comparison was made. `target` is the raw result label and `handle` the leaderboard name
(the two differ for `nernst` = `gau.nernst` and `Ravi_Theja` = `Ravi Theja`).

## QR: `qr/scored.csv`, `qr/sweep.csv`, `qr/breaks.csv`

| Column | Meaning |
|---|---|
| `target` | raw label of the entry or control |
| `round_of_case` | `scored.csv` only: `v1` or `v2`, the round whose list of scored cases this row comes from. The 7 `v1` cases are also among the 12 `v2` cases, so each target has 19 rows for 12 distinct cases. The two rows of a repeated case are separate runs: they carry the same verdicts, and their residual columns can differ slightly, since some entries' outputs vary from call to call |
| `case` | input family: one of the organizers' families (`dense`, `upper`, `diagonal`, `rankdef`, `nearrank`, `clustered`, `band`, `nearcollinear`, `rowscale`, `mixed`) or `randsvd` |
| `n`, `batch` | matrix order and number of matrices in the batch |
| `scored_batch` | `sweep.csv` and `breaks.csv` only: the batch size the competition scored for this `n`, equal to `batch` in every row. Empty in `scored.csv`, whose rows are the scored cases themselves |
| `cond` | the organizers' integer conditioning knob (column scaling by `logspace(0, -cond, n)`). Empty for `randsvd` |
| `kappa` | `randsvd` rows only: the target 2-norm condition number, which is the bisection threshold for this entry and shape in `kappa_bisect.csv` |
| `seed` | generator seed |
| `pass_v1`, `pass_v2` | verdict of the round-1 and round-2 organizer checkers on the same output |
| `broke` | `True` exactly when `status` is `OK`, `ran_path` is `kernel` and `pass_v2` is `False` (the QR break rule) |
| `status` | `OK` when the entry returned an output, otherwise the kind of failure |
| `ran_path` | `fallback_geqrf` when the output `(H, tau)` is bit-identical to `torch.geqrf` on the same input (evidence that the entry handed the input to the vendor routine), otherwise `kernel`. Both `torch.geqrf` controls, `geqrf` and `cusolver`, are `fallback_geqrf` on every row |
| `reason` | the round-2 checker's message for an output it rejects |
| `note` | the audit's note, for example `Q or R non-finite` |
| `f`, `factor_scaled`, `orth_scaled`, `recon_scaled`, `tri_scaled` | the audit's recomputation of the checker's residuals, defined below |

The residual columns use Q materialized from the entry's float32 `(H, tau)` by
`torch.linalg.householder_product` in float32 and then cast to float64, as the checkers do,
R = `triu(H)` in float64, the matrix 1-norm, the
float32 machine epsilon eps = 2^-23 (`torch.finfo(torch.float32).eps`, the checkers' `eps32`),
and maxima over the batch:

- `factor_scaled` = max over i of ‖R_i − Q_iᵀ A_i‖₁ / (n · eps · ‖A_i‖₁)
- `orth_scaled` = max over i of ‖Q_iᵀ Q_i − I‖₁ / (n · eps)
- `tri_scaled` = max over i of ‖tril(Q_iᵀ A_i, −1)‖₁ / (n · eps · max over j of ‖A_j‖₁)
- `recon_scaled` = max over i of ‖Q_i R_i − A_i‖₁ / (n · eps · max over j of ‖A_j‖₁)
- `f` = max(`factor_scaled` / 20, `orth_scaled` / 100), the two tolerance factors of the round-2 checker, so `f` ≤ 1 is the audit's recomputed pass

All five are `inf` when the output is non-finite, has the wrong shape or type, or when
`householder_product` fails. The authoritative verdicts are `pass_v1` and `pass_v2`.

## QR: `qr/kappa_bisect.csv`

| Column | Meaning |
|---|---|
| `target`, `n`, `batch` | entry and shape |
| `min_break_kappa` | smallest `randsvd` condition number at which the entry broke, searched up to 10^12. Empty when it did not break |
| `realized_kappa` | condition number of the float32 matrix actually generated at that threshold |
| `f_at_break` | the `f` value of the breaking evaluation (equal to `f` in the matching `sweep.csv` row) |
| `ran_path`, `note` | as above, except that `note` is empty here even where the matching `sweep.csv` row has one |

## QR: `qr/targets.csv` and `qr/summary.csv`

`target`, `round` (`qr_v1`, `qr_v2`, or `control`), `leaderboard_id`, `handle`, `rank`, `kind`
(`competition` or `control`), `submission_sha256` (SHA-256 of the entry file as downloaded
from the leaderboard; empty for the controls), `gpu`, `torch`. The `handle` of the `cusolver` control reads
`cuSOLVER (torch.linalg.qr)`. That label is wrong: the control calls `torch.geqrf` (see the
README). The summary has `target`, `round`, `leaderboard_id`, `handle`, `rank` and `kind`, then `scored_cases`,
`scored_pass_v1`, `scored_pass_v2` (counts over the 19 rows of `scored.csv`, which hold
12 distinct cases; the README gives the per-round counts), `sweep_evals`, `sweep_breaks` (rows with `broke` true),
`families_broken`, `families` (the families with at least one break), `min_break_kappa`
(the smallest threshold over shapes).

## Cholesky and eigendecomposition: `official.csv`, `registered.csv`, `confirmed.csv`

| Column | Meaning |
|---|---|
| `target`, `problem` | entry or control, and `cholesky` or `eigh` |
| `stage` | `official` (the organizers' scored cases), `registered` (the discovery sweep), `confirmed` (reruns) |
| `tier` | `official_test` or `official_benchmark` (the two kinds of organizer case), `discovery`, `confirmation` |
| `case_id` | identifier of the input case |
| `case_source` | origin of the case: `cholesky_task_tests`, `cholesky_task_benchmarks`, `eigh_task_tests`, `eigh_task_benchmarks` (the organizers' cases), `registered` (the discovery sweep), `confirm` (rerun of the discovery input), `confirm_heldout` (rerun with a fresh seed of the same family and difficulty) |
| `confirm_candidate` | `confirmed.csv` only: the discovery case being rerun |
| `repeat_id`, `cold_or_warm` | rerun index, and whether the call was the first in a fresh process (`cold`) or a later call in the same process (`warm`). Each candidate has three reruns: `confirm` cold, `confirm_heldout` cold, `confirm` warm. In `official.csv` and `registered.csv` every row is `cold` with `repeat_id` 0 |
| `family`, `difficulty` | input family and its parameters as JSON, for example `{"cond": 2}` |
| `n`, `batch`, `seed` | shape and generator seed |
| `input_valid`, `input_validity_evidence` | whether the generated input passed validation, and the evidence (for Cholesky, that `torch.linalg.cholesky_ex` factored the matrix in float32) |
| `vendor_pass` | the vendor routine's verdict on the same input. Empty in `official.csv` |
| `official_pass`, `official_message` | the organizers' checker verdict and its residual summary |
| `ran_path` | `fallback_vendor` when the entry's output is bit-identical to the vendor routine's on the same input (evidence that the entry handed the input to the vendor routine), `kernel` when it differs, `unknown` when no comparison was made: the scored cases, the controls, invalid inputs, and evaluations with no vendor verdict |
| `error`, `elapsed_s` | error text if the call failed, and wall time in seconds |
| `diagnostics` | the checker's scaled residuals as JSON. Eigendecomposition: `ascending_ok`, `diag_scaled`, `eigen_scaled`, `orth_scaled`, `recon_scaled`, `note`. Cholesky: `min_diagonal`, `passed_recompute`, `recon_scaled`, `triangular_scaled`, `note` |
| `env_hash`, `gpu_model`, `checker_sha256`, `submission_sha256`, `protocol_hash` | hashes identifying the environment, the GPU model, the checker file, the entry file (for the controls, `control:vendor` or `control:vendor_degraded`), and the campaign protocol |

## Cholesky and eigendecomposition: `targets.csv` and `summary.csv`

`targets.csv`: `target`, `problem`, `leaderboard_id`, `handle`, `rank`, `kind`, `submission_id`,
`submission_time`, `score` (the leaderboard score, seconds), `submission_sha256`, `lines` (the line count of the entry file).

`summary.csv`: `target`, `problem`, `scored_cases`, `scored_pass`, `discovery_evals`, `discovery_valid_inputs`,
`discovery_fail_attributable` (`official_pass` false, `vendor_pass` true, `input_valid` not
false), `discovery_fail_with_vendor` (both the entry and the vendor failed),
`confirmation_candidates`, `confirmation_runs`, `candidates_reproduced_in_all_runs`
(candidates whose every rerun failed with the vendor passing), `families_attributable`.

## `raw/`

These are the audit's result files, with GPU serial numbers, process ids and local paths
removed. `scripts/build_record.py` derives every CSV table from them.

- `raw/qr_b200/<target>.json` holds one object per QR target. `build_record.py` reads
  `label`, `gpu`, `torch`, `scored`, `sweep` and `kappa_bisect`. The `scored` and `sweep`
  entries carry the QR columns above, with `round` for `round_of_case` and a wall time
  `dt` that enters no table. The keys `target`, `grid`, `device`, `viability` (one
  preliminary evaluation) and `breaks` (the failing scored and searched evaluations)
  record the run and enter no table. `raw/qr_b200/ENVIRONMENT.txt` is the same file as
  `environment/qr_b200.txt`.
- `raw/cholesky_b200/` and `raw/eigh_b200/` hold JSONL, one object per evaluation.
  `<target>.jsonl` holds the scored cases, `<target>_registered.jsonl` the search and
  `<target>_confirmed.jsonl` the reruns. The fields are the CSV columns above, except that
  `build_record.py` derives `stage` and takes `target` from the file name. Six more fields
  enter no table: `vendor_error`, `runtime_class`, `call_sequence_id`, `presentation`,
  `checker_name` and `target_kind`.
- `raw/manifest.json` holds the leaderboard metadata and the SHA-256 of every entry file.
