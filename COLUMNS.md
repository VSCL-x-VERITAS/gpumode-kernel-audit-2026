# Columns

Booleans are `True` and `False`. `inf` marks a non-finite diagnostic. An empty cell means
not applicable. `target` is the raw result label and `handle` the leaderboard name (the two
differ for `nernst` = `gau.nernst` and `Ravi_Theja` = `Ravi Theja`).

## QR: `qr/scored.csv`, `qr/sweep.csv`, `qr/breaks.csv`

| Column | Meaning |
|---|---|
| `target` | raw label of the entry or control |
| `round_of_case` | `scored.csv` only: `v1` or `v2`, the round whose scored case this is |
| `case` | input family: one of the organizers' families (`dense`, `upper`, `diagonal`, `rankdef`, `nearrank`, `clustered`, `band`, `nearcollinear`, `rowscale`, `mixed`) or `randsvd` |
| `n`, `batch` | matrix order and number of matrices in the batch |
| `scored_batch` | the batch size the competition scored for this `n` |
| `cond` | the organizers' integer conditioning knob (column scaling by `logspace(0, -cond, n)`). Empty for `randsvd` |
| `kappa` | `randsvd` rows only: the target 2-norm condition number, which is the bisection threshold for this entry and shape in `kappa_bisect.csv` |
| `seed` | generator seed |
| `pass_v1`, `pass_v2` | verdict of the round-1 and round-2 organizer checkers on the same output |
| `broke` | the QR break rule stated in the README under How to read a verdict |
| `status` | `OK` when the entry returned an output, otherwise the kind of failure |
| `ran_path` | `kernel` when the entry ran its own code, `fallback_geqrf` when it dispatched to `torch.geqrf` |
| `reason` | the checker's message for a failing output |
| `note` | the audit's note, for example `Q or R non-finite` |
| `f`, `factor_scaled`, `orth_scaled`, `recon_scaled`, `tri_scaled` | the audit's recomputation of the checker's residuals, defined below |

The residual columns use Q materialized from the entry's `(H, tau)` by
`torch.linalg.householder_product` in float64, R = `triu(H)`, the matrix 1-norm, the
float32 unit roundoff eps = 2^-23, and maxima over the batch:

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
| `ran_path`, `note` | as above |

## QR: `qr/targets.csv` and `qr/summary.csv`

`target`, `round` (`qr_v1`, `qr_v2`, or `control`), `leaderboard_id`, `handle`, `rank`, `kind`
(`competition` or `control`), `submission_sha256` (SHA-256 of the entry file as downloaded
from the leaderboard), `gpu`, `torch`. The summary adds `scored_cases`,
`scored_pass_v1`, `scored_pass_v2`, `sweep_evals`, `sweep_breaks` (rows with `broke` true),
`families_broken`, `families` (the families with at least one break), `min_break_kappa`
(the smallest threshold over shapes).

## Cholesky and eigendecomposition: `official.csv`, `registered.csv`, `confirmed.csv`

| Column | Meaning |
|---|---|
| `target`, `problem` | entry or control, and `cholesky` or `eigh` |
| `stage` | `official` (the organizers' scored cases), `registered` (the discovery sweep), `confirmed` (fresh-container reruns) |
| `tier` | `official_test` or `official_benchmark` (the two kinds of organizer case), `discovery`, `confirmation` |
| `case_id` | identifier of the input case |
| `case_source` | origin of the case: `cholesky_task_tests`, `cholesky_task_benchmarks`, `eigh_task_tests`, `eigh_task_benchmarks` (the organizers' cases), `registered` (the discovery sweep), `confirm` (rerun of the discovery input), `confirm_heldout` (rerun with a fresh seed of the same family and difficulty) |
| `confirm_candidate` | `confirmed.csv` only: the discovery case being rerun |
| `repeat_id`, `cold_or_warm` | rerun index, and whether the call was the first in a fresh process (`cold`) or a later call in the same process (`warm`). Each candidate has three reruns: `confirm` cold, `confirm_heldout` cold, `confirm` warm |
| `family`, `difficulty` | input family and its parameters as JSON, for example `{"cond": 2}` |
| `n`, `batch`, `seed` | shape and generator seed |
| `input_valid`, `input_validity_evidence` | whether the generated input passed validation, and the evidence (for Cholesky, positive definiteness in float32) |
| `vendor_pass` | the vendor routine's verdict on the same input. Empty in `official.csv` |
| `official_pass`, `official_message` | the organizers' checker verdict and its residual summary |
| `ran_path` | `kernel`, `fallback_vendor` (the entry dispatched to the vendor routine), or `unknown` when the path could not be determined |
| `error`, `elapsed_s` | error text if the call failed, and wall time in seconds |
| `diagnostics` | the checker's scaled residuals as JSON. Eigendecomposition: `ascending_ok`, `diag_scaled`, `eigen_scaled`, `orth_scaled`, `recon_scaled`, `note`. Cholesky: `min_diagonal`, `passed_recompute`, `recon_scaled`, `triangular_scaled`, `note` |
| `env_hash`, `gpu_model`, `checker_sha256`, `submission_sha256`, `protocol_hash` | hashes identifying the environment, the GPU model, the checker file, the entry file, and the campaign protocol |

## Cholesky and eigendecomposition: `targets.csv` and `summary.csv`

`targets.csv`: `target`, `problem`, `leaderboard_id`, `handle`, `rank`, `kind`, `submission_id`,
`submission_time`, `score` (the leaderboard score, seconds), `submission_sha256`, `lines`.

`summary.csv`: `target`, `problem`, `scored_cases`, `scored_pass`, `discovery_evals`, `discovery_valid_inputs`,
`discovery_fail_attributable` (`official_pass` false, `vendor_pass` true, `input_valid` not
false), `discovery_fail_with_vendor` (both the entry and the vendor failed),
`confirmation_candidates`, `confirmation_runs`, `candidates_reproduced_in_all_runs`
(candidates whose every rerun failed with the vendor passing), `families_attributable`.
