#!/usr/bin/env python3
"""Replay one QR audit evaluation with the organizers' generator and checker.

Example, from the repository root (round 2 kernel, organizer family):
  python3 replay/replay_qr.py --reference-kernels reference-kernels \
      --submission 10billiontokens.py --case rankdef --n 32 --batch 20 --cond 2 --seed 101

Example (randsvd family, the condition number and seed from the row of qr/sweep.csv):
  python3 replay/replay_qr.py --reference-kernels ... --submission ... --case randsvd --n 512 --batch 640 \
      --kappa 794.3258949419277 --seed 7

What it does: generates the input exactly as the audit did (the organizers'
qr_v2 generate_input for every family except randsvd, which uses qr/randsvd.py),
runs the submission's custom_kernel once, and applies both organizer checkers:
round 1 (problems/linalg/qr_py) and round 2 (problems/linalg/qr_v2). It prints
the two verdicts and messages. The expected verdicts for every evaluation of
the audit are in qr/sweep.csv (columns pass_v1, pass_v2, reason).

Requires a GPU, PyTorch and Triton versions as in environment/qr_b200.txt, and a
clone of https://github.com/gpu-mode/reference-kernels at the commit named in
environment/checkers.md. Obtain the submission file as described in the README;
this script never downloads anything.
"""
import argparse, hashlib, importlib.util, os, sys


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_problem(rk_dir, problem):
    pdir = os.path.join(rk_dir, "problems", "linalg", problem)
    if not os.path.isdir(pdir):
        sys.exit(f"missing {pdir}")
    sys.path.insert(0, pdir)
    for m in ("task", "reference"):
        sys.modules.pop(m, None)
    ref = load_module(os.path.join(pdir, "reference.py"), f"reference_{problem}")
    sys.path.pop(0)
    return ref


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference-kernels", required=True, help="path to a clone of gpu-mode/reference-kernels")
    ap.add_argument("--submission", required=True, help="path to the submission .py (must define custom_kernel)")
    ap.add_argument("--case", required=True, help="dense, upper, diagonal, rankdef, nearrank, clustered, band, nearcollinear, rowscale, mixed, or randsvd")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--cond", type=int, default=None, help="organizer families (required): the cond knob of generate_input")
    ap.add_argument("--kappa", type=float, default=None, help="randsvd only: target 2-norm condition number")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--expect-sha256", default=None, help="optional sha256 of the submission file, from qr/targets.csv")
    a = ap.parse_args()

    if a.expect_sha256:
        got = hashlib.sha256(open(a.submission, "rb").read()).hexdigest()
        if got != a.expect_sha256:
            sys.exit(f"submission sha256 {got} differs from expected {a.expect_sha256}")

    import torch
    ref_v1 = load_problem(a.reference_kernels, "qr_py")
    ref_v2 = load_problem(a.reference_kernels, "qr_v2")

    if a.case == "randsvd":
        if a.kappa is None:
            sys.exit("--kappa is required for randsvd")
        here = os.path.dirname(os.path.abspath(__file__))
        rs = load_module(os.path.join(here, "..", "qr", "randsvd.py"), "randsvd")
        data = rs.randsvd(a.batch, a.n, a.kappa, a.seed, device="cuda")
    else:
        if a.cond is None:
            sys.exit("--cond is required for the organizer families")
        data = ref_v2.generate_input(a.batch, a.n, a.cond, a.seed, a.case)

    sub = load_module(a.submission, "submission_under_test")
    if not hasattr(sub, "custom_kernel"):
        sys.exit("submission has no custom_kernel")
    print(f"case={a.case} n={a.n} batch={a.batch} cond={a.cond} kappa={a.kappa} seed={a.seed}")
    try:
        out = sub.custom_kernel(data.clone())  # the kernel gets a copy; the checkers see the original input
        torch.cuda.synchronize()
    except Exception as e:  # a raising entry is a failure, not a crash of the replay
        print(f"round-1 checker: FAIL kernel raised {type(e).__name__}: {e}")
        print(f"round-2 checker: FAIL kernel raised {type(e).__name__}: {e}")
        return
    ok1, msg1 = ref_v1.check_implementation(data, out)
    ok2, msg2 = ref_v2.check_implementation(data, out)
    print(f"round-1 checker: {'PASS' if ok1 else 'FAIL'} {msg1}")
    print(f"round-2 checker: {'PASS' if ok2 else 'FAIL'} {msg2}")


if __name__ == "__main__":
    main()
