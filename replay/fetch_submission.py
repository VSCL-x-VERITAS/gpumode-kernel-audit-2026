#!/usr/bin/env python3
"""Fetch one competition entry from the organizers' public dataset and verify its hash.

The GPU MODE organizers publish the code of every submission to leaderboards
774 (qr_v2), 775 (eigh) and 776 (cholesky) in the Hugging Face dataset
GPUMODE/kernelbot-data, configuration linalg_submissions, under their
"June 9 Researcher Reciprocity License". Round 1 of QR (leaderboard 773) is
not in that dataset, and the leaderboard API returns no code, so this script
cannot fetch round-1 entries.

Usage:
  python fetch_submission.py --leaderboard 774 --user 10billiontokens \
      --expect-sha256 <value from qr/targets.csv> --out 10billiontokens.py

Streams the dataset instead of downloading it whole, keeps the rows for the user on
that leaderboard, and writes the one whose code hashes to the expected value.
Requires: pip install datasets
"""
import argparse, hashlib, sys


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--leaderboard", type=int, required=True)
    ap.add_argument("--user", required=True, help="user_name as shown on the leaderboard")
    ap.add_argument("--expect-sha256", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from datasets import load_dataset
    ds = load_dataset("GPUMODE/kernelbot-data", "linalg_submissions", split="train", streaming=True)
    seen = 0
    for row in ds:
        if row["leaderboard_id"] != a.leaderboard or row["user_name"] != a.user:
            continue
        seen += 1
        code = row["code"]
        variants = [code, code.rstrip("\n"), code.rstrip("\n") + "\n", code.replace("\r\n", "\n")]
        match = next((v for v in variants if hashlib.sha256(v.encode("utf-8")).hexdigest() == a.expect_sha256), None)
        if match is not None:
            with open(a.out, "w", encoding="utf-8", newline="") as f:
                f.write(match)  # the variant whose hash matched, so the file re-hashes to the expected value
            print(f"wrote {a.out} (submission_id {row['submission_id']}, score {row['score']})")
            return
    sys.exit(f"no row with the expected hash; {seen} rows for user {a.user} on leaderboard {a.leaderboard}")


if __name__ == "__main__":
    main()
