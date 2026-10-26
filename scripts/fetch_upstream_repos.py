#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/fetch_upstream_repos.py

Utility script to verify and fetch upstream source code repositories
cataloged in dataset/manifest_repos.csv at their pinned commit SHAs.
Governed by Item C12 (Legal, Licensing & Open Science Provenance).
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
MANIFEST_FILE = ROOT_DIR / "dataset" / "manifest_repos.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fetch or verify upstream source repositories at pinned commit SHAs."
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=ROOT_DIR / "ai-engine" / "temp_repos",
        help="Directory to clone upstream repositories into."
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Check presence and pinned commit SHAs without cloning."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not MANIFEST_FILE.exists():
        print(f"Error: Manifest file missing at {MANIFEST_FILE}", file=sys.stderr)
        sys.exit(1)

    with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        repos = list(reader)

    print(f"Loaded {len(repos)} repositories from {MANIFEST_FILE.name}")
    args.target_dir.mkdir(parents=True, exist_ok=True)

    status_counts = {"present_and_pinned": 0, "cloned": 0, "missing": 0, "errors": 0}

    for idx, r in enumerate(repos, 1):
        repo_name = r["repo_name"]
        url = r["url"]
        commit_sha = r["commit_sha"]
        safe_name = repo_name.replace("/", "__")
        dest = args.target_dir / safe_name

        print(f"[{idx:02d}/{len(repos):02d}] {repo_name} @ {commit_sha[:8]} ... ", end="")

        if dest.exists() and (dest / ".git").exists():
            # Check commit SHA
            res = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=dest,
                capture_output=True,
                text=True
            )
            head_sha = res.stdout.strip()
            if head_sha.startswith(commit_sha) or commit_sha.startswith(head_sha):
                print("MATCH (pinned commit verified)")
                status_counts["present_and_pinned"] += 1
            else:
                print(f"MISMATCH (HEAD: {head_sha[:8]} vs pinned: {commit_sha[:8]})")
                if not args.check_only:
                    subprocess.run(["git", "checkout", commit_sha], cwd=dest, capture_output=True)
                    status_counts["present_and_pinned"] += 1
        elif args.check_only:
            print("NOT PRESENT (check-only mode)")
            status_counts["missing"] += 1
        else:
            print("CLONING ... ", end="", flush=True)
            clone_res = subprocess.run(
                ["git", "clone", url, str(dest)],
                capture_output=True,
                text=True
            )
            if clone_res.returncode == 0:
                subprocess.run(["git", "checkout", commit_sha], cwd=dest, capture_output=True)
                print("DONE")
                status_counts["cloned"] += 1
            else:
                print(f"FAILED ({clone_res.stderr.strip()[:60]})")
                status_counts["errors"] += 1

    print("\nSummary:")
    for k, v in status_counts.items():
        print(f"  • {k}: {v}")


if __name__ == "__main__":
    main()
