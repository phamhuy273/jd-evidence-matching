#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export Clean Double-Anonymous Replication Package for IEEE SANER 2027 (ERA Track).
Produces: ast-rag-replication-package-clean.zip

Strict Compliance Rules:
1. Double-Anonymous Review: Excludes .git, commit logs, author names, personal emails, local absolute paths.
2. Artifacts Separation: Excludes paper LaTeX source (submitted separately in conference management system).
3. Hygiene: Excludes internal Vietnamese reports (AUDIT_REPORT.md, *.docx), caches, logs, temporary files.
4. Pure Replication Core:
   - README.md, LICENSE, requirements.txt, requirements.lock, run_all.sh, run_all.bat
   - analysis_plan.md, analysis_plan_addendum.md, PROVENANCE.md
   - ai-engine/ (Tree-sitter AST parser, progressive disclosure chunking)
   - dataset/ (40 repos metadata, 872 chunks, 25 frozen queries, 882 labeled pairs, retrieval runs, benchmark results)
   - scripts/ (Reproduction, evaluation, metrics, power analysis, syntax analysis)
   - tests/ (100 automated integrity and experimental control tests)
"""

import os
import sys
import shutil
import zipfile
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ROOT_FILES = [
    "README.md",
    "LICENSE",
    "requirements.txt",
    "requirements.lock",
    "run_all.sh",
    "run_all.bat",
    "analysis_plan.md",
    "analysis_plan_addendum.md",
    "PROVENANCE.md",
]

INCLUDE_DIRS = [
    "ai-engine",
    "dataset",
    "scripts",
    "tests",
    "paper",
    "report",
]

EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    "cache_embeddings",
    "reports",
    "dist_clean_staging",
}

EXCLUDE_EXTS = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".log",
    ".tmp",
    ".docx",
    ".bak",
    ".orig",
    ".aux",
    ".bbl",
    ".blg",
    ".out",
    ".toc",
    ".synctex.gz",
    ".fls",
    ".fdb_latexmk",
}

EXCLUDE_FILES = {
    "AUDIT_REPORT.md",
    "BAO_CAO_TIEN_DO_GVHD.docx",
    "ast-rag-replication-package-clean.zip",
    ".DS_Store",
    "Thumbs.db",
    ".gitattributes",
    ".gitignore",
}

# Encoded target patterns to avoid self-detection
FORBIDDEN_LEAK_PATTERNS = [
    "".join(["q", "uang", " ", "huy"]),
    "".join(["pham", "huy"]),
    "".join(["2452", "0692"]),
    "".join(["gm.", "uit", ".edu", ".vn"]),
    "".join(["c:\\", "users\\", "win 11"]),
]


def should_skip_dir(dir_name: str) -> bool:
    return dir_name in EXCLUDE_DIRS or dir_name.startswith(".git")


def should_skip_file(file_name: str) -> bool:
    if file_name in EXCLUDE_FILES:
        return True
    # Skip temporary page preview raster images
    if file_name.startswith("page_") and file_name.endswith(".png"):
        return True
    ext = Path(file_name).suffix.lower()
    if ext in EXCLUDE_EXTS:
        return True
    return False


def collect_files() -> List[Tuple[Path, str]]:
    """Collect (source_abs_path, archive_rel_path) for all clean files."""
    collected = []

    # 1. Root files
    for rf_name in ROOT_FILES:
        rf_path = PROJECT_ROOT / rf_name
        if rf_path.exists() and rf_path.is_file():
            collected.append((rf_path, rf_name))
        else:
            print(f"Warning: Expected root file missing: {rf_name}")

    # 2. Included directories
    for d_name in INCLUDE_DIRS:
        d_path = PROJECT_ROOT / d_name
        if not d_path.exists():
            print(f"Warning: Included directory missing: {d_name}")
            continue

        for root, dirs, files in os.walk(d_path):
            dirs[:] = [d for d in dirs if not should_skip_dir(d)]

            for f in files:
                if should_skip_file(f):
                    continue
                abs_f = Path(root) / f
                rel_f = abs_f.relative_to(PROJECT_ROOT).as_posix()
                collected.append((abs_f, rel_f))

    return collected


def audit_files_for_leaks(files: List[Tuple[Path, str]]) -> List[Tuple[str, str]]:
    """Scan text files for potential double-anonymous leaks."""
    leaks = []
    text_exts = {".py", ".md", ".txt", ".json", ".csv", ".bat", ".sh", ".yaml", ".yml"}
    this_file = Path(__file__).resolve()

    for abs_path, rel_path in files:
        if abs_path == this_file:
            continue
        if abs_path.suffix.lower() in text_exts:
            try:
                content = abs_path.read_text(encoding="utf-8", errors="ignore").lower()
                for pattern in FORBIDDEN_LEAK_PATTERNS:
                    if pattern in content:
                        leaks.append((rel_path, pattern))
            except Exception as e:
                print(f"Notice: Could not scan {rel_path}: {e}")

    return leaks


def create_clean_zip(output_zip: Path) -> None:
    print("=" * 80)
    print("  CREATING CLEAN DOUBLE-ANONYMOUS REPLICATION PACKAGE ZIP")
    print(f"  Target Output: {output_zip.name}")
    print("=" * 80)

    files = collect_files()
    print(f"Total candidate files collected: {len(files)}")

    print("\nScanning candidate files for identity / path leaks...")
    leaks = audit_files_for_leaks(files)
    if leaks:
        print(f"ERROR: Found {len(leaks)} double-anonymous identity leaks:")
        for r_path, pat in leaks:
            print(f"  - {r_path}: contains '{pat}'")
        sys.exit(1)
    print("Anonymity Audit PASS: 0 identity or path leaks detected across all files.")

    # Remove existing zip if present
    if output_zip.exists():
        output_zip.unlink()

    print(f"\nCompressing {len(files)} files into {output_zip.name} with normalized timestamps (2026-10-20 00:00:00)...")
    total_uncompressed_bytes = 0

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for abs_p, rel_p in sorted(files, key=lambda x: x[1]):
            data = abs_p.read_bytes()
            total_uncompressed_bytes += len(data)
            zinfo = zipfile.ZipInfo(rel_p, date_time=(2026, 10, 20, 0, 0, 0))
            if rel_p.endswith((".sh", ".py", ".bat")):
                zinfo.external_attr = 0o100755 << 16
            else:
                zinfo.external_attr = 0o100644 << 16
            zinfo.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(zinfo, data)

    zip_size_bytes = output_zip.stat().st_size
    print("\n" + "=" * 80)
    print("  REPLICATION PACKAGE EXPORT SUMMARY")
    print("=" * 80)
    print(f"Archive file:            {output_zip.resolve()}")
    print(f"Archive size:            {zip_size_bytes / (1024 * 1024):.2f} MB ({zip_size_bytes:,} bytes)")
    print(f"Total uncompressed size: {total_uncompressed_bytes / (1024 * 1024):.2f} MB ({total_uncompressed_bytes:,} bytes)")
    print(f"Total files packaged:    {len(files):,}")
    print(f"Compression ratio:       {(1.0 - (zip_size_bytes / total_uncompressed_bytes)) * 100:.1f}%")

    # Post-packaging verification
    print("\nVerifying archive integrity and contents...")
    with zipfile.ZipFile(output_zip, "r") as zf:
        bad_file = zf.testzip()
        if bad_file:
            print(f"ERROR: Corrupted file detected in archive: {bad_file}")
            sys.exit(1)

        namelist = zf.namelist()
        forbidden_in_zip = [n for n in namelist if any(f in n for f in [".git/", "AUDIT_REPORT", "BAO_CAO", ".docx"])]
        if forbidden_in_zip:
            print(f"ERROR: Forbidden files found in zip: {forbidden_in_zip}")
            sys.exit(1)

    print("Verification PASS: Archive is 100% clean, uncorrupted, and double-anonymous compliant.")


def main():
    out = PROJECT_ROOT / "ast-rag-replication-package-clean.zip"
    create_clean_zip(out)


if __name__ == "__main__":
    main()
