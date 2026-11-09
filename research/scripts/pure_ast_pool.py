#!/usr/bin/env python3
"""Pool the top-k of the 8 pure-AST/line-same dense and rerank runs and export ONLY the items that
have no label yet, in the same NEUTRAL format as to_label.csv (no chunk type, header, tier or score).

Outputs (in --outdir):
  to_label_pure_ast.csv   -> give to the annotators (shuffled, opaque item ids)
  pool_key_pure_ast.csv   -> keep private until labelling is finished (maps item_id to coordinates/systems)
"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd

SYSTEMS = [f"{t}_{u}_{h}" for t in ("dense", "rerank") for u in ("pure_ast", "line_same") for h in ("with_header", "no_header")]
KEY = lambda jd, r, f, s, e: (jd, r, f, int(s), int(e))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="."); ap.add_argument("--rundir", default=None); ap.add_argument("--outdir", default=None)
    ap.add_argument("--k", type=int, default=3); ap.add_argument("--seed", type=int, default=17)
    a = ap.parse_args(); root = Path(a.root)
    base = Path(a.outdir) if a.outdir else root / "dataset/pure_ast"
    rundir = Path(a.rundir) if a.rundir else base / "runs"; base.mkdir(parents=True, exist_ok=True)
    gt = pd.read_csv(root / "dataset/ground_truth_final.csv")
    labeled = {KEY(r.jd_id, r.repo_name, r.file_path, r.start_line, r.end_line) for r in gt.itertuples()}
    corpus = pd.read_parquet(root / "dataset/chunk_corpus.parquet").set_index("chunk_id")
    queries = json.load(open(root / "dataset/queries_frozen.json", encoding="utf-8-sig"))["queries"]
    pooled = {}
    for sysname in SYSTEMS:
        run = json.load(open(rundir / f"{sysname}.json"))
        for jd, items in run.items():
            for it in items[:a.k]:
                k = KEY(jd, it["repo_name"], it["file_path"], it["start_line"], it["end_line"])
                pooled.setdefault(k, {"chunk_id": it["chunk_id"], "systems": []})["systems"].append(sysname)
    new = {k: v for k, v in pooled.items() if k not in labeled}
    print(f"unique pooled coordinates: {len(pooled)} | already labeled: {len(pooled) - len(new)} | NEW to label: {len(new)}")
    if len(new) > 150:
        print("WARNING: more than 150 new items. Re-run with --k 2 or ask the annotators before continuing.")
    salt = "pure-ast-v1"; rows, key_rows = [], []
    for (jd, repo, f, s, e), v in new.items():
        iid = hashlib.sha256(f"{salt}|{jd}|{repo}|{f}|{s}|{e}".encode()).hexdigest()[:12]
        q = queries[jd]; code = corpus.loc[v["chunk_id"], "chunk_content"]
        rows.append({"item_id": iid, "jd_title": q["title"], "domain": q["domain"], "mandatory_skills": q["mandatory_skills"],
                     "file_path": f, "start_line": s, "end_line": e, "code": code,
                     "annotator_1_label": "", "annotator_2_label": "", "adjudicated_label": "", "adjudication_reason": ""})
        key_rows.append({"item_id": iid, "jd_id": jd, "repo_name": repo, "file_path": f, "start_line": s, "end_line": e,
                         "chunk_id": v["chunk_id"], "systems": ";".join(sorted(set(v["systems"])))})
    df = pd.DataFrame(rows); kdf = pd.DataFrame(key_rows)
    if len(df):
        order = np.random.default_rng(a.seed).permutation(len(df)); df = df.iloc[order].reset_index(drop=True)
    forbidden = {"chunk_type", "context_header", "tier", "score", "rank", "systems", "strategy"}
    assert not (forbidden & set(df.columns)), "to_label must stay neutral"
    df.to_csv(base / "to_label_pure_ast.csv", index=False, encoding="utf-8"); kdf.to_csv(base / "pool_key_pure_ast.csv", index=False)
    print("wrote", base / "to_label_pure_ast.csv", "and", base / "pool_key_pure_ast.csv")


if __name__ == "__main__":
    main()
