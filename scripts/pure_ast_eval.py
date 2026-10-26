#!/usr/bin/env python3
"""Evaluate the pure-AST re-run (EXPLORATORY; not part of the frozen confirmatory family).

Qrels = the 882 existing labels + any new labels from the annotators (final label = adjudicated label if
present, otherwise the two annotator labels when they agree; any other case aborts).
Restriction: only files that contain at least one pure AST chunk. Two ideals:
  shared (primary)  : every judged relevant item in those files, for both arms
  arm (sensitivity) : only items that exist in the arm's own universe
Absolute values are NOT comparable with Table I.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

K = lambda jd, r, f, s, e: (jd, r, f, int(s), int(e))
ARMS = {"pure_ast": "PureAST", "line_same": "Line(same files)"}
HDR = {"with_header": "+Header", "no_header": " No-Header"}


def dcg(rels, k):
    r = np.asarray(rels[:k], float)
    return float((r / np.log2(np.arange(2, len(r) + 2))).sum()) if len(r) else 0.0


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(o):
        run = max(run, (m - rank) * p[i]); adj[i] = min(1.0, run)
    return adj


def final_label(r):
    lab = lambda x: None if pd.isna(x) or str(x).strip() == "" else int(float(x))
    adj, a1, a2 = lab(r.adjudicated_label), lab(r.annotator_1_label), lab(r.annotator_2_label)
    if adj is not None: return adj
    if a1 is not None and a1 == a2: return a1
    raise ValueError(f"item {r.item_id}: no adjudicated label and annotators missing/disagree ({a1}, {a2})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="."); ap.add_argument("--rundir", default=None); ap.add_argument("--labels", default=None)
    ap.add_argument("--key", default=None); ap.add_argument("--outdir", default=None)
    ap.add_argument("--allow-unjudged", action="store_true"); ap.add_argument("--boot", type=int, default=10000)
    a = ap.parse_args(); root = Path(a.root); base = Path(a.outdir) if a.outdir else root / "dataset/pure_ast"
    rundir = Path(a.rundir) if a.rundir else (base / "runs" if (base / "runs").exists() else root / "dataset/retrieval_runs")
    lab_path = Path(a.labels) if a.labels else base / "to_label_pure_ast.csv"
    key_path = Path(a.key) if a.key else base / "pool_key_pure_ast.csv"
    res_dir = base / "results" if a.outdir else root / "dataset/benchmark_results/pure_ast"
    res_dir.mkdir(parents=True, exist_ok=True)

    gt_df = pd.read_csv(root / "dataset/ground_truth_final.csv")
    gt = {K(r.jd_id, r.repo_name, r.file_path, r.start_line, r.end_line): int(r.ground_truth_label) for r in gt_df.itertuples()}
    n_old = len(gt)
    if lab_path.exists() and key_path.exists():
        new = pd.read_csv(lab_path, dtype=str); keys = pd.read_csv(key_path, dtype=str).set_index("item_id")
        for r in new.itertuples():
            k = keys.loc[r.item_id]; kk = K(k.jd_id, k.repo_name, k.file_path, k.start_line, k.end_line); lbl = final_label(r)
            if kk in gt and gt[kk] != lbl: sys.exit(f"conflict with an existing label at {kk}")
            gt[kk] = lbl
        print(f"labels: {n_old} existing + {len(gt) - n_old} new = {len(gt)}")
    else:
        print(f"Using unified ground truth: {n_old} pairs")

    corpus = pd.read_parquet(root / "dataset/chunk_corpus.parquet")
    pure = corpus[(corpus.strategy == "AST") & (corpus.chunk_type != "LINE_FALLBACK")]
    F = set(zip(pure.repo_name, pure.file_path))
    line = corpus[corpus.strategy == "LINE"]; same = line[[(r, f) in F for r, f in zip(line.repo_name, line.file_path)]]
    coords = {"pure_ast": {(r.repo_name, r.file_path, int(r.start_line), int(r.end_line)) for r in pure.itertuples()},
              "line_same": {(r.repo_name, r.file_path, int(r.start_line), int(r.end_line)) for r in same.itertuples()}}
    
    # Resolve run files mapping
    def get_run(tier, arm, h):
        p1 = rundir / f"{tier}_{arm}_{h}.json"
        if p1.exists():
            return json.load(open(p1, encoding="utf-8"))
        arm_alias = "ast" if "ast" in arm else "line"
        p2 = rundir / f"{tier}_{arm_alias}_{h}.json"
        if p2.exists():
            return json.load(open(p2, encoding="utf-8"))
        raise FileNotFoundError(f"Missing run file for {tier}_{arm}_{h} at {p1} or {p2}")

    sample_run = get_run("dense", "pure_ast", "with_header")
    jds = sorted(sample_run.keys())

    def qrels(jd, mode, arm):
        out = {}
        for k, v in gt.items():
            if k[0] != jd or (k[1], k[2]) not in F: continue
            if mode == "arm" and (k[1], k[2], k[3], k[4]) not in coords[arm]: continue
            out[k] = v
        return out

    rows, gate = [], []
    for tier in ("dense", "rerank"):
        for arm in ARMS:
            for h in HDR:
                name = f"{tier}_{arm}_{h}"; run = get_run(tier, arm, h)
                for mode in ("shared", "arm"):
                    for jd in jds:
                        q = qrels(jd, mode, arm); ideal = sorted([v for v in q.values() if v > 0], reverse=True)
                        R = sum(v >= 1 for v in q.values()); N = sum(v == 0 for v in q.values())
                        items = run[jd]; labs = [gt.get(K(jd, i["repo_name"], i["file_path"], i["start_line"], i["end_line"])) for i in items]
                        rel = [0 if l is None else l for l in labs]
                        nd = lambda k: dcg(rel, k) / dcg(ideal, k) if dcg(ideal, k) > 0 else 0.0
                        nrel_above, bp = 0, 0.0
                        for l in labs:
                            if l is None: continue
                            if l >= 1: bp += 1 - (min(nrel_above, R) / min(R, N) if min(R, N) > 0 else 0)
                            else: nrel_above += 1
                        rows.append({"system": name, "tier": tier, "arm": arm, "header": h, "ideal": mode, "jd_id": jd,
                                     "ndcg_10": nd(10), "ndcg_5": nd(5), "bpref": bp / R if R else np.nan,
                                     "judged_3": np.mean([l is not None for l in labs[:3]]), "judged_10": np.mean([l is not None for l in labs[:10]]),
                                     "list_len": len(items)})
                    if min(r["judged_3"] for r in rows if r["system"] == name) < 1: gate.append(name)
    df = pd.DataFrame(rows); df.to_csv(res_dir / "pure_ast_per_query.csv", index=False)
    if gate and not a.allow_unjudged:
        sys.exit("ABORT: top-3 of these systems still contain unlabeled items (label them first): " + ", ".join(sorted(set(gate))))

    rng = np.random.default_rng(42); out = {"exploratory": True, "n_jd": len(jds), "labels_old": n_old, "labels_new": len(gt) - n_old, "unjudged_top3_systems": sorted(set(gate)), "summary": {}, "tests": {}}
    for mode in ("shared", "arm"):
        sub = df[df.ideal == mode]
        out["summary"][mode] = {s: {m: float(g[m].mean()) for m in ("ndcg_10", "ndcg_5", "bpref", "judged_3", "judged_10", "list_len")} for s, g in sub.groupby("system")}
        for tier in ("rerank", "dense"):
            pv = lambda arm, h: sub[sub.system == f"{tier}_{arm}_{h}"].set_index("jd_id")["ndcg_10"].reindex(jds).values
            comps = [("C1 PureAST+H vs Line+H", pv("pure_ast", "with_header"), pv("line_same", "with_header")),
                     ("C2 PureAST noH vs Line noH", pv("pure_ast", "no_header"), pv("line_same", "no_header")),
                     ("C3 header effect on PureAST", pv("pure_ast", "with_header"), pv("pure_ast", "no_header")),
                     ("C4 PureAST+H vs Line noH", pv("pure_ast", "with_header"), pv("line_same", "no_header"))]
            res = []
            for nme, x, y in comps:
                d = x - y; bs = [rng.choice(d, len(d)).mean() for _ in range(a.boot)]
                p = 1.0 if np.allclose(d, 0) else float(wilcoxon(x, y).pvalue)
                res.append({"comparison": nme, "delta": float(d.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                            "W/L/T": [int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum())], "p_raw": p})
            for r, ph in zip(res, holm([r["p_raw"] for r in res])): r["p_holm"] = float(ph)
            out["tests"][f"{mode}_{tier}"] = res
    json.dump(out, open(res_dir / "pure_ast_results.json", "w"), indent=2)
    lines = ["EXPLORATORY ANALYSIS (outside the frozen confirmatory family). Values are not comparable with Table I.",
             f"Labels: {n_old} existing + {len(gt) - n_old} new. Unjudged-top3 systems: {out['unjudged_top3_systems'] or 'none'}", ""]
    for mode in ("shared", "arm"):
        lines.append(f"== ideal = {mode} ==")
        for s, m in out["summary"][mode].items(): lines.append(f"{s:34s} NDCG@10={m['ndcg_10']:.3f} NDCG@5={m['ndcg_5']:.3f} Bpref={m['bpref']:.3f} J@3={m['judged_3']:.2f} J@10={m['judged_10']:.2f} len={m['list_len']:.1f}")
        for tier in ("rerank", "dense"):
            lines.append(f"-- tests ({tier}, NDCG@10) --")
            for r in out["tests"][f"{mode}_{tier}"]:
                lines.append(f"{r['comparison']:30s} delta={r['delta']:+.3f} CI=[{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}] W/L/T={r['W/L/T']} p_raw={r['p_raw']:.4f} p_Holm={r['p_holm']:.4f}")
        lines.append("")
    open(res_dir / "pure_ast_report.txt", "w", encoding="utf-8").write("\n".join(lines)); print("\n".join(lines[:22]))


if __name__ == "__main__":
    main()
