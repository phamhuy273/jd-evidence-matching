#!/usr/bin/env python3
"""Bpref (binary preference) for all 12 stored runs; EXPLORATORY secondary metric for incomplete judgments.

bpref = (1/R) * sum over relevant retrieved r of [1 - min(n_r, R) / min(R, N)]
  R = judged relevant items of the JD, N = judged non-relevant items, n_r = judged non-relevant items ranked above r.
Unjudged items are ignored (not counted as zero). Relevant = label >= 1 (also reported with label == 2).
Reads dataset/retrieval_runs/*.json (depth 20) and dataset/ground_truth_final.csv. Writes into dataset/benchmark_results/.
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

K = lambda jd, r, f, s, e: (jd, r, f, int(s), int(e))


def bpref(labels, R, N):
    if R == 0: return np.nan
    nrel, s = 0, 0.0
    for l in labels:
        if l is None: continue
        if l >= 1: s += 1 - (min(nrel, R) / min(R, N) if min(R, N) > 0 else 0.0)
        else: nrel += 1
    return s / R


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="."); ap.add_argument("--outdir", default=None); a = ap.parse_args(); root = Path(a.root)
    out = Path(a.outdir) if a.outdir else root / "dataset/benchmark_results"; out.mkdir(parents=True, exist_ok=True)
    gt = pd.read_csv(root / "dataset/ground_truth_final.csv")
    lab = {K(r.jd_id, r.repo_name, r.file_path, r.start_line, r.end_line): int(r.ground_truth_label) for r in gt.itertuples()}
    rows = []
    for t in ("bm25", "dense", "rerank"):
        for c in ("line", "ast"):
            for h in ("no_header", "with_header"):
                cfg = f"{t}_{c}_{h}"; run = json.load(open(root / f"dataset/retrieval_runs/{cfg}.json"))
                for jd, items in run.items():
                    R1 = sum(1 for k, v in lab.items() if k[0] == jd and v >= 1); N0 = sum(1 for k, v in lab.items() if k[0] == jd and v == 0)
                    R2 = sum(1 for k, v in lab.items() if k[0] == jd and v == 2); N02 = sum(1 for k, v in lab.items() if k[0] == jd and v < 2)
                    ls = [lab.get(K(jd, i["repo_name"], i["file_path"], i["start_line"], i["end_line"])) for i in items]
                    rows.append({"config": cfg, "tier": t, "chunk": c, "header": h, "jd_id": jd,
                                 "bpref": bpref(ls, R1, N0), "bpref@10": bpref(ls[:10], R1, N0),
                                 "bpref_lvl2": bpref([None if l is None else int(l == 2) * 2 for l in ls], R2, N02)})
    df = pd.DataFrame(rows); df.to_csv(out / "bpref_per_query.csv", index=False)
    summ = df.groupby("config")[["bpref", "bpref@10", "bpref_lvl2"]].mean().round(4); summ.to_csv(out / "bpref_summary.csv")
    piv = df[df.tier == "rerank"].pivot(index="jd_id", columns="config", values="bpref")
    tests = {}
    for name, (x, y) in {"C1": ("rerank_ast_with_header", "rerank_line_no_header"), "C2": ("rerank_ast_no_header", "rerank_line_no_header"),
                         "C3": ("rerank_ast_with_header", "rerank_ast_no_header"), "C4": ("rerank_line_with_header", "rerank_line_no_header"),
                         "C5": ("rerank_ast_with_header", "rerank_line_with_header")}.items():
        d = (piv[x] - piv[y]).dropna(); tests[name] = {"delta": float(d.mean()), "p_raw": float(wilcoxon(piv[x].loc[d.index], piv[y].loc[d.index]).pvalue)}
    json.dump({"exploratory": True, "summary": summ.to_dict("index"), "rerank_tests_bpref": tests}, open(out / "bpref_report.json", "w"), indent=2)
    txt = ["Bpref (EXPLORATORY secondary metric; unjudged items ignored). Relevant = label >= 1.", summ.to_string(), "", "Rerank tier, paired Wilcoxon on per-JD bpref (raw p, exploratory):"]
    txt += [f"{k}: delta={v['delta']:+.3f} p_raw={v['p_raw']:.4f}" for k, v in tests.items()]
    open(out / "bpref_report.txt", "w").write("\n".join(txt)); print("\n".join(txt))


if __name__ == "__main__":
    main()
