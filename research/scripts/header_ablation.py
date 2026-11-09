#!/usr/bin/env python3
"""Header-component ablation by RE-SCORING already judged pairs (closed pool; EXPLORATORY).

Why closed pool: re-running retrieval with a new header would surface unlabeled items, so we only re-score
the 882 pairs that already have a human label. Per JD we rank its judged pairs under each header variant and
compute ROC-AUC (relevant = label>=1 vs 0; baseline 0.5) and NDCG@5 (graded). NDCG@10 is NOT used because
most JDs have <=10 judged pairs, so the random-order NDCG@10 baseline is already about 0.8.

Subsets / variants (header lines are taken from the corpus column context_header):
  all (882 pairs)       : none, path, full
  pure_ast (438 pairs)  : none, path, class, method, path_class, no_lines, full
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pure_ast_run import Stub, BGE, load  # noqa: E402

K = lambda jd, r, f, s, e: (jd, r, f, int(s), int(e))
PREFIX = {"path": ["// File:"], "class": ["// Class:"], "method": ["// Method:"], "path_class": ["// File:", "// Class:"],
          "no_lines": ["// File:", "// Class:", "// Method:"], "full": ["// File:", "// Class:", "// Method:", "// Lines:"]}
SUBSETS = {"all": ["none", "path", "full"], "pure_ast": ["none", "path", "class", "method", "path_class", "no_lines", "full"]}


def variant_text(row, v):
    if v == "none": return row.chunk_content
    lines = [l for l in str(row.context_header).split("\n") if l.strip()]
    sel = [l for l in lines if any(l.startswith(p) for p in PREFIX[v])]
    return ("\n".join(sel) + "\n\n" + row.chunk_content) if sel else row.chunk_content


def auc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l >= 1]; neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg: return np.nan
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def ndcg(scores, labels, k=5):
    order = np.argsort(-np.asarray(scores), kind="stable"); rel = np.asarray(labels, float)
    dcg = lambda r: float((r[:k] / np.log2(np.arange(2, len(r[:k]) + 2))).sum())
    ideal = dcg(np.sort(rel)[::-1]); return dcg(rel[order]) / ideal if ideal > 0 else np.nan


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(o):
        run = max(run, (m - rank) * p[i]); adj[i] = min(1.0, run)
    return adj


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="."); ap.add_argument("--backend", choices=["bge", "stub"], default="bge")
    ap.add_argument("--outdir", default=None); ap.add_argument("--boot", type=int, default=10000)
    a = ap.parse_args(); root = Path(a.root)
    out = Path(a.outdir) if a.outdir else root / ("dataset/header_ablation_STUB" if a.backend == "stub" else "dataset/header_ablation")
    if a.backend == "stub" and "stub" not in out.name.lower(): sys.exit("stub results must be written to a directory whose name contains the word stub")
    out.mkdir(parents=True, exist_ok=True)
    corpus, queries = load(root); gt = pd.read_csv(root / "dataset/ground_truth_final.csv")
    byc = {}
    for r in corpus.itertuples(): byc.setdefault((r.repo_name, r.file_path, int(r.start_line), int(r.end_line)), r)
    pairs, skipped = [], 0
    for g in gt.itertuples():
        r = byc.get((g.repo_name, g.file_path, int(g.start_line), int(g.end_line)))
        if r is None: skipped += 1; continue  # judged under an earlier extraction; chunk no longer in the corpus
        assert r.text_with_header == variant_text(r, "full"), "header format differs from corpus text_with_header"
        pairs.append({"jd": g.jd_id, "row": r, "label": int(g.ground_truth_label), "pure": r.strategy == "AST" and r.chunk_type != "LINE_FALLBACK"})
    be = Stub(corpus.text_with_header.tolist() + corpus.text_no_header.tolist()) if a.backend == "stub" else BGE(cache=out / "cache" / "emb.npz")
    print(f"judged pairs with a chunk in the current corpus: {len(pairs)} (skipped {skipped} without a matching chunk)")
    rng = np.random.default_rng(42); rows, per_pair, results = [], [], {"exploratory": True, "backend": be.name, "n_pairs_used": len(pairs), "n_pairs_skipped": skipped, "subsets": {}}
    for sname, variants in SUBSETS.items():
        sel = [p for p in pairs if (sname == "all" or p["pure"])]
        jds = sorted({p["jd"] for p in sel}); results["subsets"][sname] = {"n_pairs": len(sel), "n_jd": len(jds), "tests": {}, "means": {}}
        scores = {}
        for v in variants:
            texts = [variant_text(p["row"], v) for p in sel]; E = be.embed(texts)
            for tier in ("dense", "rerank"):
                s = np.zeros(len(sel))
                for jd in jds:
                    idx = [i for i, p in enumerate(sel) if p["jd"] == jd]
                    s[idx] = be.sims(queries[jd], E[idx]) if tier == "dense" else be.rerank(queries[jd], [texts[i] for i in idx])
                scores[(tier, v)] = s
        for (tier, v), s in scores.items():
            for jd in jds:
                idx = [i for i, p in enumerate(sel) if p["jd"] == jd]; lab = [sel[i]["label"] for i in idx]
                rand = np.nanmean([ndcg(rng.permutation(len(idx)), lab) for _ in range(300)])
                rows.append({"subset": sname, "tier": tier, "variant": v, "jd_id": jd, "n": len(idx), "auc": auc(s[idx], lab),
                             "ndcg_5": ndcg(s[idx], lab), "ndcg_5_random": rand})
        for i, p in enumerate(sel):
            d = {"subset": sname, "jd_id": p["jd"], "repo_name": p["row"].repo_name, "file_path": p["row"].file_path,
                 "start_line": p["row"].start_line, "end_line": p["row"].end_line, "label": p["label"]}
            d.update({f"{t}_{v}": float(s[i]) for (t, v), s in scores.items()}); per_pair.append(d)
        df = pd.DataFrame([r for r in rows if r["subset"] == sname])
        for tier in ("rerank", "dense"):
            for metric in ("auc", "ndcg_5"):
                piv = df[df.tier == tier].pivot(index="jd_id", columns="variant", values=metric)
                results["subsets"][sname]["means"][f"{tier}_{metric}"] = {v: float(piv[v].mean()) for v in variants}
                comps = []
                for v in variants[1:]:
                    ok = piv[["none", v]].dropna(); d = (ok[v] - ok["none"]).values
                    bs = [rng.choice(d, len(d)).mean() for _ in range(a.boot)]
                    p = 1.0 if np.allclose(d, 0) else float(wilcoxon(ok[v], ok["none"]).pvalue)
                    comps.append({"comparison": f"{v} vs none", "n_jd": len(d), "delta": float(d.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "p_raw": p})
                for c, ph in zip(comps, holm([c["p_raw"] for c in comps])): c["p_holm"] = float(ph)
                results["subsets"][sname]["tests"][f"{tier}_{metric}"] = comps
    pd.DataFrame(rows).to_csv(out / "header_ablation_per_jd.csv", index=False); pd.DataFrame(per_pair).to_csv(out / "header_ablation_scores.csv", index=False)
    json.dump(results, open(out / "header_ablation_results.json", "w"), indent=2)
    L = ["EXPLORATORY closed-pool re-scoring of judged pairs. AUC baseline = 0.5. Not comparable with Table I.", f"backend: {be.name}; pairs used: {len(pairs)}; skipped (no chunk in current corpus): {skipped}", ""]
    for sname, S in results["subsets"].items():
        L.append(f"== subset {sname}: {S['n_pairs']} pairs, {S['n_jd']} JDs ==")
        for key, m in S["means"].items(): L.append(f"{key:14s} " + "  ".join(f"{v}={x:.3f}" for v, x in m.items()))
        for key, comps in S["tests"].items():
            if key.endswith("auc") or key.endswith("ndcg_5"):
                for c in comps: L.append(f"  [{key}] {c['comparison']:18s} delta={c['delta']:+.3f} CI=[{c['ci95'][0]:+.3f},{c['ci95'][1]:+.3f}] p_raw={c['p_raw']:.4f} p_Holm={c['p_holm']:.4f}")
        L.append("")
    open(out / "header_ablation_report.txt", "w", encoding="utf-8").write("\n".join(L)); print("\n".join(L[:24]))


if __name__ == "__main__":
    main()
