#!/usr/bin/env python3
"""Pure-AST full re-run (EXPLORATORY). Re-runs dense (BGE-M3) and rerank (bge-reranker-base)
retrieval on restricted corpora:
  pure_ast  : AST chunks that are NOT line-window fallbacks (strategy=AST, chunk_type!=LINE_FALLBACK)
  line_same : 50-LOC line windows from the same files that contain at least one pure AST chunk
Each universe is run with and without the context header, using the frozen queries.
 
  --check-repro : re-run on the FULL AST and line corpora and compare with the stored runs in
                  dataset/retrieval_runs/. Must pass before trusting any new run.
  --backend stub: TF-IDF/BM25 stand-ins for a smoke test ONLY (never use for results).
"""
import argparse, hashlib, json, platform, sys, time
from pathlib import Path
import numpy as np, pandas as pd

DEPTH = 20
COLS = ["rank", "score", "dense_rank", "dense_score", "chunk_id", "repo_name", "file_path", "start_line", "end_line"]


def sha_bytes(b): return hashlib.sha256(b).hexdigest()
def sha_text(s): return sha_bytes(s.encode("utf-8"))


def load(root):
    c = pd.read_parquet(root / "dataset/chunk_corpus.parquet")
    q = json.load(open(root / "dataset/queries_frozen.json", encoding="utf-8-sig"))["queries"]
    return c, {k: q[k]["primary_query"] for k in sorted(q)}


def universes(c):
    pure = c[(c.strategy == "AST") & (c.chunk_type != "LINE_FALLBACK")]
    files = set(zip(pure.repo_name, pure.file_path))
    line = c[c.strategy == "LINE"]
    same = line[[(r, f) in files for r, f in zip(line.repo_name, line.file_path)]]
    return {"pure_ast": pure, "line_same": same, "ast_full": c[c.strategy == "AST"], "line_full": line}


class Stub:
    name = "stub"
    def __init__(self, all_texts):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.v = TfidfVectorizer(sublinear_tf=True, token_pattern=r"[A-Za-z_][A-Za-z0-9_]+").fit(all_texts)
    def embed(self, texts): return self.v.transform(texts)
    def sims(self, qtext, E): return (E @ self.v.transform([qtext]).T).toarray().ravel()
    def rerank(self, qtext, texts):
        from rank_bm25 import BM25Okapi
        tok = lambda s: [t.lower() for t in __import__("re").findall(r"[A-Za-z_][A-Za-z0-9_]+", s)]
        bm = BM25Okapi([tok(t) for t in texts]); s = np.asarray(bm.get_scores(tok(qtext)), float)
        return s / (s + 1.0)


class BGE:
    name = "bge"
    def __init__(self, dense="BAAI/bge-m3", rerank="BAAI/bge-reranker-base", max_len=512, cache=None):
        from sentence_transformers import SentenceTransformer, CrossEncoder
        self.dense_name, self.rerank_name, self.max_len = dense, rerank, max_len
        self.st = SentenceTransformer(dense); self.st.max_seq_length = max_len
        self.ce = CrossEncoder(rerank, max_length=max_len)
        self.cache_path = cache; self.cache = {}
        if cache and Path(cache).exists():
            z = np.load(cache, allow_pickle=True); self.cache = dict(zip(z["k"].tolist(), z["v"]))
    def embed(self, texts):
        keys = [sha_text(f"{self.dense_name}|{self.max_len}|{t}") for t in texts]
        miss = [i for i, k in enumerate(keys) if k not in self.cache]
        if miss:
            V = self.st.encode([texts[i] for i in miss], normalize_embeddings=True, batch_size=16, show_progress_bar=False)
            for i, v in zip(miss, V): self.cache[keys[i]] = v
            if self.cache_path:
                Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
                np.savez(self.cache_path, k=np.array(list(self.cache)), v=np.array(list(self.cache.values())))
        return np.stack([self.cache[k] for k in keys])
    def sims(self, qtext, E): return E @ self.embed([qtext])[0]
    def rerank(self, qtext, texts):
        return np.asarray(self.ce.predict([(qtext, t) for t in texts], batch_size=16), float)


def item(row, rank, score, drank, dscore):
    return {"rank": rank, "score": float(score), "dense_rank": int(drank), "dense_score": float(dscore),
            "chunk_id": row.chunk_id, "repo_name": row.repo_name, "file_path": row.file_path,
            "start_line": int(row.start_line), "end_line": int(row.end_line)}


def run(df, col, queries, be):
    df = df.reset_index(drop=True); texts = df[col].tolist(); E = be.embed(texts)
    dense, rer = {}, {}
    for jd, q in queries.items():
        s = be.sims(q, E); order = np.argsort(-s, kind="stable")[:DEPTH]
        dense[jd] = [{k: v for k, v in item(df.iloc[i], r + 1, s[i], r + 1, s[i]).items() if k not in ("dense_rank", "dense_score")}
                     for r, i in enumerate(order)]
        rs = be.rerank(q, [texts[i] for i in order]); ro = np.argsort(-rs, kind="stable")
        rer[jd] = [item(df.iloc[order[j]], k + 1, rs[j], j + 1, s[order[j]]) for k, j in enumerate(ro)]
    return dense, rer


def check_repro(root, c, queries, be, out):
    U = universes(c); rows = []
    for u, arm in (("ast_full", "ast"), ("line_full", "line")):
        for h, col in (("with_header", "text_with_header"), ("no_header", "text_no_header")):
            d, r = run(U[u], col, queries, be)
            for tier, new in (("dense", d), ("rerank", r)):
                old = json.load(open(root / f"dataset/retrieval_runs/{tier}_{arm}_{h}.json"))
                ov20, ov10, dd = [], [], []
                for jd in queries:
                    a = {x["chunk_id"]: x for x in old[jd]}; b = {x["chunk_id"]: x for x in new[jd]}
                    ov20.append(len(set(a) & set(b)) / max(1, len(a)))
                    ov10.append(len({x["chunk_id"] for x in old[jd][:10]} & {x["chunk_id"] for x in new[jd][:10]}) / 10)
                    sc = "score" if tier == "dense" else "dense_score"
                    dd += [abs(a[k][sc] - b[k][sc]) for k in set(a) & set(b)]
                rows.append({"config": f"{tier}_{arm}_{h}", "top20_overlap": float(np.mean(ov20)),
                             "top10_overlap": float(np.mean(ov10)), "mean_abs_dense_score_diff": float(np.mean(dd)) if dd else None})
    ok = all(r["top20_overlap"] >= 0.95 and (r["mean_abs_dense_score_diff"] or 0) <= 1e-3 for r in rows)
    json.dump({"backend": be.name, "passed": ok, "rows": rows}, open(out / "repro_check.json", "w"), indent=2)
    for r in rows: print(r)
    print("REPRODUCTION CHECK:", "PASSED" if ok else "FAILED (do NOT use new runs; fix the pipeline first)")
    return 0 if ok and be.name == "bge" else 2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="."); ap.add_argument("--backend", choices=["bge", "stub"], default="bge")
    ap.add_argument("--outdir", default=None); ap.add_argument("--check-repro", action="store_true")
    a = ap.parse_args(); root = Path(a.root)
    out = Path(a.outdir) if a.outdir else root / ("dataset/pure_ast_STUB" if a.backend == "stub" else "dataset/pure_ast")
    if a.backend == "stub" and "stub" not in out.name.lower():
        sys.exit("stub results must be written to a directory whose name contains the word stub")
    (out / "runs").mkdir(parents=True, exist_ok=True)
    c, queries = load(root); U = universes(c)
    be = Stub(c.text_with_header.tolist() + c.text_no_header.tolist()) if a.backend == "stub" else BGE(cache=out / "cache" / "emb.npz")
    if a.check_repro: sys.exit(check_repro(root, c, queries, be, out))
    for u in ("pure_ast", "line_same"):
        for h, col in (("with_header", "text_with_header"), ("no_header", "text_no_header")):
            d, r = run(U[u], col, queries, be)
            json.dump(d, open(out / "runs" / f"dense_{u}_{h}.json", "w")); json.dump(r, open(out / "runs" / f"rerank_{u}_{h}.json", "w"))
            print("wrote", u, h, "n_chunks =", len(U[u]))
    meta = {"backend": be.name, "models": {"dense": getattr(be, "dense_name", None), "rerank": getattr(be, "rerank_name", None)},
            "max_len": 512, "depth": DEPTH, "n_pure_ast": len(U["pure_ast"]), "n_line_same": len(U["line_same"]),
            "corpus_sha256": sha_bytes(open(root / "dataset/chunk_corpus.parquet", "rb").read()),
            "queries_sha256": sha_bytes(open(root / "dataset/queries_frozen.json", "rb").read()),
            "python": platform.python_version(), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(meta, open(out / "run_meta.json", "w"), indent=2); print(meta)


if __name__ == "__main__":
    main()
