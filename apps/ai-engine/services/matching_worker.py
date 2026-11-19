import json
import logging
import os
from urllib.parse import quote

import requests
import psycopg

from config import DATABASE_URL
from parser.tree_sitter_loader import ast_loader
from services.github_fetcher import fetch_repo_files, public_snapshot

logger = logging.getLogger("ai-engine.worker")
tasks_store = {}


def embed(texts):
    url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    payload = {"model": "bge-m3", "input": texts, "truncate": True}
    response = requests.post(f"{url}/api/embed", json=payload, timeout=(10, 300))
    if response.status_code == 413:
        # Leave headroom for special tokens below Ollama's 2048-token physical batch.
        payload["options"] = {"num_ctx": 1536}
        response = requests.post(f"{url}/api/embed", json=payload, timeout=(10, 300))
    if response.status_code == 404:
        raise ValueError("Chưa tải BGE-M3. Chạy docker compose exec embeddings ollama pull bge-m3.")
    response.raise_for_status()
    vectors = response.json()["embeddings"]
    if len(vectors) != len(texts) or any(len(v) != 1024 for v in vectors):
        raise ValueError("BGE-M3 trả về vector sai kích thước.")
    return vectors


def index_snapshot(cur, owner, repo, sha, repo_url):
    cur.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s,0))", (repo_url + sha,))
    cur.execute("""SELECT id, indexed_file_count FROM repositories
                   WHERE repo_url=%s AND commit_hash=%s
                   ORDER BY indexed_file_count DESC NULLS LAST LIMIT 1""", (repo_url, sha))
    row = cur.fetchone()
    if row and row[1] is not None:
        return row[0]
    if row:
        repo_id = row[0]
    else:
        cur.execute("INSERT INTO repositories (repo_url, commit_hash) VALUES (%s,%s) RETURNING id", (repo_url, sha))
        repo_id = cur.fetchone()[0]

    # Previously committed chunks belong to whole files of this same immutable commit.
    cur.execute("SELECT DISTINCT file_path FROM code_chunks WHERE repo_id=%s", (repo_id,))
    indexed_paths = {path for (path,) in cur.fetchall()}
    pending = []
    file_count = 0
    has_chunks = bool(indexed_paths)
    for source in fetch_repo_files(owner, repo, sha):
        file_count += 1
        if source["file_path"] in indexed_paths:
            continue
        for chunk in ast_loader.parse_and_chunk(source["file_path"], source["content"], source["language"]):
            pending.append(chunk)
            has_chunks = True
            if len(pending) == 16:
                store_chunks(cur, repo_id, pending)
                pending.clear()
    if not file_count or not has_chunks:
        raise ValueError("Repo không có đoạn mã Java/JavaScript/TypeScript có thể lập chỉ mục.")
    if pending:
        store_chunks(cur, repo_id, pending)
    cur.execute("UPDATE repositories SET indexed_file_count=%s, last_parsed_at=NOW() WHERE id=%s", (file_count, repo_id))
    logger.info("Indexed full snapshot %s@%s: %d source files", repo_url, sha, file_count)
    return repo_id


def store_chunks(cur, repo_id, batch):
    vectors = embed([c["context_header"] + "\n" + c["chunk_content"] for c in batch])
    cur.executemany("""INSERT INTO code_chunks
        (repo_id,file_path,class_name,method_name,start_line,end_line,context_header,chunk_content,chunk_type,embedding)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::vector)""",
        [(repo_id, c["file_path"], c["class_name"], c["method_name"], c["start_line"], c["end_line"],
          c["context_header"], c["chunk_content"], c["chunk_type"], json.dumps(v)) for c,v in zip(batch,vectors)])



def run_matching_pipeline(task_id, job_id, candidate_name, github_repo_url, commit_hash):
    task = tasks_store[task_id]
    try:
        with psycopg.connect(DATABASE_URL) as conn, conn.cursor() as cur:
            cur.execute("SELECT skill_name,weight FROM required_skills WHERE jd_id=%s ORDER BY skill_name", (job_id,))
            skills = cur.fetchall()
            if not skills:
                raise ValueError("JD chưa có kỹ năng. Hãy lưu JD và kỹ năng trước.")
            owner, repo, sha = public_snapshot(github_repo_url, commit_hash)
            url = f"https://github.com/{owner}/{repo}"
            task["progress_percent"] = 15
            repo_id = index_snapshot(cur, owner, repo, sha, url)
            task["progress_percent"] = 75
            assessments = []
            weighted_score = 0
            weight_sum = 0
            for (skill, weight), vector in zip(skills, embed([name for name, _ in skills])):
                cur.execute("""SELECT file_path,start_line,end_line,chunk_type,chunk_content,
                               1-(embedding <=> %s::vector) AS similarity
                               FROM code_chunks WHERE repo_id=%s
                               ORDER BY similarity DESC, file_path,start_line,end_line,id LIMIT 5""",
                            (json.dumps(vector), repo_id))
                evidence = []
                for path, start, end, kind, snippet, similarity in cur.fetchall():
                    evidence.append(dict(file_path=path, start_line=start, end_line=end, chunk_type=kind,
                        code_snippet=snippet, similarity=similarity,
                        source_url=f"{url}/blob/{sha}/{quote(path)}#L{start}-L{end}"))
                score = round(100 * max(0, min(1, evidence[0]["similarity"])), 2)
                assessments.append(dict(skill_name=skill, score=score, evidence_code_chunk=evidence[0],
                                        evidence=evidence, llm_explanation=""))
                weighted_score += score * float(weight)
                weight_sum += float(weight)
            if weight_sum <= 0:
                raise ValueError("Tổng trọng số kỹ năng phải lớn hơn 0.")
            cur.execute("""INSERT INTO evaluation_results
                        (jd_id,repo_id,candidate_name,total_score,skills_detail)
                        VALUES (%s,%s,%s,%s,%s::jsonb)""",
                        (job_id, repo_id, candidate_name, round(weighted_score/weight_sum, 2), json.dumps(assessments)))
        task.update(progress_percent=100, progress_status="COMPLETED")
    except Exception as exc:
        logger.exception("Matching failed for task %s", task_id)
        task.update(progress_status="FAILED", error=str(exc) if isinstance(exc, ValueError) else "Lỗi khi truy cập dịch vụ đối sánh.")


def rankings_for_job(job_id):
    with psycopg.connect(DATABASE_URL) as conn:
        rows = conn.execute("""SELECT candidate_name,total_score,skills_detail FROM
            (SELECT DISTINCT ON (candidate_name,repo_url) candidate_name,total_score,skills_detail,repo_url
             FROM evaluation_results e JOIN repositories r ON r.id=e.repo_id
             WHERE jd_id=%s AND status='COMPLETED'
             ORDER BY candidate_name,repo_url,e.created_at DESC,e.id DESC) latest
            ORDER BY total_score DESC,candidate_name,repo_url""", (job_id,)).fetchall()
    return [dict(rank=i, candidate_name=name, total_score=score, skills_assessment=details)
            for i,(name,score,details) in enumerate(rows, 1)]
