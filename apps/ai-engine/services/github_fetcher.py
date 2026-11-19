from urllib.parse import quote
import re
import tarfile
import logging
import os
import urllib.request
import urllib.error
import json
from typing import Optional

logger = logging.getLogger("ai-engine.github_fetcher")

# Chỉ lấy các ngôn ngữ mà tree_sitter_loader hỗ trợ
SUPPORTED_EXTENSIONS = {
    ".java": "java",
    ".ts":   "typescript",
    ".tsx":  "tsx",
    ".js":   "javascript",
    ".jsx":  "javascript",
}

# Bỏ qua các thư mục không cần thiết
SKIP_DIRS = {"node_modules", ".git", "dist", "build", "target", ".idea", "__pycache__"}

def fetch_repo_files(owner: str, repo_name: str, revision: str):
    """Stream regular source files from the public commit archive, one file at a time."""
    url = f"https://codeload.github.com/{owner}/{repo_name}/tar.gz/{quote(revision, safe='')}"
    with urllib.request.urlopen(url, timeout=90) as response:
        with tarfile.open(fileobj=response, mode="r|gz") as archive:
            for entry in archive:
                if not entry.isfile():
                    continue
                _, _, path = entry.name.partition("/")
                parts = path.split("/")
                language = SUPPORTED_EXTENSIONS.get(os.path.splitext(parts[-1])[1].lower())
                if not language or any(part in SKIP_DIRS for part in parts[:-1]):
                    continue
                with archive.extractfile(entry) as source:
                    content = source.read().decode("utf-8", errors="replace")
                yield {"file_path": path, "content": content, "language": language}

def _api_get(url: str, headers: dict) -> Optional[dict]:
    """Helper: gọi GET request đến GitHub API, trả về JSON dict."""
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        logger.error(f"GitHub API HTTP {e.code}: {url}")
        return None
    except Exception as e:
        logger.error(f"GitHub API lỗi: {e} — {url}")
        return None

def public_snapshot(url: str, ref: str):
    match = re.fullmatch(r"https://github\.com/([A-Za-z0-9-]+)/([A-Za-z0-9_.-]+)/?", url)
    if not match or not re.fullmatch(r"HEAD|[0-9a-fA-F]{7,40}", ref):
        raise ValueError("Cần URL repo GitHub và commit SHA hợp lệ hoặc HEAD.")
    owner, repo = match.groups()
    repo = repo.removesuffix(".git")
    if repo in ("", ".", ".."):
        raise ValueError("Tên repo không hợp lệ.")
    base = f"https://api.github.com/repos/{owner}/{repo}"
    headers = {"Accept": "application/vnd.github+json"}
    token = os.getenv("GITHUB_PERSONAL_TOKEN") or os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    metadata = _api_get(base, headers)
    if not metadata or metadata.get("private") is not False:
        raise ValueError("Không truy cập được repo GitHub công khai.")
    commit = _api_get(f"{base}/commits/{ref}", headers) or {}
    sha = commit.get("sha", "")
    if not re.fullmatch(r"[a-f0-9]{40}", sha):
        raise ValueError("Không xác định được commit của repo.")
    return owner, repo, sha
