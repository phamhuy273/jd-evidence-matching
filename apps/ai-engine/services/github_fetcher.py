"""
Module: apps/ai-engine/services/github_fetcher.py
Chức năng: Gọi GitHub REST API để:
  1. Lấy danh sách file trong repo (tree)
  2. Fetch nội dung từng file .java/.ts/.js/.tsx
"""

import base64
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

# Giới hạn số file tối đa để không bị timeout / rate limit
MAX_FILES = 50


def fetch_repo_files(owner: str, repo_name: str) -> list[dict]:
    """
    Hàm chính: nhận owner + repo_name → trả danh sách file với nội dung.
    Mỗi phần tử: { file_path, content, language }

    VD: fetch_repo_files("phamhuy273", "jd-matching")
    """
    token = os.getenv("GITHUB_PERSONAL_TOKEN") or os.getenv("GITHUB_TOKEN")  # lấy từ .env
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    # Bước 1: lấy cây file
    file_paths = _get_file_tree(owner, repo_name, headers)
    if not file_paths:
        logger.warning(f"Repo {owner}/{repo_name}: không tìm thấy file phù hợp")
        return []

    logger.info(f"Repo {owner}/{repo_name}: tìm thấy {len(file_paths)} file, fetch tối đa {MAX_FILES}")

    # Bước 2: fetch nội dung từng file (tối đa MAX_FILES)
    results = []
    for path in file_paths[:MAX_FILES]:
        content = _fetch_file_content(owner, repo_name, path, headers)
        if content is None:
            continue
        ext = "." + path.rsplit(".", 1)[-1].lower()
        language = SUPPORTED_EXTENSIONS.get(ext, "java")
        results.append({"file_path": path, "content": content, "language": language})

    logger.info(f"Repo {owner}/{repo_name}: fetch xong {len(results)} file")
    return results


def _get_file_tree(owner: str, repo_name: str, headers: dict) -> list[str]:
    """
    Gọi GitHub Trees API để lấy toàn bộ cây file (1 request).
    Lọc chỉ lấy extension được hỗ trợ, bỏ thư mục trong SKIP_DIRS.
    """
    url = f"https://api.github.com/repos/{owner}/{repo_name}/git/trees/HEAD?recursive=1"
    data = _api_get(url, headers)
    if not data or "tree" not in data:
        return []

    matched = []
    for item in data["tree"]:
        # item["type"] == "blob" nghĩa là file (không phải folder)
        if item.get("type") != "blob":
            continue
        path = item["path"]

        # Bỏ qua nếu nằm trong thư mục cần skip
        parts = path.split("/")
        if any(p in SKIP_DIRS for p in parts):
            continue

        # Chỉ lấy extension được hỗ trợ
        if "." not in parts[-1]:
            continue
        ext = "." + parts[-1].rsplit(".", 1)[-1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            matched.append(path)

    return matched


def _fetch_file_content(owner: str, repo_name: str, path: str, headers: dict) -> Optional[str]:
    """
    Gọi GitHub Contents API để lấy nội dung 1 file (base64 encoded).
    Trả về string nội dung, hoặc None nếu lỗi.
    """
    url = f"https://api.github.com/repos/{owner}/{repo_name}/contents/{path}"
    data = _api_get(url, headers)
    if not data or data.get("encoding") != "base64":
        return None
    try:
        # GitHub trả về content với \n xen kẽ → cần xóa trước khi decode
        raw = data["content"].replace("\n", "")
        return base64.b64decode(raw).decode("utf-8", errors="replace")
    except Exception as e:
        logger.warning(f"Decode lỗi file {path}: {e}")
        return None


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
