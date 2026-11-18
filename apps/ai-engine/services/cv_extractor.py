"""
Module: apps/ai-engine/services/cv_extractor.py
Chức năng: Đọc file CV dạng PDF, trích xuất:
  - Tên ứng viên (heuristic từ 5 dòng đầu)
  - Danh sách GitHub repo URL (2 layer: regex text + URI annotation)
"""

import re
import logging
from typing import Optional
from models.schemas import CVExtractData, GitHubRepoItem

logger = logging.getLogger("ai-engine.cv_extractor")

# Regex khớp URL GitHub dạng https://github.com/owner/repo
GITHUB_URL_PATTERN = re.compile(
    r'https?://github\.com/([a-zA-Z0-9_-]+)/([a-zA-Z0-9_.\-]+)'
)


def extract_cv_info(pdf_bytes: bytes) -> CVExtractData:
    """
    Hàm chính: nhận bytes của file PDF, trả về CVExtractData.
    Gọi nội bộ _extract_text(), _find_github_in_text(), _find_github_in_annotations()
    """
    try:
        from pypdf import PdfReader
        import io
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except Exception as e:
        logger.error(f"Không thể đọc file PDF: {e}")
        return CVExtractData()

    full_text = _extract_text(reader)
    candidate_name = _guess_candidate_name(full_text)

    # Layer 1: tìm GitHub URL trong text thô
    repos_from_text = _find_github_in_text(full_text)

    # Layer 2: tìm GitHub URL trong link ẩn (annotation)
    repos_from_annotation = _find_github_in_annotations(reader)

    # Gom lại, ưu tiên annotation nếu URL trùng
    all_repos = _merge_repos(repos_from_text, repos_from_annotation)

    logger.info(f"CV: candidate='{candidate_name}', repos={len(all_repos)}")
    return CVExtractData(candidate_name=candidate_name, github_repos=all_repos)


def _extract_text(reader) -> str:
    """Ghép text từ tất cả các trang PDF thành 1 chuỗi."""
    pages_text = []
    for page in reader.pages:
        try:
            pages_text.append(page.extract_text() or "")
        except Exception:
            pages_text.append("")
    return "\n".join(pages_text)


def _guess_candidate_name(full_text: str) -> Optional[str]:
    """
    Heuristic: tên ứng viên thường nằm trong 5 dòng đầu CV.
    Lấy dòng đầu tiên có 2-5 từ, không phải email/URL/số điện thoại.
    """
    lines = [l.strip() for l in full_text.splitlines() if l.strip()]
    for line in lines[:5]:
        words = line.split()
        # Tên người: 2-5 từ, không chứa @, /, số điện thoại dài
        if 2 <= len(words) <= 5 and "@" not in line and "/" not in line and not re.search(r'\d{5,}', line):
            return line
    return None


def _find_github_in_text(full_text: str) -> list[GitHubRepoItem]:
    """Layer 1: Regex tìm GitHub URL xuất hiện trực tiếp trong text PDF."""
    repos = []
    seen_urls = set()
    for match in GITHUB_URL_PATTERN.finditer(full_text):
        url = match.group(0).rstrip(".,;)")   # bỏ dấu câu dính cuối URL
        owner = match.group(1)
        repo_name = match.group(2).rstrip(".,;)")
        if url not in seen_urls:
            seen_urls.add(url)
            repos.append(GitHubRepoItem(url=url, owner=owner, repo_name=repo_name, source="text"))
    return repos


def _find_github_in_annotations(reader) -> list[GitHubRepoItem]:
    """
    Layer 2: Quét URI annotation ẩn trong từng page của PDF.
    Nhiều CV xuất từ Canva/Notion ẩn hyperlink vào ký tự thường.
    """
    repos = []
    seen_urls = set()
    for page in reader.pages:
        try:
            annots = page.get("/Annots")
            if not annots:
                continue
            for annot in annots:
                obj = annot.get_object()
                subtype = obj.get("/Subtype")
                if subtype != "/Link":
                    continue
                action = obj.get("/A")
                if not action:
                    continue
                uri = action.get("/URI")
                if not uri:
                    continue
                url = str(uri)
                match = GITHUB_URL_PATTERN.match(url)
                if match and url not in seen_urls:
                    seen_urls.add(url)
                    repos.append(GitHubRepoItem(
                        url=url,
                        owner=match.group(1),
                        repo_name=match.group(2).rstrip(".,;)"),
                        source="annotation"
                    ))
        except Exception as e:
            logger.debug(f"Lỗi đọc annotation trang PDF: {e}")
    return repos


def _merge_repos(text_repos: list, annot_repos: list) -> list[GitHubRepoItem]:
    """
    Gom 2 danh sách lại, bỏ trùng URL.
    annotation ưu tiên hơn text (vì chắc chắn là link thật).
    """
    seen_urls = set()
    result = []
    for repo in annot_repos + text_repos:   # annotation trước
        if repo.url not in seen_urls:
            seen_urls.add(repo.url)
            result.append(repo)
    return result
