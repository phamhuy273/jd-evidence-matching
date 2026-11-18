"""
Module: apps/ai-engine/tests/test_cv_extractor.py
Mục đích: Unit tests kiểm thử module bóc tách CV (Layer 1 Text, Layer 2 Hyperlinks)
và các API endpoints liên quan.
"""

import io
from fastapi.testclient import TestClient
from pypdf import PdfWriter
from pypdf.annotations import Link

from main import app
from services.cv_extractor import (
    _find_github_in_text,
    _guess_candidate_name,
    extract_cv_info
)

client = TestClient(app)


def _create_sample_pdf(with_link: bool = True) -> bytes:
    """Helper tạo file PDF nhị phân trong bộ nhớ để phục vụ test."""
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)
    if with_link:
        link = Link(
            rect=(10, 10, 200, 30),
            url="https://github.com/phamhuy273/hidden-portfolio"
        )
        writer.add_annotation(page_number=0, annotation=link)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def test_regex_github_urls_in_text():
    sample_text = (
        "Xin chào, tôi là Nguyễn Văn A.\n"
        "Dự án chính: https://github.com/nguyenvana/spring-ecommerce.\n"
        "Repo khác: https://github.com/nguyenvana/react-dashboard, xem thêm tại...\n"
        "Link trùng: https://github.com/nguyenvana/spring-ecommerce"
    )
    repos = _find_github_in_text(sample_text)
    assert len(repos) == 2
    assert repos[0].owner == "nguyenvana"
    assert repos[0].repo_name == "spring-ecommerce"
    assert repos[1].repo_name == "react-dashboard"


def test_guess_candidate_name():
    cv_text = (
        "\n\n"
        "Trần Thị Bích Ngọc\n"
        "Backend Developer | Java Specialist\n"
        "Email: ngoc.tran@example.com | SĐT: 0912345678\n"
        "GitHub: https://github.com/bichngoc/demo\n"
    )
    name = _guess_candidate_name(cv_text)
    assert name == "Trần Thị Bích Ngọc"


def test_annotation_link_detection():
    pdf_bytes = _create_sample_pdf(with_link=True)
    result = extract_cv_info(pdf_bytes)
    assert len(result.github_repos) >= 1
    assert result.github_repos[0].owner == "phamhuy273"
    assert result.github_repos[0].repo_name == "hidden-portfolio"
    assert result.github_repos[0].source == "annotation"


def test_api_extract_cv_endpoint():
    pdf_bytes = _create_sample_pdf(with_link=True)
    response = client.post(
        "/api/v1/cv/extract",
        files={"file": ("my_cv.pdf", pdf_bytes, "application/pdf")}
    )
    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "SUCCESS"
    assert len(res["data"]["github_repos"]) >= 1


def test_api_extract_invalid_file_type():
    response = client.post(
        "/api/v1/cv/extract",
        files={"file": ("cv.docx", b"fake word doc content", "application/vnd.openxmlformats")}
    )
    assert response.status_code == 400
    assert "chỉ hỗ trợ định dạng file PDF" in response.json()["detail"]


def test_api_evaluate_with_cv():
    pdf_bytes = _create_sample_pdf(with_link=True)
    response = client.post(
        "/api/v1/cv/evaluate-with-cv",
        data={"job_id": "job-abc-123", "candidate_name": "Phạm Huy"},
        files={"file": ("cv.pdf", pdf_bytes, "application/pdf")}
    )
    assert response.status_code == 202
    res = response.json()
    assert res["status"] == "ACCEPTED"
    assert "task_id" in res["data"]
