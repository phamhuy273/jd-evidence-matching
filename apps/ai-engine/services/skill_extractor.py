"""
Module: apps/ai-engine/services/skill_extractor.py
Chức năng: Bóc tách danh sách kỹ năng kỹ thuật từ Job Description (JD) hoàn toàn bằng Google Gemini LLM.
Không sử dụng từ điển tĩnh hay heuristic fallback.
"""

import json
import logging
import os
import re
import time
from typing import List
import requests

from config import GEMINI_API_KEY
from models.schemas import SkillItem

logger = logging.getLogger("ai-engine.skill_extractor")

PROMPT_TEMPLATE = """Bạn là chuyên gia phân tích yêu cầu tuyển dụng kỹ thuật (Technical Recruiter & Solutions Architect).
Nhiệm vụ của bạn là đọc Tiêu đề công việc (Job Title) và Bản mô tả công việc (Job Description - JD), sau đó trích xuất danh sách các kỹ năng kỹ thuật trọng tâm để đối sánh với mã nguồn GitHub của ứng viên.

Vị trí: {job_title}
Nội dung JD:
\"\"\"
{raw_jd_text}
\"\"\"

YÊU CẦU ĐẦU RA:
Trả về DUY NHẤT một mảng JSON (JSON array), mỗi phần tử có định dạng sau:
[
  {{
    "skill_name": "Tên kỹ năng chuẩn hóa (vd: Java, Spring Boot, React, Docker, PostgreSQL, v.v.)",
    "category": "Một trong các nhóm: LANGUAGE, FRAMEWORK, DATABASE, DEVOPS, TESTING, ARCHITECTURE, TOOL, OTHER",
    "importance": "MANDATORY (nếu là bắt buộc/yêu cầu cốt lõi) hoặc PREFERRED (nếu là ưu tiên/điểm cộng)",
    "weight": 0.25
  }}
]

QUY TẮC QUAN TRỌNG:
1. CHỈ trích xuất kỹ năng kỹ thuật (Hard Skills) có thể chứng minh qua mã nguồn (code). KHÔNG trích xuất kỹ năng mềm (giao tiếp, làm việc nhóm, tiếng Anh, v.v.).
2. Số lượng kỹ năng: từ 3 đến 8 kỹ năng tiêu biểu nhất.
3. Trọng số 'weight' là số thực từ 0.05 đến 0.5, tổng các weight nên xấp xỉ 1.0.
4. Chỉ xuất chuỗi JSON thuần túy, KHÔNG kèm văn bản giải thích ngoài lề.
"""


def extract_skills_from_jd(job_title: str, raw_jd_text: str) -> List[SkillItem]:
    """
    Trích xuất kỹ năng 100% từ Google Gemini LLM.
    Nếu không có API Key hoặc LLM lỗi, sẽ báo lỗi trực tiếp (không dùng từ điển hay fallback).
    """
    api_key = os.getenv("GEMINI_API_KEY") or GEMINI_API_KEY

    if not api_key or not api_key.strip():
        raise ValueError("Chưa cấu hình GEMINI_API_KEY trong file .env!")

    return _call_gemini_api(job_title=job_title, raw_jd_text=raw_jd_text, api_key=api_key.strip())


def _call_gemini_api(job_title: str, raw_jd_text: str, api_key: str) -> List[SkillItem]:
    """Gọi trực tiếp Google Gemini REST API và parse kết quả trả về từ LLM."""
    models_to_try = ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite"]
    prompt = PROMPT_TEMPLATE.format(job_title=job_title, raw_jd_text=raw_jd_text)

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.2
        }
    }
    headers = {"Content-Type": "application/json"}

    last_error = None
    for model_name in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        for attempt in range(3):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=15)
                if response.status_code == 200:
                    data = response.json()
                    candidates = data.get("candidates", [])
                    if not candidates:
                        raise ValueError(f"Gemini {model_name} không trả về candidate nào.")

                    raw_text = candidates[0]["content"]["parts"][0]["text"].strip()
                    if raw_text.startswith("```"):
                        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
                        raw_text = re.sub(r"\s*```$", "", raw_text)

                    skills_data = json.loads(raw_text)
                    if not isinstance(skills_data, list):
                        raise ValueError("Dữ liệu LLM trả về không phải dạng danh sách JSON.")

                    result: List[SkillItem] = []
                    for item in skills_data:
                        name = item.get("skill_name")
                        if not name:
                            continue
                        category = item.get("category", "OTHER").upper()
                        importance = item.get("importance", "MANDATORY").upper()
                        if importance not in ("MANDATORY", "PREFERRED"):
                            importance = "MANDATORY"
                        weight = float(item.get("weight", 0.2))

                        result.append(SkillItem(
                            skill_name=name,
                            category=category,
                            importance=importance,
                            weight=round(weight, 2)
                        ))

                    if result:
                        logger.info(f"Gemini LLM ({model_name}) bóc tách thành công {len(result)} kỹ năng.")
                        return result

                elif response.status_code == 503:
                    time.sleep(1)
                    last_error = f"HTTP 503 Spike ({model_name})"
                else:
                    last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                    break
            except Exception as e:
                last_error = str(e)
                time.sleep(1)

    raise RuntimeError(f"Lỗi gọi Gemini LLM: {last_error}")
