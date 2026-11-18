"""
Module: apps/ai-engine/routers/cv.py
Chức năng: Các API endpoint xử lý CV (PDF) và kích hoạt luồng đối sánh tự động.
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status, BackgroundTasks
from typing import Optional
import uuid
import logging

from models.schemas import (
    StandardResponse,
    CVExtractData,
    EvaluateAsyncData
)
from services.cv_extractor import extract_cv_info
from services.matching_worker import tasks_store, run_matching_pipeline

logger = logging.getLogger("ai-engine.routers.cv")

router = APIRouter(prefix="/api/v1/cv", tags=["CV Processing"])


@router.post("/extract", response_model=StandardResponse[CVExtractData])
async def extract_cv_endpoint(file: UploadFile = File(...)):
    """
    Endpoint 1: Upload file CV (PDF) -> Trả về thông tin ứng viên và danh sách GitHub repos.
    Dùng để hiển thị preview trên UI trước khi người dùng bấm xác nhận phân tích.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Hệ thống hiện tại chỉ hỗ trợ định dạng file PDF."
        )

    try:
        content = await file.read()
        extracted_data = extract_cv_info(content)
        
        return StandardResponse(
            code=status.HTTP_200_OK,
            status="SUCCESS",
            message="Bóc tách thông tin CV thành công.",
            data=extracted_data
        )
    except Exception as e:
        logger.error(f"Lỗi xử lý file CV {file.filename}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi đọc file CV: {str(e)}"
        )


@router.post("/evaluate-with-cv", status_code=status.HTTP_202_ACCEPTED, response_model=StandardResponse[EvaluateAsyncData])
async def evaluate_with_cv_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    job_id: str = Form(...),
    candidate_name: Optional[str] = Form(None)
):
    """
    Endpoint 2: Upload CV + job_id -> Tự động trích xuất GitHub repo & đưa vào hàng đợi đánh giá ngầm.
    Phản hồi ngay HTTP 202 Accepted kèm task_id.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Vui lòng tải lên file định dạng PDF."
        )

    content = await file.read()
    extracted_data = extract_cv_info(content)

    if not extracted_data.github_repos:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Không tìm thấy liên kết GitHub nào trong CV này để đối sánh năng lực kỹ thuật."
        )

    # Ưu tiên tên người dùng nhập, nếu không có thì lấy tên tự động đoán từ CV
    final_name = candidate_name or extracted_data.candidate_name or "Ứng viên ẩn danh"
    
    # Chọn repo đầu tiên tìm thấy để đánh giá
    primary_repo = extracted_data.github_repos[0]
    task_id = str(uuid.uuid4())

    # Lưu task vào store
    tasks_store[task_id] = {
        "job_id": job_id,
        "candidate_name": final_name,
        "progress_status": "PROCESSING",
        "progress_percent": 0
    }

    # Đưa vào BackgroundTasks xử lý ngầm
    background_tasks.add_task(
        run_matching_pipeline,
        task_id=task_id,
        job_id=job_id,
        candidate_name=final_name,
        github_repo_url=primary_repo.url,
        commit_hash="HEAD"
    )

    return StandardResponse(
        code=status.HTTP_202_ACCEPTED,
        status="ACCEPTED",
        message=f"Đã phát hiện repo {primary_repo.url} và tiếp nhận yêu cầu phân tích.",
        data=EvaluateAsyncData(
            task_id=task_id,
            status="PROCESSING",
            is_cached=False
        )
    )
