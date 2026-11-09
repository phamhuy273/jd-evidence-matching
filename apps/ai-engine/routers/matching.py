from fastapi import APIRouter, BackgroundTasks, HTTPException, status
import uuid
from models.schemas import (
    EvaluateAsyncRequest, EvaluateAsyncData,
    TaskStatusData, JobResultsData, StandardResponse
)
from services.matching_worker import tasks_store, jobs_ranking_store, run_matching_pipeline

router = APIRouter(prefix="/api/v1/matching", tags=["Matching"])

@router.post("/evaluate-async", status_code=status.HTTP_202_ACCEPTED, response_model=StandardResponse[EvaluateAsyncData])
async def evaluate_async(request: EvaluateAsyncRequest, background_tasks: BackgroundTasks):
    """
    DOC-03 Section 4.3 & 1.2.2:
    Kích hoạt tiến trình đối sánh kho mã nguồn ứng viên ngầm qua BackgroundTasks,
    phản hồi HTTP 202 Accepted kèm task_id dưới 100ms.
    """
    task_id = str(uuid.uuid4())

    # Khởi tạo trạng thái tác vụ trong kho lưu trữ
    tasks_store[task_id] = {
        "job_id": request.job_id,
        "candidate_name": request.candidate_name,
        "progress_status": "PROCESSING",
        "progress_percent": 0
    }

    # Đưa tác vụ nặng vào hàng đợi xử lý ngầm
    background_tasks.add_task(
        run_matching_pipeline,
        task_id=task_id,
        job_id=request.job_id,
        candidate_name=request.candidate_name,
        github_repo_url=request.github_repo_url,
        commit_hash=request.commit_hash
    )

    return StandardResponse(
        code=status.HTTP_202_ACCEPTED,
        status="ACCEPTED",
        message="Đã tiếp nhận yêu cầu phân tích.",
        data=EvaluateAsyncData(
            task_id=task_id,
            status="PROCESSING",
            is_cached=False
        )
    )

@router.get("/tasks/{task_id}/status", response_model=StandardResponse[TaskStatusData])
async def get_task_status(task_id: str):
    """
    DOC-03 Section 4.4 - Endpoint 1:
    Truy vấn trạng thái thực thi hiện tại của worker ngầm để cập nhật thanh tiến trình.
    """
    if task_id not in tasks_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Không tìm thấy task_id: {task_id}"
        )

    task_info = tasks_store[task_id]
    return StandardResponse(
        code=status.HTTP_200_OK,
        status="SUCCESS",
        message="Truy vấn trạng thái tác vụ thành công.",
        data=TaskStatusData(
            task_id=task_id,
            progress_status=task_info["progress_status"],
            progress_percent=task_info["progress_percent"]
        )
    )

@router.get("/jobs/{job_id}/results", response_model=StandardResponse[JobResultsData])
async def get_job_results(job_id: str):
    """
    DOC-03 Section 4.4 - Endpoint 2:
    Lấy danh sách ứng viên đã được đối sánh cho vị trí JD, xếp hạng theo điểm tổng.
    """
    rankings = jobs_ranking_store.get(job_id, [])
    return StandardResponse(
        code=status.HTTP_200_OK,
        status="SUCCESS",
        message="Truy vấn kết quả xếp hạng thành công.",
        data=JobResultsData(
            job_id=job_id,
            ranking=rankings
        )
    )
