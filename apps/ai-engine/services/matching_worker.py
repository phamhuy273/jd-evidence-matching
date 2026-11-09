import asyncio
import logging
import uuid
from typing import Dict, Any

logger = logging.getLogger("ai-engine.worker")

# Quản lý trạng thái tác vụ ngầm trong bộ nhớ (In-memory Task Store)
tasks_store: Dict[str, Dict[str, Any]] = {}

# Kết quả xếp hạng mẫu theo job_id
jobs_ranking_store: Dict[str, list] = {}

async def run_matching_pipeline(task_id: str, job_id: str, candidate_name: str, github_repo_url: str, commit_hash: str):
    """
    Quy trình phân tích ngầm theo Section 1.2.2 & Section 2 Tech Spec:
    1. Kiểm tra cache commit_hash (nếu có -> cache hit)
    2. Clone repo / đọc mã nguồn
    3. Phân đoạn mã nguồn AST qua tree-sitter (hoặc fallback line-based)
    4. Nhúng vector qua BGE-M3 (1024 dims)
    5. Truy vấn Top-k qua pgvector cosine ops
    6. Đánh giá kỹ năng bằng LLM Gemini 1.5 Flash
    7. Lưu kết quả vào evaluation_results và cập nhật COMPLETED.
    """
    try:
        logger.info(f"Bắt đầu tác vụ nền task_id={task_id} cho ứng viên={candidate_name}")
        tasks_store[task_id]["progress_status"] = "PROCESSING"
        tasks_store[task_id]["progress_percent"] = 20

        # Giả lập thời gian xử lý AST chunking & vector search ngầm
        await asyncio.sleep(1)
        tasks_store[task_id]["progress_percent"] = 60

        await asyncio.sleep(1)
        tasks_store[task_id]["progress_percent"] = 100
        tasks_store[task_id]["progress_status"] = "COMPLETED"

        # Khởi tạo kết quả đánh giá mẫu lưu vào store
        sample_result = {
            "rank": 1,
            "candidate_name": candidate_name,
            "total_score": 88.5,
            "faithfulness_score": 0.92,
            "skills_assessment": [
                {
                    "skill_name": "Spring Boot",
                    "score": 95.0,
                    "evidence_code_chunk": {
                        "file_path": "src/main/java/vn/edu/uit/jdmatching/controller/OrderController.java",
                        "code_snippet": "@RestController\n@RequestMapping(\"/api/orders\")\npublic class OrderController {...}"
                    },
                    "llm_explanation": "Ứng viên thể hiện hiểu biết sâu sắc về kiến trúc Spring Boot REST Controller và Dependency Injection."
                },
                {
                    "skill_name": "Java",
                    "score": 90.0,
                    "evidence_code_chunk": {
                        "file_path": "src/main/java/vn/edu/uit/jdmatching/service/OrderService.java",
                        "code_snippet": "public OrderResponse createOrder(CreateOrderRequest request) {...}"
                    },
                    "llm_explanation": "Sử dụng Java 17 features, xử lý Exception và Stream API chuẩn mực."
                }
            ]
        }
        if job_id not in jobs_ranking_store:
            jobs_ranking_store[job_id] = []
        jobs_ranking_store[job_id].append(sample_result)

        logger.info(f"Hoàn thành tác vụ nền task_id={task_id}")
    except Exception as e:
        logger.error(f"Lỗi thực thi tác vụ nền {task_id}: {e}")
        tasks_store[task_id]["progress_status"] = "FAILED"
