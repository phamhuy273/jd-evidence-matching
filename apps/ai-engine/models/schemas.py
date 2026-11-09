from pydantic import BaseModel, Field
from typing import List, Optional, Any, Generic, TypeVar
from datetime import datetime, timezone

T = TypeVar("T")

class StandardResponse(BaseModel, Generic[T]):
    code: int
    status: str
    message: str = "Thao tác thực hiện thành công."
    data: Optional[T] = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

# ===================================================================
# Extract Skills Models 
# ===================================================================
class ExtractSkillsRequest(BaseModel):
    job_title: str
    raw_jd_text: str

class SkillItem(BaseModel):
    skill_name: str
    category: str
    importance: str = "MANDATORY"
    weight: float = 1.0

class ExtractSkillsData(BaseModel):
    skills: List[SkillItem]

# ===================================================================
# Evaluate Async Models
# ===================================================================
class EvaluateAsyncRequest(BaseModel):
    job_id: str
    candidate_name: str
    github_repo_url: str
    commit_hash: str

class EvaluateAsyncData(BaseModel):
    task_id: str
    status: str = "PROCESSING"
    is_cached: bool = False

# ===================================================================
# Task Status & Results Models (DOC-03 Section 4.4)
# ===================================================================
class TaskStatusData(BaseModel):
    task_id: str
    progress_status: str  # PROCESSING, COMPLETED, FAILED
    progress_percent: int

class EvidenceCodeChunk(BaseModel):
    file_path: str
    code_snippet: str

class SkillAssessment(BaseModel):
    skill_name: str
    score: float
    evidence_code_chunk: EvidenceCodeChunk
    llm_explanation: str

class CandidateRanking(BaseModel):
    rank: int
    candidate_name: str
    total_score: float
    faithfulness_score: float
    skills_assessment: List[SkillAssessment]

class JobResultsData(BaseModel):
    job_id: str
    ranking: List[CandidateRanking]
