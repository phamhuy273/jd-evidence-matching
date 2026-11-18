from fastapi import APIRouter, HTTPException, status
from models.schemas import ExtractSkillsRequest, ExtractSkillsData, SkillItem, StandardResponse
from services.skill_extractor import extract_skills_from_jd

router = APIRouter(prefix="/api/v1/jobs", tags=["Jobs"])

@router.post("/extract-skills", response_model=StandardResponse[ExtractSkillsData])
async def extract_skills(request: ExtractSkillsRequest):
    """
    DOC-03 Section 4.2:
    Tiếp nhận văn bản JD, sử dụng Gemini 1.5 Flash (hoặc Heuristic fallback)
    để bóc tách danh sách kỹ năng kỹ thuật, phân loại và phân bổ trọng số.
    """
    if not request.raw_jd_text or len(request.raw_jd_text.split()) < 5:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Nội dung JD quá ngắn để trích xuất kỹ năng."
        )

    # Gọi bộ bóc tách kỹ năng qua Gemini LLM
    try:
        skills = extract_skills_from_jd(
            job_title=request.job_title,
            raw_jd_text=request.raw_jd_text
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi gọi Gemini LLM: {str(e)}"
        )

    return StandardResponse(
        code=status.HTTP_200_OK,
        status="SUCCESS",
        message="Trích xuất kỹ năng từ JD thành công.",
        data=ExtractSkillsData(skills=skills)
    )
