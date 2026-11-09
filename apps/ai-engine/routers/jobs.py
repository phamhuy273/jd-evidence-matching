from fastapi import APIRouter, HTTPException, status
from models.schemas import ExtractSkillsRequest, ExtractSkillsData, SkillItem, StandardResponse

router = APIRouter(prefix="/api/v1/jobs", tags=["Jobs"])

@router.post("/extract-skills", response_model=StandardResponse[ExtractSkillsData])
async def extract_skills(request: ExtractSkillsRequest):
    """
    DOC-03 Section 4.2:
    Tiếp nhận văn bản JD, sử dụng LLM bóc tách danh sách kỹ năng bắt buộc và ưu tiên.
    """
    if not request.raw_jd_text or len(request.raw_jd_text.split()) < 5:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Nội dung JD quá ngắn để trích xuất kỹ năng."
        )

    # Skeleton: Trả về kết quả bóc tách kỹ năng chuẩn mẫu
    skills = [
        SkillItem(skill_name="Java", category="LANGUAGE", importance="MANDATORY", weight=0.3),
        SkillItem(skill_name="Spring Boot", category="FRAMEWORK", importance="MANDATORY", weight=0.3),
        SkillItem(skill_name="PostgreSQL", category="DATABASE", importance="MANDATORY", weight=0.2),
        SkillItem(skill_name="Docker", category="DEVOPS", importance="PREFERRED", weight=0.1),
        SkillItem(skill_name="JUnit 5", category="TESTING", importance="PREFERRED", weight=0.1)
    ]

    return StandardResponse(
        code=status.HTTP_200_OK,
        status="SUCCESS",
        message="Trích xuất kỹ năng từ JD thành công.",
        data=ExtractSkillsData(skills=skills)
    )
