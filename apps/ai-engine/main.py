import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import jobs, matching
from config import AI_ENGINE_PORT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ai-engine.main")

app = FastAPI(
    title="Evidence-based JD-GitHub Matching - AI Engine",
    description="Dịch vụ phân tích mã nguồn AST, nhúng vector BGE-M3 và đối sánh năng lực kỹ thuật với JD",
    version="1.0.0"
)

# Cấu hình CORS cho phép Spring Boot Backend và React Web truy cập
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gắn các router nghiệp vụ
app.include_router(jobs.router)
app.include_router(matching.router)

@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "UP",
        "service": "ai-engine",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=AI_ENGINE_PORT, reload=True)
