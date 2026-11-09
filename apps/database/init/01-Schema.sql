-- ===================================================================
-- DATABASE INITIALIZATION SCRIPT (DOC-03: TECH SPEC SECTION 3)
-- Project: Evidence-based JD-GitHub Matching (Enterprise MVP)
-- CSDL: PostgreSQL 16 + pgvector 0.7+
-- ===================================================================

-- 1. Kích hoạt tiện ích mở rộng pgvector & UUID
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 2. Bảng 1: job_descriptions (Bản mô tả công việc)
CREATE TABLE IF NOT EXISTS job_descriptions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(255) NOT NULL,
    domain VARCHAR(50) NOT NULL,
    raw_content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 3. Bảng 2: required_skills (Kỹ năng yêu cầu bóc tách từ JD)
CREATE TABLE IF NOT EXISTS required_skills (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    jd_id UUID NOT NULL REFERENCES job_descriptions(id) ON DELETE CASCADE,
    skill_name VARCHAR(100) NOT NULL,
    category VARCHAR(50) NOT NULL,
    importance VARCHAR(20) NOT NULL DEFAULT 'MANDATORY',
    weight NUMERIC(3,2) NOT NULL DEFAULT 1.00,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 4. Bảng 3: repositories (Kho mã nguồn ứng viên & Khóa Caching)
CREATE TABLE IF NOT EXISTS repositories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    repo_url VARCHAR(255) NOT NULL,
    commit_hash VARCHAR(40) NOT NULL,
    default_branch VARCHAR(50) NOT NULL DEFAULT 'main',
    has_pom BOOLEAN NOT NULL DEFAULT FALSE,
    has_package_json BOOLEAN NOT NULL DEFAULT FALSE,
    readme_content TEXT NULL,
    last_parsed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 5. Bảng 4: code_chunks (Khối mã nguồn & Vector nhúng)
CREATE TABLE IF NOT EXISTS code_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    repo_id UUID NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
    file_path VARCHAR(500) NOT NULL,
    class_name VARCHAR(150) NULL,
    method_name VARCHAR(150) NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    context_header TEXT NOT NULL,
    chunk_content TEXT NOT NULL,
    chunk_type VARCHAR(30) NOT NULL DEFAULT 'AST_METHOD',
    embedding VECTOR(1024) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 6. Bảng 5: evaluation_results (Kết quả đánh giá đối sánh)
CREATE TABLE IF NOT EXISTS evaluation_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    jd_id UUID NOT NULL REFERENCES job_descriptions(id) ON DELETE CASCADE,
    repo_id UUID NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
    candidate_name VARCHAR(150) NOT NULL,
    total_score NUMERIC(5,2) NOT NULL DEFAULT 0.00,
    faithfulness_score NUMERIC(3,2) NOT NULL DEFAULT 0.00,
    skills_detail JSONB NOT NULL,
    status VARCHAR(30) NOT NULL DEFAULT 'COMPLETED',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ===================================================================
-- TẠO CHỈ MỤC TỐI ƯU HÓA (INDEXES)
-- ===================================================================

-- Chỉ mục HNSW cho vector nhúng trên bảng code_chunks (Toán tử vector_cosine_ops)
CREATE INDEX IF NOT EXISTS idx_code_chunks_embedding_hnsw
ON code_chunks
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Chỉ mục tìm kiếm nhanh cho khóa Caching theo commit_hash
CREATE INDEX IF NOT EXISTS idx_repositories_commit_hash
ON repositories (repo_url, commit_hash);

-- Chỉ mục khóa ngoại hỗ trợ truy vấn nhanh
CREATE INDEX IF NOT EXISTS idx_required_skills_jd_id ON required_skills (jd_id);
CREATE INDEX IF NOT EXISTS idx_code_chunks_repo_id ON code_chunks (repo_id);
CREATE INDEX IF NOT EXISTS idx_evaluation_results_jd_id ON evaluation_results (jd_id);
CREATE INDEX IF NOT EXISTS idx_evaluation_results_repo_id ON evaluation_results (repo_id);
