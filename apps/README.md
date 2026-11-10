# Apps - Enterprise Platform Ecosystem

Thư mục `apps/` chứa toàn bộ mã nguồn của hệ thống phần mềm doanh nghiệp 3 tầng phục vụ việc đối sánh năng lực ứng viên từ Job Description (JD) với mã nguồn GitHub.

---

## 🏛️ Kiến Trúc Các Tầng Dịch Vụ

```text
apps/
├── backend-service/      [Port 8080] Spring Boot 3, Java 21: Quản lý nghiệp vụ, API, xác thực, điều phối
├── ai-engine/            [Port 8001] FastAPI, Python 3.11+: Phân tích cú pháp AST Tree-sitter, nhúng BGE-M3
└── database/             [Port 5432] PostgreSQL 16 + pgvector: Lưu trữ thực thể và vector ngữ nghĩa
```

---

## 🚀 Hướng Dẫn Khởi Chạy Nhanh

### 1. Khởi chạy toàn bộ hệ thống bằng Docker Compose

Từ thư mục gốc dự án hoặc thư mục `apps/`:

```bash
docker compose up -d
```

### 2. Kiểm tra trạng thái hoạt động (Healthcheck)

* **Backend Service (Spring Boot):** `http://localhost:8080/actuator/health`
* **AI Engine (FastAPI):** `http://localhost:8001/health`
* **PostgreSQL pgvector:** `localhost:5432` (Database: `evidence_db`)

---

## ⚙️ Cấu Hình Môi Trường

Sao chép `.env.example` thành `.env` tại thư mục gốc và cấu hình các thông số bảo mật, API key và kết nối database trước khi vận hành thực tế.
