# JD–GitHub Evidence Matching (Đồ Án 1 - UIT & IEEE SANER 2027)

> **Hệ thống Đối sánh Năng lực Ứng viên từ Mã nguồn GitHub dựa trên Mô hình RAG và Phân đoạn AST**  
> *Đề tài Đồ án 1 / Nghiên cứu Khoa học – Trường Đại học Công nghệ Thông tin (UIT - VNU-HCM)*  
> *Bài báo khoa học: IEEE SANER 2027 (ERA Track)*

---

## 🏛️ Cấu trúc Dự án (Project Monorepo Structure)

Dự án được tổ chức gọn gàng thành **3 phân hệ độc lập**:

```text
jd-evidence-matching/
│
├── 📁 apps/                  ──► [HỆ THỐNG ỨNG DỤNG DOANH NGHIỆP]
│   ├── frontend/             (Giao diện Web: React + Vite + TailwindCSS)
│   ├── backend-service/      (Cổng nghiệp vụ: Spring Boot 3 & Java 21, Port 8080)
│   ├── ai-engine/            (Tầng phân tích AI: FastAPI & Tree-sitter AST, Port 8001)
│   ├── database/             (Cơ sở dữ liệu: PostgreSQL 16 + pgvector)
│   └── docker-compose.yml    (Khởi chạy toàn bộ hệ thống Web)
│
├── 📁 research/              ──► [NGHIÊN CỨU KHOA HỌC & SANER 2027 REPLICATION]
│   ├── dataset/              (872 chunks, 882 ground-truth pairs, 25 JDs, 40 repos)
│   ├── scripts/              (Kịch bản tái lập: calculate_kappa, evaluate_retrieval...)
│   ├── tests/                (Bộ 100/100 unit tests tự động - Pytest)
│   ├── paper/                (Bản thảo LaTeX & PDF bài báo 5 trang đã nộp)
│   ├── report/               (Báo cáo kỹ thuật chi tiết & Tables compendium)
│   ├── requirements.txt      (Môi trường Python chạy thực nghiệm)
│   └── README.md             (Hướng dẫn chi tiết tái lập 1-click cho Reviewer)
│
├── 📁 docs/                  ──► [HỒ SƠ HỌC THUẬT & TÀI LIỆU ĐỒ ÁN UIT]
│   └── Đồ án 1/              (Tech Spec DOC-03, Đặc tả yêu cầu, Đề cương, Mẫu UIT...)
│
├── docker-compose.yml        (Lệnh 1-click chạy toàn bộ dịch vụ web từ root)
├── .env                      (Cấu hình biến môi trường)
└── .gitignore                (Quy tắc loại trừ mã nguồn rác)
```

---

## 🚀 Hướng Dẫn Khởi Chạy

### 1. Khởi chạy Ứng dụng Web Doanh nghiệp (`apps/`)
Yêu cầu đã cài đặt **Docker** và **Docker Compose**:
```bash
# Khởi động toàn bộ Database, AI Engine và Backend
docker-compose up -d

# Hoặc khởi động từ thư mục apps:
cd apps
docker-compose up -d
```
* **Backend API (Spring Boot):** `http://localhost:8080`
* **AI Engine Swagger UI (FastAPI):** `http://localhost:8001/docs`
* **PostgreSQL pgvector:** `localhost:5433`

---

### 2. Kiểm thử & Tái lập Nghiên cứu Khoa học (`research/`)
Yêu cầu Python 3.11+:
```bash
cd research

# Cài đặt thư viện thực nghiệm
pip install -r requirements.txt

# Chạy toàn bộ 100 unit tests kiểm định tính toàn vẹn và không gán cứng
python -m pytest tests/

# Tái lập Bảng 1, Bảng 2 và kiểm định thống kê Wilcoxon
python scripts/evaluate_retrieval.py
```
*Chi tiết các kịch bản đo đạc xem tại [`research/README.md`](research/README.md).*

---

## 👥 Nhóm Tác Giả & Nghiên Cứu
* **Sinh viên thực hiện:**
  1. Phạm Viết Quang Huy – MSSV: 24520692 (Khoa CNPM, UIT)
  2. Nguyễn Ngọc Thu An  – MSSV: 24520063 (Khoa CNPM, UIT)
* **Cán bộ hướng dẫn:** ThS. Trần Thị Hồng Yến
