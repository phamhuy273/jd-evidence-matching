# JD–GitHub Evidence Matching (Đồ Án 1 - UIT & IEEE SANER 2027)

<div align="center">

[![Tests](https://img.shields.io/badge/Tests-100%2F100%20PASSED-brightgreen.svg?style=flat-square)](research/tests/)
[![Paper](https://img.shields.io/badge/IEEE%20SANER%202027-ERA%20Track%20(Submitted)-blue.svg?style=flat-square)](research/paper/main.pdf)
[![Data](https://img.shields.io/badge/Ground%20Truth-882%20pairs%20(%CE%BA%3D0.9822)-purple.svg?style=flat-square)](research/dataset/ground_truth_final.csv)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg?style=flat-square)](research/requirements.txt)
[![Java](https://img.shields.io/badge/Java-21%20(Spring%20Boot%203)-orange.svg?style=flat-square)](apps/backend-service/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16%20%2B%20pgvector-blue.svg?style=flat-square)](apps/database/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](research/LICENSE)

**HỆ THỐNG ĐỐI SÁNH NĂNG LỰC ỨNG VIÊN TỪ MÃ NGUỒN GITHUB DỰA TRÊN MÔ HÌNH RAG VÀ PHÂN ĐOẠN AST**  
*Đề tài Đồ án 1 / Nghiên cứu Khoa học Sinh viên – Trường Đại học Công nghệ Thông tin (UIT - VNU-HCM)*  
*Công trình nghiên cứu: "An Empirical Evaluation of Source Code Chunking and Context Headers for Requirement Evidence Retrieval" (IEEE SANER 2027 ERA Track)*

</div>

---

## 📖 Giới Thiệu Dự Án

Trong quy trình tuyển dụng kỹ sư phần mềm, việc thẩm định năng lực thực tế qua CV thường gặp hiện tượng phóng đại kỹ năng (keyword stuffing). Thay vào đó, việc đối sánh yêu cầu công việc (Job Description - JD) trực tiếp với mã nguồn trên GitHub của ứng viên mang lại bằng chứng thực nghiệm khách quan nhất.

Dự án này giải quyết bài toán trên thông qua hai trụ cột:
1. **Trụ cột Kỹ thuật Công nghệ (Enterprise Platform - `apps/`):** Hệ thống phần mềm doanh nghiệp 3 tầng hỗ trợ nhà tuyển dụng tải lên JD, quét kho lưu trữ GitHub của ứng viên, tự động phân tích cú pháp mã nguồn bằng **Tree-sitter AST**, tính toán vector nhúng ngữ nghĩa qua **BGE-M3** và đối sánh tìm ra bằng chứng năng lực thực tế.
2. **Trụ cột Nghiên cứu Khoa học (Research & Empirical Benchmark - `research/`):** Thực nghiệm ma trận nhân tố $2 \times 2$ (AST vs. Line-based 50 LOC $\times$ Có Header vs. Không Header) trên **40 repository** (Java Spring Boot & React/TypeScript), **25 JDs**, đánh giá qua **882 cặp nhãn vàng mù đôi** ($\kappa_w = 0.9822$).

---

## 🏛️ Cấu Trúc Dự Án (Monorepo Architecture)

Dự án được phân tách rạch ròi thành 3 miền độc lập, sạch sẽ và chuyên nghiệp:

```text
jd-evidence-matching/
│
├── 📁 apps/                  ──► [HỆ THỐNG PHẦN MỀM DOANH NGHIỆP]
│   ├── frontend/             (Giao diện Web: React + Vite + TailwindCSS - Đang phát triển)
│   ├── backend-service/      (Cổng nghiệp vụ: Spring Boot 3 & Java 21, Port 8080)
│   ├── ai-engine/            (Tầng AI & Parser: FastAPI, Tree-sitter AST, BGE-M3, Port 8001)
│   ├── database/             (Cơ sở dữ liệu Vector: PostgreSQL 16 + pgvector init)
│   └── docker-compose.yml    (Khởi chạy toàn bộ hệ sinh thái ứng dụng)
│
├── 📁 research/              ──► [NGHIÊN CỨU THỰC NGHIỆM & SANER 2027 REPLICATION]
│   ├── dataset/              (Kho ngữ liệu kép 872 chunks, 882 ground-truth pairs, 12 retrieval runs)
│   ├── scripts/              (Kịch bản đo lường: evaluate_retrieval, evaluate_syntax, calculate_kappa...)
│   ├── tests/                (Bộ 100/100 unit tests tự động - kiểm tra toán học và tính bất biến)
│   ├── paper/                (Bản thảo LaTeX modular & File PDF 5 trang chính thức đã nộp)
│   ├── report/               (Báo cáo kỹ thuật chi tiết results_report.pdf & tables_compendium.pdf)
│   ├── requirements.txt      (Môi trường Python phục vụ tái lập nghiên cứu)
│   ├── run_all.bat / .sh     (Lệnh 1-click tái lập toàn bộ thí nghiệm)
│   └── README.md             (Đặc tả chi tiết 34.5 KB dành cho Reviewer hội nghị)
│
├── 📁 docs/                  ──► [HỒ SƠ HỌC THUẬT & TÀI LIỆU ĐỒ ÁN UIT]
│   └── Đồ án 1/              (Tài liệu Thiết kế kỹ thuật DOC-03, Đặc tả yêu cầu FR/NFR, Đề cương)
│
├── docker-compose.yml        (Lệnh 1-click chạy toàn bộ dịch vụ web từ root)
├── pytest.ini                (Cấu hình test tự động từ thư mục gốc)
├── .env                      (Biến môi trường)
└── .gitignore                (Quy tắc loại trừ mã nguồn rác)
```

---

## 📊 Kết Quả Nghiên Cứu Thực Nghiệm Cốt Lõi (SANER 2027)

Thực nghiệm trên 25 JDs, 40 repositories và 3 tầng truy xuất (BM25, Dense Bi-Encoder BGE-M3, Cross-Encoder Reranker) mang lại những phát hiện quan trọng:

1. **Hiệu ứng đột phá của Progressive Context Header đối với AST:**  
   Gắn Context Header (`// File: ... // Class: ... // Method: ...`) giúp AST tăng vọt $\text{NDCG@10}$ từ **0.341 lên 0.487** ($\Delta = +0.146, p_{\text{Holm}} < 0.001$, thắng 23/25 JDs).
2. **Ưu thế mật độ token của Line-based (50 LOC):**  
   Cửa sổ 50 dòng có Header đạt hiệu năng truy xuất tổng thể cao nhất ($\text{NDCG@10} = \mathbf{0.621}$).
3. **Bảo toàn ranh giới cú pháp:**  
   AST bảo toàn **87.2% – 90.9%** ranh giới cú pháp nguyên vẹn (với tỷ lệ fallback lỗi là **0.0%**), trong khi Line-based chém đứt **56.0%** ranh giới cú pháp (cắt ngang hàm, `if`, `try-catch`).

| Cấu hình Thực nghiệm (Reranking Tier) | Context Header | NDCG@10 [95% CI] | NDCG@1 | MRR@10 | bpref | Tính Toàn Vẹn Cú Pháp |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Line-based (50 LOC)** | Có | **0.621** [0.555, 0.682] | **0.800** | 0.950 | **0.353** | 44.0% (56.0% bị vỡ) |
| **Line-based (50 LOC)** | Không | 0.580 [0.505, 0.649] | **0.800** | **0.953** | 0.300 | 44.0% (56.0% bị vỡ) |
| **AST Progressive (Method-level)** | Có | 0.487 [0.419, 0.551] | 0.740 | 0.896 | 0.241 | **87.2% – 90.9%** (Nguyên vẹn) |
| **AST Progressive (Method-level)** | Không | 0.341 [0.263, 0.421] | 0.340 | 0.699 | 0.170 | **87.2% – 90.9%** (Nguyên vẹn) |

---

## 🚀 Hướng Dẫn Cài Đặt & Khởi Chạy

### 1. Khởi chạy Ứng dụng Web Doanh nghiệp (`apps/`)
Yêu cầu đã cài đặt **Docker** và **Docker Compose**:
```bash
# Khởi động đồng bộ PostgreSQL (pgvector), FastAPI (AI Engine) và Spring Boot (Backend):
docker-compose up -d
```
* **Backend Service (Spring Boot):** `http://localhost:8080`
* **AI Engine Swagger API (FastAPI):** `http://localhost:8001/docs`
* **Cơ sở dữ liệu PostgreSQL Vector:** `localhost:5433`

---

### 2. Tái Lập Thí Nghiệm & Chạy Bộ Kiểm Thử (`research/`)
Yêu cầu Python 3.11+:
```bash
# Cài đặt các thư viện cần thiết
pip install -r research/requirements.txt

# 1. Chạy toàn bộ bộ 100/100 unit tests (Toán học, Mã băm bảo toàn nhãn, Kiểm toán văn bản):
python -m pytest

# 2. Đo hệ số đồng thuận liên thẩm định viên (Bảng 3.1, Kappa = 0.9822):
python research/scripts/calculate_kappa.py

# 3. Đo lường ranh giới cú pháp Tree-sitter & độ dài token (Bảng 3.2):
python research/scripts/evaluate_syntax.py

# 4. Tái lập ma trận 2x2 Factorial & Kiểm định Wilcoxon Holm-Bonferroni (Bảng 1, Bảng 2):
python research/scripts/evaluate_retrieval.py

# 5. Chạy kịch bản kiểm toán toàn diện tính trung thực khoa học:
python research/scripts/audit_code_and_data_integrity.py
```

---

## 👥 Nhóm Tác Giả & Đơn Vị Thực Hiện

* **Sinh viên nghiên cứu & thực hiện (Khoa Công nghệ Phần mềm, UIT - VNU-HCM):**
  * **Phạm Viết Quang Huy** – MSSV: 24520692
  * **Nguyễn Ngọc Thu An**  – MSSV: 24520063
* **Cán bộ hướng dẫn khoa học:**
  * **ThS. Trần Thị Hồng Yến** – Giảng viên Khoa Công nghệ Phần mềm, Trường Đại học Công nghệ Thông tin, ĐHQG-HCM
* **Trích dẫn nghiên cứu (BibTeX):**
```bibtex
@inproceedings{huy2027empirical,
  title={An Empirical Evaluation of Source Code Chunking and Context Headers for Requirement Evidence Retrieval},
  author={Anonymous Authors},
  booktitle={Proceedings of the 34th IEEE International Conference on Software Analysis, Evolution and Reengineering (SANER 2027) - Early Research Achievements (ERA) Track},
  year={2027}
}
```
