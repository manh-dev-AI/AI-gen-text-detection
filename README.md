# AI-Generated Text Detection

Dự án nghiên cứu và triển khai hệ thống phát hiện văn bản do AI tạo ra bằng Machine Learning và Natural Language Processing (NLP).

Mục tiêu là phân loại một đoạn văn bản thành **Human** hoặc **AI-generated**, so sánh các phương pháp khác nhau, sau đó triển khai mô hình tối ưu trực tiếp vào **Chrome Extension** để phân tích văn bản người dùng chọn trên trang web.

---

## 📊 Dữ liệu Thực nghiệm

Dataset chính: `data/AI_Human.csv` (~487k mẫu, đã chia phân tầng thành `train.csv` và `val.csv` tại `data/processed/`).

| Cột | Mô tả |
| :--- | :--- |
| `text` | Nội dung văn bản đầu vào |
| `generated` | Nhãn phân loại: `0` là Human, `1` là AI-generated |

---

## 🚀 Các Pipeline Nghiên cứu & Kết quả Thực nghiệm

| Pipeline | Phương pháp & Mô hình | Macro-F1 | AI Precision | AI Recall | Latency | Ghi chú & Đặc điểm nổi bật |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Pipeline 1** | **Word TF-IDF + Linear SVM** | `0.9993` | `0.9996` | `0.9986` | **1.77 ms** | Baseline Unigram + Bigram, đã đóng gói chạy Offline trên Extension |
| **Pipeline 2** | **Word + Char TF-IDF + Linear SVM** | **`0.9996`** | **`0.9997`** | **`0.9993`** | `72.82 ms` | Bắt hình thái từ và n-gram ký tự vi mô, độ chính xác cao nhất |
| **Pipeline 3** | **Stylometry + LSA SVD + LightGBM** | `0.9928` | `0.9943` | `0.9877` | **3.12 ms** | 20 đặc trưng phong cách ngôn ngữ, **giải thích được mô hình**, kháng học vẹt chủ đề |
| **Pipeline 4** | **Fine-tuned Transformer** | *Đang làm* | *—* | *—* | *—* | Pretrained Transformer (DistilRoBERTa / DeBERTa) |

> 📖 **Xem chi tiết phương pháp luận, công thức và biểu đồ:** [`docs/pipelines_methodology.md`](docs/pipelines_methodology.md)

---

## 📁 Cấu trúc Thư mục

- `notebooks/`: Notebooks EDA và thực nghiệm huấn luyện từng pipeline (`01` đến `06`).
- `data/`: Dataset gốc và dữ liệu tiền xử lý (`data/processed/`).
- `models/`: Artifacts của các pipeline (`pipeline1/`, `pipeline2/`, `pipeline3/`).
- `results/`: Bảng đánh giá [`results/evaluation.csv`](results/evaluation.csv) và biểu đồ trực quan hóa (`results/figures/`).
- `docs/`: Báo cáo EDA ([`info_data.md`](docs/info_data.md)), phương pháp luận chi tiết ([`pipelines_methodology.md`](docs/pipelines_methodology.md)), và bảng chọn mô hình ([`model_selection.md`](docs/model_selection.md)).
- `extension/`: Chrome Extension giao tiếp trực tiếp với mô hình (hoạt động 100% Offline).
- `scripts/`: Scripts tự động hóa huấn luyện và xuất mô hình.
- `api/`: API FastAPI tùy chọn nếu cần tích hợp hệ thống bên ngoài.

---

## 🛠️ Hướng dẫn Cài đặt & Thiết lập Môi trường vào Máy

Dành cho nhà phát triển muốn chạy lại thực nghiệm, mở Jupyter Notebooks hoặc huấn luyện lại mô hình:

### 1. Yêu cầu Tiên quyết (Prerequisites)
- **Python:** Phiên bản `3.10` trở lên (đã kiểm thử mượt mà trên Python 3.10 – 3.14).
- **Git:** Đã cài đặt trên máy.

### 2. Các bước Cài đặt Chi tiết

1. **Clone repository về máy:**
   ```bash
   git clone https://github.com/manh-dev-AI/AI-gen-text-detection.git
   cd AI-gen-text-detection
   ```

2. **Khởi tạo và Kích hoạt Môi trường ảo (Virtual Environment):**
   - **Trên Windows (PowerShell):**
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
     > *Nếu gặp lỗi PowerShell chặn script ExecutionPolicy, chạy lệnh:*  
     > `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`
   - **Trên macOS / Linux:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Cài đặt các gói phụ thuộc (Dependencies):**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 💻 Hướng dẫn Chạy Dự án

### Cách 1: Sử dụng Chrome Extension (Tiện lợi nhất - Không cần chạy Server)

Chrome Extension đã được tích hợp sẵn mô hình Word TF-IDF + Linear SVM ([`extension/model_pipeline1.json`](extension/model_pipeline1.json)), chạy trực tiếp 100% trong trình duyệt (Offline) mà **không cần cài đặt Python hay mở terminal**:

1. Mở trình duyệt Google Chrome (hoặc Edge / Brave / Cốc Cốc), truy cập:
   ```text
   chrome://extensions
   ```
2. Bật công tắc **Developer mode** (Chế độ cho nhà phát triển) ở góc trên bên phải.
3. Nhấn **Load unpacked** (Tải tiện ích đã giải nén) và chọn thư mục [`extension/`](extension).
4. Bôi đen văn bản bất kỳ trên trang web, mở extension, chọn **Lấy phần bôi đen** rồi bấm **Phân tích** (hoặc dán văn bản trực tiếp).

### Cách 2: Chạy Jupyter Notebooks (Nghiên cứu & Trực quan hóa)

Để xem báo cáo EDA và tự tay chạy lại các pipeline huấn luyện:
```powershell
jupyter lab
# hoặc
jupyter notebook
```
Mở các notebook trong thư mục `notebooks/` theo thứ tự:
- `01_visualization.ipynb`: Khám phá dữ liệu và trực quan hóa phân phối.
- `02_pipeline1_tfidf.ipynb`: Huấn luyện Pipeline 1 (Word TF-IDF).
- `03_pipeline2_word_char.ipynb`: Huấn luyện Pipeline 2 (Word + Char n-grams).
- `04_pipeline3_stylometry.ipynb`: Huấn luyện Pipeline 3 (Stylometry & LightGBM).

### Cách 3: Chạy Huấn luyện Tự động & Xuất Mô hình

Chạy script Python tự động hóa:
```powershell
# Chạy huấn luyện và đánh giá Pipeline 3:
python scripts/train_pipeline3.py

# Xuất mô hình mới sang Chrome Extension:
python scripts/export_to_extension.py
```

### Cách 4: (Tùy chọn) Chạy Backend API với FastAPI

Nếu bạn muốn phục vụ mô hình qua HTTP REST API:
```powershell
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```
- API Docs tương tác: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Endpoint dự đoán: `POST http://127.0.0.1:8000/predict`

---

## 📌 Trạng thái Dự án

- [x] Tiền xử lý dữ liệu và phân tách tập train/val/test chống rò rỉ (`data/processcing.py`).
- [x] Phân tích khám phá dữ liệu toàn diện (EDA).
- [x] Huấn luyện hoàn thiện Pipeline 1 (Word TF-IDF baseline).
- [x] Huấn luyện hoàn thiện Pipeline 2 (Dual-channel Word + Char).
- [x] Huấn luyện hoàn thiện Pipeline 3 (Stylometry + LSA + Tree models).
- [x] Đóng gói mô hình chạy Offline 100% trong Chrome Extension.
- [ ] Huấn luyện Pipeline 4 (Fine-tuned Transformer).
- [ ] Báo cáo so sánh đối đầu tổng hợp cuối cùng (`06_model_comparison.ipynb`).
