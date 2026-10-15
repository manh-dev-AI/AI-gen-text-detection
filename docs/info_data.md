# 📊 BÁO CÁO PHÂN TÍCH KHÁM PHÁ DỮ LIỆU (EDA REPORT)
> **Tập dữ liệu:** `AI_Human.csv`  
> **Dự án:** AI-Generated Text Detection Using Machine Learning & NLP with Chrome Extension  
> **Tài liệu nguồn:** `data/trucquan.ipynb`

---

## 1. TỔNG QUAN VỀ TẬP DỮ LIỆU (DATASET OVERVIEW)

Tập dữ liệu gốc là tập phân loại nhị phân (Binary Classification) gồm 2 cột: `text` (nội dung văn bản) và `generated` (nhãn phân loại: 0 cho Human, 1 cho AI).

### 1.1. Thống kê số lượng & Chất lượng dữ liệu
Toàn bộ file CSV (~1.1 GB) được quét và kiểm toán chất lượng (Audit):

| Chỉ số kiểm toán | Giá trị | Tỷ lệ (%) | Ghi chú & Đánh giá |
| :--- | :---: | :---: | :--- |
| **Tổng số dòng CSV** | `487,235` | 100.0% | Dung lượng thô ~1.1 GB |
| **Số dòng hợp lệ** | `487,229` | **99.998%** | Dữ liệu cực kỳ sạch về mặt kỹ thuật |
| **Văn bản thiếu (Missing/NaN)** | `0` | 0.0% | Không có ô trống |
| **Văn bản quá ngắn ($\le 10$ ký tự)**| `6` | 0.001% | Dữ liệu rác/lỗi ghi nhận, đã loại bỏ |
| **Nhãn không hợp lệ (ngoài 0 và 1)** | `0` | 0.0% | Nhãn hoàn toàn chuẩn nhị phân |
| **Trùng lặp (Duplicate text)** | `308` | ~0.51% (trong mẫu 60k) | **Cần thực hiện `drop_duplicates`** trước khi huấn luyện mô hình |

### 1.2. Phân bố nhãn (Class Distribution)
- **Human (0):** `305,797` mẫu (**62.76%**)
- **AI (1):** `181,432` mẫu (**37.24%**)
- **Tỷ lệ mất cân bằng:** ~ `1.68 : 1` (Human chiếm gần 2/3).
- **Hệ quả cho mô hình:** Mức chênh lệch nhẹ, không cần kỹ thuật Oversampling (SMOTE) phức tạp, nhưng bắt buộc:
  - Chia tập dữ liệu có phân tầng (**Stratified Split / Stratified 5-Fold CV**).
  - Không đánh giá mô hình bằng **Accuracy** đơn thuần; phải dùng **Macro-F1, AI Recall, AI Precision và PR-AUC**.

---

## 2. PHÂN PHỐI ĐỘ DÀI VĂN BẢN (TEXT LENGTH ANALYSIS)

Phân tích số từ (`word_count`) giữa hai nhóm:

- **Khoảng phân vị phổ biến (IQR):**
  - Phân vị 25% ($Q_1$): **279 từ**
  - Phân vị 75% ($Q_3$): **470 từ**
- **Trung vị (Median):**
  - Văn bản Human: **390 từ**
  - Văn bản AI: **338 từ**
- **Hiệu ứng khác biệt:** Cohen's $d = -0.494$ (văn bản Human dài hơn rõ rệt).

> [!WARNING]
> ### RỦI RO "BIẾN GÂY NHIỄU" (CONFOUNDING VARIABLE)
> Vì văn bản Human dài hơn văn bản AI trong tập dữ liệu, mô hình rất dễ **"học vẹt"** quy luật: *Bài dài $\rightarrow$ Human, bài ngắn $\rightarrow$ AI*.
> 
> **Quy tắc bắt buộc khi thiết kế đặc trưng:**
> 1. Tuyệt đối **KHÔNG dùng `word_count`** làm đặc trưng phân loại độc lập.
> 2. Mọi đặc trưng số lượng (dấu câu, chữ hoa, ký tự số) đều phải **chuẩn hóa theo tỷ lệ mật độ** (trên 100 từ hoặc trên tổng ký tự).

---

## 3. ĐẶC TRƯNG PHONG CÁCH NGÔN NGỮ (STYLOMETRIC FEATURES)

Bảng xếp hạng mức độ khác biệt chuẩn hóa (**Cohen's $d$**) giữa AI và Human:

| Đặc trưng (Feature) | Trung vị Human | Trung vị AI | Cohen's $d$ (AI - Human) | Đánh giá mức độ phân biệt |
| :--- | :---: | :---: | :---: | :--- |
| **`avg_word_len`** (Độ dài từ TB) | `4.404` | `4.979` | **$+1.072$** | 🔥 **RẤT MẠNH** (Đặc trưng then chốt) |
| **`word_count`** (Số từ) | `390` | `338` | $-0.494$ | Trung bình (Biến gây nhiễu, cần loại bỏ) |
| **`digit_density`** (Mật độ chữ số) | `0.069` | `0.000` | $-0.402$ | Trung bình (Người dùng số liệu cụ thể hơn) |
| **`newline_density`** (Mật độ xuống dòng) | `1.887` | `2.611` | $+0.307$ | Nhẹ - Trung bình (AI hay chia đoạn nhỏ) |
| **`uppercase_ratio`** (Tỷ lệ chữ hoa) | `2.606` | `2.060` | $-0.257$ | Nhẹ (Người viết hoa tự do / viết tắt) |
| **`avg_sentence_len`** (Độ dài câu TB) | `19.259` | `19.294` | $-0.181$ | Rất yếu (Cả 2 bên đều ~19 từ/câu) |
| **`punct_density`** (Mật độ dấu câu) | `11.321` | `12.389` | $+0.184$ | Nhẹ (AI ngắt câu ổn định hơn) |
| **`root_ttr`** (Vốn từ phong phú) | `8.490` | `8.300` | **$-0.027$** | ❌ **KHÔNG CÓ TÁC DỤNG** (Gần như bằng 0) |

### Kiểm chứng khi kiểm soát độ dài (Controlled Experiment - 10k bài mỗi lớp, 279–470 từ):
1. **`avg_word_len` vẫn giữ nguyên sự phân tách:** Đỉnh của Human ở mức 4.38, trong khi AI ở mức 5.05. AI có thói quen cố hữu sử dụng từ vựng dài hơn, nhiều âm tiết và mang tính học thuật cao.
2. **`root_ttr` chồng khít 95%:** Khi bài viết cùng độ dài, độ phong phú từ vựng giữa AI và Người là như nhau. Quan niệm "AI vốn từ rộng hơn" là không chính xác trong tập dữ liệu này.

---

## 4. TỪ VỰNG & CỤM TỪ ĐẶC TRƯNG (TF-IDF N-GRAMS)

Chênh lệch điểm số TF-IDF trung bình ($\text{TF-IDF}_{\text{AI}} - \text{TF-IDF}_{\text{Human}}$) đã bộc lộ rõ **"dấu vân tay ngôn ngữ"** của hai nhóm:

### 4.1. Đặc trưng của Con người (Human) — Giá trị âm lớn
- **Khẩu ngữ & Từ ngữ đời thường:** `just`, `kids`, `lot` (a lot of), `good`.
- **Động từ thể hiện cái tôi & suy nghĩ cá nhân:** `think` (*I think*), `want` (*I want*), `know` (*You know*).
- **Từ ngữ nói / Viết tắt:** `don` (*don't*), `going` (*going to*).
- **Danh từ cụ thể theo tình huống:** `car`, `people`, `kids`.

### 4.2. Đặc trưng của Trí tuệ nhân tạo (AI) — Giá trị dương lớn
- **Từ nối lập luận khuôn mẫu (Discourse Markers):** `additionally` (từ nối kinh điển của ChatGPT/LLMs).
- **Tính từ đánh giá chuẩn mực:** `important`, `essential`, `significant`, `potential`.
- **Từ ngữ định hướng giải pháp & phát triển bài luận:** `develop`, `benefits`, `impact`, `success`, `skills`, `provide`, `lead`.

---

## 5. HẠN CHẾ CỦA TẬP DỮ LIỆU & RỦI RO MÔ HÌNH (LIMITATIONS & RISKS)

1. **Rủi ro rò rỉ chủ đề (Topic Shortcut / Shortcut Learning):**
   - Các từ như `car`, `kids` đứng đầu phía Human cho thấy tập dữ liệu có thiên vị về chủ đề (ví dụ đề thi luận tiếng Anh về xe tự lái hoặc học sinh).
   - Nếu mô hình chỉ dựa vào Word TF-IDF, khi người dùng kiểm tra một bài AI viết về "car", mô hình có thể đoán nhầm thành Human.
2. **Thiếu Metadata:**
   - Dữ liệu không cung cấp nhãn nguồn (văn bản AI được sinh từ GPT-3.5, GPT-4, Llama hay Claude; văn bản Human là của học sinh cấp 3, sinh viên hay nhà báo).
   - Do đó không thể thực hiện Leave-One-Prompt-Out (LOPO) Cross-Validation.

---

## 6. ĐỊNH HƯỚNG CHO CÁC PIPELINE MÔ HÌNH (ACTIONABLE IMPLICATIONS)

Từ toàn bộ phát hiện EDA, 4 pipeline mô hình cần tuân thủ các chiến lược sau:

| Pipeline | Hướng tiếp cận | Tận dụng phát hiện từ EDA |
| :--- | :--- | :--- |
| **Pipeline 1: Word TF-IDF** | Baseline nhẹ | Tận dụng các từ vựng học thuật (`additionally`, `essential`) vs từ khẩu ngữ (`don`, `think`, `just`) qua **Logistic Regression / Linear SVM**. |
| **Pipeline 2: Word + Char TF-IDF** | **Ứng viên ưu tiên deploy** | Thêm **Character 3-5 grams** để bắt trọn: hậu tố từ dài (`avg_word_len`), nhịp điệu dấu câu chuẩn mực và viết hoa mà không bị phụ thuộc vào chủ đề bài viết. |
| **Pipeline 3: Stylometry + Tree** | Mô hình giải thích được | **Chỉ chọn các đặc trưng còn hiệu quả:** `avg_word_len`, `newline_density`, `digit_density`, `punct_density`. **Loại bỏ hoàn toàn:** `root_ttr` và `word_count`. Huấn luyện bằng **LightGBM / XGBoost**. |
| **Pipeline 4: Transformer** | Mô hình ngữ cảnh sâu | Fine-tune **DeBERTa-v3-small / DistilRoBERTa** trên GPU T4 để bắt trọn tính mạch lạc và ngữ cảnh câu mà các phương pháp n-grams bỏ sót. |
