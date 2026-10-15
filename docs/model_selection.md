# Lựa chọn mô hình & Bảng so sánh thực nghiệm

Bảng tổng hợp kết quả thực nghiệm trên tập kiểm định độc lập (Validation Set: 73,095 mẫu):

| Pipeline | Model | Macro-F1 | AI Precision | AI Recall | PR-AUC | Latency (ms/1k) | Kích thước | Điểm mạnh & Ghi chú |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| **Pipeline 1** | Linear SVM (Word TF-IDF) | **0.9993** | 0.9996 | 0.9986 | 0.9999 | 1.77 ms | 2.7 MB | Siêu nhanh, đã được đóng gói chạy Offline trên Chrome Extension |
| **Pipeline 2** | Linear SVM (Word + Char TF-IDF) | **0.9996** | 0.9997 | 0.9993 | 0.9999 | 72.82 ms | 4.1 MB | Bắt cả hình thái từ và phong cách ký tự vi mô, điểm cao nhất |
| **Pipeline 3** | LightGBM (Stylometry + LSA SVD) | **0.9928** | 0.9943 | 0.9877 | 0.9994 | 3.12 ms | 0.85 MB | **Giải thích được đặc trưng ngôn ngữ** (Feature Importance), chống học vẹt theo chủ đề |
| **Pipeline 4** | Fine-tuned Transformer | *Đang chờ* | *—* | *—* | *—* | *—* | *—* | DeBERTa / DistilRoBERTa ngữ cảnh sâu |

### Nhận định chuyên sâu về Pipeline 3 (Stylometry):
1. **Giá trị độc bản:** Trong khi Pipeline 1 & 2 dùng túi từ (Bag-of-Words) dễ bị rủi ro *Topic Shortcut Learning*, Pipeline 3 đo đạc cấu trúc phong cách câu và hình thái từ (Stylometry) kết hợp 30 chiều LSA semantic.
2. **Top đặc trưng then chốt (Feature Importance):**
   - `newline_density`: AI tổ chức bài theo từng đoạn văn/bullet points đều đặn.
   - `avg_word_len` & `long_word_ratio`: AI có xu hướng dùng từ dài và từ học thuật nhiều âm tiết hơn con người (khớp với kết luận Cohen's $d = +1.072$ từ báo cáo EDA).
   - `std_sent_len` (Burstiness): Nhịp điệu câu của con người biến thiên mạnh mẽ hơn AI.
   - `transition_density`: LLM lặp lại nhiều từ nối kinh điển (*additionally*, *furthermore*, *crucial*, *essential*).
   - `first_person_density` & `contraction_density`: Dấu ấn con người qua đại từ xưng hô cá nhân và từ viết tắt (*don't*, *it's*).
