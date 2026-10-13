# AI-Generated Text Detection

Dự án nghiên cứu và triển khai hệ thống phát hiện văn bản do AI tạo ra bằng Machine Learning và Natural Language Processing (NLP).

Mục tiêu là phân loại một đoạn văn bản thành **Human** hoặc **AI-generated**, so sánh các phương pháp khác nhau, sau đó triển khai mô hình phù hợp nhất qua API để Chrome Extension có thể phân tích văn bản người dùng chọn trên trang web.

## Dữ liệu

Dataset chính: `data/AI_Human.csv`

| Cột | Mô tả |
| --- | --- |
| `text` | Nội dung văn bản đầu vào |
| `generated` | Nhãn: `0` là Human, `1` là AI-generated |

## Các pipeline thử nghiệm

1. **Word TF-IDF**: baseline dùng word n-grams với Logistic Regression, Naive Bayes, Linear SVM hoặc SGDClassifier.
2. **Word + Character TF-IDF**: kết hợp đặc trưng từ và ký tự để nhận diện cả nội dung lẫn phong cách viết.
3. **Stylometry + Feature Engineering**: dùng độ dài câu/từ, lexical diversity, dấu câu và các đặc trưng phong cách; có thể kết hợp TF-IDF + SVD.
4. **Fine-tuned Transformer**: fine-tune các mô hình pretrained như DistilRoBERTa hoặc DeBERTa cho bài toán phân loại nhị phân.

## Đánh giá và lựa chọn mô hình

Mô hình được đánh giá bằng Macro-F1, AI Precision, AI Recall, PR-AUC, latency và kích thước model. Tập test được tách riêng; việc chọn hyperparameter và threshold chỉ thực hiện trên dữ liệu train/validation hoặc cross-validation.

Mô hình cuối cùng không nhất thiết là mô hình có điểm số cao nhất, mà cần cân bằng chất lượng phát hiện với tốc độ và chi phí triển khai cho Chrome Extension.

## Cấu trúc thư mục

- `notebooks/`: EDA, huấn luyện từng pipeline và so sánh kết quả.
- `data/`: dataset gốc.
- `models/`: artifacts của các pipeline và mô hình được chọn.
- `results/`: bảng đánh giá, hình biểu đồ và báo cáo thực nghiệm.
- `docs/`: báo cáo EDA (`info_data.md`), phương pháp luận chi tiết (`pipelines_methodology.md`), và bảng chọn mô hình (`model_selection.md`).
- `api/`: API nhận văn bản và trả về dự đoán.
- `extension/`: Chrome Extension giao tiếp với API.

## Chạy Chrome Extension (Không cần chạy Server)

Chrome Extension đã được tích hợp sẵn mô hình Word TF-IDF + Linear SVM (`extension/model_pipeline1.json`), chạy trực tiếp 100% trong trình duyệt (Offline) mà **không cần cài đặt Python hay mở server backend**.

1. Mở trình duyệt Chrome (hoặc Edge / Brave / Cốc Cốc), truy cập `chrome://extensions`
2. Bật **Developer mode** (Chế độ cho nhà phát triển) ở góc trên bên phải.
3. Nhấn **Load unpacked** (Tải tiện ích đã giải nén) và chọn thư mục `extension/`.
4. Bôi đen văn bản trên trang web, mở extension, chọn **Lấy phần bôi đen** rồi bấm **Phân tích** (hoặc paste văn bản trực tiếp).

> **Lưu ý**: Thư mục `api/` là API backend tùy chọn (FastAPI) nếu cần tích hợp hệ thống bên ngoài (`uvicorn api.main:app --reload`), không bắt buộc cho Chrome Extension.

## Trạng thái

Project đang ở giai đoạn chuẩn bị cấu trúc và thực nghiệm mô hình. Kết quả benchmark cuối cùng sẽ được bổ sung sau khi các pipeline được huấn luyện theo cùng một quy trình đánh giá.

