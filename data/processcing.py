import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. THIẾT LẬP ĐƯỜNG DẪN
BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_PATH = BASE_DIR / "AI_Human.csv"
PROCESSED_DIR = BASE_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print(f"🔄 Đang đọc dữ liệu gốc từ: {RAW_DATA_PATH} ...")
df_raw = pd.read_csv(RAW_DATA_PATH)

# 2. LÀM SẠCH CƠ BẢN
# - Xóa null
df_clean = df_raw.dropna(subset=['text', 'generated']).copy()
# - Lọc bỏ văn bản rác quá ngắn (<= 50 ký tự)
df_clean = df_clean[df_clean['text'].str.strip().str.len() > 50].copy()
# - Khử trùng lặp (tránh rò rỉ dữ liệu giữa các tập)
dups_count = df_clean['text'].duplicated().sum()
df_clean = df_clean.drop_duplicates(subset=['text']).reset_index(drop=True)
df_clean['target'] = df_clean['generated'].astype(int)

print(f"✅ Đã loại bỏ {dups_count:,} bài viết trùng lặp. Còn lại {len(df_clean):,} mẫu sạch.")

# 3. PHÂN CHIA TẬP DỮ LIỆU CÓ PHÂN TẦNG (STRATIFIED SPLIT)
# Tách riêng 15% Test Set (Cất riêng, tuyệt đối không rò rỉ)
train_val_df, test_df = train_test_split(
    df_clean, test_size=0.15, stratify=df_clean['target'], random_state=42
)

# Tách Train (70% tổng) và Val (15% tổng ~ 17.65% của train_val)
train_df, val_df = train_test_split(
    train_val_df, test_size=0.1765, stratify=train_val_df['target'], random_state=42
)

# Chỉ giữ 2 cột cần thiết nhất để tối ưu dung lượng đĩa
cols = ['text', 'target']
train_df = train_df[cols].reset_index(drop=True)
val_df = val_df[cols].reset_index(drop=True)
test_df = test_df[cols].reset_index(drop=True)

# 4. LƯU RA THƯ MỤC DATA/PROCESSED
print("💾 Đang lưu dữ liệu ra thư mục data/processed/ ...")
train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
val_df.to_csv(PROCESSED_DIR / "val.csv", index=False)
test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)

print(f"🎉 Hoàn thành!")
print(f"📁 Train set: {len(train_df):,} mẫu -> {PROCESSED_DIR / 'train.csv'}")
print(f"📁 Val set:   {len(val_df):,} mẫu -> {PROCESSED_DIR / 'val.csv'}")
print(f"📁 Test set:  {len(test_df):,} mẫu -> {PROCESSED_DIR / 'test.csv'}")
