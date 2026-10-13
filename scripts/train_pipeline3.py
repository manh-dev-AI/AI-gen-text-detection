"""Pipeline 3: Stylometry + Feature Engineering + LSA (TF-IDF + TruncatedSVD)
Huấn luyện và đánh giá các mô hình Tree-based (LightGBM, HistGradientBoosting, RandomForest)
kết hợp giải thích đặc trưng (Feature Importance).
"""

import sys
import gc
import json
import os
import re
import time
from pathlib import Path

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "pipeline3"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
RANDOM_STATE = 42

TRANSITION_WORDS = {
    "additionally", "furthermore", "moreover", "consequently", "therefore",
    "in conclusion", "specifically", "nevertheless", "crucial", "essential",
    "more importantly", "on the other hand", "in summary", "overall", "fundamental",
    "further", "thus", "hence", "namely"
}

FIRST_PERSON = {"i", "me", "my", "myself", "we", "our", "ours", "us"}

BELIEF_WORDS = {
    "think", "believe", "feel", "opinion", "guess", "personally", "agree", "disagree"
}

MODAL_VERBS = {
    "can", "could", "may", "might", "must", "shall", "should", "will", "would"
}

CONTRACTIONS = re.compile(
    r"\b(can't|cannot|don't|doesn't|didn't|it's|i'm|i've|we're|they're|aren't|isn't|wasn't|weren't|won't|wouldn't|shouldn't|couldn't|haven't|hasn't|hadn't)\b",
    re.IGNORECASE,
)

WORD_RE = re.compile(r"\b[a-zA-Z]{2,}\b")
SENT_RE = re.compile(r"[^.!?]+[.!?]?")

FEATURE_NAMES = [
    "avg_word_len",
    "std_word_len",
    "long_word_ratio",
    "short_word_ratio",
    "avg_sent_len",
    "std_sent_len",
    "newline_density",
    "digit_density",
    "uppercase_ratio",
    "comma_density",
    "semicolon_colon_density",
    "dash_density",
    "quote_density",
    "question_density",
    "exclamation_density",
    "transition_density",
    "first_person_density",
    "belief_density",
    "contraction_density",
    "modal_verb_density",
]


class StylometryExtractor(BaseEstimator, TransformerMixin):
    """Trích xuất 20 đặc trưng phong cách ngôn ngữ (Stylometry) chuẩn hóa theo mật độ,

    hoàn toàn miễn nhiễm với rò rỉ độ dài văn bản (word_count confounder).
    """

    def __init__(self):
        self.feature_names_ = FEATURE_NAMES

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for text in X:
            text = str(text) if text is not None else ""
            text_len = max(len(text), 1)
            words = WORD_RE.findall(text)
            lower_words = [w.lower() for w in words]
            num_words = max(len(words), 1)

            # 1. Cấu trúc từ vựng
            word_lens = [len(w) for w in words]
            avg_word_len = float(np.mean(word_lens)) if word_lens else 0.0
            std_word_len = float(np.std(word_lens)) if word_lens else 0.0
            long_word_ratio = sum(1 for l in word_lens if l >= 7) / num_words
            short_word_ratio = sum(1 for l in word_lens if l <= 3) / num_words

            # 2. Cấu trúc và nhịp điệu câu (Burstiness)
            raw_sents = [s.strip() for s in SENT_RE.findall(text) if len(s.strip()) > 3]
            sent_word_counts = [len(WORD_RE.findall(s)) for s in raw_sents] or [num_words]
            avg_sent_len = float(np.mean(sent_word_counts))
            std_sent_len = float(np.std(sent_word_counts))

            # 3. Mật độ định dạng & ký tự đặc biệt (per 1,000 ký tự)
            newline_density = text.count("\n") * 1000.0 / text_len
            digit_density = sum(c.isdigit() for c in text) * 1000.0 / text_len
            uppercase_ratio = sum(c.isupper() for c in text) / text_len

            # 4. Phân phối dấu câu (per 1,000 ký tự)
            comma_density = text.count(",") * 1000.0 / text_len
            semicolon_colon_density = (text.count(";") + text.count(":")) * 1000.0 / text_len
            dash_density = (text.count("-") + text.count("—")) * 1000.0 / text_len
            quote_density = (text.count('"') + text.count("'")) * 1000.0 / text_len
            question_density = text.count("?") * 1000.0 / text_len
            exclamation_density = text.count("!") * 1000.0 / text_len

            # 5. Đặc trưng từ ngữ lập luận & phong cách (per 1,000 từ)
            lower_text = text.lower()
            transition_count = sum(lower_text.count(t) for t in TRANSITION_WORDS)
            transition_density = transition_count * 1000.0 / num_words

            first_person_count = sum(1 for w in lower_words if w in FIRST_PERSON)
            first_person_density = first_person_count * 1000.0 / num_words

            belief_count = sum(1 for w in lower_words if w in BELIEF_WORDS)
            belief_density = belief_count * 1000.0 / num_words

            contraction_count = len(CONTRACTIONS.findall(text))
            contraction_density = contraction_count * 1000.0 / num_words

            modal_count = sum(1 for w in lower_words if w in MODAL_VERBS)
            modal_verb_density = modal_count * 1000.0 / num_words

            features.append([
                avg_word_len,
                std_word_len,
                long_word_ratio,
                short_word_ratio,
                avg_sent_len,
                std_sent_len,
                newline_density,
                digit_density,
                uppercase_ratio,
                comma_density,
                semicolon_colon_density,
                dash_density,
                quote_density,
                question_density,
                exclamation_density,
                transition_density,
                first_person_density,
                belief_density,
                contraction_density,
                modal_verb_density,
            ])
        return np.array(features, dtype=np.float32)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_)


def evaluate_model(model_name, y_true, y_pred, y_prob=None, train_time=0.0, latency_ms=0.0):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0, 1])
    macro_f1 = (f1[0] + f1[1]) / 2.0
    roc_auc = roc_auc_score(y_true, y_prob) if y_prob is not None else 0.0
    pr_auc = average_precision_score(y_true, y_prob) if y_prob is not None else 0.0

    return {
        "Model": model_name,
        "Accuracy": float(acc),
        "Macro-F1": float(macro_f1),
        "AI Precision": float(prec[1]),
        "AI Recall": float(rec[1]),
        "AI F1": float(f1[1]),
        "Human F1": float(f1[0]),
        "ROC-AUC": float(roc_auc),
        "PR-AUC": float(pr_auc),
        "Train Time (s)": round(train_time, 2),
        "Latency (ms/1k)": round(latency_ms, 2),
    }


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("🚀 PIPELINE 3: STYLOMETRY + FEATURE ENGINEERING + LSA SVD")
    print("=" * 80)

    # 1. Nạp dữ liệu
    print("📂 Nạp dữ liệu từ data/processed/...")
    t0 = time.time()
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "val.csv")

    # Giới hạn mẫu đại diện 60,000 bài train tương tự Pipeline 2 để tối ưu thời gian và RAM
    SAMPLE_LIMIT = 60_000
    if SAMPLE_LIMIT and len(train_df) > SAMPLE_LIMIT:
        print(f"⚡ Lấy mẫu đại diện phân tầng: {SAMPLE_LIMIT:,} mẫu Train để huấn luyện...")
        from sklearn.model_selection import train_test_split
        train_df, _ = train_test_split(
            train_df,
            train_size=SAMPLE_LIMIT,
            stratify=train_df["target"],
            random_state=RANDOM_STATE
        )
        train_df = train_df.reset_index(drop=True)

    X_train_text = train_df["text"].tolist()
    y_train = train_df["target"].to_numpy()


    X_val_text = val_df["text"].tolist()
    y_val = val_df["target"].to_numpy()

    print(f"✅ Đã nạp Train: {len(X_train_text):,} mẫu | Val: {len(X_val_text):,} mẫu ({time.time() - t0:.2f}s)")

    # 2. Trích xuất Stylometry
    print("\n🔍 Đang trích xuất 20 đặc trưng Stylometry & Linguistic...")
    t_sty = time.time()
    sty_extractor = StylometryExtractor()
    X_train_sty = sty_extractor.transform(X_train_text)
    X_val_sty = sty_extractor.transform(X_val_text)
    print(f"✅ Stylometry hoàn tất: Shape {X_train_sty.shape} trong {time.time() - t_sty:.2f}s")

    # 3. Trích xuất Semantic LSA (TF-IDF + TruncatedSVD 30 dimensions)
    print("\n🧠 Đang trích xuất đặc trưng ngữ nghĩa LSA (TF-IDF + TruncatedSVD 30 chiều)...")
    t_svd = time.time()
    tfidf = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.9,
        sublinear_tf=True,
        max_features=20_000,
        dtype=np.float32,
    )
    X_train_tfidf = tfidf.fit_transform(X_train_text)
    X_val_tfidf = tfidf.transform(X_val_text)

    svd = TruncatedSVD(n_components=30, random_state=RANDOM_STATE)
    X_train_svd = svd.fit_transform(X_train_tfidf)
    X_val_svd = svd.transform(X_val_tfidf)
    print(f"✅ LSA hoàn tất: TF-IDF {X_train_tfidf.shape[1]} dims -> SVD {X_train_svd.shape[1]} dims ({time.time() - t_svd:.2f}s)")

    # Chuẩn hóa Stylometry với StandardScaler
    scaler = StandardScaler()
    X_train_sty_scaled = scaler.fit_transform(X_train_sty)
    X_val_sty_scaled = scaler.transform(X_val_sty)

    # Ghép toàn bộ đặc trưng: 20 Stylometry + 30 SVD = 50 chiều
    X_train_comb = np.hstack([X_train_sty_scaled, X_train_svd])
    X_val_comb = np.hstack([X_val_sty_scaled, X_val_svd])
    all_feature_names = FEATURE_NAMES + [f"svd_dim_{i}" for i in range(30)]
    print(f"📊 Tổng không gian đặc trưng kết hợp: {X_train_comb.shape[1]} đặc trưng.")

    # 4. Huấn luyện các mô hình đối đầu
    print("\n" + "=" * 80)
    print("🏋️‍♂️ HUẤN LUYỆN VÀ ĐÁNH GIÁ CÁC MÔ HÌNH PIPELINE 3")
    print("=" * 80)

    models = {
        "LightGBM Classifier": lgb.LGBMClassifier(
            n_estimators=250,
            learning_rate=0.08,
            num_leaves=31,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            verbose=-1,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.08,
            max_leaf_nodes=31,
            max_depth=6,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "Logistic Regression (L2)": LogisticRegression(
            C=1.0,
            max_iter=1000,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }

    benchmark_results = []
    trained_models = {}

    for name, model in models.items():
        print(f"⏳ Đang huấn luyện {name} ...", end=" ", flush=True)
        t_start = time.time()
        model.fit(X_train_comb, y_train)
        t_train = time.time() - t_start

        # Đo latency trên 1,000 mẫu
        t_lat = time.time()
        sample_x = X_val_comb[:1000]
        _ = model.predict(sample_x)
        lat_ms = (time.time() - t_lat) * 1000.0

        y_pred = model.predict(X_val_comb)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_val_comb)[:, 1]
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X_val_comb)
            y_prob = 1.0 / (1.0 + np.exp(-scores))
        else:
            y_prob = None

        metrics = evaluate_model(name, y_val, y_pred, y_prob, train_time=t_train, latency_ms=lat_ms)
        benchmark_results.append(metrics)
        trained_models[name] = (model, y_pred, y_prob)
        print(f"Xong trong {t_train:.1f}s | Macro-F1: {metrics['Macro-F1']:.4f} | AI Recall: {metrics['AI Recall']:.4f}")

    results_df = pd.DataFrame(benchmark_results).sort_values(by="Macro-F1", ascending=False).reset_index(drop=True)
    print("\n🏆 BẢNG XẾP HẠNG CÁC MÔ HÌNH PIPELINE 3:")
    print(results_df.to_string(index=False))

    # 5. Phân tích độ quan trọng của đặc trưng (Feature Importance)
    lgb_model = trained_models["LightGBM Classifier"][0]
    importances = lgb_model.feature_importances_
    feat_imp = pd.DataFrame({
        "Feature": all_feature_names,
        "Importance": importances,
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    print("\n🔝 TOP 15 ĐẶC TRƯNG QUAN TRỌNG NHẤT (LIGHTGBM):")
    print(feat_imp.head(15).to_string(index=False))

    # 6. Lưu biểu đồ trực quan hóa
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.barplot(data=results_df, x="Macro-F1", y="Model", hue="Model", palette="Blues_r", legend=False, ax=axes[0])
    axes[0].set_title("Pipeline 3: So sánh Macro-F1 giữa các mô hình", fontweight="bold")
    axes[0].set_xlim(0.85, 1.0)
    for p in axes[0].patches:
        axes[0].annotate(f"{p.get_width():.4f}", (p.get_width() - 0.02, p.get_y() + p.get_height() / 2),
                         color="white", fontweight="bold", va="center")

    top_feats = feat_imp.head(12)
    sns.barplot(data=top_feats, x="Importance", y="Feature", hue="Feature", palette="viridis", legend=False, ax=axes[1])
    axes[1].set_title("Top 12 Đặc trưng Stylometry & LSA quan trọng nhất", fontweight="bold")


    plt.tight_layout()
    chart_path = FIGURES_DIR / "pipeline3_benchmark.png"
    plt.savefig(chart_path, dpi=200)
    plt.close()
    print(f"📊 Đã lưu biểu đồ tại: {chart_path}")

    # 7. Lưu Confusion Matrix của mô hình chiến thắng
    best_name = results_df.iloc[0]["Model"]
    best_model_obj, best_y_pred, best_y_prob = trained_models[best_name]
    cm = confusion_matrix(y_val, best_y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=False,
                xticklabels=["Human (0)", "AI (1)"], yticklabels=["Human (0)", "AI (1)"], ax=ax)
    ax.set_title(f"Confusion Matrix: {best_name}\n(Val set: {len(y_val):,} bài)", fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    cm_path = FIGURES_DIR / "pipeline3_confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"📊 Đã lưu Confusion Matrix tại: {cm_path}")

    # 8. Đóng gói và lưu artifacts vào models/pipeline3/
    print("\n📦 Đang đóng gói và xuất mô hình vào models/pipeline3/...")
    joblib.dump(best_model_obj, MODEL_DIR / "best_model.joblib")
    joblib.dump(scaler, MODEL_DIR / "stylometry_scaler.joblib")
    joblib.dump(tfidf, MODEL_DIR / "tfidf_vectorizer.joblib")
    joblib.dump(svd, MODEL_DIR / "svd_transformer.joblib")

    # Lưu metrics và metadata
    with open(MODEL_DIR / "pipeline3_metrics.json", "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2, ensure_ascii=False)

    with open(MODEL_DIR / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(all_feature_names, f, indent=2, ensure_ascii=False)

    with open(MODEL_DIR / "feature_importances.json", "w", encoding="utf-8") as f:
        json.dump(feat_imp.to_dict(orient="records"), f, indent=2, ensure_ascii=False)

    print(f"✅ Hoàn tất xuất artifacts vào {MODEL_DIR}!")

    # 9. Cập nhật kết quả vào results/evaluation.csv
    eval_csv_path = RESULTS_DIR / "evaluation.csv"
    best_metric = results_df.iloc[0]
    model_size_mb = (MODEL_DIR / "best_model.joblib").stat().st_size / (1024 * 1024)

    new_row = pd.DataFrame([{
        "pipeline": "Pipeline 3",
        "model": f"{best_metric['Model']} (Stylometry + LSA SVD)",
        "macro_f1": best_metric["Macro-F1"],
        "ai_precision": best_metric["AI Precision"],
        "ai_recall": best_metric["AI Recall"],
        "pr_auc": best_metric["PR-AUC"],
        "latency_ms": best_metric["Latency (ms/1k)"],
        "model_size_mb": round(model_size_mb, 2),
        "notes": "20 Stylometry features + 30 LSA components, giải thích được đặc trưng ngôn ngữ",
    }])

    if eval_csv_path.exists() and eval_csv_path.stat().st_size > 100:
        existing_df = pd.read_csv(eval_csv_path)
        existing_df = existing_df[existing_df["pipeline"] != "Pipeline 3"]
        combined_eval = pd.concat([existing_df, new_row], ignore_index=True)
    else:
        combined_eval = new_row

    combined_eval.to_csv(eval_csv_path, index=False)
    print(f"✅ Đã cập nhật kết quả vào {eval_csv_path}")


if __name__ == "__main__":
    main()
