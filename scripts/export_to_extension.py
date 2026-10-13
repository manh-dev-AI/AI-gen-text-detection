"""Export Pipeline 1 (TF-IDF + LinearSVC) model artifacts to extension/model_pipeline1.json for offline in-browser inference."""

import json
from pathlib import Path
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "pipeline1"
VECTORIZER_PATH = MODEL_DIR / "word_vectorizer.joblib"
CLASSIFIER_PATH = MODEL_DIR / "best_model.joblib"
OUT_PATH = PROJECT_ROOT / "extension" / "model_pipeline1.json"


def export():
    print(f"Loading model artifacts from {MODEL_DIR}...")
    vectorizer = joblib.load(VECTORIZER_PATH)
    classifier = joblib.load(CLASSIFIER_PATH)

    vocab = vectorizer.vocabulary_
    idf = vectorizer.idf_
    coef = classifier.coef_[0]
    stop_words = sorted(list(vectorizer.get_stop_words()))

    # Build compact token dictionary: token -> [idf, coef]
    weights = {
        token: [round(float(idf[idx]), 4), round(float(coef[idx]), 4)]
        for token, idx in vocab.items()
    }

    model_data = {
        "version": "pipeline1",
        "algorithm": "LinearSVC",
        "vectorizer": "Word TF-IDF (1, 2)",
        "intercept": round(float(classifier.intercept_[0]), 5),
        "stop_words": stop_words,
        "vocabulary_size": len(vocab),
        "weights": weights,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(model_data, f, ensure_ascii=False, separators=(",", ":"))

    size_mb = OUT_PATH.stat().st_size / (1024 * 1024)
    print(f"Successfully exported model to {OUT_PATH} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    export()
