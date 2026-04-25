"""
model.py
--------
TF-IDF + Logistic Regression pipeline for multi-emotion detection.
Using dair-ai/emotion dataset for improved accuracy.

Created by: Likith Abhiram Jaldu and Vishwanath Reddy

TF-IDF recap
~~~~~~~~~~~~
  TF  (term frequency)   = how often a word appears in *this* document
  IDF (inverse doc freq) = log(N / df) — penalises words in many documents
  TF-IDF = TF × IDF  → rare but locally frequent words score highest

Why bigrams?  "not happy" and "love you" are far more informative
              than "not" or "happy" alone.
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from data_preprocessing import prepare_data, preprocess_text

MODEL_PATH = "emotion_model.pkl"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def build_pipeline() -> Pipeline:
    # Optimized for 6-emotion classification (dair-ai/emotion dataset)
    tfidf = TfidfVectorizer(
        max_features=15_000,      # Reduced from 20k for focused features
        ngram_range=(1, 2),        # Keep bigrams for context
        sublinear_tf=True,         # Apply sublinear term frequency
        min_df=2,                  # Keep words appearing 2+ times
        max_df=0.95,               # Remove very common words
        strip_accents="unicode",
        lowercase=True,
        use_idf=True,
        norm="l2",
    )
    
    lr = LogisticRegression(
        class_weight="balanced",   # Handle class imbalance
        max_iter=3_000,            # More iterations for convergence
        C=1.0,                     # Regularization parameter (C=1 is default)
        solver="lbfgs",            # Good for multiclass
        random_state=42,
        multi_class="multinomial",
    )
    
    return Pipeline([("tfidf", tfidf), ("classifier", lr)])


# ---------------------------------------------------------------------------
# Interpretability
# ---------------------------------------------------------------------------

def show_top_words(pipeline: Pipeline, n: int = 10) -> dict:
    tfidf      = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]
    features   = np.array(tfidf.get_feature_names_out())
    classes    = classifier.classes_

    print("\n" + "=" * 60)
    print(f"  TOP {n} WORDS PER EMOTION")
    print("=" * 60)

    top_words: dict = {}
    for i, emotion in enumerate(classes):
        coefs    = classifier.coef_[i]
        top_idx  = np.argsort(coefs)[-n:][::-1]
        words    = features[top_idx].tolist()
        top_words[emotion] = words
        print(f"\n  {emotion.upper():18s}: {', '.join(words)}")

    return top_words


# ---------------------------------------------------------------------------
# Training + evaluation
# ---------------------------------------------------------------------------

def train_and_evaluate(cache_path: str = "processed_data.csv") -> Pipeline:
    df = prepare_data(cache_path)

    X, y = df["text"], df["emotion"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train):,}  |  Test: {len(X_test):,}")
    print(f"Emotion classes: {y.nunique()}")

    print("Fitting pipeline ...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred   = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy : {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)
    fig, ax = plt.subplots(figsize=(max(10, len(pipeline.classes_)), max(8, len(pipeline.classes_) - 2)))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=pipeline.classes_,
                yticklabels=pipeline.classes_,
                cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix — Emotion Detection")
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=120)
    print("Confusion matrix saved → confusion_matrix.png")

    show_top_words(pipeline)

    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModel saved → {MODEL_PATH}")

    return pipeline


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def load_model() -> Pipeline:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at '{MODEL_PATH}'. Run  python model.py  first."
        )
    return joblib.load(MODEL_PATH)


def predict_emotion(text: str, pipeline: Pipeline | None = None):
    """Return (top_emotion, confidence, {emotion: prob, ...})."""
    if pipeline is None:
        pipeline = load_model()

    text   = preprocess_text(text)
    proba  = pipeline.predict_proba([text])[0]
    classes = pipeline.classes_
    best_i = int(np.argmax(proba))

    return classes[best_i], float(proba[best_i]), dict(zip(classes, proba.tolist()))


def get_top_words_for_emotion(emotion: str, pipeline: Pipeline, n: int = 12) -> list[str]:
    tfidf      = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]
    features   = np.array(tfidf.get_feature_names_out())
    idx        = list(classifier.classes_).index(emotion)
    top_idx    = np.argsort(classifier.coef_[idx])[-n:][::-1]
    return features[top_idx].tolist()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pipeline = train_and_evaluate()

    print("\n" + "=" * 60)
    print("  SAMPLE PREDICTIONS")
    print("=" * 60)
    samples = [
        "I just got the promotion I have been working towards for years!",
        "This is completely unacceptable. I am absolutely furious.",
        "Thank you so much for your help. I really appreciate it.",
        "I am terrified and cannot stop shaking right now.",
        "I miss them so much. I feel so lonely and heartbroken.",
        "I wonder how this technology actually works under the hood.",
    ]
    for text in samples:
        e, c, _ = predict_emotion(text, pipeline)
        print(f"\n  [{e.upper():14s}  {c:.0%}]  {text}")
