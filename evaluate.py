"""Evaluation and reporting utilities for EvoSelect."""
from __future__ import annotations

import base64
import io
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def evaluate_selected_features(
    df,
    target_name,
    selected_features,
    cv=5,
    random_state=42,
):
    """Evaluate a fixed feature subset with leakage-safe preprocessing."""
    if not selected_features:
        raise ValueError("No features were selected.")

    X = df[selected_features]
    y = df[target_name]
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
    ])
    splitter = StratifiedKFold(
        n_splits=cv, shuffle=True, random_state=random_state
    )
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision_weighted",
        "recall": "recall_weighted",
        "f1": "f1_weighted",
    }
    scores = cross_validate(
        model, X, y, cv=splitter, scoring=scoring, n_jobs=1
    )
    return {
        "accuracy": float(np.mean(scores["test_accuracy"])),
        "precision": float(np.mean(scores["test_precision"])),
        "recall": float(np.mean(scores["test_recall"])),
        "f1": float(np.mean(scores["test_f1"])),
        "n_features": len(selected_features),
    }


def compare_and_stats(df, target_name, methods_results):
    summary = {}
    for result in methods_results:
        method = result.get("method", "unknown")
        selected = result.get("selected_features", []) or []
        score = result.get("cv_score", result.get("final_score"))
        summary[method] = {
            "method": method,
            "fitness_score": float(score) if score is not None else None,
            "n_features": len(selected),
            "selected_features": selected,
        }
    return summary


def plot_results_base64(df, target_name, methods_results):
    summary = compare_and_stats(df, target_name, methods_results)
    names = list(summary)
    scores = [
        summary[name]["fitness_score"] or 0.0
        for name in names
    ]

    output = {}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(names, scores)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("CV score")
    ax.set_xlabel("Method")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140)
    plt.close(fig)
    buf.seek(0)
    output["comparison_bar"] = base64.b64encode(buf.read()).decode()

    for result in methods_results:
        if result.get("method") == "genetic" and result.get("history"):
            history = result["history"]
            x = [item["generation"] for item in history]
            y = [item["best_fitness"] for item in history]
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(x, y, marker="o")
            ax.set_xlabel("Generation")
            ax.set_ylabel("Best fitness")
            ax.set_title("GA fitness evolution")
            fig.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=140)
            plt.close(fig)
            buf.seek(0)
            output["ga_history"] = base64.b64encode(buf.read()).decode()
            break
    return output
