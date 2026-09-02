from __future__ import annotations
import time
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from ga_module import GAConfig, run_genetic_algorithm

RANDOM_STATE = 42
OUTER_FOLDS = 3
DATA_PATH = Path("data") / "sonar.all-data"
RESULTS_DIR = Path("benchmark_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

GA_CONFIG = GAConfig(
    pop_size=30, generations=20, crossover_rate=0.8, mutation_rate=0.02,
    selection_pressure=3, feature_penalty=0.01, cv_folds=5,
    random_state=RANDOM_STATE, patience=5,
)

def load_sonar():
    if not DATA_PATH.exists():
        raise SystemExit(f"Dataset not found: {DATA_PATH.resolve()}")
    df = pd.read_csv(DATA_PATH, header=None)
    if df.shape[1] != 61:
        raise ValueError(f"Unexpected dataset shape {df.shape}; expected 61 columns.")
    X = df.iloc[:, :-1].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        raise ValueError("Dataset contains invalid feature values.")
    y_raw = df.iloc[:, -1].astype(str).str.strip()
    y = pd.Series(pd.Categorical(y_raw).codes, name="target")
    X.columns = [f"feature_{i+1:02d}" for i in range(60)]
    return X, y

def make_classifier():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1))
    ])

def evaluate(X_train, y_train, X_test, y_test, selected):
    if not selected:
        raise ValueError("No features selected.")
    model = make_classifier()
    model.fit(X_train[selected], y_train)
    pred = model.predict(X_test[selected])
    return {
        "accuracy": accuracy_score(y_test, pred),
        "f1": f1_score(y_test, pred, average="weighted", zero_division=0),
    }

def traditional_selection(method, X_train, y_train):
    k = min(15, X_train.shape[1])
    if method == "chi2":
        selector = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", MinMaxScaler()),
            ("selector", SelectKBest(chi2, k=k)),
        ])
    elif method == "mutual_info":
        selector = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("selector", SelectKBest(mutual_info_classif, k=k)),
        ])
    elif method == "f_classif":
        selector = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("selector", SelectKBest(f_classif, k=k)),
        ])
    elif method == "l1_logistic":
        est = LogisticRegression(penalty="l1", solver="liblinear", C=0.1,
                                 max_iter=2000, random_state=RANDOM_STATE)
        selector = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("selector", SelectFromModel(est, threshold="mean")),
        ])
    elif method == "embedding_rf":
        est = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
        selector = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("selector", SelectFromModel(est, max_features=k)),
        ])
    elif method == "rfe_rf":
        est = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
        selector = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("selector", RFE(estimator=est, n_features_to_select=k)),
        ])
    else:
        raise ValueError(f"Unknown method: {method}")
    selector.fit(X_train, y_train)
    return X_train.columns[selector.named_steps["selector"].get_support()].tolist()

def main():
    X, y = load_sonar()
    print("=" * 78)
    print("EvoSelect - OFFLINE NESTED BENCHMARK")
    print("=" * 78)
    print(f"Dataset: {DATA_PATH}")
    print(f"Samples: {len(X)} | Features: {X.shape[1]} | Outer folds: {OUTER_FOLDS}")
    print(f"GA: population={GA_CONFIG.pop_size}, generations={GA_CONFIG.generations}, "
          f"inner_cv={GA_CONFIG.cv_folds}, penalty={GA_CONFIG.feature_penalty}")
    print("=" * 78)

    outer_cv = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True,
                               random_state=RANDOM_STATE)
    methods = ["full_features", "genetic", "chi2", "mutual_info",
               "f_classif", "l1_logistic", "embedding_rf", "rfe_rf"]
    rows = []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), 1):
        print(f"\n{'='*25} OUTER FOLD {fold}/{OUTER_FOLDS} {'='*25}")
        X_train, X_test = X.iloc[train_idx].reset_index(drop=True), X.iloc[test_idx].reset_index(drop=True)
        y_train, y_test = y.iloc[train_idx].reset_index(drop=True), y.iloc[test_idx].reset_index(drop=True)

        for method in methods:
            print(f"\n[{method}]")
            start = time.perf_counter()
            if method == "full_features":
                selected = X_train.columns.tolist()
            elif method == "genetic":
                train_df = X_train.copy()
                train_df["target"] = y_train
                result = run_genetic_algorithm(train_df, "target", config=GA_CONFIG, verbose=True)
                selected = result["selected_features"]
            else:
                selected = traditional_selection(method, X_train, y_train)

            metrics = evaluate(X_train, y_train, X_test, y_test, selected)
            elapsed = time.perf_counter() - start
            reduction = 100.0 * (1.0 - len(selected) / X.shape[1])
            rows.append({
                "dataset": "UCI Sonar", "fold": fold, "method": method,
                "accuracy": metrics["accuracy"], "f1": metrics["f1"],
                "n_features": len(selected),
                "feature_reduction_percent": reduction,
                "runtime_seconds": elapsed,
                "selected_features": "|".join(selected),
            })
            print(f"Result: accuracy={metrics['accuracy']:.4f} | f1={metrics['f1']:.4f} "
                  f"| features={len(selected)}/60 | reduction={reduction:.1f}% | time={elapsed:.2f}s")

    raw = pd.DataFrame(rows)
    raw_path = RESULTS_DIR / "sonar_nested_raw.csv"
    raw.to_csv(raw_path, index=False)

    summary = raw.groupby("method", as_index=False).agg(
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        features_mean=("n_features", "mean"),
        reduction_mean=("feature_reduction_percent", "mean"),
        runtime_mean_seconds=("runtime_seconds", "mean"),
    ).sort_values(["accuracy_mean", "f1_mean"], ascending=False)

    summary_path = RESULTS_DIR / "sonar_nested_summary.csv"
    summary.to_csv(summary_path, index=False)
    print("\n" + "=" * 78)
    print("FINAL SUMMARY")
    print("=" * 78)
    print(summary.to_string(index=False))
    print("\nSaved:")
    print(raw_path)
    print(summary_path)

if __name__ == "__main__":
    main()
