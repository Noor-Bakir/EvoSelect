"""Feature-selection baselines for EvoSelect.

The important design rule is that selection and preprocessing are fitted
inside the CV pipeline. This prevents feature-selection leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    SelectKBest,
    chi2,
    f_classif,
    mutual_info_classif,
    SelectFromModel,
    VarianceThreshold,
    RFE,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer


def _cv(model, X, y, folds=5, seed=42):
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return float(np.mean(cross_val_score(
        model, X, y, cv=cv, scoring="accuracy", n_jobs=-1
    )))


def _base_classifier(seed=42):
    return RandomForestClassifier(
        n_estimators=100, random_state=seed, n_jobs=-1
    )


def _result(name, selected, score, **meta):
    return {
        "method": name,
        "selected_features": list(selected),
        "cv_score": score,
        "meta": meta,
    }


def variance_threshold(df, target_name, threshold=0.0, cv=5, random_state=42):
    X, y = df.drop(columns=[target_name]), df[target_name]
    selector = VarianceThreshold(threshold=threshold)
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("selector", selector),
        ("classifier", _base_classifier(random_state)),
    ])
    score = _cv(model, X, y, cv, random_state)
    # Fit only for reporting selected features; score above remains CV-safe.
    fitted = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("selector", selector),
    ]).fit(X, y)
    selected = X.columns[fitted.named_steps["selector"].get_support()].tolist()
    return _result("variance_threshold", selected, score, threshold=threshold)


def filter_chi2(df, target_name, k=10, cv=5, random_state=42):
    X, y = df.drop(columns=[target_name]), df[target_name]
    k = min(int(k), X.shape[1])
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", MinMaxScaler()),
        ("selector", SelectKBest(chi2, k=k)),
        ("classifier", _base_classifier(random_state)),
    ])
    score = _cv(model, X, y, cv, random_state)
    fitted = model.fit(X, y)
    selected = X.columns[fitted.named_steps["selector"].get_support()].tolist()
    return _result("filter_chi2", selected, score, k=k)


def mutual_info(df, target_name, k=10, cv=5, random_state=42):
    X, y = df.drop(columns=[target_name]), df[target_name]
    k = min(int(k), X.shape[1])
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("selector", SelectKBest(mutual_info_classif, k=k)),
        ("classifier", _base_classifier(random_state)),
    ])
    score = _cv(model, X, y, cv, random_state)
    fitted = model.fit(X, y)
    selected = X.columns[fitted.named_steps["selector"].get_support()].tolist()
    return _result("mutual_info", selected, score, k=k)


def f_classif_filter(df, target_name, k=10, cv=5, random_state=42):
    X, y = df.drop(columns=[target_name]), df[target_name]
    k = min(int(k), X.shape[1])
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("selector", SelectKBest(f_classif, k=k)),
        ("classifier", _base_classifier(random_state)),
    ])
    score = _cv(model, X, y, cv, random_state)
    fitted = model.fit(X, y)
    selected = X.columns[fitted.named_steps["selector"].get_support()].tolist()
    return _result("f_classif", selected, score, k=k)


def embedding_rf(df, target_name, top_k=None, cv=5, random_state=42):
    X, y = df.drop(columns=[target_name]), df[target_name]
    top_k = max(1, min(int(top_k or max(1, X.shape[1] // 4)), X.shape[1]))
    selector_model = RandomForestClassifier(
        n_estimators=100, random_state=random_state, n_jobs=-1
    )
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("selector", SelectFromModel(selector_model, max_features=top_k)),
        ("classifier", _base_classifier(random_state)),
    ])
    score = _cv(model, X, y, cv, random_state)
    fitted = model.fit(X, y)
    selected = X.columns[fitted.named_steps["selector"].get_support()].tolist()
    return _result("embedding_rf", selected, score, top_k=top_k)


def l1_logistic(df, target_name, C=0.1, cv=5, random_state=42):
    X, y = df.drop(columns=[target_name]), df[target_name]
    selector_model = LogisticRegression(
        penalty="l1", solver="liblinear", C=float(C),
        max_iter=2000, random_state=random_state
    )
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("selector", SelectFromModel(selector_model, threshold="mean")),
        ("classifier", _base_classifier(random_state)),
    ])
    score = _cv(model, X, y, cv, random_state)
    fitted = model.fit(X, y)
    selected = X.columns[fitted.named_steps["selector"].get_support()].tolist()
    return _result("l1_logistic", selected, score, C=C)


def rfe_rf(df, target_name, n_features_to_select=None, cv=5, random_state=42):
    X, y = df.drop(columns=[target_name]), df[target_name]
    n = max(1, min(
        int(n_features_to_select or max(1, X.shape[1] // 4)),
        X.shape[1],
    ))
    estimator = RandomForestClassifier(
        n_estimators=100, random_state=random_state, n_jobs=-1
    )
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("selector", RFE(estimator=estimator, n_features_to_select=n)),
        ("classifier", _base_classifier(random_state)),
    ])
    score = _cv(model, X, y, cv, random_state)
    fitted = model.fit(X, y)
    selected = X.columns[fitted.named_steps["selector"].get_support()].tolist()
    return _result("rfe_rf", selected, score, n_features_to_select=n)
