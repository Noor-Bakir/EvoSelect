"""Flask API for EvoSelect: leakage-aware feature-selection experiments."""
from __future__ import annotations

import hashlib
import io
import json
import os
import threading
from dataclasses import asdict

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

from data_utils import (
    generate_random_dataset,
    prepare_xy,
    read_csv_text,
    validate_dataset,
)
from evaluate import compare_and_stats, plot_results_base64
from ga_module import GAConfig, run_genetic_algorithm
import traditional_methods as tm

app = Flask(__name__)
_CACHE: dict[str, dict] = {}
_CACHE_LOCK = threading.Lock()
_MAX_CACHE_ITEMS = 64


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _dataset_from_payload(payload):
    raw = payload.get("df")
    if not raw:
        raise ValueError("No dataset was supplied.")
    df = read_csv_text(raw)
    target = payload.get("target") or df.columns[-1]
    validate_dataset(df, target)
    return df, target


def _dataset_key(df, target):
    canonical = df.to_csv(index=False).encode()
    return hashlib.sha256(target.encode() + b"|" + canonical).hexdigest()


def _cache_get(key):
    with _CACHE_LOCK:
        return _CACHE.get(key)


def _cache_put(key, value):
    with _CACHE_LOCK:
        if len(_CACHE) >= _MAX_CACHE_ITEMS:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = value


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "version": "2.0"})


@app.post("/api/generate")
def generate():
    try:
        payload = request.get_json(silent=True) or {}
        df, target = generate_random_dataset(
            n_rows=int(payload.get("nSamples", 200)),
            n_cols=int(payload.get("nFeatures", 20)),
            n_informative=int(payload.get("nInformative", 5)),
            random_state=int(payload.get("random_state", 42)),
        )
        return jsonify({
            "csv": df.to_csv(index=False),
            "target": target,
            "message": "Dataset generated successfully.",
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/upload")
def upload():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file was supplied."}), 400
        file = request.files["file"]
        name = (file.filename or "").lower()
        if name.endswith(".csv"):
            df = pd.read_csv(file)
        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file)
        else:
            return jsonify({"error": "Only CSV and Excel files are supported."}), 400

        target = request.form.get("target") or df.columns[-1]
        validate_dataset(df, target)
        return jsonify({
            "csv": df.to_csv(index=False),
            "target": target,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/fetch")
def fetch_disabled():
    # Remote URL ingestion was deliberately removed from v2 to avoid SSRF.
    return jsonify({
        "error": "Remote URL ingestion is disabled in v2. Upload the dataset instead."
    }), 410


METHODS = {
    "embedding_rf": tm.embedding_rf,
    "filter_chi2": tm.filter_chi2,
    "mutual_info": tm.mutual_info,
    "f_classif": tm.f_classif_filter,
    "l1_logistic": tm.l1_logistic,
    "rfe_rf": tm.rfe_rf,
    "variance_threshold": tm.variance_threshold,
}


@app.post("/api/ga")
def run_ga():
    try:
        payload = request.get_json(silent=True) or {}
        df, target = _dataset_from_payload(payload)

        config = GAConfig(
            pop_size=int(payload.get("pop_size", 30)),
            generations=int(payload.get("generations", 20)),
            crossover_rate=float(payload.get("crossover_rate", 0.8)),
            mutation_rate=float(payload.get("mutation_rate", 0.02)),
            feature_penalty=float(payload.get("feature_penalty", 0.01)),
            cv_folds=int(payload.get("cv_folds", 5)),
            random_state=int(payload.get("random_state", 42)),
            patience=int(payload.get("patience", 5)),
        )

        key = "ga:" + hashlib.sha256(
            (_dataset_key(df, target) + json.dumps(asdict(config), sort_keys=True)).encode()
        ).hexdigest()

        cached = _cache_get(key)
        if cached is not None and payload.get("use_cache", True):
            return jsonify({**cached, "cached": True})

        result = run_genetic_algorithm(df, target, config=config)
        result["dataset"] = {
            "target": target,
            "n_samples": len(df),
            "n_features": df.shape[1] - 1,
        }
        _cache_put(key, result)
        return jsonify({**_json_safe(result), "cached": False})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/traditional/run")
def run_traditional():
    try:
        payload = request.get_json(silent=True) or {}
        df, target = _dataset_from_payload(payload)
        method = payload.get("method")
        if method not in METHODS:
            return jsonify({"error": f"Unknown method: {method}"}), 400

        params = payload.get("params") or {}
        # Include params in the cache key; v1 did not, which could return stale results.
        key_data = {
            "dataset": _dataset_key(df, target),
            "method": method,
            "params": params,
        }
        key = "method:" + hashlib.sha256(
            json.dumps(key_data, sort_keys=True, default=str).encode()
        ).hexdigest()

        cached = _cache_get(key)
        if cached is not None and payload.get("use_cache", True):
            return jsonify({**cached, "cached": True})

        result = METHODS[method](df, target, **params)
        _cache_put(key, result)
        return jsonify({**_json_safe(result), "cached": False})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/run_all")
def run_all():
    try:
        payload = request.get_json(silent=True) or {}
        df, target = _dataset_from_payload(payload)
        results = []

        for method_name in METHODS:
            try:
                results.append(
                    METHODS[method_name](
                        df,
                        target,
                        **(payload.get("params", {}).get(method_name, {})),
                    )
                )
            except Exception as exc:
                results.append({
                    "method": method_name,
                    "selected_features": [],
                    "error": str(exc),
                })

        # GA is deliberately separate because it is much more expensive.
        return jsonify({"methods": _json_safe(results)})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/compare")
def compare():
    try:
        payload = request.get_json(silent=True) or {}
        df, target = _dataset_from_payload(payload)
        methods = payload.get("methods") or []
        stats = compare_and_stats(df, target, methods)
        plots = plot_results_base64(df, target, methods)
        return jsonify({
            "stats": _json_safe(stats),
            "plots": plots,
            "methods_used": [m.get("method") for m in methods],
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/cache/clear_all")
def clear_cache():
    with _CACHE_LOCK:
        _CACHE.clear()
    return jsonify({"success": True})


@app.get("/api/cache/status")
def cache_status():
    with _CACHE_LOCK:
        return jsonify({"items": len(_CACHE), "max_items": _MAX_CACHE_ITEMS})



@app.post("/api/run_all_traditional")
def run_all_traditional_compat():
    payload = request.get_json(silent=True) or {}
    try:
        df, target = _dataset_from_payload(payload)
        out = []
        for name in ["embedding_rf", "l1_logistic", "rfe_rf"]:
            try:
                out.append(METHODS[name](df, target))
            except Exception as exc:
                out.append({"method": name, "selected_features": [], "error": str(exc)})
        return jsonify({"methods": _json_safe(out)})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/run_all_statistical")
def run_all_statistical_compat():
    payload = request.get_json(silent=True) or {}
    try:
        df, target = _dataset_from_payload(payload)
        out = []
        for name in ["filter_chi2", "mutual_info", "f_classif", "variance_threshold"]:
            try:
                out.append(METHODS[name](df, target))
            except Exception as exc:
                out.append({"method": name, "selected_features": [], "error": str(exc)})
        return jsonify({"methods": _json_safe(out)})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/api/results/ga")
def results_ga_compat():
    return jsonify({"ga_result": None, "message": "Results are request-scoped in v2."})


@app.get("/api/results/traditional")
def results_traditional_compat():
    return jsonify({"methods": []})


@app.get("/api/results/statistical")
def results_statistical_compat():
    return jsonify({"methods": []})


if __name__ == "__main__":
    # Production deployment should use gunicorn; debug is never enabled here.
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "5000")), debug=False)
