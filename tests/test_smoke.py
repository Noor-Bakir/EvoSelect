import pandas as pd

from data_utils import generate_random_dataset, validate_dataset
from ga_module import GAConfig, run_genetic_algorithm
from traditional_methods import mutual_info


def test_dataset_generation():
    df, target = generate_random_dataset(
        n_rows=80, n_cols=8, n_informative=3, random_state=42
    )
    info = validate_dataset(df, target)
    assert info.n_samples == 80
    assert info.n_features == 8


def test_ga_is_reproducible():
    df, target = generate_random_dataset(
        n_rows=80, n_cols=8, n_informative=3, random_state=42
    )
    config = GAConfig(
        pop_size=6, generations=3, cv_folds=3,
        random_state=42, patience=10
    )
    a = run_genetic_algorithm(df, target, config)
    b = run_genetic_algorithm(df, target, config)
    assert a["selected_features"] == b["selected_features"]
    assert a["final_score"] == b["final_score"]


def test_baseline_returns_cv_score():
    df, target = generate_random_dataset(
        n_rows=80, n_cols=8, n_informative=3, random_state=42
    )
    result = mutual_info(df, target, k=3, cv=3)
    assert result["selected_features"]
    assert 0.0 <= result["cv_score"] <= 1.0
