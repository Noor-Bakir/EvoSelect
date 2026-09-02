"""Leakage-aware genetic feature selection for BIA - V3.

V3 keeps the V2 stability penalty and elitism, while reducing the cost of
fitness evaluation. The benchmark's final model/evaluation is unchanged.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class GAConfig:
    pop_size: int = 30
    generations: int = 20
    crossover_rate: float = 0.8
    mutation_rate: float = 0.02
    selection_pressure: int = 3
    feature_penalty: float = 0.01
    stability_penalty: float = 0.25
    cv_folds: int = 5
    random_state: int = 42
    patience: int = 5
    elite_size: int = 2
    fitness_estimators: int = 30


def _rngs(seed: int):
    return random.Random(seed), np.random.default_rng(seed)


def make_individual(
    n_features: int,
    rng: random.Random,
    p_select: float = 0.25,
) -> list[int]:
    chromosome = [
        1 if rng.random() < p_select else 0
        for _ in range(n_features)
    ]
    if not any(chromosome):
        chromosome[rng.randrange(n_features)] = 1
    return chromosome


def _fitness(
    chromosome: list[int],
    X: pd.DataFrame,
    y: pd.Series,
    config: GAConfig,
    *,
    cache: dict[tuple[int, ...], float],
) -> float:
    key = tuple(chromosome)
    if key in cache:
        return cache[key]

    mask = np.asarray(chromosome, dtype=bool)
    if not mask.any():
        return -1.0

    # Cheaper RF is used ONLY while searching chromosomes.
    # The outer benchmark still evaluates the selected features using
    # benchmark.py's unchanged final classifier.
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=config.fitness_estimators,
                    max_depth=None,
                    random_state=config.random_state,
                    n_jobs=1,
                ),
            ),
        ]
    )

    cv = StratifiedKFold(
        n_splits=config.cv_folds,
        shuffle=True,
        random_state=config.random_state,
    )

    try:
        cv_scores = cross_val_score(
            model,
            X.loc[:, mask],
            y,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
        )
        mean_score = float(np.mean(cv_scores))
        std_score = float(np.std(cv_scores))

        # Prefer solutions that are accurate AND stable across CV folds.
        score = mean_score - config.stability_penalty * std_score
    except Exception:
        score = 0.0

    fraction_selected = float(mask.sum() / len(mask))
    value = score - config.feature_penalty * fraction_selected

    cache[key] = value
    return value


def _tournament(
    population,
    fitnesses,
    rng: random.Random,
    k: int,
):
    indices = rng.sample(
        range(len(population)),
        min(k, len(population)),
    )
    best = max(indices, key=lambda i: fitnesses[i])
    return population[best][:]


def _crossover(a, b, rng: random.Random):
    if len(a) < 2:
        return a[:], b[:]

    point = rng.randint(1, len(a) - 1)
    return (
        a[:point] + b[point:],
        b[:point] + a[point:],
    )


def _mutate(
    chromosome,
    rate,
    rng: random.Random,
):
    child = [
        1 - bit if rng.random() < rate else bit
        for bit in chromosome
    ]

    if not any(child):
        child[rng.randrange(len(child))] = 1

    return child


def run_genetic_algorithm(
    df: pd.DataFrame,
    target_name: str,
    config: GAConfig | None = None,
    verbose: bool = False,
):
    config = config or GAConfig()

    if config.pop_size < 4:
        raise ValueError("pop_size must be >= 4.")
    if config.generations < 1:
        raise ValueError("generations must be >= 1.")
    if not 0 <= config.crossover_rate <= 1:
        raise ValueError(
            "crossover_rate must be between 0 and 1."
        )
    if not 0 <= config.mutation_rate <= 1:
        raise ValueError(
            "mutation_rate must be between 0 and 1."
        )
    if config.cv_folds < 2:
        raise ValueError("cv_folds must be >= 2.")
    if config.stability_penalty < 0:
        raise ValueError("stability_penalty must be >= 0.")
    if config.elite_size < 0:
        raise ValueError("elite_size must be >= 0.")
    if config.elite_size >= config.pop_size:
        raise ValueError(
            "elite_size must be smaller than pop_size."
        )
    if config.fitness_estimators < 1:
        raise ValueError(
            "fitness_estimators must be >= 1."
        )

    X = df.drop(columns=[target_name]).copy()
    y = df[target_name].copy()

    if not pd.api.types.is_numeric_dtype(y):
        y = pd.Series(
            pd.Categorical(y).codes,
            index=y.index,
        )

    rng, _ = _rngs(config.random_state)

    population = [
        make_individual(X.shape[1], rng)
        for _ in range(config.pop_size)
    ]

    fitness_cache: dict[tuple[int, ...], float] = {}

    best_solution = None
    best_fitness = -np.inf
    history = []
    stagnant = 0

    for generation in range(config.generations):
        fitnesses = [
            _fitness(
                ind,
                X,
                y,
                config,
                cache=fitness_cache,
            )
            for ind in population
        ]

        best_idx = int(np.argmax(fitnesses))
        generation_best = float(fitnesses[best_idx])
        generation_solution = population[best_idx][:]

        if generation_best > best_fitness + 1e-12:
            best_fitness = generation_best
            best_solution = generation_solution
            stagnant = 0
        else:
            stagnant += 1

        history.append(
            {
                "generation": generation + 1,
                "best_fitness": generation_best,
                "selected_count": int(sum(generation_solution)),
                "evaluated_solutions": len(fitness_cache),
            }
        )

        if verbose:
            print(
                f"[GA] generation={generation + 1} "
                f"fitness={generation_best:.4f} "
                f"features={sum(generation_solution)}"
            )

        if stagnant >= config.patience:
            break

        # Elitism: preserve the strongest chromosomes.
        elite_count = min(
            config.elite_size,
            config.pop_size,
        )

        elite_indices = sorted(
            range(len(population)),
            key=lambda i: fitnesses[i],
            reverse=True,
        )[:elite_count]

        new_population = [
            population[i][:]
            for i in elite_indices
        ]

        while len(new_population) < config.pop_size:
            p1 = _tournament(
                population,
                fitnesses,
                rng,
                config.selection_pressure,
            )
            p2 = _tournament(
                population,
                fitnesses,
                rng,
                config.selection_pressure,
            )

            if rng.random() < config.crossover_rate:
                c1, c2 = _crossover(p1, p2, rng)
            else:
                c1, c2 = p1, p2

            new_population.extend(
                [
                    _mutate(
                        c1,
                        config.mutation_rate,
                        rng,
                    ),
                    _mutate(
                        c2,
                        config.mutation_rate,
                        rng,
                    ),
                ]
            )

        population = new_population[:config.pop_size]

    assert best_solution is not None

    selected_features = [
        X.columns[i]
        for i, bit in enumerate(best_solution)
        if bit
    ]

    return {
        "method": "genetic",
        "selected_features": selected_features,
        "final_score": float(best_fitness),
        "history": history,
        "best_chromosome": best_solution,
        "best_chromosome_str": "".join(
            map(str, best_solution)
        ),
        "best_selected_count": int(sum(best_solution)),
        "config": config.__dict__,
        "cache_size": len(fitness_cache),
    }
