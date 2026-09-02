# EvoSelect — Evolutionary Feature Selection Benchmark

**EvoSelect** is a leakage-aware feature-selection framework that uses a Genetic Algorithm (GA) to search for compact feature subsets and compares the result with classical statistical and model-based selection methods.

The project is designed as an experimental benchmark rather than a claim of state-of-the-art performance.

## What EvoSelect does

Given a supervised classification dataset, EvoSelect can:

- generate synthetic datasets or accept uploaded CSV/Excel data through the web interface
- search feature subsets with a Genetic Algorithm
- compare the GA with classical feature-selection methods
- evaluate selected features with a common machine-learning classifier
- report predictive performance, feature reduction, and runtime
- run leakage-aware cross-validation workflows

## Feature-selection methods

The benchmark currently includes:

- **Genetic Algorithm** — binary chromosome, tournament selection, single-point crossover, bit-flip mutation, elitism, stability-aware fitness, early stopping, and fitness memoization
- **Chi-square (Chi²)** filter
- **Mutual Information** filter
- **ANOVA F-test** filter
- **L1 Logistic Regression** embedded selection
- **Random Forest** embedded selection
- **Recursive Feature Elimination (RFE)** with Random Forest
- **Full-feature baseline**

## Genetic Algorithm objective

For a candidate subset `S`, the optimization objective combines predictive quality, stability, and model compactness:

```text
fitness = mean_CV_accuracy
          - stability_penalty * CV_accuracy_std
          - feature_penalty * (|S| / |F|)
```

where `F` is the complete feature set.

The stability term discourages subsets whose cross-validation performance varies substantially between folds, while the feature penalty encourages smaller subsets.

## Experimental integrity

EvoSelect was refactored specifically to avoid common feature-selection leakage:

1. Imputation and scaling are performed inside cross-validation pipelines.
2. Classical selectors are fitted on training data rather than on the complete benchmark dataset.
3. The nested benchmark runs feature selection inside each outer training fold.
4. The outer test fold remains untouched until final evaluation.
5. The same final Random Forest evaluation procedure is used for the compared methods.
6. Experiments use fixed random seeds for reproducibility.
7. Dataset identity and method configuration can be incorporated into experiment caching.
8. Remote URL ingestion is disabled in the Flask application to avoid SSRF exposure.
9. Flask debug mode is disabled for normal execution.

## Benchmark result: UCI Sonar

The current reference experiment uses the UCI Sonar dataset (208 samples, 60 numeric features) with 3 outer stratified folds. The GA is run inside each outer training fold with the V2 configuration.

| Method | Accuracy | F1 | Features | Reduction |
|---|---:|---:|---:|---:|
| Full features | **82.24%** | **81.76%** | 60 | 0.0% |
| Chi² | 76.94% | 76.52% | 15 | 75.0% |
| RFE + RF | 75.99% | 75.69% | 15 | 75.0% |
| **Genetic Algorithm (V2)** | **81.25%** | **81.14%** | **16.3** | **72.8%** |
| L1 Logistic | 74.53% | 74.12% | 9.7 | 83.9% |
| F-test | 74.07% | 73.75% | 15 | 75.0% |
| Mutual Information | 74.06% | 73.57% | 15 | 75.0% |
| RF Embedded | 72.64% | 72.23% | 15 | 75.0% |

### Interpretation

The reference GA result is not the highest-accuracy method. Its main result is the trade-off between predictive performance and dimensionality: approximately **81.25% mean accuracy using about 16 of 60 features**, compared with **82.24% using all 60 features**.

This means the GA sacrifices about one percentage point of mean accuracy while removing roughly 73% of the features in this experiment.

These results should not be generalized as proof that Genetic Algorithms outperform other feature-selection methods. A broader claim would require experiments across multiple datasets and repeated/nested validation.

## Project structure

```text
EvoSelect/
├── app.py                  # Flask web application and API
├── data_utils.py           # Dataset validation and preprocessing utilities
├── evaluate.py             # Evaluation and result helpers
├── ga_module.py            # Genetic Algorithm implementation
├── traditional_methods.py  # Classical feature-selection methods
├── benchmark.py            # Offline nested Sonar benchmark
├── requirements.txt
├── templates/
│   └── index.html
├── static/
│   ├── css/
│   └── js/
└── tests/
    └── test_smoke.py
```

## Run locally

### 1. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the application

```bash
python app.py
```

Then open the local Flask address shown in the terminal.

## Run the benchmark

The offline benchmark expects the Sonar file at:

```text
data/sonar.all-data
```

Then run:

```bash
python benchmark.py
```

The benchmark writes:

```text
benchmark_results/sonar_nested_raw.csv
benchmark_results/sonar_nested_summary.csv
```

> The benchmark can take several minutes because the GA performs an evolutionary search inside each outer training fold. This is intentional: the benchmark prioritizes a fair evaluation over demo-speed execution.

## API

Main endpoints include:

- `GET /api/health`
- `POST /api/generate`
- `POST /api/upload`
- `POST /api/ga`
- `POST /api/traditional/run`
- `POST /api/run_all`
- `POST /api/compare`
- `GET /api/cache/status`
- `POST /api/cache/clear_all`

## Limitations

The current benchmark is a research prototype, not a production AutoML system. In particular:

- the main reported experiment uses one public dataset
- only a small number of outer folds are used in the reference benchmark
- the GA can be computationally expensive
- the current evidence does not establish superiority over classical methods
- broader conclusions require multiple datasets and repeated/nested evaluation

## Portfolio positioning

A defensible project description is:

> **EvoSelect is a leakage-aware evolutionary feature-selection benchmark that compares a Genetic Algorithm with classical statistical and model-based selection methods using nested cross-validation.**

Avoid describing EvoSelect as state-of-the-art or claiming that the GA is universally superior.

## License

Add the license that matches your intended GitHub distribution before publishing the repository.
