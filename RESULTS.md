# EvoSelect — Reference Results

## UCI Sonar

Reference nested-CV experiment using the V2 Genetic Algorithm configuration.

| Method | Accuracy mean | Accuracy std | F1 mean | F1 std | Features mean | Reduction | Runtime mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full features | 0.822429 | 0.058368 | 0.817602 | 0.063891 | 60.00 | 0.00% | 0.27 s |
| Chi² | 0.769427 | 0.061846 | 0.765225 | 0.064469 | 15.00 | 75.00% | 0.26 s |
| RFE + RF | 0.759903 | 0.083334 | 0.756909 | 0.084346 | 15.00 | 75.00% | 14.15 s |
| Genetic Algorithm (V2) | 0.812491 | 0.001554 | 0.811362 | 0.000362 | 16.33 | 72.78% | 521.86 s |
| L1 Logistic | 0.745273 | 0.045821 | 0.741221 | 0.041189 | 9.67 | 83.89% | 0.26 s |
| F-test | 0.740718 | 0.063823 | 0.737540 | 0.063282 | 15.00 | 75.00% | 0.28 s |
| Mutual Information | 0.764527 | — | — | — | 15.00 | 75.00% | — |
| RF Embedded | 0.726432 | 0.092170 | 0.722341 | 0.095030 | 15.00 | 75.00% | 0.59 s |

The GA result is best interpreted as a compactness/performance trade-off rather than an accuracy win: about 81.25% accuracy with roughly 16/60 features versus 82.24% with all 60.

The values above are the reference V2 run used for the project narrative. Re-run the benchmark before publishing if exact reproducibility of the CSV artifacts is required.
