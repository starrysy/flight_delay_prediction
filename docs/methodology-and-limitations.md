# Methodology and limitations

## Prediction definition

The task is binary classification of `DEP_DEL15`, which equals 1 when a US domestic flight departed at least 15 minutes late. The intended scoring point is before departure, using schedule, carrier, airport, and aircraft-rotation context. The data contains 50,205 January 2019 flights; 8,707 (17.343%) are positive.

## Audit of the original workflow

The original project first made a stratified 70/30 split, then transformed `CARRIER_NAME` and `PREVIOUS_AIRPORT` to the mean `DEP_DEL15` for each category separately in the complete training and test files. This does not prevent leakage:

1. A test row's feature value was calculated using test labels, including its own label.
2. Cross-validation was run after the complete training file was encoded, so each validation fold contributed labels to its own features.
3. The transformation mappings were not packaged with the saved classifier, so the historical model could not accept raw records safely.

The processed files were checked directly: their category codes reproduce category-specific target means within the same file. Historical reported model metrics are therefore excluded from the current README and results.

Additional issues found during the audit:

- Original precision, recall, and F1 were weighted across both classes; with an 82.7% majority class these obscure delayed-flight performance.
- The report and saved Orange workflow disagreed on k-NN neighbours (23 versus 21) and random-forest maximum depth (described as enabled but disabled in the workflow).
- The random forest did not have a fixed seed.
- The serialized classifier relied on already processed columns and was not a deployable raw-data pipeline.
- The original illustrative business-value calculation treated a national annual delay-cost estimate as model revenue without establishing dataset coverage, preventability, intervention effectiveness, or implementation cost.

## Current evaluation design

The refactored workflow uses a single raw CSV and random seed 42:

| Partition | Rows | Purpose |
| --- | ---: | --- |
| Training | 40,164 | Five-fold stratified model comparison and final refit |
| Final test | 10,041 | One evaluation after model selection |

All learned transformations are inside each sklearn pipeline. Numeric columns are median-imputed and standardised. Categorical columns are most-frequent-imputed, categories occurring fewer than five times are grouped, and values are one-hot encoded with unknown-category handling. During cross-validation, each fitted transformer sees only four training folds. After selection, a fresh pipeline is fitted to the complete training partition and applied unchanged to the test set.

The four fixed candidates are:

- prior-probability `DummyClassifier` baseline;
- class-balanced L2 logistic regression;
- distance-weighted 21-neighbour k-NN;
- 300-tree random forest with balanced bootstrap weights and minimum leaf size 5.

No holdout metrics are used for model or hyperparameter selection. Mean five-fold training ROC-AUC is the selection criterion. Average precision, accuracy, delayed-class precision, recall, and F1 are secondary diagnostics. Standard deviations in the result file describe fold variability, not formal confidence intervals.

## Prediction-time feature policy

Included features are schedule or relatively stable contextual fields: departure block, day of week, distance group, aircraft segment, concurrent flights, seats, carrier, departure and previous airport, passenger-volume context, staffing ratios, plane age, and airport coordinates.

Seven fields are conservatively excluded:

| Excluded field(s) | Reason |
| --- | --- |
| `AIRLINE_FLIGHTS_MONTH`, `AIRLINE_AIRPORT_FLIGHTS_MONTH` | They appear to aggregate the same month being predicted; prior-period or scheduled values are not identified. |
| `PRCP`, `SNOW`, `SNWD`, `AWND` | They are realized daily weather summaries, not timestamped forecasts. |
| `TMAX` | A realized daily maximum is not known before many departures. |

The retained passenger and staffing fields are treated as static reference attributes, but the source file does not state their reference period. This assumption should be verified before deployment.

## Recomputed results

Five-fold training cross-validation:

| Model | ROC-AUC | Average precision | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.500 +/- 0.000 | 0.173 +/- 0.000 | 0.827 | 0.000 | 0.000 | 0.000 |
| Logistic regression | 0.653 +/- 0.010 | 0.272 +/- 0.008 | 0.608 | 0.248 | 0.618 | 0.353 |
| k-nearest neighbours | 0.636 +/- 0.009 | 0.263 +/- 0.009 | 0.823 | 0.365 | 0.029 | 0.054 |
| Random forest | **0.678 +/- 0.010** | **0.304 +/- 0.014** | 0.700 | 0.289 | 0.503 | **0.367** |

Final random-forest holdout evaluation:

| ROC-AUC | Average precision | Accuracy | Precision | Recall | F1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.684 | 0.314 | 0.699 | 0.292 | 0.517 | 0.373 |

Confusion matrix counts are 6,119 true negatives, 2,181 false positives, 841 false negatives, and 900 true positives. Accuracy is lower than the constant baseline because the class-balanced model flags substantially more flights; this trade-off is visible in its 51.7% delayed-flight recall. Without a validated cost matrix, the default 0.5 threshold is a reproducible reference rather than an operational recommendation.

Exact machine-readable values are stored in `results/cross_validation_metrics.csv` and `results/test_metrics.json`.

## Interpretation

Permutation importance was calculated after final evaluation on a fixed 5,000-row sample of the holdout, using decrease in ROC-AUC. The leading features were:

| Feature | Mean ROC-AUC decrease |
| --- | ---: |
| Departure block | 0.0676 |
| Previous airport | 0.0137 |
| Segment number | 0.0122 |
| Day of week | 0.0055 |
| Concurrent flights | 0.0041 |

The result describes reliance of this fitted model, not causal effects or stable operational drivers. Correlated airport, geographic, carrier, and capacity fields can divide importance, and near-zero importance does not establish that a variable has no relationship with delays.

## Remaining limitations

- The missing date and flight identifier prevent chronological splitting and make it impossible to determine whether 94 exact duplicate rows are true repeated flights or duplicate records. They are retained rather than silently removed.
- One January sample cannot establish seasonal, route-level, or future-year transportability.
- Random splitting can place similar airport/carrier combinations in both partitions; future work should add out-of-time and grouped robustness checks when dates and flight identifiers are available.
- No forecast weather, destination, inbound delay, scheduled timestamp, cancellation, or tail identifier is available.
- Class weighting changes the operating trade-off and can affect probability calibration. Calibration and threshold choice require a separate validation design and explicit use costs.
- The test-set permutation analysis is post-selection interpretation; it must not be used to revise the model and then re-report the same holdout as final.
