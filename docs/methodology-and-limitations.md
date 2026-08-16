# Methodology and limitations

## Historical methodology

The original project used a stratified 70/30 split:

| Dataset | Rows | Delayed flights | Delay rate |
| --- | ---: | ---: | ---: |
| Training | 35,144 | 6,095 | 17.343% |
| Testing | 15,061 | 2,612 | 17.343% |
| Total | 50,205 | 8,707 | 17.343% |

The analysis compared k-nearest neighbours, logistic regression, random forest,
and a constant baseline. Numerical scaling was configured for the distance- and
coefficient-based models. The selected random forest used 40 trees, five
features per split, and a minimum split size of five in the saved Orange
workflow.

## Known validation issue

`PREVIOUS_AIRPORT` and `CARRIER_NAME` were target-encoded by calculating the
mean of `DEP_DEL15` for each category. The transformation was performed
separately on the complete training and testing datasets.

This creates two forms of target leakage:

1. Test-set feature values are calculated using test-set labels.
2. Cross-validation feature values are calculated before folds are formed, so
   each validation fold contributes labels to its own encoded features.

The encoded CSVs confirm this behavior: every category code equals the target
mean for the corresponding category in that same file, within floating-point
precision. Consequently, the historical test AUC of 0.716 and accuracy of 0.835
should not be presented as leakage-free estimates.

## Leakage-safe evaluation design

For each cross-validation split:

1. Learn category-to-target mappings from the fold's training partition.
2. Transform its validation partition using only those mappings.
3. Replace unseen categories with the training partition's overall target
   mean, or another explicitly documented fallback.
4. Fit scaling and any feature selection on the training partition only.
5. Evaluate on the transformed validation partition.

After selecting a model, fit preprocessing on the full training set exactly
once and apply it unchanged to the held-out test set. A final deliverable should
serialize preprocessing and classification together.

## Other limitations

- The data covers only January 2019, so seasonal generalization is unknown.
- The positive class represents only 17.34% of observations; accuracy and
  weighted metrics can hide poor delayed-flight recall.
- The saved random forest does not use a fixed random state.
- The report says k-NN used 23 neighbours, while the saved workflow specifies
  21.
- The report describes a maximum tree depth of three, but that option is
  disabled in the saved workflow.
- The inference workflow expects already processed features and does not
  contain a deployable raw-data preprocessing stage.
- The business-value calculation is illustrative and does not scale annual
  delay cost by coverage, preventability, intervention effectiveness, or
  implementation cost.

