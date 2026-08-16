# Flight Delay Prediction

An exploratory machine-learning project that predicts whether a US domestic
flight will depart at least 15 minutes late. The project uses 50,205 flights
from January 2019 and compares k-nearest neighbours, logistic regression, and
random forest classifiers in Orange Data Mining.

This repository is a reorganized version of a university team project. It is
preserved as a portfolio case study and documents the validation limitations
that must be addressed before the model is used in production.

## Project highlights

- Explored operational, carrier, airport, and weather factors associated with
  departure delays.
- Built and compared three binary classifiers against a constant baseline.
- Used a stratified 70/30 train/test split and cross-validation in Orange.
- Selected a 40-tree random forest in the original analysis.
- Considered false-positive and false-negative costs in the model-selection
  discussion.

The original report records a test AUC of **0.716** and accuracy of **0.835**
for the random forest. These figures are historical results, not validated
production estimates: the original target-encoding procedure leaks label
information into evaluation data. See
[Methodology and limitations](docs/methodology-and-limitations.md).

## Repository structure

```text
.
├── docs/
│   ├── flight-delay-prediction-report.pdf
│   └── methodology-and-limitations.md
└── workflows/
    ├── evaluation/
    │   ├── Evaluation Workflow.ows
    │   ├── training for team9.csv
    │   ├── training_data_coded.csv
    │   └── testing_data_coded.csv
    └── inference/
        ├── Final Model Workflow.ows
        └── final_model_rf.pkcls
```

## Dataset

The raw dataset contains 50,205 flight records and 24 columns. The target,
`DEP_DEL15`, equals 1 when departure was delayed by at least 15 minutes and 0
otherwise. Delayed flights account for approximately 17.34% of the data.

The processed training and testing files contain 35,144 and 15,061 rows,
respectively. They are included to reproduce the historical Orange workflow,
not as examples of a leakage-safe preprocessing pipeline.

## Running the historical workflows

1. Install [Orange Data Mining](https://orangedatamining.com/).
2. Open `workflows/evaluation/Evaluation Workflow.ows` to inspect model
   comparison and evaluation.
3. If Orange cannot resolve a saved input automatically, select the CSV stored
   in the same directory as the workflow.
4. Open `workflows/inference/Final Model Workflow.ows` to inspect the saved-model
   prediction flow.

Orange workflows store application state, including paths from the machines on
which they were created. The relevant data and model files remain beside their
workflows to make relinking straightforward.

> [!CAUTION]
> The `.pkcls` model is a Python pickle. Only load it if you trust this
> repository; unpickling an untrusted file can execute arbitrary code.

## Recommended next iteration

- Put target encoding inside each cross-validation fold and learn all mappings
  from training data only.
- Apply the training mappings to the test set and use a documented fallback for
  unseen categories.
- Package preprocessing and prediction as one reproducible pipeline.
- Report delayed-class precision, recall, F1, PR-AUC, calibration, and
  threshold-specific confusion matrices.
- Pin the Orange/Python environment and record random seeds.
- Validate on additional months to measure seasonality and temporal drift.

## Contributors

Original university project by Songlin Yang, Qianting Yang, Shijia Rong, and
Xinyi Ji (2021).

