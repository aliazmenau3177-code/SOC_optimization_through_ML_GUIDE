# Title: SOC_Optimization through ML - Microsoft GUIDE Dataset

Reproducible code for the paper *"An ensemble machine learning framework for reducing false positives in security operations centers using big data analytics"* (PeerJ Computer Science).

# Description

This Python code is the result of our research on optimising Security Operations Centres (SOCs) by combining big data analysis, data mining (DM) and machine learning (ML).

The code applies a layered ensemble of DM/ML algorithms to Microsoft GUIDE, the SOC-specific triage dataset released by Microsoft for public research:

* supervised classification (Random Forest and XGBoost);
* unsupervised anomaly detection (Isolation Forest);
* Apriori class-association rules mined over a causal 60-minute incident context;
* weighted score fusion with validation-selected weights and thresholds;
* three-tier alert routing (auto-dismiss / analyst review / escalate).

The aim is a substantial reduction of false-positive alerts reaching SOC analysts while true threats are still detected. On the official GUIDE_Test file (4,147,992 alerts), the framework attains a false-positive rate of **6.63%** (95% CI 5.96–7.32%) at a recall of **0.962**, against 38.95% for a detector-reputation rule baseline (relative reduction 83.0%).

The script carries out every step automatically when it is given the two GUIDE files:

* loading the data and building the features;
* training the models and evaluating them;
* running all supplementary analyses requested during peer review;
* writing all tables and figures of the paper.

The research is intended as one step towards a more automated, context-aware SOC. The evaluation is offline and retrospective, and the framework is designed as decision support with analysts kept in the loop.

# Dataset Information

The code uses Microsoft's **GUIDE** dataset, the SOC-specific dataset released for public research. It contains real-world SOC triage telemetry. Each record is one piece of evidence of an alert, and each incident is graded by analysts as TruePositive, BenignPositive or FalsePositive.

GUIDE is distributed as two separate files, both used **complete and unmodified**:

| File | Records | Columns | Incidents (OrgId, IncidentId) | Organisations | Threat share |
|---|---|---|---|---|---|
| GUIDE_Train.csv | 9,516,837 (9,465,497 with a triage grade) | 45 | 730,778 | 5,769 | 35.10% |
| GUIDE_Test.csv | 4,147,992 | 46 (incl. `Usage`) | 303,220 | 4,400 | 35.98% |

The columns fall into five main categories of information:
1. Identifiers (Id, OrgId, IncidentId, AlertId and anonymised entity identifiers)
2. Alert metadata (Timestamp, DetectorId, AlertTitle, Category, SuspicionLevel, ...)
3. MITRE ATT&CK annotations (MitreTechniques)
4. Host / network context (DeviceId, IpAddress, AccountName, OSFamily, CountryCode, ...)
5. Enrichment labels (IncidentGrade, LastVerdict, ActionGrouped, ActionGranular)

The script processes the dataset in the following steps:
1. Removal of attributes that do not affect alert correctness or help infer whether an alert is a threat (record identifiers such as Id, Sha256, Url, registry and folder paths).
2. Removal of leakage attributes: post-triage outcomes (IncidentGrade as target, LastVerdict, ActionGrouped, ActionGranular) that would bias the learning process.
3. Removal of redundancy and noise through feature engineering: causal, label-free context features computed only from past events.
4. Imputation and encoding of values: median imputation of numeric fields; one-hot and frequency encoding of categorical fields.
5. Label transformation: TruePositive → 1 (threat); BenignPositive and FalsePositive → 0 (non-threat).
6. Encoding of the important features for use by the ML algorithms: **151 model features** after encoding.

The number of records is not reduced, except that 51,340 GUIDE_Train records without a triage grade are removed. GUIDE_Train is split, by whole incidents, into a training set (85%) and a validation set (15%). GUIDE_Test is used once, as the independent test set. The steps are explained in detail under the heading Methodology.

# Code Information

The code file in this repository is **`guide_soc_experiment.py`**. It is a single Python script that runs from a terminal or from any Python IDE (we used PyCharm on Ubuntu). It contains the following execution steps:
1. Importing of libraries and the configuration (`CONFIG`, all hyper-parameters in one place)
2. Loading of data (both GUIDE files, read in memory-efficient chunks)
3. Creating binary labels (TruePositive = 1; BenignPositive / FalsePositive = 0)
4. Removing leakage and identifier columns from the dataset
5. Feature engineering (temporal, MITRE and causal incident-context features)
6. Splitting features and labels; incident-disjoint training/validation split of GUIDE_Train
7. Categorical encoding of features (fitted on the training split only)
8. Execution of the Random Forest algorithm
9. Execution of the XGBoost algorithm (early stopping on the validation split)
10. Execution of Isolation Forest for anomaly detection
11. Execution of the Apriori algorithm (class-association rules for alert correlation)
12. Fusion of models (weights and decision threshold selected on the validation split) and three-tier alert routing
13. Evaluation of results on GUIDE_Test: all metrics, 95% confidence intervals, McNemar tests, baselines and ablation study
14. Identifying important features and the top association rules
15. Generating graphs: confusion matrix, ablation, ROC/PR curves, risk distribution, scalability and operating points, feature importance
16. Execution of the label-shuffle sanity test and the leakage check
17. Supplementary analyses: temporal hold-out, unseen organisations, robustness to missing fields, latency, scalability, sensitivity, cross-validation and seed variation

# Method of Access, Usage Instructions and Loading of Data

The dataset (Microsoft GUIDE) can be downloaded from Kaggle. The two files are several gigabytes in size, and both are needed, unchanged.

## Method of Access - Dataset Link

<https://www.kaggle.com/datasets/Microsoft/microsoft-security-incident-prediction>

Download `GUIDE_Train.csv` and `GUIDE_Test.csv` and put them in one folder, for example:

```
GUIDE/
  GUIDE_Train.csv
  GUIDE_Test.csv
  guide_soc_experiment.py
```

## Loading of Data

You don't need to write any loading code. The script finds the two files in the folder given with `--data-dir`. To check that the files are complete before a long run, you can load a few columns as follows:

```python
import pandas as pd

DATA_DIR = "/path/to/GUIDE"     # folder containing the two GUIDE files

for name in ["GUIDE_Train.csv", "GUIDE_Test.csv"]:
    n = 0
    for chunk in pd.read_csv(f"{DATA_DIR}/{name}", usecols=["IncidentGrade"], chunksize=1_000_000):
        n += len(chunk)
    print(name, "rows:", n)      # expected: 9,516,837 and 4,147,992
```

## Usage Instructions

Once the two files are in place, run the script from a terminal:

```bash
cd /path/to/GUIDE
python3 guide_soc_experiment.py --data-dir .
```

For a long run on a remote or virtual machine, run it in the background and follow the log:

```bash
nohup python3 guide_soc_experiment.py --data-dir . > run_output.txt 2>&1 &
tail -f guide_results/run.log
```

The script carries out all the steps listed under Code Information. It writes every result to the folder `guide_results/`, inside the folder where you run the command. The run is divided into stages. Each stage saves its own result file, so an interrupted run can be continued with `--resume`:

```bash
python3 guide_soc_experiment.py --data-dir . --resume
```

| Option | Meaning |
|---|---|
| `--data-dir` | Folder with GUIDE_Train.csv and GUIDE_Test.csv |
| `--out` | Output folder (default `guide_results`) |
| `--stages` | Stages to run (default: all, see below) |
| `--resume` | Skip stages whose result file already exists |
| `--train-frac` | Share of GUIDE_Train incidents to use (default 1.0 = all; use 0.5 on machines with less than 16 GB RAM) |
| `--aux-frac` | Share of GUIDE_Train incidents used by the auxiliary stages (default 0.25) |
| `--quick` | Small, fast configuration for an installation check only (not for results) |

A quick installation check takes a few minutes and is not intended for results:

```bash
python3 guide_soc_experiment.py --data-dir . --quick --train-frac 0.02 --aux-frac 0.02 --out smoke_test
```

# Requirements

The script needs the following Python libraries (versions used for the reported results in brackets). `requirements.txt` lists them:

```
numpy (2.2.6)
pandas (2.3.3)
scipy (1.15.3)
scikit-learn (1.7.2)
xgboost (3.2.0)
mlxtend (0.23.4)
matplotlib (3.10.8)
openpyxl (optional, for the Excel file of all tables)
```

To install them in a virtual environment:

```bash
python3 -m venv soc_env
source soc_env/bin/activate
pip install -r requirements.txt
```

The script imports them itself; for example:

```python
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
```

The processing / computation requirements for the complete run are as follows:
1. A machine with at least 4 CPU cores (8 recommended)
2. **20 GB of RAM** recommended (the peak memory of the reported run was 15.6 GB). With 16 GB, close other programs and add swap space, or use `--train-frac 0.5`.
3. About 10 GB of free disk space for caches and outputs
4. Linux (tested on Ubuntu 22.04) with Python 3.10–3.13. We ran the code in a Python virtual environment, from PyCharm and from the terminal.

# Methodology

The processes applied to the original Microsoft GUIDE files to obtain a machine-learning-ready dataset are as follows:

1. **Column normalisation after loading:** column names are stripped. Both files are read in chunks, and categorical fields are stored in compressed form so that the complete files fit in memory.
2. **Binary label construction:** TruePositive → 1; BenignPositive and FalsePositive → 0.
3. **Incident key:** GUIDE reuses incident numbers across organisations, so an incident is identified by the pair (OrgId, IncidentId).
4. **Leakage attribute removal:** IncidentGrade (the target), LastVerdict, ActionGrouped and ActionGranular.
5. **Irrelevant attribute removal:** record identifiers such as Id, Sha256, Url, registry keys/values, FolderPath, account SIDs/UPNs/object IDs, NetworkMessageId and Usage.
6. **Feature engineering:**
   a. Temporal feature engineering: the raw timestamp is converted into hour of day and day of week.
   b. MITRE feature engineering: the multi-valued MITRE ATT&CK technique string is converted into the number of techniques (MitreCount); individual techniques are used as items for Apriori.
   c. Causal incident-context features, using only events at or before the current alert:
      * alerts and evidence so far in the incident;
      * incident age;
      * alerts of the same organisation in the previous 60 minutes;
      * other alert categories in the same incident in the previous 60 minutes.
7. **Incident-disjoint split:** whole incidents of GUIDE_Train are assigned to training (85%) or validation (15%) with a deterministic, class-stratified assignment. GUIDE_Test is the test set. No incident appears in two splits, and this is verified.
8. **Encoding (fitted on the training split only):**
   * one-hot encoding of categorical fields, keeping the 30 most frequent training categories per field with at least 50 occurrences; rarer and unseen values become "infrequent", and missing values get an "NA" indicator;
   * frequency encoding of high-cardinality identifiers;
   * median imputation of numeric fields.
9. **Model layers:**
   * Random Forest (200 trees, trained on a fixed sample of 1,000,000 training records);
   * XGBoost (trained on all 8,045,675 training records, with early stopping on validation);
   * Isolation Forest (anomaly score of Eq. 11);
   * Apriori class-association rules (support 0.01, confidence 0.60, lift 1.10).
10. **Fusion and routing:**
    * weights for Risk = α·RF + β·XGB + γ·IF + δ·Apriori and the decision threshold are chosen on the validation split, minimising the false-positive rate subject to recall ≥ 0.95;
    * two further thresholds route alerts into auto-dismiss (≤ 1% of threats), analyst review and escalation (precision ≥ 0.95).
11. **Evaluation:** GUIDE_Test is scored once. 95% confidence intervals use a cluster bootstrap over incidents, and McNemar tests compare each model with the full model.

# Reproducing the Paper's Results

The default command runs all stages in order:

| Stage | What it does | Reviewer points | Time on our machine |
|---|---|---|---|
| `audit` | Exact counts, grades, incidents, organisations and train/test overlap of both files | R1.6, R2.1, R2.8 | 3 min |
| `main` | Training on GUIDE_Train and a single evaluation on GUIDE_Test. Covers:<br>• baselines and the full ablation<br>• confidence intervals, McNemar tests and routing<br>• latency, robustness to missing fields and the label-shuffle test<br>• Apriori, Isolation Forest and operating-point sensitivity<br>• scalability and interpretability | R1.2–R1.8, R2.1–R2.7 | 156 min |
| `temporal` | Training on earlier incidents and testing on later ones, with and without retraining | R2.7, R2.8 | 16 min |
| `orgshift` | Testing on organisations never seen in training, with and without tenant identifiers | R2.8 | 11 min |
| `leakage` | Incident-disjoint vs. row-level random split | R2.5 | 7 min |
| `cv` | Incident-grouped 5-fold cross-validation (mean ± SD, paired t-tests) | R2.4 | 41 min |
| `seeds` | Repetition with three random seeds | R2.4 | 30 min |
| `report` | All tables (CSV and Excel), figures (PNG, 300 dpi) and `RESULTS_SUMMARY.md` | – | 1 min |

The complete run took about **4 h 23 min**. The auxiliary stages (`temporal`, `orgshift`, `leakage`, `cv`, `seeds`) retrain the models many times. To keep the run time manageable, they use a fixed, deterministic 25% sample of GUIDE_Train incidents (2,252,039 records). Stages can be run selectively, for example `--stages audit,main,report`.

# Outputs

All outputs are written to `guide_results/`:

* `RESULTS_SUMMARY.md`: human-readable summary of the main results
* `tables/*.csv` and `all_tables.xlsx`: every result table. Correspondence with the paper:

| Output file | Paper |
|---|---|
| `table_dataset_audit.csv`, `table_split.csv`, `table_train_test_overlap.csv` | Section 7.2, Table 11 |
| `table_config.csv` | Table 10 |
| `table_test_results_all_models.csv` | Tables 12 and 13 |
| `table_layers.csv` | Section 7.4 (layer ROC-AUC and Brier scores) |
| `table_fp_reduction.csv` | Section 7.7 |
| `table_routing.csv` | Table 15 |
| `table_cv.csv`, `table_seeds.csv` | Table 14 |
| `table_robustness.csv`, `table_temporal.csv`, `table_unseen_orgs.csv` | Table 16 |
| `table_leakage_split.csv`, `table_checks.csv` | Sections 7.9, 7.10 and 7.12 |
| `table_training_time.csv`, `table_scalability.csv`, `table_latency.csv` | Table 17 |
| `table_sens_apriori.csv`, `table_sens_iforest.csv`, `table_operating_points.csv` | Table 18, Section 7.12 |
| `table_top_rules.csv`, `table_feature_importance.csv` | Table 19, Section 7.13 |

* `figures/`:
  * `fig_confusion_matrix.png` (Figure 8)
  * `fig_ablation.png` (Figure 9)
  * `fig_scalability_operating_points.png` (Figure 10)
  * `fig_roc_pr.png`, `fig_risk_distribution.png` and `fig_feature_importance.png` (supplementary)
* `stage_*.json`, `results_all.json`: all raw numbers; `scores_test.npz`: test-set risk scores
* `environment.json`: software versions, CPU, RAM and command-line options; `run.log`: complete log, including memory use after every major step

# Main Results (GUIDE_Test, 4,147,992 alerts)

| Model | Precision | Recall | FPR % (95% CI) | ROC-AUC |
|---|---|---|---|---|
| Forward-all (SIEM as-is) | 0.360 | 1.000 | 100 | – |
| Detector-reputation rule | 0.579 | 0.954 | 38.95 (37.20–40.65) | 0.911 |
| Logistic regression | 0.456 | 0.965 | 64.76 (63.03–66.57) | 0.864 |
| Random Forest | 0.781 | 0.962 | 15.19 (14.05–16.53) | 0.982 |
| XGBoost | 0.890 | 0.964 | 6.72 (6.07–7.51) | 0.991 |
| **Full layered model** | **0.891** | **0.962** | **6.63 (5.96–7.32)** | **0.989** |

On in-distribution data, XGBoost provides most of the discriminative power. On a temporal hold-out of later incidents, the fused model keeps a more stable operating point: FPR 18.50% at recall 0.962, against 23.92% for RF + XGBoost at similar recall. Performance does not transfer to organisations never seen in training (ROC-AUC 0.706), so models should be trained on the target SOC's own triage history. See the paper for the full analysis.

# Methodological Safeguards

* Incidents are identified by (OrgId, IncidentId). Training, validation and test sets never share an incident; the overlap is verified and reported.
* Every learned transformation (encoders, frequency maps, medians, Apriori rules, Isolation Forest, model weights) is fitted on the training split only. Fusion weights and thresholds are selected on the validation split. GUIDE_Test is scored once.
* Context features are label-free and causal: they use only events at or before the current alert.
* Leakage columns (IncidentGrade, LastVerdict, ActionGrouped, ActionGranular) and record identifiers are removed.
* No synthetic oversampling (SMOTE), PCA or test-set rescaling is used. Class imbalance is handled by class weights and validation-based thresholds.
* A label-shuffle test confirms chance-level performance (ROC-AUC 0.510) when labels are randomised.
* All random operations use fixed seeds.

# Configuration

All hyper-parameters are defined in the `CONFIG` dictionary at the top of `guide_soc_experiment.py` and exported with every run to `tables/table_config.csv` (Table 10 of the paper).

The main settings are:
* **Random Forest:** 200 trees, minimum 5 samples per leaf, balanced subsample class weights.
* **XGBoost:** up to 1,500 rounds, learning rate 0.1, depth 6, early stopping after 50 rounds.
* **Isolation Forest:** 200 trees, ψ = 256.
* **Apriori:** support 0.01, confidence 0.60, lift 1.10, maximum itemset length 3.
* **Fusion:** grid step 0.05, target recall 0.95.
* **Routing:** at most 1% of threats auto-dismissed; escalation precision at least 0.95.

# Environment Used for the Reported Results

* VMware Workstation virtual machine, Ubuntu 22.04 (Linux kernel 6.8), 8 virtual CPUs, 20 GB RAM; host laptop with a 10th-generation Intel Core i5 processor
* Python 3.10.12; numpy 2.2.6; pandas 2.3.3; scipy 1.15.3; scikit-learn 1.7.2; xgboost 3.2.0; mlxtend 0.23.4
* Complete run: about 4 h 23 min; peak memory 15.6 GB

# Citation

If you use this code, please cite:

Mahmud, A. R., Rashid, I., Tahir, S., Tufail, A., Zuhairi, M. F., and Ahmed, A. An ensemble machine learning framework for reducing false positives in security operations centers using big data analytics. *PeerJ Computer Science* (under review).

The GUIDE dataset: Microsoft (2024). Microsoft Security Incident Prediction (GUIDE) dataset. https://www.kaggle.com/datasets/Microsoft/microsoft-security-incident-prediction

