# BT-KIDS: Breach-Tolerant Kubernetes Intrusion Detection System

This repository accompanies the work **"BT-KIDS: Breach-Tolerant Kubernetes Intrusion Detection System Using Trusted External Resource Telemetry"**. It contains:

- A **Kubernetes + Prometheus + Grafana testbed** that collects **trusted external metrics** (CPU, memory, disk writes) for each pod.
- A **Kali/Ubuntu stress pod setup** for simulating post-compromise attacks (cryptomining-like CPU saturation, memory flooding, disk-write bursts, and low-and-slow variants).
- A **labeled dataset and ML/DL models** (Random Forest, XGBoost, LSTM, CNN, CNN+BiLSTM) implementing the intrusion detection pipeline described in the paper.

The code is organized into two main areas:

- [Final_Data_with_models](Final_Data_with_models): offline data, feature construction, and model training/evaluation.
- [kdeploy](kdeploy): Kubernetes deployment manifests and helper scripts for monitoring stack and attack pods.

---

## 1. High-Level System Overview

The system follows the architecture described in the paper:

1. **Trusted external telemetry**
   - Prometheus scrapes **cAdvisor** and **kube-state-metrics** from outside the workload namespaces.
   - Metrics include per-pod **CPU usage**, **memory usage**, and **disk write throughput**.

2. **Attack and normal workloads**
   - A privileged **Kali** or **Ubuntu stress** pod runs controlled stress workloads to emulate:
     - Cryptomining-like CPU saturation.
     - Memory flooding and leaks.
     - Ransomware-style disk write bursts.
     - Low-rate, throttled ("low-and-slow") attacks.

3. **Dataset creation**
   - Telemetry is exported and aggregated into Excel files.
   - Labeled as `isAttacked` ∈ {0 (normal), 1 (attack)}.
   - Optionally enriched with a temporal **timestamp** column.

4. **Models**
   - **Classical ML** on static feature vectors:
     - Random Forest, XGBoost.
   - **Time-series deep learning** on metric sequences:
     - LSTM / GRU, pure CNN, and optimized CNN+BiLSTM.

5. **Goal**
   - Detect post-compromise, stealthy attacks **using only trusted external metrics**, assuming the attacker fully controls one pod but cannot tamper with Prometheus/Grafana.

---

## 2. Data and Models: `Final_Data_with_models`

This directory contains the datasets and Python code for generating time-series features and training all models.

### 2.1. Files and Folders

- [Final_Data_with_models/timeseriesgenerator.py](Final_Data_with_models/timeseriesgenerator.py)
- [Final_Data_with_models/Combined data and models](Final_Data_with_models/Combined%20data%20and%20models)
  - [Final_Data_with_models/Combined data and models/Combined-Dataset.xlsx](Final_Data_with_models/Combined%20data%20and%20models/Combined-Dataset.xlsx)
  - [Final_Data_with_models/Combined data and models/RandomForest_oncombined.py](Final_Data_with_models/Combined%20data%20and%20models/RandomForest_oncombined.py)
  - [Final_Data_with_models/Combined data and models/xgboost_oncombined.py](Final_Data_with_models/Combined%20data%20and%20models/xgboost_oncombined.py)
- [Final_Data_with_models/Separate data](Final_Data_with_models/Separate%20data)
  - [Final_Data_with_models/Separate data/Attack-FinalDataSet.xlsx](Final_Data_with_models/Separate%20data/Attack-FinalDataSet.xlsx)
  - [Final_Data_with_models/Separate data/Modded-Real-FinalDataSet.xlsx](Final_Data_with_models/Separate%20data/Modded-Real-FinalDataSet.xlsx)
- [Final_Data_with_models/Time series and models](Final_Data_with_models/Time%20series%20and%20models)
  - [Final_Data_with_models/Time series and models/Combined-Dataset-with-timestamps.xlsx](Final_Data_with_models/Time%20series%20and%20models/Combined-Dataset-with-timestamps.xlsx)
  - [Final_Data_with_models/Time series and models/CNNonly.py](Final_Data_with_models/Time%20series%20and%20models/CNNonly.py)
  - [Final_Data_with_models/Time series and models/lstm.py](Final_Data_with_models/Time%20series%20and%20models/lstm.py)
  - [Final_Data_with_models/Time series and models/optimized_cnn_bilstm_attack_detection.py](Final_Data_with_models/Time%20series%20and%20models/optimized_cnn_bilstm_attack_detection.py)

> Note: Excel files are not described line-by-line, but their structure is inferred from how they are used in the code.

### 2.2. `timeseriesgenerator.py`

**Path**: [Final_Data_with_models/timeseriesgenerator.py](Final_Data_with_models/timeseriesgenerator.py)

**Purpose**:

- Augments the combined dataset with a **synthetic, monotonically increasing timestamp column**, which is required by the sequence models (LSTM/CNN/CNN+BiLSTM) for chronological ordering and sliding-window sequence construction.

**Key behavior**:

- Reads an input Excel file (by default `Combined-Dataset.xlsx`).
- Creates a timestamp series starting at `2025-10-25 09:00`, with **5-minute intervals** between rows.
- Inserts this timestamp as the first column named `timestamp`.
- Writes out a new file `Combined-Dataset-with-timestamps.xlsx`.

This corresponds to the **"Metrics and labeling"** and **"Feature representation"** sections of the paper, where the dataset is prepared as a time series.

### 2.3. `Combined data and models`

This folder contains the **combined, labeled dataset** (normal + attack) and scripts for the **classical machine learning baselines**.

#### 2.3.1. `Combined-Dataset.xlsx`

**Path**: [Final_Data_with_models/Combined data and models/Combined-Dataset.xlsx](Final_Data_with_models/Combined%20data%20and%20models/Combined-Dataset.xlsx)

**Role**:

- Single Excel file aggregating **external resource metrics** (CPU, memory, disk writes, and possibly derived features) across both normal and attack modes.
- Contains a binary label column `isAttacked`:
  - `0` → normal behavior.
  - `1` → attack behavior.

This dataset is used as input to the Random Forest and XGBoost models described in the **"Model comparison and evaluation"** section of the paper.

#### 2.3.2. `RandomForest_oncombined.py`

**Path**: [Final_Data_with_models/Combined data and models/RandomForest_oncombined.py](Final_Data_with_models/Combined%20data%20and%20models/RandomForest_oncombined.py)

**Purpose**:

- Trains and evaluates a **Random Forest** classifier for binary **attack vs. normal** detection using the combined dataset.

**Main steps**:

1. **Load data** from `Combined-Dataset.xlsx`.
2. **Feature/label split**:
   - `X` = all columns except `isAttacked`.
   - `y` = `isAttacked`.
3. Stratified **train/test split** (80/20).
4. Train a `RandomForestClassifier` with tuned hyperparameters (number of trees, max depth, etc.).
5. Evaluate:
   - Accuracy.
   - `classification_report` (precision, recall, F1 for each class).
   - Confusion matrix and its heatmap.
   - Top 10 feature importances plotted as a bar chart.
6. Persist model as `random_forest_attack_detector.pkl` using `joblib`.

**Relation to paper**:

- Implements the **Random Forest** baseline in Section "Results: Classical ML models" and contributes to the metrics reported in Table  (`Random Forest` row).

#### 2.3.3. `xgboost_oncombined.py`

**Path**: [Final_Data_with_models/Combined data and models/xgboost_oncombined.py](Final_Data_with_models/Combined%20data%20and%20models/xgboost_oncombined.py)

**Purpose**:

- Trains and evaluates an **XGBoost** classifier on the same combined dataset to compare with Random Forest.

**Main steps**:

1. Load `Combined-Dataset.xlsx`.
2. Split into `X` / `y` (same as Random Forest script).
3. Stratified train/test split (80/20).
4. Configure `xgb.XGBClassifier` for binary logistic classification (with tuned hyperparameters: `n_estimators`, `max_depth`, learning rate, subsampling, etc.).
5. Train the model.
6. Evaluate:
   - Accuracy.
   - `classification_report` and confusion matrix.
   - Confusion matrix heatmap.
   - XGBoost feature importance plot (top 10 features).

**Relation to paper**:

- This is the **XGBoost** baseline in the evaluation section, with results summarized in the table and ROC/PR analysis.

### 2.4. `Separate data`

This folder isolates **attack-only** and **(modified) real/benign** samples:

- [Final_Data_with_models/Separate data/Attack-FinalDataSet.xlsx](Final_Data_with_models/Separate%20data/Attack-FinalDataSet.xlsx)
- [Final_Data_with_models/Separate data/Modded-Real-FinalDataSet.xlsx](Final_Data_with_models/Separate%20data/Modded-Real-FinalDataSet.xlsx)

These are useful for:

- Building balanced or specific evaluation sets.
- Sanity-checking that attack signatures differ from real workloads.
- Potential future work (e.g., multi-class or anomaly detection variants).

Even though they are not directly invoked in the current training scripts, they map conceptually to the **"Dataset modes"** section (normal vs. attack) of the paper.

### 2.5. `Time series and models`

This folder contains:

- Time-augmented dataset.
- Time-series deep learning models (CNN, LSTM/GRU, and CNN+BiLSTM) used for **sequence-based intrusion detection**.

#### 2.5.1. `Combined-Dataset-with-timestamps.xlsx`

**Path**: [Final_Data_with_models/Time series and models/Combined-Dataset-with-timestamps.xlsx](Final_Data_with_models/Time%20series%20and%20models/Combined-Dataset-with-timestamps.xlsx)

**Role**:

- Same core features and label as `Combined-Dataset.xlsx`, but with an extra **`timestamp`** column.
- Sorted chronologically and used to construct rolling windows of length `SEQ_LEN`.

In the paper, this corresponds to the **"Metric sequences"** feature representation used by LSTM and CNN models.

#### 2.5.2. `CNNonly.py`

**Path**: [Final_Data_with_models/Time series and models/CNNonly.py](Final_Data_with_models/Time%20series%20and%20models/CNNonly.py)

**Purpose**:

- Implements a **pure 1D CNN** model for time-series classification of pod resource usage.

**Key logic**:

1. Load and sort `Combined-Dataset-with-timestamps.xlsx` by `timestamp`.
2. Prepare features and labels:
   - Drop `timestamp` and `isAttacked` to form numeric feature matrix.
   - Standardize features with `StandardScaler`.
3. Construct sequences of length `SEQ_LEN` (e.g., 36 steps) using a sliding window:
   - Each sequence corresponds to `SEQ_LEN` consecutive time steps of metrics.
   - Label is the `isAttacked` value at the final time index.
4. Chronological train/test split (no leakage from the future to the past).
5. Define a **multi-layer 1D CNN**:
   - Several `Conv1D` blocks with BatchNorm, MaxPooling, Dropout.
   - Flatten + dense layers with Dropout.
   - Sigmoid output for binary classification.
6. Train with early stopping on validation loss.
7. Evaluate on test set:
   - Classification report.
   - Confusion matrix.
   - Accuracy.

**Relation to paper**:

- Implements the **CNN** setting in "Time-series models" and contributes to the CNN performance numbers and ROC/PR curves.

#### 2.5.3. `lstm.py`

**Path**: [Final_Data_with_models/Time series and models/lstm.py](Final_Data_with_models/Time%20series%20and%20models/lstm.py)

**Purpose**:

- Trains and evaluates **LSTM** and **GRU** models for sequence-based intrusion detection.

**Key logic**:

1. Load and chronologically sort `Combined-Dataset-with-timestamps.xlsx`.
2. Preprocess:
   - Drop `timestamp` and `isAttacked` to get numeric features.
   - Standardize features using `StandardScaler`.
3. Build sequences of length `SEQ_LEN` (e.g., last 24 timesteps, conceptually "2 hours") via a sliding window.
4. Chronological train/test split.
5. Two architectures:
   - **LSTM** model: stacked LSTMs with Dropout, sigmoid output.
   - **GRU** model: stacked GRUs with Dropout, sigmoid output.
6. Train each with early stopping on validation loss.
7. Evaluate each on the test set:
   - Classification report.
   - Confusion matrix.

**Relation to paper**:

- This corresponds directly to the **LSTM** section in "Time-series models" and its high recall results in the evaluation.
- The GRU results can be used as an additional point of comparison, even if not all metrics appear in the final table.

#### 2.5.4. `optimized_cnn_bilstm_attack_detection.py`

**Path**: [Final_Data_with_models/Time series and models/optimized_cnn_bilstm_attack_detection.py](Final_Data_with_models/Time%20series%20and%20models/optimized_cnn_bilstm_attack_detection.py)

**Purpose**:

- Implements an **optimized hybrid CNN + Bidirectional LSTM** (BiLSTM) model to capture both local patterns and long-term temporal dependencies.

**Key logic**:

1. Load and sort `Combined-Dataset-with-timestamps.xlsx`.
2. Standardize features; create sequences of length `SEQ_LEN` via sliding window.
3. Chronological train/test split.
4. Model architecture:
   - Initial 1D convolution + BatchNorm + additional Conv1D + MaxPool + Dropout.
   - Two stacked **Bidirectional LSTM** layers with Dropout.
   - Dense layer with ReLU + Dropout.
   - Final dense layer with sigmoid activation for binary classification.
5. Training with early stopping on validation loss.
6. Evaluation using classification report and confusion matrix.

**Relation to paper**:

- Represents an advanced time-series architecture that combines strengths of CNN (local motifs) and LSTM (long-term temporal behavior), aligning with the paper’s emphasis on time-series models for stealthy attack detection.

---

## 3. Kubernetes Monitoring and Attack Setup: `kdeploy`

This directory contains scripts and manifests to deploy the **trusted monitoring stack** and **attack/stress pods** on a Kubernetes cluster (e.g., Minikube). Together, they implement the testbed described in the **"System Configuration and Data Pipeline"** and **"Scenario, Threat Model, and Adversary Goals"** sections.

### 3.1. Files

- [kdeploy/deploy-monitoring.sh](kdeploy/deploy-monitoring.sh)
- [kdeploy/grafana-configs.yaml](kdeploy/grafana-configs.yaml)
- [kdeploy/dashboard_to_import.yaml](kdeploy/dashboard_to_import.yaml)
- [kdeploy/kali-pod.yaml](kdeploy/kali-pod.yaml)
- [kdeploy/linux-container.sh](kdeploy/linux-container.sh)
- [kdeploy/safe_stress_test.sh](kdeploy/safe_stress_test.sh)
- [kdeploy/freeupspace.txt](kdeploy/freeupspace.txt)

### 3.2. `deploy-monitoring.sh`

**Path**: [kdeploy/deploy-monitoring.sh](kdeploy/deploy-monitoring.sh)

**Purpose**:

- One-click deployment of the **monitoring stack** for the testbed:
  - `monitoring` namespace.
  - `kube-state-metrics` in `kube-system`.
  - Prometheus configuration and deployment (scraping cAdvisor and kube-state-metrics).
  - Grafana deployment and exposure.

**What it does** (high level):

1. Creates a **`monitoring`** namespace.
2. Deploys **kube-state-metrics** with RBAC and a Service.
3. Creates a **Prometheus ConfigMap** (`prometheus-config`) with `prometheus.yml`, configuring scrape jobs:
   - `kubernetes-nodes-cadvisor`: scrapes cAdvisor metrics from each node via the kubelet proxy.
   - `kube-state-metrics`: scrapes cluster state metrics.
   - `kubernetes-pods`: scrapes annotated pods.
4. Creates Prometheus **ClusterRole** and **ClusterRoleBinding** granting it access to the required Kubernetes APIs and `/metrics` endpoints.
5. Deploys a single-node **Prometheus** instance in the `monitoring` namespace with:
   - ConfigMap volume for `prometheus.yml`.
   - `NodePort` service for external access.
6. Deploys a single **Grafana** instance in the `monitoring` namespace with:
   - Default admin credentials (`admin` / `admin`).
   - `NodePort` service for access.
7. Waits for all relevant deployments to be ready.
8. Uses `minikube service` to open Prometheus and Grafana UIs in the browser.

**Relation to paper**:

- Implements the **trusted external metrics pipeline** (Prometheus + cAdvisor + kube-state-metrics) described in the **"Trusted metric collection"** and **"External metrics surfaces"** subsections.

### 3.3. `grafana-configs.yaml`

**Path**: [kdeploy/grafana-configs.yaml](kdeploy/grafana-configs.yaml)

**Purpose**:

- Provides Grafana **ConfigMaps** for:
  - A Prometheus **data source**.
  - A pre-built pod-level **dashboard** visualizing CPU, memory, and disk write metrics per pod.

**Key components**:

1. **ConfigMap `grafana-datasources`**:
   - Declares a Grafana data source named `Prometheus` pointing to `http://prometheus:9090`.
   - Marks it as `isDefault: true`.

2. **ConfigMap `grafana-dashboards`**:
   - Contains a JSON dashboard `kubernetes-pods-cpu-memory.json` that defines panels for:
     - Per-pod **CPU usage** in millicores (`container_cpu_usage_seconds_total` rate).
     - **CPU usage as % of request** (ratio of usage to requested CPU).
     - **Memory working set** in bytes and **% of limit**.
     - **Memory cache**.
     - **Pod restart counts**.
     - **Disk write rate (Bytes/sec)** based on `container_fs_writes_bytes_total`.

**Relation to paper**:

- Provides visualization for the **signals used for detection** (Table “Signals used for detection”), especially CPU, memory, and disk writes.
- Demonstrates how these metrics look during both normal and attack modes.

### 3.4. `dashboard_to_import.yaml`

**Path**: [kdeploy/dashboard_to_import.yaml](kdeploy/dashboard_to_import.yaml)

**Purpose**:

- Contains an **importable Grafana dashboard JSON** focusing on **cluster-wide** resource usage:
  - CPU mCores per pod/namespace.
  - Aggregate CPU% of total cores.
  - Memory usage per pod (MB) and cluster-wide memory utilization.
  - Disk read/write throughput (MB/s).
  - Disk I/O utilization (%) (read and write).

**Usage**:

- Can be manually imported into Grafana to get a high-level overview of cluster behavior and resource patterns during experiments.

**Relation to paper**:

- Provides complementary visualization to verify the **external metric behavior** used later for offline model training.

### 3.5. `kali-pod.yaml`

**Path**: [kdeploy/kali-pod.yaml](kdeploy/kali-pod.yaml)

**Purpose**:

- Defines a privileged **Kali Linux pod** used to **simulate a compromised pod** with root access and tooling to launch various attack/stress scenarios.

**Key characteristics**:

- `hostNetwork: true` – exposes host networking to the container.
- `securityContext.privileged: true` – full privileges, modeling the **post-compromise** scenario.
- `postStart` lifecycle hook:
  - Runs an initialization script that installs a wide set of tools: `stress-ng`, networking utilities, shell tools, editors, etc.
  - Marks completion by creating `/opt/kali-init/kalipackages.installed`.
- `readinessProbe` checks that the initialization file exists before marking the pod as ready.
- Volume mounts for ephemeral storage (`/mnt/storage`, `/fluct_io`, `/opt/kali-init`).

**Relation to paper**:

- Embodies the **"attacker has root inside exactly one pod"** threat model.
- Used to run CPU/memory/disk-heavy workloads that are observed from outside through the Prometheus/cAdvisor pipeline.

### 3.6. `linux-container.sh`

**Path**: [kdeploy/linux-container.sh](kdeploy/linux-container.sh)

**Purpose**:

- Helper script to deploy a **privileged Ubuntu 22.04 "ubuntu-stress" pod** with full access, for controlled stress testing.

**What it does**:

1. Applies a Kubernetes `Pod` manifest that:
   - Runs an Ubuntu container as root.
   - Uses `hostPID` and `hostNetwork`.
   - Grants capabilities like `SYS_ADMIN`, `SYS_RESOURCE`, `NET_ADMIN`, `NET_RAW`.
2. Waits until the pod is ready.
3. Enters the pod to install tools:
   - `stress-ng`, `htop`, `curl`, `wget`, `iputils-ping`, `net-tools`, `vim`, `python3`, etc.
4. Prints helper commands for accessing and deleting the pod.

**Relation to paper**:

- Alternative to the Kali pod for generating **attack mode** workloads used to build the dataset (cryptomining-style stress, disk I/O, etc.).

### 3.7. `safe_stress_test.sh`

**Path**: [kdeploy/safe_stress_test.sh](kdeploy/safe_stress_test.sh)

**Purpose**:

- **Safe, looped stress workload** script designed to be run inside a container (Kali/Ubuntu pod) to generate controlled CPU, memory, and disk pressure while respecting resource limits.

**Key behavior**:

1. Detects available CPU cores and total memory from `/proc/meminfo`.
2. Sets memory usage target to ~80% of total memory.
3. Sets a per-cycle disk write limit (10 GB) for safety.
4. Creates a temporary directory for stress files.
5. Enters an infinite loop of **stress cycles**:
   - Each cycle runs `stress-ng` with parameters for:
     - CPU stress (`--cpu` for all cores).
     - Memory stress (`--vm`, `--vm-bytes`).
     - Disk stress (`--hdd`, `--hdd-bytes`) limited to the defined threshold.
   - Each cycle runs up to 30 minutes and then cleans up temporary files.
   - Sleeps 60 seconds between cycles.

**Relation to paper**:

- Produces the **resource-usage patterns** used in the dataset: CPU saturation, memory flooding, disk bursts.
- Can be configured/observed to produce both "noisy" and more stealthy (lower intensity) patterns.

### 3.8. `freeupspace.txt`

**Path**: [kdeploy/freeupspace.txt](kdeploy/freeupspace.txt)

**Purpose**:

- Contains a set of `rm -rf` commands targeting Docker Desktop-related directories under `~/.docker`.
- Used as a **manual clean-up note** to free disk space on local development machines when experimenting heavily with containers.

> Use with caution; this removes local Docker artifacts.

---

## 4. How Components Map to the Paper

- **Threat Model & Scenario**
  - **Compromised pod**: [kdeploy/kali-pod.yaml](kdeploy/kali-pod.yaml) and [kdeploy/linux-container.sh](kdeploy/linux-container.sh) instantiate pods with root access and privileged capabilities.
  - **External, trusted metrics**: [kdeploy/deploy-monitoring.sh](kdeploy/deploy-monitoring.sh) + cAdvisor + Prometheus + kube-state-metrics implement the trusted perimeter.

- **Signal Collection & Dataset Modes**
  - External resource metrics (CPU, memory, disk writes) visualized via [kdeploy/grafana-configs.yaml](kdeploy/grafana-configs.yaml) and [kdeploy/dashboard_to_import.yaml](kdeploy/dashboard_to_import.yaml).
  - Normal vs. attack modes are reflected in:
    - Combined dataset: [Final_Data_with_models/Combined data and models/Combined-Dataset.xlsx](Final_Data_with_models/Combined%20data%20and%20models/Combined-Dataset.xlsx).
    - Separate datasets: [Final_Data_with_models/Separate data/Attack-FinalDataSet.xlsx](Final_Data_with_models/Separate%20data/Attack-FinalDataSet.xlsx), [Final_Data_with_models/Separate data/Modded-Real-FinalDataSet.xlsx](Final_Data_with_models/Separate%20data/Modded-Real-FinalDataSet.xlsx).

- **Feature Representation**
  - Static features for classical ML: constructed implicitly in the Random Forest and XGBoost scripts from the columns in `Combined-Dataset.xlsx`.
  - Time-series sequences: generated by `create_sequences` functions in [Final_Data_with_models/Time series and models/CNNonly.py](Final_Data_with_models/Time%20series%20and%20models/CNNonly.py), [Final_Data_with_models/Time series and models/lstm.py](Final_Data_with_models/Time%20series%20and%20models/lstm.py), and [Final_Data_with_models/Time series and models/optimized_cnn_bilstm_attack_detection.py](Final_Data_with_models/Time%20series%20and%20models/optimized_cnn_bilstm_attack_detection.py), using the timestamped dataset.

- **Model Comparison & Evaluation**
  - Classical ML scripts implement the **Random Forest** and **XGBoost** models with accuracy, precision, recall, and F1 scores.
  - Time-series scripts implement **LSTM**, **GRU**, **CNN**, and **CNN+BiLSTM** architectures, matching the deep-learning configurations discussed in the paper and supporting the ROC/PR analyses.

---

## 5. Example Reproduction Workflow (High Level)

This is a sketch of how you would typically use this repo end-to-end (exact commands may depend on your specific environment and data export process):

1. **Start Kubernetes cluster** (e.g., Minikube) and deploy monitoring:
   - Run [kdeploy/deploy-monitoring.sh](kdeploy/deploy-monitoring.sh).

2. **Deploy attack/experiment pod**:
   - Apply [kdeploy/kali-pod.yaml](kdeploy/kali-pod.yaml) or run [kdeploy/linux-container.sh](kdeploy/linux-container.sh).

3. **Generate attack workloads** inside the pod:
   - Exec into the pod and run [kdeploy/safe_stress_test.sh](kdeploy/safe_stress_test.sh) to create CPU/memory/disk stress.
   - Observe metrics in Grafana using dashboards from [kdeploy/grafana-configs.yaml](kdeploy/grafana-configs.yaml) or import [kdeploy/dashboard_to_import.yaml](kdeploy/dashboard_to_import.yaml).

4. **Export metrics** from Prometheus/cAdvisor into CSV/Excel format and construct:
   - Combined dataset in [Final_Data_with_models/Combined data and models/Combined-Dataset.xlsx](Final_Data_with_models/Combined%20data%20and%20models/Combined-Dataset.xlsx).
   - Optional separate normal/attack datasets.

5. **Generate timestamped dataset**:
   - Run [Final_Data_with_models/timeseriesgenerator.py](Final_Data_with_models/timeseriesgenerator.py) to produce `Combined-Dataset-with-timestamps.xlsx` in the time-series folder.

6. **Train and evaluate models**:
   - Classical ML: run
     - [Final_Data_with_models/Combined data and models/RandomForest_oncombined.py](Final_Data_with_models/Combined%20data%20and%20models/RandomForest_oncombined.py)
     - [Final_Data_with_models/Combined data and models/xgboost_oncombined.py](Final_Data_with_models/Combined%20data%20and%20models/xgboost_oncombined.py)
   - Time-series DL: run
     - [Final_Data_with_models/Time series and models/lstm.py](Final_Data_with_models/Time%20series%20and%20models/lstm.py)
     - [Final_Data_with_models/Time series and models/CNNonly.py](Final_Data_with_models/Time%20series%20and%20models/CNNonly.py)
     - [Final_Data_with_models/Time series and models/optimized_cnn_bilstm_attack_detection.py](Final_Data_with_models/Time%20series%20and%20models/optimized_cnn_bilstm_attack_detection.py)

7. **Analyze results** and compare with the performance reported in the paper (accuracy, precision, recall, F1, ROC-AUC, PR-AUC).

---