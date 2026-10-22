# GrahakNiti: Data-Driven Customer Targeting & Strategic Campaign Allocation Engine

[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red.svg)](https://streamlit.io/)
[![Optimization: SciPy](https://img.shields.io/badge/Operations%20Research-SciPy%20HiGHS-green.svg)](https://scipy.org/)
[![XAI: SHAP](https://img.shields.io/badge/Explainability-SHAP-purple.svg)](https://shap.readthedocs.io/)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-GrahakNiti%20App-brightgreen.svg)](https://grahakniti.streamlit.app/)

> **An enterprise-grade, end-to-end Machine Learning and Decision Science Platform that unifies customer conversion propensity modeling with linear programming for optimal marketing budget allocation.**

---

## 📌 Executive Summary & Business Problem

Direct marketing campaigns for retail and financial products represent a classic high-volume, asymmetric-cost decision problem. In benchmark marketing campaign datasets (Moro et al., 2014), positive conversion rates hover around **~11.2%**, meaning roughly **9 out of 10 targeted customer interactions yield no conversion**.

Traditional marketing analytics initiatives often fail due to two fundamental structural gaps:
1. **The Static Scoring Trap:** Machine learning teams deploy propensity models that output raw probabilities or classify leads using an arbitrary default threshold ($\tau = 0.50$). This ignores asymmetric business unit economics—specifically, the direct contact cost of placing an outreach call versus the net lifetime customer value of a secured conversion.
2. **The Budget & Capacity Disconnect:** Even with calibrated probabilities, campaign managers operate under hard monetary budgets and outreach capacity limits. Prioritizing leads without mathematical optimization leads to **customer burnout** (diminishing returns and brand erosion from excessive contacts) and suboptimal Return on Marketing Investment (ROMI).

**GrahakNiti** (*Grahak* = Customer, *Niti* = Strategy) solves this by unifying:
- **Phase 1: Advanced Behavioral Feature Engineering** capturing contact intensity, recency decay, financial liquidity, and contact burnout.
- **Phase 2: Imbalance-Trained Predictive Classifiers** tuned with Bayesian optimization (**Optuna**) evaluated against **PR-AUC (Average Precision)** and **Brier Score Loss** rather than misleading raw accuracy.

---

## 📐 Mathematical Foundations

### 1. Advanced Behavioral Feature Engineering
To move beyond static demographic snapshots, CampaignOptima derives domain-engineered behavioral features implemented in [`src/features.py`](src/features.py):

#### a. Campaign Intensity Ratio
Measures telemarketing outreach pressure during the current campaign relative to historical interactions:
$$I_i = \frac{\text{campaign}_i}{\max(\text{previous}_i, 0) + 1.0}$$
*Interpretation:* High values signal aggressive, high-pressure contact cadences that trigger customer resistance and diminishing response rates.

#### b. Recency Weight Score
Applies exponential decay half-life weighting to historical touches, penalized for elapsed intervals and amplified by multi-touch engagement:

For cold leads ($\text{pdays}_i < 0$):
$$R_i = 0.0$$

For contacted leads ($\text{pdays}_i \ge 0$):
$$R_i = \exp(-\lambda \cdot \text{clip}(\text{pdays}_i, 0, 500)) \cdot \left(1 + \ln(1 + \max(\text{previous}_i, 0))\right)$$

where $\lambda = 0.015$ (half-life $t_{1/2} \approx 46.2$ days).
#### c. Financial Stability Index (FSI)
A composite index reflecting customer liquidity and existing debt burden:
1. Account balance is winsorized between bounds $[-1000.0, 25000.0]$ and min-max normalized into $\widetilde{B}_i \in [-1, 1]$.
2. Debt liability penalties are subtracted:
   $$\text{FSI}_i = \widetilde{B}_i - 0.35 \cdot \mathbb{I}(\text{housing} = \text{yes}) - 0.40 \cdot \mathbb{I}(\text{loan} = \text{yes}) - 0.50 \cdot \mathbb{I}(\text{default} = \text{yes})$$

#### d. Supplementary Behavioral Signals
- **Log Call Duration:** `log_duration` = $\ln(1 + \max(\text{duration}, 0))$ to normalize right-skewed interaction durations.
- **Contact Burnout Indicator:** `is_high_frequency` = $\mathbb{I}(\text{campaign} \ge 4)$ flagging leads subjected to excessive touches.
---

### 2. Decision Science: Cost-Benefit Threshold Optimization
Standard machine learning assigns $\hat{y}_i = 1$ when predicted probability $\hat{p}_i \ge 0.50$. In marketing economics, however, the financial payoffs are highly asymmetric ([`src/config.py`](src/config.py)):
- **Direct contact cost per call ($C_{\text{call}}$):** \$15.00
- **Net revenue value per conversion ($V_{\text{deposit}}$):** \$300.00
- **Excess contact penalty ($C_{\text{burnout}}$):** \$5.00 per excess call ($>3$ calls)

The expected net profit for contacting customer $i$ is:

$$
\mathbb{E}[\text{Profit}_i] = \hat{p}_i \cdot V_{\text{deposit}} - C_{\text{call}}
$$

The theoretical zero-profit break-even probability threshold is:

$$
\tau_{\text{breakeven}} = \frac{C_{\text{call}}}{V_{\text{deposit}}} = \frac{15.00}{300.00} = 0.05 \quad (5.0\%)
$$

CampaignOptima sweeps $\tau \in [0.01, 0.95]$ over empirical validation observations to identify the optimal cutoff $\tau^*$:

$$
\tau^* = \arg\max_{\tau} \sum_{i=1}^N \mathbb{I}(\hat{p}_i \ge \tau) \left( y_i \cdot V_{\text{deposit}} - C_{\text{call}} \right)
$$

*Empirical Calibration Result ([`models/optimal_threshold.json`](models/optimal_threshold.json)):*
- **Optimal Decision Threshold ($\tau^*$):** **$0.029$**
- **Theoretical Break-even Threshold ($\tau_{\text{breakeven}}$):** **$0.050$**
- **Net Profit Uplift over Default ($\tau = 0.50$):** **+\$5,295.00 (+60.55%)**

---

### 3. Operations Research: Knapsack Budget Allocation
When deploying campaigns under real-world budgetary and call capacity ceilings, CampaignOptima formulates resource allocation as a **Bounded Linear Program / 0-1 Knapsack Problem** implemented in [`src/optimization.py`](src/optimization.py):

$$
\begin{aligned}
\max_{\mathbf{x}} \quad & \sum_{i=1}^N x_i \cdot \left( \hat{p}_i \cdot V_{\text{deposit}} - C_{\text{call}} \right) \\
\text{subject to} \quad & \sum_{i=1}^N x_i \cdot C_{\text{call}} \le B \quad && (\text{Total Campaign Budget}) \\
& \sum_{i=1}^N x_i \le M_{\text{calls}} \quad && (\text{Call Capacity / Burnout Ceiling}) \\
& 0 \le x_i \le 1 \quad \forall i \in \{1, \dots, N\}
\end{aligned}
$$

#### Solvers Implemented:
1. **Linear Programming (HiGHS):** Solved via `scipy.optimize.linprog(method="highs")` with decision vector rounded to binary $x_i = \mathbb{I}(x_i^* \ge 0.5)$.
2. **Greedy Marginal Efficiency Knapsack (Fallback / Rapid Heuristic):** Ranks leads descending by marginal return per dollar:

   $$
   \text{Efficiency}_i = \frac{\hat{p}_i \cdot V_{\text{deposit}} - C_{\text{call}}}{C_{\text{call}}}
   $$

   Sequentially targets positive-return prospects until budget $B$ or contact quota $M_{\text{calls}}$ is fully exhausted.

---

## 🏆 Model Performance Benchmark

Models were benchmarked on stratified holdout test partitions following SMOTE minority oversampling on the training set. Because bank term deposit conversions are rare (~11%), **PR-AUC (Average Precision)** and **Brier Score Loss** are the primary selection metrics ([`models/benchmark_report.json`](models/benchmark_report.json)):

| Model Architecture | PR-AUC (Primary) 🎯 | ROC-AUC | F1-Score | Precision | Recall | Brier Loss 📉 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tuned XGBoost (Optuna)** 🏆 | **0.4179** | **0.7970** | **0.4476** | **0.5614** | **0.3721** | **0.0672** |
| XGBoost (Baseline) | 0.4457 | 0.8161 | 0.4744 | 0.5286 | 0.4302 | 0.0721 |
| LightGBM | 0.4085 | 0.7928 | 0.4557 | 0.5000 | 0.4186 | 0.0740 |
| Random Forest | 0.3902 | 0.8182 | 0.4667 | 0.4468 | 0.4884 | 0.0847 |
| Logistic Regression (L2) | 0.4599 | 0.8188 | 0.3625 | 0.2511 | 0.6512 | 0.1465 |

> **Champion Selection Rationale:** `Tuned XGBoost` was chosen as the production champion model because it achieves the highest Precision (**56.14%**) and lowest Brier Score Loss (**0.0672**), ensuring well-calibrated probabilities essential for financial threshold optimization and minimum false-positive call spend.

---

## 📂 Repository Architecture

```bash
CampaignOptima/
├── app.py                          # Multi-tab Streamlit Analytics & Decision Science Web App
├── requirements.txt                # Production Python dependencies
├── README.md                       # Enterprise system documentation
├── data/
│   ├── raw/
│   │   └── bank.csv                # Semicolon-delimited Bank Marketing benchmark dataset
│   └── processed/                  # Processed cache directory
├── models/
│   ├── best_model.joblib           # Serialized champion Tuned XGBoost classifier
│   ├── feature_pipeline.joblib     # Scikit-Learn behavioral transformer pipeline
│   ├── optimal_threshold.json      # Calibrated business unit economics & optimal threshold
│   └── benchmark_report.json       # Multi-model evaluation benchmark metrics
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb         # EDA, feature distributions, burnout analysis
│   ├── 02_model_training_and_tuning.ipynb         # Model training, SMOTE, Optuna tuning
│   ├── 03_decision_science_and_optimization.ipynb # Cost-benefit thresholding & HiGHS LP
│   └── 04_xai_shap_explainability.ipynb           # Global & local SHAP tree-based explanations
└── src/
    ├── __init__.py                 # Package version definition (v1.0.0)
    ├── config.py                   # Centralized dataclass configurations & unit economics
    ├── data_generator.py           # Realistic multivariate benchmark dataset generator
    ├── data_loader.py              # Semicolon-delimited ingestion, validation, splitting
    ├── features.py                 # Behavioral feature engineering & SMOTE pipeline
    ├── models.py                   # Model training, Optuna tuning, and evaluation engine
    ├── optimization.py             # ThresholdOptimizer & CampaignBudgetOptimizer (HiGHS LP)
    ├── explainability.py           # Robust ModelExplainabilityEngine (SHAP TreeExplainer)
    └── pipeline.py                 # End-to-end execution orchestrator
```

---

## 🚀 Quickstart & Setup Guide

### 1. Environment Setup
```bash
git clone https://github.com/your-org/CampaignOptima.git
cd CampaignOptima

# Create and activate virtual environment (Python 3.9, 3.10, or 3.11 recommended)
python -m venv venv

# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the End-to-End Pipeline
To run data ingestion, behavioral feature engineering, SMOTE balancing, Optuna hyperparameter tuning, cost-benefit threshold optimization, and artifact serialization:
```bash
python src/pipeline.py
```

### 3. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to access the three dashboard modules:
- **📊 Tab 1: Executive KPI & Campaign Analytics:** Cohort conversion rates across professions, balance tiers, contact methods, and telemarketing contact burnout curves.
- **🎯 Tab 2: Real-Time Conversion Predictor & SHAP:** Interactive customer intake form, dynamic Behavioral Feature calculation, propensity gauge, decision recommendation card, and local SHAP waterfall driver attribution.
- **🧮 Tab 3: Operations Research & Budget Simulator:** Marketing budget and call capacity sliders, Linear Programming solver (`scipy.optimize.linprog` with HiGHS), efficient frontier curves, and downloadable prioritized lead call sheets.

---

## 🔍 Model Explainability (XAI)

CampaignOptima utilizes [`src/explainability.py`](src/explainability.py) with a version-resilient `shap.TreeExplainer` wrapper:
- **Global Feature Attribution:** Summarizes population-level drivers. Features such as `duration` (call engagement length), `poutcome_success` (prior positive interactions), and `balance` consistently emerge as strong positive catalysts, while high contact frequency triggers negative SHAP contributions.
- **Local Customer Attribution:** Provides individualized waterfall and force breakdowns showing exact positive ($+$) and negative ($-$) feature contributions, equipping relationship managers with transparent reasoning before placing outreach calls.

---

## 📄 License
This project is open-source under the [MIT License](https://opensource.org/licenses/MIT).
