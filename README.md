# GCI World Final Assignment: Telecommunications Customer Churn & Retention Optimization
**Matsuo-Iwasawa Laboratory | Graduate School of Engineering | The University of Tokyo**  
*Global Consumer Intelligence (GCI) World 2026 Winter Final Assignment*

---

## 📌 Executive Summary & Project Context

This project presents a data-driven **Proof of Concept (PoC)** for **Company A**, an anonymous telecommunications service provider. The objective is to analyze historical customer demographics and usage history, develop predictive machine learning models to forecast subscriber churn propensity, and formulate an analytically defensible, high-ROI retention strategy.

### Critical Dataset, Target Variable & Data Availability Qualifications
To maintain analytical rigor, this project explicitly distinguishes observed empirical facts from operational assumptions and hypotheses:

1. **Target Variable Definition (`churn`):**
   - As documented in Company A's dataset specification, `churn` represents an **instance of churn occurring between 31 and 60 days after the observation date**.
   - It is a **30-day target window outcome**, **not an annual churn rate**.
   - Across the 100,000 customer records in the dataset sample, **49,562 accounts churned** during this window, representing an observed churn rate of **49.56%** (49.562%).
2. **Account Population vs. Active Subscribers:**
   - The dataset contains 100,000 customer account records (`Customer_ID`), not 100,000 active individual subscribers. Active subscribers per account (`actvsubs`) range from 0 to 11 (mean: 1.36); 81 accounts have 0 active subscribers.
3. **Data-Availability Timing & Leakage Boundaries:**
   - **Preprocessing Leakage Prevention:** Learned transformations (median imputation, categorical encoding, standard scaling) are fitted **strictly on the training partition** after an 80/20 stratified split. Customer ID and target `churn` are kept strictly out of the feature matrix.
   - **Feature Availability Timing:** The dataset overview does not provide calendar timestamps, billing cycle cutoffs, or ETL pipeline execution schedules. Therefore, availability of all source features at prediction time is an **assumption** based on column descriptions and standard telecom churn modeling architectures (where Month 0 is observation date, Month 1 is retention campaign buffer, and Month 2 is 31–60 day outcome observation). Timing availability cannot be claimed as definitive and must be confirmed with Company A before live operational deployment.
   - **Audited Uncertain Fields:**
     - `change_mou` / `change_rev`: Measure usage/revenue changes vs. prior 3-month average. In production, monthly billing closure often carries a multi-day reconciliation lag; whether prior-month usage is finalized on Day 0 depends on the account's billing cycle calendar.
     - Billing-adjusted metrics (`adjrev`, `adjmou`, `adjqty`): Billing adjustments (disputes, credits) posted retroactively after the observation date would represent timing leakage if present. Live operational definitions must be confirmed with Company A.
     - Undefined metrics (`recv_sms_Mean`, `iwylis_vce_Mean`): Described as `NaN` (no definition) in Company A's documentation. While numerical in the data, their exact calculation window remains unconfirmed.
     - *Assessment:* None of the features exhibit correlations indicating direct post-churn contamination ($|r| < 0.12$ across all features). All are retained for this PoC, but an operational ETL timing audit with Company A is required prior to deployment.
4. **Revenue Loss Clarification:**
   - Previous claims that Company A loses $34.9M in annual recurring revenue conflated the 31–60 day churn proportion with an annual churn rate for 100,000 active subscribers. Any annualized financial figures in this report are **illustrative scenario projections with declared assumptions**, not verified accounting losses.
5. **Correlation vs. Causation:**
   - Statistical patterns connecting equipment age (`eqpdays`), usage decline (`change_mou`), and refurbished handsets (`refurb_new`) with churn are **empirical associations and predictive signals**. They are not proven causal mechanisms, as contract terms, network telemetry, and customer care transcripts are unobserved.

---

## 🏆 Verified Machine Learning & Targeting Performance

Both models were trained and evaluated on an **identical, leak-free 80/20 stratified holdout split** (80,000 training records, 20,000 test records; `random_state=42`, `stratify=y`). Preprocessing transformations (median imputation, one-hot encoding, standard scaling) were fit **strictly on the training partition**.

### Out-of-Sample Holdout Set Performance (N = 20,000)

| Evaluation Metric / Dimension | Baseline Model: Regularized Logistic Regression (L2) | Primary Champion: LightGBM Classifier (300 Trees) | Lift / Performance Delta |
| :--- | :---: | :---: | :---: |
| **Exact Algorithm Name** | Scikit-learn LogisticRegression (`C=0.1`, `lbfgs`) | **LightGBM Classifier (`num_leaves=31`, `max_depth=6`)** | Primary Champion |
| **ROC-AUC Score** | `0.6526` | **`0.6924`** | **+0.0398 (+6.1% rel.)** |
| **Accuracy** | `61.10%` | **`63.49%`** | **+2.39 percentage points** |
| **Precision (Macro / Binary)** | `0.6076` | **`0.6289`** | **+0.0213** |
| **Recall (Sensitivity)** | `0.6071` | **`0.6426`** | **+0.0355** |
| **F1-Score** | `0.6074` | **`0.6356`** | **+0.0282** |
| **Precision @ Top 10% (Decile 0)** | `70.20%` (1,404 / 2,000) | **`78.30%` (1,566 / 2,000)** | **+8.10 percentage points** |
| **Churn Capture @ Top 10% (Recall@10%)** | `14.16%` (1,404 / 9,912) | **`15.80%` (1,566 / 9,912)** | **+1.64 percentage points** |
| **Decile 0 Lift** | `1.42x` baseline | **`1.58x` baseline** | **+0.16x Lift** |
| **Precision @ Top 20% (Deciles 0 & 1)** | `67.35%` (2,694 / 4,000) | **`72.85%` (2,914 / 4,000)** | **+5.50 percentage points** |
| **Cumulative Churn Capture @ Top 20%** | `27.18%` (2,694 / 9,912) | **`29.40%` (2,914 / 9,912)** | **+2.22 percentage points** |
| **Cumulative Lift @ Top 20%** | `1.36x` baseline | **`1.47x` baseline** | **+0.11x Lift** |

---

## 🔍 Verified Exploratory Data Analysis (EDA) Findings

All empirical statistics below were recalculated directly from the supplied CSV datasets:

1. **Equipment Age (`eqpdays`) Association:**
   - **Observed Association:** Churn is lowest among subscribers with equipment aged 180–300 days (**37.53%**, n = 21,761) and surges to **58.57%** (n = 23,645) for equipment aged 300–420 days (+21.04 percentage points).
   - **Context & Sample Sizes:**
     - `0–180 days` (0–6 mo): n = 18,161 | Churn = 41.81%
     - `180–300 days` (6–10 mo): n = 21,761 | Churn = 37.53%
     - `300–420 days` (10–14 mo): n = 23,645 | Churn = 58.57%
     - `420–600 days` (14–20 mo): n = 17,055 | Churn = 53.25%
     - `600–1000 days` (20–33 mo): n = 16,514 | Churn = 55.66%
     - `>1000 days` (33+ mo): n = 2,697 | Churn = 60.18%
     - `Missing`: n = 1 | Churn = 0.00%
   - **Interpretation Boundary:** This pattern represents an empirical **association**. The dataset documentation does not contain contract dates; therefore, this spike cannot be claimed as proven contract expiration or anniversary cliff. It serves as an empirical risk marker.
2. **Usage Trajectory (`change_mou`) as an Early Predictive Signal:**
   - **Observed Association:** Accounts experiencing severe drop in monthly minutes of use relative to their previous 3-month average exhibit higher churn:
     - `Severe Drop (<-150 min)`: n = 16,311 | Churn = 54.49%
     - `Moderate Drop (-150 to -30 min)`: n = 22,906 | Churn = 49.74%
     - `Stable (+/-30 min)`: n = 27,507 | Churn = 49.61%
     - `Moderate Growth (+30 to +150 min)`: n = 18,650 | Churn = 45.32%
     - `High Surge (>+150 min)`: n = 13,735 | Churn = 47.35%
     - `Missing`: n = 891 | Churn = 53.09%
   - **Interpretation Boundary:** Behavioral usage deterioration correlates with subsequent churn and serves as a strong predictive signal, but does not prove direct causation.
3. **Handset Source Category (`refurb_new`) Correlation:**
   - **Observed Counts & Rates:**
     - Brand New (`N`): n = 85,593 (85.6% of base) | Churn = 48.91%
     - Refurbished (`R`): n = 14,406 (14.4% of base) | Churn = 53.44%
     - Missing: n = 1 | Churn = 0.00%
   - **Interpretation Boundary:** Accounts with refurbished handsets exhibit a +4.53 percentage point higher churn rate. This is described as an empirical correlation/predictive signal, not a proven hardware defect penalty.
4. **Monthly Recurring Charges (`totmrc_Mean`) & Effective Unit Rate:**
   - Churned customers have slightly lower recurring charges (mean $44.33 vs. $47.96) and lower monthly minutes (478 vs. 550 min), but higher overage charges ($14.54 vs. $12.59) and higher effective per-minute costs (`rev_Mean / (mou_Mean + 1)`).

---

## 💰 Defensible Financial Impact & Sensitivity Analysis

### Framework & Declared Assumptions
- **Time Horizons:** Evaluated across both a **6-month retention horizon** and an illustrative **12-month retention horizon**.
- **Customer Value Proxy:** Mean monthly revenue (`rev_Mean`) across all records is **$58.72** ($58.719985).
- **Contribution Margin Proxy:** Because true cost-to-serve is unavailable, we model a realistic **60% contribution margin proxy** ($35.23/mo retained margin) to avoid treating gross revenue as net profit. An upper-bound 100% gross revenue proxy is also reported and explicitly labeled as revenue, NOT profit.
- **Intervention Cost:** Assumed at **$45.00 per targeted customer** (loyalty voucher subsidy + outbound care operations).
- **Retention Save Rate:** Assumed at **25.0%** (tested across 15%–35% in sensitivity analysis).

### Scenario A: Out-of-Sample Holdout Set (N = 20,000 Accounts, Top 20% Targeted = 4,000)

| Financial Metric | 6-Month Horizon (60% Margin) | 12-Month Horizon (60% Margin) | 12-Month Gross Revenue Proxy |
| :--- | :---: | :---: | :---: |
| **Targeted Accounts (Top 20%)** | 4,000 accounts | 4,000 accounts | 4,000 accounts |
| **Captured Churners (Precision = 72.85%)** | 2,914 accounts | 2,914 accounts | 2,914 accounts |
| **Retained Accounts (25% Save Rate)** | 728.5 accounts | 728.5 accounts | 728.5 accounts |
| **Program Campaign Cost ($45/contact)** | $180,000.00 | $180,000.00 | $180,000.00 |
| **Preserved Economic Value** | $153,999.03 | $307,998.06 | $513,330.11 (Revenue proxy) |
| **Net Financial Margin** | **-$26,000.97** | **+$127,998.06** | **+$333,330.11** |
| **Program ROI** | **-14.4%** | **+71.1%** | **+185.2%** |
| **Required Break-Even Save Rate** | **29.22%** | **14.61%** | **8.77%** |

### Scenario B: Scaled Illustrative Portfolio (Full 100,000 Accounts, Top 20% Targeted = 20,000)
*Caveat: Scaling assumes identical capture rates apply across the entire base over an illustrative horizon.*

| Financial Metric | 6-Month Horizon (60% Margin) | 12-Month Horizon (60% Margin) | 12-Month Gross Revenue Proxy |
| :--- | :---: | :---: | :---: |
| **Scaled Targeted Accounts (Top 20%)** | 20,000 accounts | 20,000 accounts | 20,000 accounts |
| **Scaled Captured Churners (29.40% Capture)** | 14,571 accounts | 14,571 accounts | 14,571 accounts |
| **Scaled Retained Accounts (25% Save Rate)** | 3,643 accounts | 3,643 accounts | 3,643 accounts |
| **Scaled Campaign Investment ($45/contact)** | $900,000.00 | $900,000.00 | $900,000.00 |
| **Preserved Economic Value** | $770,026.23 | $1,540,052.47 | $2,566,754.12 (Revenue proxy) |
| **Net Financial Margin** | **-$129,973.77** | **+$640,052.47** | **+$1,666,754.12** |
| **Estimated Program ROI** | **-14.4%** | **+71.1%** | **+185.2%** |
| **Required Break-Even Save Rate** | **29.22%** | **14.61%** | **8.77%** |

### Break-Even Formula & Sensitivity Matrices
The break-even retention success rate $s^*$ is given by:
$$s^* = \frac{\text{Campaign Cost per Contact}}{\text{Precision@K} \times \text{Customer Value Proxy}}$$

**Holdout Set Sensitivity Matrix (6-Month Horizon, 60% Contribution Margin Proxy):**

| Cost per Contact ($) | Save Rate (%) | Retained Accounts | Preserved Margin ($) | Net Financial Margin ($) | Program ROI (%) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$35.00** | 15% | 437.1 | $92,399 | -$47,601 | -34.0% |
| **$35.00** | 20% | 582.8 | $123,199 | -$16,801 | -12.0% |
| **$35.00** | 25% | 728.5 | $153,999 | +$13,999 | +10.0% |
| **$45.00** | 15% | 437.1 | $92,399 | -$87,601 | -48.7% |
| **$45.00** | 20% | 582.8 | $123,199 | -$56,801 | -31.6% |
| **$45.00** | 25% | 728.5 | $153,999 | -$26,001 | -14.4% |
| **$45.00** | 30% | 874.2 | $184,799 | +$4,799 | +2.7% |
| **$60.00** | 25% | 728.5 | $153,999 | -$86,001 | -35.8% |
| **$60.00** | 30% | 874.2 | $184,799 | -$55,201 | -23.0% |
| **$60.00** | 35% | 1019.9 | $215,599 | -$24,401 | -10.2% |

**Holdout Set Sensitivity Matrix (12-Month Horizon, 60% Contribution Margin Proxy):**

| Cost per Contact ($) | Save Rate (%) | Retained Accounts | Preserved Margin ($) | Net Financial Margin ($) | Program ROI (%) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$35.00** | 15% | 437.1 | $184,799 | +$44,799 | +32.0% |
| **$35.00** | 20% | 582.8 | $246,398 | +$106,398 | +76.0% |
| **$35.00** | 25% | 728.5 | $307,998 | +$167,998 | +120.0% |
| **$45.00** | 15% | 437.1 | $184,799 | +$4,799 | +2.7% |
| **$45.00** | 20% | 582.8 | $246,398 | +$66,398 | +36.9% |
| **$45.00** | 25% | 728.5 | $307,998 | +$127,998 | +71.1% |
| **$45.00** | 30% | 874.2 | $369,598 | +$189,598 | +105.3% |
| **$60.00** | 20% | 582.8 | $246,398 | +$6,398 | +2.7% |
| **$60.00** | 25% | 728.5 | $307,998 | +$67,998 | +28.3% |
| **$60.00** | 30% | 874.2 | $369,598 | +$129,598 | +54.0% |

---

## 🚀 Proposed Business Solution & Randomized Pilot (RCT) Design

### 1. Operational Retention Workflow
- **Batch Scoring (Day 0):** The automated LightGBM model scores all active accounts monthly on the observation date.
- **Lead Triage & Dispatch (Days 1–5):** Top 20% risk accounts (Deciles 0 & 1) are flagged in the CRM and assigned to retention queues.
- **Tailored Intervention Window (Days 5–25, prior to the 31–60 day churn window):**
  - **Aging Equipment (`eqpdays` > 300):** $45 handset upgrade voucher paired with renewal incentives.
  - **Usage Deterioration (`change_mou` < -50m):** Proactive rate-plan consultation to right-size base allowances and eliminate excessive per-minute unit costs.
  - **Refurbished Handsets (`refurb_new` == 'R'):** Device diagnostic check and certified replacement discount.

### 2. Prospective Randomized Controlled Trial (RCT) Pilot
To establish causal incremental impact and prevent the **post hoc attribution fallacy** (attributing all retained customers to the campaign), Company A should execute a pilot on the top 20% cohort (e.g., 4,000 accounts):
- **Treatment Group (85%: 3,400 accounts):** Receives proactive outreach and $45 voucher/plan realignment.
- **Randomized Control Group (15%: 600 accounts):** Receives standard Business-As-Usual (BAU) treatment (no proactive intervention).

### 3. Pilot KPI Measurement
1. **Contact Rate:** Target >85% of treatment accounts reached.
2. **Offer Acceptance Rate:** Target >25% voucher/plan uptake.
3. **Incremental Retention Rate:** Direct difference between control and treatment churn rates ($\text{Churn}_{\text{Control}} - \text{Churn}_{\text{Treatment}}$).
4. **Cost per Incrementally Retained Account:** $\frac{\text{Total Campaign Spend}}{\text{Incremental Retained Accounts}}$.
5. **Net Contribution Lift:** Preserved contribution margin minus total program costs.

---

## ⚠️ Methodological Limitations & Future Requirements

1. **Preprocessing Leakage vs. Feature Availability Timing:**
   - Preprocessing leakage from the holdout partition is strictly eliminated by the scikit-learn training-only transformer workflow.
   - Availability timing of source features at prediction time is assumed based on column descriptions and must be confirmed with Company A before live operational deployment.
   - Audited uncertain fields (`change_mou`, `change_rev`, `adjrev`, `adjmou`, `adjqty`, `recv_sms_Mean`, `iwylis_vce_Mean`) carry operational reconciliation dependencies that require an ETL audit.
2. **Absence of Calendar Dates & Temporal Split:**
   - The dataset is a cross-sectional extract without calendar timestamps. Model validation was conducted via stratified random holdout (80/20). Longitudinal validation across future calendar quarters is required to monitor temporal drift.
3. **Target Specificity:**
   - The label reflects churn between 31 and 60 days post-observation. It does not measure immediate 1–30 day churn or multi-year retention.
4. **Unobserved Variables:**
   - Contract start/expiration dates, detailed network telemetry, competitor pricing, and call center voice transcripts were not provided.
5. **Experimental Necessity:**
   - Causal campaign effectiveness must be validated via the proposed randomized pilot before enterprise-scale deployment.

---

## 📂 Repository Deliverables & File Structure

```
d:/GCI World 2026 Final Assignment/
│
├── README.md                             # Comprehensive Project Documentation (This File)
├── gci_telecom_final_assignment.ipynb    # Fully Executed Production Jupyter Notebook
├── build_nb.py                           # Reproducible Notebook Generation & Execution Script
├── sim.py                                # Standalone Financial Simulation & Sensitivity Script
│
├── telecom/                              # Company A Dataset Directory
│   ├── Client.csv                        # Subscriber Demographics & Handset Data (100k x 50)
│   └── Record.csv                        # Usage Records & Churn Target Data (100k x 51)
│
├── ENG_Company _A_ Dataset Overview.docx # Official Client Dataset Dictionary & Documentation
├── docx_text.txt                         # Extracted Plain Text of Dataset Documentation
│
├── executive_eda_overview.png            # 4-Panel Verified EDA Visualizations
├── model_performance_evaluation.png      # ROC Curves & LightGBM Top 15 Feature Importances
└── decile_lift_analysis.png              # Decile Risk Concentration & Cumulative Capture Plot
```

### Direct Workspace Links
- [Jupyter Notebook: `gci_telecom_final_assignment.ipynb`](file:///d:/GCI%20World%202026%20Final%20Assignment/gci_telecom_final_assignment.ipynb)
- [Notebook Builder Script: `build_nb.py`](file:///d:/GCI%20World%202026%20Final%20Assignment/build_nb.py)
- [Financial Simulation Script: `sim.py`](file:///d:/GCI%20World%202026%20Final%20Assignment/sim.py)
- [Client Dataset: `Client.csv`](file:///d:/GCI%20World%202026%20Final%20Assignment/telecom/Client.csv)
- [Record Dataset: `Record.csv`](file:///d:/GCI%20World%202026%20Final%20Assignment/telecom/Record.csv)
- [Client Dataset Overview Docx: `ENG_Company _A_ Dataset Overview.docx`](file:///d:/GCI%20World%202026%20Final%20Assignment/ENG_Company%20_A_%20Dataset%20Overview.docx)

---

## 📚 Traceable References & Citations

1. **Course Guidelines & Client Data Documentation:**
   - Matsuo-Iwasawa Laboratory, Graduate School of Engineering, The University of Tokyo. *Global Consumer Intelligence (GCI) World 2026 Winter Final Assignment Requirements & Guidelines*.
   - Company A. *Telecommunications Subscriber Client and Usage Records Overview (Dataset Documentation)*.
2. **Machine Learning & Statistical Methods:**
   - Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research (JMLR), 12, 2825–2830.
   - Ke, G., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. Advances in Neural Information Processing Systems (NeurIPS 30), 3146–3154.
   - Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions (SHAP)*. Advances in Neural Information Processing Systems (NeurIPS 30), 4765–4774.
3. **Telecommunications Churn Economics & Uplift Modeling:**
   - Ascarza, E. (2018). *Retention First, but for Whom? Identifying the Target of Retention Interventions*. Journal of Marketing Research, 55(1), 1–16.
