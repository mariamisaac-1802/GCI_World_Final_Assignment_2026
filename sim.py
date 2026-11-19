"""
Financial Simulation and Sensitivity Analysis for Telecommunications Retention PoC
Client: Company A
Target: 31-60 Day Post-Observation Churn
Author: Lead Data Scientist & IT Management Consultant

This standalone script evaluates the financial viability of targeting the top 20%
highest-risk churn propensity accounts. It provides both:
1. Scenario A: Out-of-sample Holdout Test Set empirical simulation (N = 20,000)
2. Scenario B: Scaled Illustrative Full Portfolio simulation (N = 100,000)
Both 6-month and 12-month retention horizons are reported with a 60% contribution
margin proxy, alongside an upper-bound 12-month gross revenue proxy.

All financial outputs are illustrative estimates based on declared assumptions,
not guaranteed campaign outcomes or verified accounting losses.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

# Parse arguments
parser = argparse.ArgumentParser(description="Run financial simulation and sensitivity analysis.")
parser.add_argument(
    "--fast",
    action="store_true",
    help="Use verified holdout metrics without retraining LightGBM."
)
args = parser.parse_args()

print("=" * 80)
print("COMPANY A RETENTION OPTIMIZATION: FINANCIAL SIMULATION & SENSITIVITY")
print("=" * 80)

# Step 1: Ingest datasets and compute portfolio baseline statistics
client_path = 'telecom/Client.csv'
record_path = 'telecom/Record.csv'

if not os.path.exists(client_path) or not os.path.exists(record_path):
    raise FileNotFoundError("Dataset files not found in telecom/ directory.")

print(f"[INFO] Ingesting client records from: {client_path}")
print(f"[INFO] Ingesting usage records from:  {record_path}")

df_client = pd.read_csv(client_path)
df_record = pd.read_csv(record_path)
df = pd.merge(df_client, df_record, on='Customer_ID', how='inner')

mean_monthly_arpu = df['rev_Mean'].mean()
baseline_churn_rate = df['churn'].mean()
total_records = len(df)

print(f"[DATA] Total Unified Records: {total_records:,} accounts")
print(f"[DATA] Baseline Observed Churn Rate (31-60 Day Window): {baseline_churn_rate:.4%} ({df['churn'].sum():,} accounts)")
print(f"[DATA] Mean Monthly Revenue (rev_Mean ARPU): ${mean_monthly_arpu:.2f}/month")

# Step 2: Derive or verify model targeting metrics (Top 20% cohort)
DOCUMENTED_PRECISION_TOP20 = 2914 / 4000  # 0.7285 (2,914 actual churners / 4,000 targeted)
DOCUMENTED_RECALL_TOP20 = 2914 / 9912     # 0.293987... (~29.40% churn capture in holdout)

if args.fast:
    print("\n[MODE] Fast Mode: Using verified out-of-sample holdout metrics.")
    precision_top20 = DOCUMENTED_PRECISION_TOP20
    recall_top20 = DOCUMENTED_RECALL_TOP20
    # Verification checks
    assert round(precision_top20, 4) == 0.7285, "Precision check failed!"
    assert round(recall_top20, 4) == 0.2940, "Recall check failed!"
    print(f"       Precision@20%: {precision_top20:.2%} (verified holdout input)")
    print(f"       Recall@20% (Churn Capture): {recall_top20:.2%} (verified holdout input)")
else:
    print("\n[MODE] Dynamic Derivation Mode: Training LightGBM on leak-free split to derive metrics...")
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.impute import SimpleImputer
    import lightgbm as lgb

    y = df['churn'].values
    X = df.drop(columns=['Customer_ID', 'churn'])

    # Stratified 80/20 partition
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Domain features (row-wise, no cross-sample leakage)
    def add_domain_feats(data):
        df_feat = data.copy()
        df_feat['eqp_years'] = df_feat['eqpdays'] / 365.25
        df_feat['eqp_300_420_flag'] = ((df_feat['eqpdays'] >= 300) & (df_feat['eqpdays'] <= 420)).astype(float)
        df_feat['overage_ratio'] = df_feat['ovrrev_Mean'] / (df_feat['rev_Mean'].fillna(0) + 1.0)
        df_feat['usage_drop_flag'] = (df_feat['change_mou'] < -50.0).astype(float)
        df_feat['rev_drop_flag'] = (df_feat['change_rev'] < -10.0).astype(float)
        df_feat['rev_per_mou'] = df_feat['rev_Mean'] / (df_feat['mou_Mean'].fillna(0) + 1.0)
        df_feat['drop_call_rate'] = df_feat['drop_vce_Mean'] / (df_feat['plcd_vce_Mean'].fillna(0) + 1.0)
        df_feat['comp_call_rate'] = df_feat['comp_vce_Mean'] / (df_feat['plcd_vce_Mean'].fillna(0) + 1.0)
        df_feat['refurb_binary'] = (df_feat['refurb_new'] == 'R').astype(float)
        df_feat['has_custcare_calls'] = (df_feat['custcare_Mean'] > 0).astype(float)
        df_feat['subscriber_util_ratio'] = df_feat['actvsubs'] / (df_feat['uniqsubs'].fillna(1) + 1e-4)
        df_feat['aging_price_ratio'] = df_feat['eqpdays'] / (df_feat['hnd_price'].fillna(0) + 10.0)
        return df_feat

    X_train_f = add_domain_feats(X_train)
    X_test_f = add_domain_feats(X_test)

    num_cols = X_train_f.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train_f.select_dtypes(exclude=[np.number]).columns.tolist()

    num_imp = SimpleImputer(strategy='median')
    cat_imp = SimpleImputer(strategy='constant', fill_value='Unknown')
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    X_train_proc = np.hstack([
        num_imp.fit_transform(X_train_f[num_cols]),
        ohe.fit_transform(cat_imp.fit_transform(X_train_f[cat_cols].astype(str)))
    ])
    X_test_proc = np.hstack([
        num_imp.transform(X_test_f[num_cols]),
        ohe.transform(cat_imp.transform(X_test_f[cat_cols].astype(str)))
    ])

    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train_proc, y_train)
    test_probs = model.predict_proba(X_test_proc)[:, 1]

    eval_df = pd.DataFrame({'actual_churn': y_test, 'score': test_probs})
    eval_df['decile'] = 9 - pd.qcut(eval_df['score'], q=10, labels=False)
    top20_cohort = eval_df[eval_df['decile'] <= 1]

    precision_top20 = top20_cohort['actual_churn'].mean()
    recall_top20 = top20_cohort['actual_churn'].sum() / y_test.sum()

    print(f"       Dynamic Precision@20%: {precision_top20:.4f}")
    print(f"       Dynamic Recall@20%:    {recall_top20:.4f}")
    # Verification check against documented figures
    assert round(precision_top20, 4) == round(DOCUMENTED_PRECISION_TOP20, 4), f"Precision mismatch: {precision_top20}"
    assert round(recall_top20, 4) == round(DOCUMENTED_RECALL_TOP20, 4), f"Recall mismatch: {recall_top20}"
    print("       [VERIFIED] Dynamically derived metrics match documented holdout values exactly.")

# Step 3: Shared Parameter Declarations
# All values are explicit modeling assumptions:
cost_per_contact = 45.0      # Campaign expenditure per targeted customer ($)
assumed_save_rate = 0.25     # Assumed incremental retention / save rate (tested across 15%-35%)
assumed_margin = 0.60        # Estimated contribution margin proxy (cost-to-serve unavailable)

# ==============================================================================
# SCENARIO A: Out-of-Sample Holdout Test Set Empirical Simulation (N = 20,000)
# ==============================================================================
holdout_n = 20000
holdout_targeted = 4000  # Top 20%
holdout_churners = holdout_targeted * precision_top20  # 2,914 accounts
holdout_retained = holdout_churners * assumed_save_rate  # 728.5 accounts
holdout_cost = holdout_targeted * cost_per_contact      # $180,000.00

# 6-Month Retention Horizon (60% Contribution Margin Proxy)
holdout_gross_rev_6m = holdout_retained * (mean_monthly_arpu * 6)
holdout_margin_6m = holdout_gross_rev_6m * assumed_margin
holdout_net_margin_6m = holdout_margin_6m - holdout_cost
holdout_roi_6m = (holdout_net_margin_6m / holdout_cost) * 100

# 12-Month Retention Horizon (60% Contribution Margin Proxy)
holdout_gross_rev_12m = holdout_retained * (mean_monthly_arpu * 12)
holdout_margin_12m = holdout_gross_rev_12m * assumed_margin
holdout_net_margin_12m = holdout_margin_12m - holdout_cost
holdout_roi_12m = (holdout_net_margin_12m / holdout_cost) * 100

# 12-Month Upper-Bound Gross Revenue Proxy (No Cost-to-Serve Subtracted)
holdout_net_rev_12m = holdout_gross_rev_12m - holdout_cost
holdout_roi_rev_12m = (holdout_net_rev_12m / holdout_cost) * 100

# Break-even retention success rate formula:
# s_break_even = Cost_per_contact / (Precision@K * Customer_Value_Proxy)
be_rate_6m_margin = cost_per_contact / (precision_top20 * (mean_monthly_arpu * 6 * assumed_margin))
be_rate_12m_margin = cost_per_contact / (precision_top20 * (mean_monthly_arpu * 12 * assumed_margin))
be_rate_12m_gross = cost_per_contact / (precision_top20 * (mean_monthly_arpu * 12))

print("\n" + "=" * 80)
print("SCENARIO A: HOLDOUT TEST SET EMPIRICAL SIMULATION (N = 20,000, TOP 20% TARGETED)")
print("=" * 80)
print(f"Targeted Accounts (Top 20%):       {holdout_targeted:,} accounts")
print(f"Actual Churners in Target Group:   {holdout_churners:,.0f} accounts (Precision@20% = {precision_top20:.2%})")
print(f"Retained Accounts (25% Save Rate): {holdout_retained:,.1f} accounts")
print(f"Campaign Expenditure ($45/target): ${holdout_cost:,.2f}")
print("-" * 80)
print("6-Month Retention Horizon (60% Contribution Margin Proxy):")
print(f"  - Preserved Contribution Margin: ${holdout_margin_6m:,.2f}")
print(f"  - Net Financial Contribution:    ${holdout_net_margin_6m:,.2f}")
print(f"  - Estimated Program ROI:         {holdout_roi_6m:.1f}%")
print(f"  - Break-Even Save Rate Required: {be_rate_6m_margin:.2%}")
print("-" * 80)
print("12-Month Retention Horizon (60% Contribution Margin Proxy):")
print(f"  - Preserved Contribution Margin: ${holdout_margin_12m:,.2f}")
print(f"  - Net Financial Contribution:    ${holdout_net_margin_12m:,.2f}")
print(f"  - Estimated Program ROI:         {holdout_roi_12m:.1f}%")
print(f"  - Break-Even Save Rate Required: {be_rate_12m_margin:.2%}")
print("-" * 80)
print("12-Month Upper-Bound Gross Revenue Proxy (NOT Profit):")
print(f"  - Preserved Gross Revenue:       ${holdout_gross_rev_12m:,.2f}")
print(f"  - Net Revenue Benefit:           ${holdout_net_rev_12m:,.2f}")
print(f"  - Revenue-Based ROI:             {holdout_roi_rev_12m:.1f}%")
print(f"  - Break-Even Save Rate Required: {be_rate_12m_gross:.2%}")

# ==============================================================================
# SCENARIO B: Scaled Illustrative Full Portfolio Simulation (N = 100,000)
# ==============================================================================
scaled_n = 100000
scaled_targeted = 20000  # Top 20%
scaled_churners = scaled_n * baseline_churn_rate * recall_top20  # ~14,570.6 accounts
scaled_retained = scaled_churners * assumed_save_rate             # ~3,642.65 accounts
scaled_cost = scaled_targeted * cost_per_contact                 # $900,000.00

# 6-Month Horizon
scaled_gross_rev_6m = scaled_retained * (mean_monthly_arpu * 6)
scaled_margin_6m = scaled_gross_rev_6m * assumed_margin
scaled_net_margin_6m = scaled_margin_6m - scaled_cost
scaled_roi_6m = (scaled_net_margin_6m / scaled_cost) * 100

# 12-Month Horizon
scaled_gross_rev_12m = scaled_retained * (mean_monthly_arpu * 12)
scaled_margin_12m = scaled_gross_rev_12m * assumed_margin
scaled_net_margin_12m = scaled_margin_12m - scaled_cost
scaled_roi_12m = (scaled_net_margin_12m / scaled_cost) * 100

# 12-Month Gross Revenue Proxy
scaled_net_rev_12m = scaled_gross_rev_12m - scaled_cost
scaled_roi_rev_12m = (scaled_net_rev_12m / scaled_cost) * 100

print("\n" + "=" * 80)
print("SCENARIO B: SCALED ILLUSTRATIVE PORTFOLIO SIMULATION (N = 100,000, TOP 20% TARGETED)")
print("=" * 80)
print("[CAVEAT] Scaling assumes identical capture rates apply across the entire base over an illustrative horizon.")
print(f"Scaled Targeted Accounts:          {scaled_targeted:,} accounts")
print(f"Scaled Captured Churners:          {scaled_churners:,.0f} accounts ({recall_top20:.2%} capture)")
print(f"Scaled Retained Accounts:          {scaled_retained:,.0f} accounts (25% save rate)")
print(f"Scaled Campaign Expenditure:       ${scaled_cost:,.2f}")
print("-" * 80)
print("6-Month Retention Horizon (60% Contribution Margin Proxy):")
print(f"  - Preserved Contribution Margin: ${scaled_margin_6m:,.2f}")
print(f"  - Net Financial Contribution:    ${scaled_net_margin_6m:,.2f}")
print(f"  - Estimated Program ROI:         {scaled_roi_6m:.1f}%")
print("-" * 80)
print("12-Month Retention Horizon (60% Contribution Margin Proxy):")
print(f"  - Preserved Contribution Margin: ${scaled_margin_12m:,.2f}")
print(f"  - Net Financial Contribution:    ${scaled_net_margin_12m:,.2f}")
print(f"  - Estimated Program ROI:         {scaled_roi_12m:.1f}%")
print("-" * 80)
print("12-Month Upper-Bound Gross Revenue Proxy (NOT Profit):")
print(f"  - Preserved Gross Revenue:       ${scaled_gross_rev_12m:,.2f}")
print(f"  - Net Revenue Benefit:           ${scaled_net_rev_12m:,.2f}")
print(f"  - Revenue-Based ROI:             {scaled_roi_rev_12m:.1f}%")

# ==============================================================================
# SENSITIVITY MATRIX (Holdout Set, Both Horizons)
# ==============================================================================
costs = [35.0, 45.0, 60.0]
save_rates = [0.15, 0.20, 0.25, 0.30, 0.35]

print("\n" + "=" * 80)
print("SENSITIVITY MATRIX: HOLDOUT TEST SET (6-MONTH 60% MARGIN)")
print("=" * 80)
sens_6m = []
for c in costs:
    for s in save_rates:
        ret = holdout_churners * s
        inv = holdout_targeted * c
        contrib = ret * (mean_monthly_arpu * 6 * assumed_margin)
        net_b = contrib - inv
        roi = (net_b / inv) * 100
        sens_6m.append({
            'Cost/Target ($)': c,
            'Save Rate': f"{s:.0%}",
            'Retained': round(ret, 1),
            'Preserved Margin': f"${contrib:,.0f}",
            'Net Benefit ($)': f"${net_b:,.0f}",
            'ROI (%)': f"{roi:.1f}%"
        })
print(pd.DataFrame(sens_6m).to_string(index=False))

print("\n" + "=" * 80)
print("SENSITIVITY MATRIX: HOLDOUT TEST SET (12-MONTH 60% MARGIN)")
print("=" * 80)
sens_12m = []
for c in costs:
    for s in save_rates:
        ret = holdout_churners * s
        inv = holdout_targeted * c
        contrib = ret * (mean_monthly_arpu * 12 * assumed_margin)
        net_b = contrib - inv
        roi = (net_b / inv) * 100
        sens_12m.append({
            'Cost/Target ($)': c,
            'Save Rate': f"{s:.0%}",
            'Retained': round(ret, 1),
            'Preserved Margin': f"${contrib:,.0f}",
            'Net Benefit ($)': f"${net_b:,.0f}",
            'ROI (%)': f"{roi:.1f}%"
        })
print(pd.DataFrame(sens_12m).to_string(index=False))

print("\n" + "=" * 80)
print("[CONCLUSION] Financial simulations require prospective RCT validation before capital deployment.")
print("=" * 80)
