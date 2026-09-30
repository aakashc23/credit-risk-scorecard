"""Single source of truth for paths, dates, feature lists and model constants."""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
FIGURES = ROOT / "reports" / "figures"
SQL_DIR = ROOT / "sql"

RAW_FILENAME = "accepted_2007_to_2018Q4.csv.gz"  # Kaggle: wordsforthewise/lending-club
RAW_PATH = DATA_RAW / RAW_FILENAME

RANDOM_STATE = 42

# --------------------------------------------------------------------------- sample definition
# Only 36-month loans: the snapshot ends 2018Q4, so a 36m loan issued by Dec-2015 has
# reached contractual maturity and its outcome is (almost always) fully observed.
TERM_MONTHS = 36
DEV_START = "2012-01-01"   # development window (train + in-time test)
DEV_END = "2014-12-31"
OOT_START = "2015-01-01"   # out-of-time validation window
OOT_END = "2015-12-31"
TEST_SIZE = 0.30           # in-time test share of the development window

# --------------------------------------------------------------------------- target
BAD_STATUSES = {
    "Charged Off",
    "Default",
    "Does not meet the credit policy. Status:Charged Off",
}
GOOD_STATUSES = {
    "Fully Paid",
    "Does not meet the credit policy. Status:Fully Paid",
}
# Anything else (Current, In Grace Period, Late (16-30 days), Late (31-120 days)) is
# "indeterminate" and excluded; the count is logged.
TARGET = "bad"

# --------------------------------------------------------------------------- columns
ID_COLS = ["id", "issue_d", "term", "loan_status"]

# Application-time fields allowed as model inputs (before feature engineering).
CANDIDATE_FEATURES = [
    "loan_amnt",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "purpose",
    "application_type",
    "dti",
    "delinq_2yrs",
    "earliest_cr_line",          # -> credit_history_months
    "fico_range_low",            # -> fico_mid
    "fico_range_high",
    "inq_last_6mths",
    "mths_since_last_delinq",
    "mths_since_last_record",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "mort_acc",
    "pub_rec_bankruptcies",
    "acc_open_past_24mths",
    "bc_util",
    "num_actv_rev_tl",
    "percent_bc_gt_75",
    "tot_cur_bal",
    "total_rev_hi_lim",
    "mo_sin_rcnt_tl",
    "num_tl_op_past_12m",
    "avg_cur_bal",
    "mths_since_recent_inq",
    "tax_liens",
    "collections_12_mths_ex_med",
]

# Known at origination but deliberately NOT model features.
BENCHMARK_COLS = ["grade", "sub_grade", "int_rate", "installment"]  # LC's own pricing model
EXCLUDED_FAIR_LENDING = ["zip_code", "addr_state"]

# Post-origination fields. NEVER model features. A subset is loaded ONLY to calibrate
# LGD/EAD on training-window defaults and to measure realised loss for back-testing.
LOSS_CALIBRATION_COLS = ["funded_amnt", "total_rec_prncp", "recoveries", "collection_recovery_fee"]

LEAKAGE_COLUMNS = {
    "recoveries", "collection_recovery_fee", "total_pymnt", "total_pymnt_inv",
    "total_rec_prncp", "total_rec_int", "total_rec_late_fee", "last_pymnt_d",
    "last_pymnt_amnt", "next_pymnt_d", "last_credit_pull_d", "last_fico_range_high",
    "last_fico_range_low", "out_prncp", "out_prncp_inv", "pymnt_plan", "loan_status",
    "debt_settlement_flag", "debt_settlement_flag_date", "settlement_status",
    "settlement_date", "settlement_amount", "settlement_percentage", "settlement_term",
    "hardship_flag", "hardship_type", "hardship_reason", "hardship_status",
    "deferral_term", "hardship_amount", "hardship_start_date", "hardship_end_date",
    "payment_plan_start_date", "hardship_length", "hardship_dpd", "hardship_loan_status",
    "orig_projected_additional_accrued_interest", "hardship_payoff_balance_amount",
    "hardship_last_payment_amount", "funded_amnt_inv",
}

# Candidate raw fields dropped if missing share in the development window exceeds this.
MAX_MISSING_SHARE = 0.30

# --------------------------------------------------------------------------- binning / selection
MAX_FINE_BINS = 10
MAX_COARSE_BINS = 6
MIN_BIN_SHARE = 0.05         # each numeric bin holds >= 5% of rows
MIN_CATEGORY_SHARE = 0.01    # rarer categories pooled into "Other"
WOE_SMOOTHING = 0.5          # added to good/bad counts to avoid log(0)
MIN_IV = 0.02                # drop weaker features
SUSPICIOUS_IV = 0.50         # flag for leakage review
MAX_CORR = 0.70              # |corr| between WOE features; keep the higher-IV one
MAX_MODEL_FEATURES = 15

# --------------------------------------------------------------------------- score scaling
# score = OFFSET + FACTOR * ln(odds_good), FACTOR = PDO / ln 2
# Anchor: 650 points at 10:1 good:bad odds (PD ~9%); every +50 points doubles the odds.
BASE_SCORE = 650
BASE_ODDS = 10.0
PDO = 50
SCORE_MIN, SCORE_MAX = 300, 850

# --------------------------------------------------------------------------- cut-off grid
CUTOFF_STEP = 5
N_DECILES = 10
