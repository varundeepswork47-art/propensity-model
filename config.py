"""
config.py
---------
Single source of truth for file paths, column names, and per-segment
feature lists — matched to the real sample_.csv structure.

Model design: TWO models, not three.
  - "health"     : trained on customers currently holding Health -> predicts
                    propensity to convert into ANOTHER Health product
                    (health -> health cross-sell).
  - "non_health" : trained on customers currently holding Non-Health ->
                    predicts propensity to convert into a Health product
                    (non_health -> health cross-sell).

Both models use the SAME raw label column (TARGET_COLUMN_RAW) as-is, for
both segments — there is no per-segment inversion of the label anywhere
in the pipeline. This only works because the raw label already means
"converted into a Health product" for both segments (per the source
campaign mapping) — confirm that's still true if the label's source
definition ever changes, since nothing in the code enforces it; the
pipeline just trusts whatever the raw column already encodes.

One thing worth a gut-check on the real data if you haven't already: does
label=1 for a health-segment row ever fire on a customer just renewing
the SAME product code they already hold? If so that's not really a
cross-sell and would quietly inflate the health segment's positive rate.
Worth a quick spot-check, not a blocker.

Segments are derived from PRODUCT_CODE: a fixed list of codes counts as
"health", everything else is "non_health". See HEALTH_PRODUCT_CODES below.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# 1. PATHS
# ---------------------------------------------------------------------------
# MODEL_DIR is anchored to THIS file's own location (not a relative path
# dependent on the process's current working directory) — on some
# deployment platforms (including Streamlit Cloud, depending on repo
# layout), the working directory at runtime isn't guaranteed to be the
# repo root, which silently breaks a plain Path("models") lookup.
#
# Set to the repo root (same folder as app.py/config.py) since that's
# where health_model.joblib / non_health_model.joblib were uploaded.
# If you later move them into an actual "models/" subfolder on GitHub,
# change the line below to:
#   MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR = Path(__file__).resolve().parent

# Point this at wherever the labeled training extract actually lives.
# Update this path any time you get a new monthly extract.
TRAINING_FILE = Path(r"C:\Users\Hp\Downloads\June_Segment_part_1.xlsx")

# ---------------------------------------------------------------------------
# 2. IDENTIFIER / PII COLUMNS — never used as features
# ---------------------------------------------------------------------------
PII_COLUMNS = [
    "INSURED_NAME", "user_id", "KYC_Mobile", "KYC_Email",
]

# Metadata / constant / non-predictive columns observed in the sample —
# confirm these stay constant/irrelevant in the full 1.4 crore file before
# dropping blindly; if PolicyStatus or event_name show real variance at
# full scale, reconsider.
DROP_COLUMNS = [
    "Source.Name", "event_name", "CAMPAIGN_NAME", "event_time",
    "Registration_NO", "MODEL_NO", "NRMR_Code",
    "KYC_DOB", "YearsAge",          # 90%+ missing in sample — too sparse to trust
    "Segemnt",                      # no longer used for routing — segmentation now
                                     # comes from PRODUCT_CODE (see HEALTH_PRODUCT_CODES)
]

TARGET_COLUMN_RAW = "Mapping to Leads & Read User in June"
TARGET_COLUMN = "label"

# ---------------------------------------------------------------------------
# 3. DATE COLUMNS
#    Stored as Excel serial numbers (e.g. 45160) in the raw extract —
#    must be converted before use. See data_loader.convert_excel_dates().
# ---------------------------------------------------------------------------
EXCEL_SERIAL_DATE_COLUMNS = ["RelationShip_start_Date", "POLICY_END_Date"]

# ---------------------------------------------------------------------------
# 4. SEGMENT DERIVATION
# ---------------------------------------------------------------------------
# Segment is derived from PRODUCT_CODE — a row is "health" if its product
# code is in HEALTH_PRODUCT_CODES, and "non_health" for every other code
# (including codes not seen before). See segment_builder.py for the exact
# matching logic (handles numeric/string/float-string variants like
# 2824, "2824", "2824.0").
PRODUCT_CODE_COLUMN = "PRODUCT_CODE"

HEALTH_PRODUCT_CODES = [
    "2824", "2825", "2835", "2849", "2851", "2868", "2876",
]

SEGMENTS = ["health", "non_health"]

# ---------------------------------------------------------------------------
# 5. FEATURE LISTS
# ---------------------------------------------------------------------------
COMMON_FEATURES = [
    "STATE",
    "PINCODE",
    "TOTAL_PREMIUM",
    "SUM_INSURED",
    "Total_Policy_Count",
    "Total_Active_Policy",
    "Total_Inactive_Policy",
    "POLICY_TENURE",
    "BusinessTypeActual",
    "SubChannel",
    "whatsapp_opt_in",
    "Mobile Lenght",
    "PolicyStatus",          # kept as a feature — see note in README on inactive coverage
    "tenure_days",            # derived from RelationShip_start_Date
    "days_to_policy_expiry",  # derived from POLICY_END_Date
    "sub_channel_is_web_aggregator",  # derived flag pulled out of SubChannel, weighted separately below
]

SEGMENT_FEATURES = {
    "health": [
        "PED",
        "Family_Combination",
        "Total_Insured",
        "Current_Claim_Status",
        "claim_history_present",   # derived flag: Total_NO_Claim/Total_Claim_Amount populated
        "Total_NO_Claim",
        "Total_Claim_Amount",
    ],
    "non_health": [
        "vehicle_data_present",    # derived flag: MAKE/Vehicle_Age/etc populated
        "MAKE",
        "Vehicle_Age",
        "Fuel_Type",
        "RTO",
        "travel_intent_present",   # derived flag: NameOfCountryVisiting populated
    ],
}

# ---------------------------------------------------------------------------
# 6. POLICY STATUS WEIGHTING
# ---------------------------------------------------------------------------
# IMPORTANT: the current 90k training sample is 100% PolicyStatus == ACTIVE.
# The model will have literally never seen an Inactive example, so it
# cannot have learned genuine behavioral differences for that group yet.
# Two things are done about this (see train_model.py and app.py):
#   1. PolicyStatus is still included as a feature/sample-weight so the
#      pipeline is ready the moment Inactive-labeled rows exist in training.
#   2. Until then, apply a manual confidence discount at SCORING time for
#      Inactive leads — the model is extrapolating, not truly predicting,
#      for that group. Adjust this multiplier as real inactive outcomes
#      become available.
POLICY_STATUS_TRAIN_WEIGHT = {
    "ACTIVE": 1.0,
    "INACTIVE": 1.0,   # placeholder — revisit once inactive-labeled data exists
}
POLICY_STATUS_SCORING_CONFIDENCE_MULTIPLIER = {
    "ACTIVE": 1.0,
    "INACTIVE": 0.7,   # conservative discount — model is extrapolating here
}

# ---------------------------------------------------------------------------
# 7. RARE-EVENT / IMBALANCE SETTINGS
#    ~1,500 positives out of 90,000 rows (~1.7%) overall. Non-Health segment
#    has fewer positives than Health — keep an eye on per-segment positive
#    counts (train_model.py prints this) and favor simpler models
#    (shallower trees, more regularization) for whichever segment is thinner.
# ---------------------------------------------------------------------------
N_CV_FOLDS = 5
N_CV_REPEATS = 3   # repeated stratified CV — averages out fold-assignment luck when positives are thin
N_CV_REPEATS_FOR_SEARCH = 1  # lighter repeat count while comparing candidates, to keep search fast
RANDOM_STATE = 42

XGB_PARAMS = {
    "n_estimators": 500,          # capped by early stopping in practice
    "max_depth": 4,               # shallow — limited positives, avoid overfitting
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "eval_metric": "aucpr",
    "random_state": RANDOM_STATE,
    "enable_categorical": True,   # native categorical handling — see feature_engineering.encode_categoricals
    "tree_method": "hist",        # required for enable_categorical
}
EARLY_STOPPING_ROUNDS = 30

# ---------------------------------------------------------------------------
# HYPERPARAMETER SEARCH — same candidates, same process, for BOTH segments.
# ---------------------------------------------------------------------------
# Rather than assuming Non-Health needs more regularization just because it
# has fewer positives, both segments are run through this identical grid via
# cross-validation, and whichever candidate wins for THAT segment's data is
# used. This keeps the comparison fair — the data decides the complexity
# each segment can support, not a prior assumption about it.
CANDIDATE_XGB_OVERRIDES = [
    {"label": "baseline",       "params": {}},
    {"label": "conservative",   "params": {"max_depth": 3, "min_child_weight": 8}},
    {"label": "more_conservative", "params": {"max_depth": 3, "min_child_weight": 8, "learning_rate": 0.02}},
    {"label": "expressive",     "params": {"max_depth": 5, "min_child_weight": 3}},
]

# Business capacity: what fraction of scored leads can the sales team
# realistically work? Used to pick the Yes/No threshold post-training.
TOP_K_PERCENT_CAPACITY = 0.10

# ---------------------------------------------------------------------------
# 7b. TRAINING-TIME FEATURE WEIGHTS
# ---------------------------------------------------------------------------
# XGBoost's `feature_weights` biases which columns get chosen during split
# search (works together with colsample_bytree/bylevel, both <1.0 in
# XGB_PARAMS, which is what makes this have any effect at all — a column
# that's never sampled can never be picked, no matter its true predictive
# power). This is NOT the same as scaling a column's values, which does
# nothing in a tree model.
#
# The last trained health_model.joblib put ~59% of all split usage on
# SUM_INSURED alone — that's a single-feature-dependent model, which is a
# real risk for generalizing to the full campaign base if SUM_INSURED's
# distribution there differs at all from the 90k training sample. Weights
# below suppress it and boost days_to_policy_expiry / SubChannel /
# sub_channel_is_web_aggregator. Any feature not listed here defaults to
# a weight of 1.0 (xgboost's own default — unweighted, chosen uniformly
# at random alongside every other feature during column subsampling).
#
# These are RELATIVE sampling weights, not percentages — "SUM_INSURED: 0.3"
# means it's picked ~3x less often than a default (1.0) feature during
# column subsampling, not "SUM_INSURED gets exactly 30% of importance."
# The resulting importance split still depends on the data; retrain and
# re-check model.feature_importances_ after changing these.
FEATURE_TRAINING_WEIGHTS = {
    "SUM_INSURED": 0.3,                    # was dominating splits (~59%) — suppressed, not removed
    "days_to_policy_expiry": 2.5,
    "SubChannel": 2.0,
    "sub_channel_is_web_aggregator": 3.0,  # isolated web-aggregator signal, weighted highest
}

# ---------------------------------------------------------------------------
# 7c. HEALTH -> HEALTH CROSS-SELL TARGET (corrected business rule)
# ---------------------------------------------------------------------------
# Business rule as of the latest correction:
#   - non_health customers -> target = converted into a HEALTH product (unchanged)
#   - health customers     -> target = converted into a DIFFERENT HEALTH
#                             product (NOT into Non-Health, which is what
#                             the model was predicting before)
#
# Building the health->health target requires knowing what product code a
# customer actually converted into, separate from PRODUCT_CODE (their
# EXISTING product). Set this to that column's real name in your raw file.
# Left as None on purpose: guessing at a plausible-sounding column name
# and silently training on it is worse than failing loudly, since a wrong
# guess here produces a model that looks fine in CV but is trained on a
# meaningless label. See target_builder.py.
PURCHASED_PRODUCT_CODE_COLUMN = None  # e.g. "New_PRODUCT_CODE" / "Converted_Product_Code" — set me

# ---------------------------------------------------------------------------
# 8. DASHBOARD TOP-K% CUTOFF (scoring time — app.py only)
# ---------------------------------------------------------------------------
# This is DIFFERENT from TOP_K_PERCENT_CAPACITY above. TOP_K_PERCENT_CAPACITY
# is baked into the saved model bundle at TRAINING time (against the training
# holdout). DASHBOARD_TOP_K_DEFAULT drives the live cutoff shown in app.py,
# recomputed fresh off whatever file the user just uploaded — so the actual
# probability score behind "top 3%" will legitimately differ upload to
# upload, and will differ from the training-time threshold too.
#
# Defaults reflect each segment's real working capacity for the sales team:
# Health = top 10%, Non-Health = top 6%. The dashboard exposes a free-form
# slider (not fixed steps) so a user can dial in any whole-percent value
# in [DASHBOARD_TOP_K_MIN, DASHBOARD_TOP_K_MAX] — e.g. 4% or 12% — for
# capacity what-ifs, without touching the model itself.
DASHBOARD_TOP_K_DEFAULT = {
    "health": 10,
    "non_health": 6,
}
DASHBOARD_TOP_K_MIN = 1
DASHBOARD_TOP_K_MAX = 50
