"""
config.py
---------
Single source of truth for file paths, column names, and per-segment
feature lists — matched to the real sample_.csv structure.

Model design: TWO models, not three.
  - "health"     : trained on customers currently holding Health -> predicts
                    propensity to cross-sell into Non-Health (motor/travel/etc).
  - "non_health" : trained on customers currently holding Non-Health ->
                    predicts propensity to cross-sell into Health.

Segments are derived (not hard-coded) from PRODUCT_CODE / SubChannel so the
logic still works once the full 1.4 crore file is loaded.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# 1. PATHS
# ---------------------------------------------------------------------------
MODEL_DIR = Path("models")

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
    # NOTE: "Segemnt" is deliberately NOT dropped here — it's the column
    # segment_builder.py uses to route each row to the right model. It's
    # dropped automatically later (after segmenting) since it isn't listed
    # in COMMON_FEATURES or SEGMENT_FEATURES.
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
# Segment is read directly from the existing "Segemnt" column (values like
# "Health - High Intent 260526" / "Non Health - High Intent 260526") rather
# than inferred from product codes — more reliable since it's already
# business-defined. See segment_builder.py for the exact matching logic
# (handles the trailing campaign-date suffix and casing/spacing variants).
SEGMENT_COLUMN_RAW = "Segemnt"

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
