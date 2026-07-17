"""
feature_engineering.py
------------------------
Turns a raw (dated, PII-stripped) dataframe into a model-ready feature
matrix for a given segment ("health" or "non_health").

Kept as a pure function (df -> df) reused identically at training and
scoring time, so there's no train/serve skew.
"""

import pandas as pd
import config

SNAPSHOT_DATE = pd.Timestamp.today()  # overridden by caller if a specific "as of" date is needed


def add_derived_fields(df: pd.DataFrame, snapshot_date: pd.Timestamp = SNAPSHOT_DATE) -> pd.DataFrame:
    df = df.copy()

    if "RelationShip_start_Date" in df.columns:
        df["tenure_days"] = (snapshot_date - df["RelationShip_start_Date"]).dt.days

    if "POLICY_END_Date" in df.columns:
        df["days_to_policy_expiry"] = (df["POLICY_END_Date"] - snapshot_date).dt.days

    # Presence flags for sparse fields — the flag itself carries signal
    # ("do we even have this data point?") independent of the value.
    if "Total_NO_Claim" in df.columns or "Total_Claim_Amount" in df.columns:
        df["claim_history_present"] = (
            df.get("Total_NO_Claim").notna() | df.get("Total_Claim_Amount").notna()
        ).astype(int)

    if any(c in df.columns for c in ["MAKE", "Vehicle_Age", "Fuel_Type", "RTO"]):
        vehicle_cols = [c for c in ["MAKE", "Vehicle_Age", "Fuel_Type", "RTO"] if c in df.columns]
        df["vehicle_data_present"] = df[vehicle_cols].notna().any(axis=1).astype(int)

    if "NameOfCountryVisiting" in df.columns:
        df["travel_intent_present"] = df["NameOfCountryVisiting"].notna().astype(int)

    return df


def select_features(df: pd.DataFrame, segment: str) -> pd.DataFrame:
    feature_cols = config.COMMON_FEATURES + config.SEGMENT_FEATURES[segment]
    available = [c for c in feature_cols if c in df.columns]
    missing = set(feature_cols) - set(available)
    if missing:
        print(f"[feature_engineering] Columns not found for segment '{segment}', skipped: {missing}")

    keep_cols = available.copy()
    if config.TARGET_COLUMN in df.columns:
        keep_cols.append(config.TARGET_COLUMN)
    if "PolicyStatus" in df.columns and "PolicyStatus" not in keep_cols:
        keep_cols.append("PolicyStatus")  # ensure available for sample-weighting even if excluded as a feature

    return df[keep_cols]


def missing_columns_report(df: pd.DataFrame, segment: str) -> list:
    """
    Returns the list of expected feature columns (COMMON_FEATURES +
    that segment's SEGMENT_FEATURES) that are absent from df — without
    printing anything. Used by app.py to show an upfront, human-readable
    warning about header mismatches before scoring runs, instead of
    letting a missing/renamed column surface later as a cryptic error.
    """
    feature_cols = config.COMMON_FEATURES + config.SEGMENT_FEATURES[segment]
    return [c for c in feature_cols if c not in df.columns]


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    numeric_cols = df.select_dtypes(include="number").columns
    categorical_cols = df.select_dtypes(include="object").columns

    df[numeric_cols] = df[numeric_cols].fillna(0)
    df[categorical_cols] = df[categorical_cols].fillna("Unknown")
    return df


def encode_categoricals(df: pd.DataFrame, category_maps: dict = None):
    """
    Converts object columns to pandas 'category' dtype instead of one-hot
    encoding. With high-cardinality fields (STATE, RTO, MAKE, SubChannel)
    and only a few hundred positives per segment, one-hot encoding blows
    up column count (e.g. 14 features -> 80+ dummy columns on a 70-row
    segment) and invites overfitting. XGBoost's native categorical
    support (enable_categorical=True in train_model.py) handles these
    directly via histogram splits instead.

    category_maps: if provided (at scoring time), aligns each column's
    categories to what the model was TRAINED on — any new/unseen category
    in fresh data becomes NaN, which XGBoost handles natively as missing,
    rather than silently creating a column mismatch.

    Returns (df, category_maps) — category_maps must be saved alongside
    the model and passed back in at scoring time.
    """
    df = df.copy()
    cat_cols = df.select_dtypes(include="object").columns
    maps = {} if category_maps is None else category_maps

    for col in cat_cols:
        if category_maps is not None and col in category_maps:
            df[col] = pd.Categorical(df[col], categories=category_maps[col])
        else:
            df[col] = df[col].astype("category")
            maps[col] = list(df[col].cat.categories)

    return df, maps


def build_feature_matrix(raw_df: pd.DataFrame, segment: str, snapshot_date: pd.Timestamp = SNAPSHOT_DATE,
                          category_maps: dict = None):
    """
    Full pipeline. Returns (X, y, sample_weight, policy_status_series, category_maps).

    Pass category_maps=None at TRAINING time (it will be built from the
    data and returned — save it alongside the model). Pass the SAVED
    category_maps back in at SCORING time so new data is aligned to the
    categories the model actually learned.
    """
    df = add_derived_fields(raw_df, snapshot_date)
    df = select_features(df, segment)

    policy_status = df["PolicyStatus"].copy() if "PolicyStatus" in df.columns else None

    df = handle_missing_values(df)
    df, category_maps = encode_categoricals(df, category_maps)

    y = df[config.TARGET_COLUMN] if config.TARGET_COLUMN in df.columns else None
    X = df.drop(columns=[config.TARGET_COLUMN]) if config.TARGET_COLUMN in df.columns else df

    sample_weight = None
    if policy_status is not None:
        sample_weight = policy_status.map(config.POLICY_STATUS_TRAIN_WEIGHT).fillna(1.0)

    return X, y, sample_weight, policy_status, category_maps


def align_to_model_columns(X: pd.DataFrame, feature_columns: list, category_maps: dict) -> pd.DataFrame:
    """
    Aligns a scoring-time feature matrix to the EXACT columns/order the
    model was trained on — safely handling a column that's entirely
    absent from the uploaded file (e.g. a renamed or dropped header):

      - Missing NUMERIC column     -> filled with 0 (neutral/absent value).
      - Missing CATEGORICAL column -> filled with NaN using the column's
        trained category set (via pd.Categorical(..., categories=...)),
        so XGBoost treats every row as "missing" for that feature — its
        native handling for unknown/absent categorical data.

    Previously, a naive `X.reindex(columns=feature_columns, fill_value=0)`
    filled missing categorical columns with a raw integer 0, which doesn't
    match the categorical dtype the model was trained on and crashes
    XGBoost with a dtype-mismatch error. This avoids that by fixing the
    dtype per-column instead of applying one fill value to everything.

    Also drops any column not in feature_columns (e.g. leftover PII or
    metadata that slipped through), and returns columns in the exact
    trained order so the model sees a consistent feature layout.
    """
    X = X.copy()
    for col in feature_columns:
        if col in X.columns:
            continue
        if col in category_maps:
            X[col] = pd.Categorical([None] * len(X), categories=category_maps[col])
        else:
            X[col] = 0
    return X[feature_columns]
