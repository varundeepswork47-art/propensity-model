"""
target_builder.py
--------------------
Builds the corrected cross-sell label:

  - non_health rows -> target = 1 if the customer converted into a HEALTH
    product. (Unchanged from the original design.)
  - health rows      -> target = 1 if the customer converted into a
    DIFFERENT HEALTH product than the one in PRODUCT_CODE. (Changed —
    previously this segment's label meant "converted into Non-Health.")

Replaces the old approach in data_loader.load_training_data(), which just
renamed the raw target column as-is and relied on that column already
encoding the right direction. It doesn't for the health segment under the
corrected rule, so this module builds the label explicitly per row instead.

REQUIRES config.PURCHASED_PRODUCT_CODE_COLUMN to be set to whatever column
in your raw file records the NEW/converted product's code (distinct from
PRODUCT_CODE, the EXISTING product). Raises loudly if it isn't set or isn't
present in the file — see the comment in config.py for why.
"""

import numpy as np
import pandas as pd
import config
import segment_builder


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expects df to already have a 'segment' column (i.e. called after
    segment_builder.derive_segment()). Adds/overwrites config.TARGET_COLUMN
    with the corrected label. Returns a copy.
    """
    df = df.copy()

    if "segment" not in df.columns:
        df = segment_builder.derive_segment(df)

    if config.PURCHASED_PRODUCT_CODE_COLUMN is None or \
            config.PURCHASED_PRODUCT_CODE_COLUMN not in df.columns:
        raise KeyError(
            "config.PURCHASED_PRODUCT_CODE_COLUMN is not set (or not found in "
            "this file). Building the corrected health->health label needs the "
            "column that records which product code a customer actually "
            "converted into. Set it in config.py — look for a column separate "
            "from PRODUCT_CODE that holds a NEW/purchased/converted product "
            "code in the raw extract, and tell me its name if you're not sure "
            "which one it is. I'm intentionally not guessing at a column name "
            "here: a wrong guess would train a real model on a meaningless "
            "label without any obvious failure signal in CV."
        )

    purchased_raw = df[config.PURCHASED_PRODUCT_CODE_COLUMN]
    purchased_norm = purchased_raw.apply(segment_builder._normalize_code)
    existing_norm = df[config.PRODUCT_CODE_COLUMN].apply(segment_builder._normalize_code)
    health_codes = set(config.HEALTH_PRODUCT_CODES)

    purchased_is_health = purchased_norm.isin(health_codes)
    purchased_is_different = purchased_norm != existing_norm
    converted_at_all = purchased_norm != ""  # blank purchased-code = no conversion this period

    is_health_row = df["segment"] == "health"
    is_non_health_row = df["segment"] == "non_health"

    # health row -> hit only if converted into a DIFFERENT health product
    health_target = is_health_row & converted_at_all & purchased_is_health & purchased_is_different

    # non_health row -> hit if converted into ANY health product (unchanged rule)
    non_health_target = is_non_health_row & converted_at_all & purchased_is_health

    df[config.TARGET_COLUMN] = np.where(health_target | non_health_target, 1, 0)

    # Sanity print — mirrors segment_builder.report_segment_positive_counts,
    # run this too after changing the rule so a near-zero positive count
    # (e.g. wrong column, or almost everyone "converts" into the exact same
    # product renewal) is obvious immediately rather than discovered after
    # a full training run.
    summary = df.groupby("segment")[config.TARGET_COLUMN].agg(["count", "sum", "mean"])
    print("[target_builder] Corrected-label positive counts per segment:")
    print(summary)

    return df
