"""
segment_builder.py
--------------------
Derives which segment (health / non_health) each row currently belongs to,
based on the "PRODUCT_CODE" column: a row is "health" if its product code
is in config.HEALTH_PRODUCT_CODES, and "non_health" for every other code —
including codes not on the list at all, per the business rule that
Non-Health is "everything else".

Matching is done on a normalized string form of the code (stripped
whitespace, trailing ".0" removed) so it doesn't matter whether the code
arrives as an int (2824), a string ("2824"), or a float-like string
("2824.0", which pandas produces when a numeric column has any NaNs and
gets read as float64).
"""

import numpy as np
import pandas as pd
import config


def _normalize_code(value) -> str:
    """Normalizes a single product code value to a plain string for comparison."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def derive_segment(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if config.PRODUCT_CODE_COLUMN not in df.columns:
        candidates = [c for c in df.columns if "product" in c.lower() or "code" in c.lower()]
        raise KeyError(
            f"Expected product code column '{config.PRODUCT_CODE_COLUMN}' not found. "
            f"Closest matches in this file: {candidates or 'none found'}. "
            f"Update config.PRODUCT_CODE_COLUMN to match the real column name."
        )

    normalized = df[config.PRODUCT_CODE_COLUMN].apply(_normalize_code)
    health_codes = set(config.HEALTH_PRODUCT_CODES)

    is_health = normalized.isin(health_codes)
    df["segment"] = np.where(is_health, "health", "non_health")

    missing_count = (normalized == "").sum()
    if missing_count:
        print(f"[segment_builder] WARNING: {missing_count} rows had a missing/blank "
              f"'{config.PRODUCT_CODE_COLUMN}' value. These default to 'non_health' under "
              f"the current 'everything else is non_health' rule — review if this count "
              f"is large, since a missing code isn't the same as a confirmed non-health code.")

    return df


def split_by_segment(df: pd.DataFrame) -> dict:
    """Returns {'health': df_health, 'non_health': df_non_health}."""
    df = derive_segment(df)
    return {seg: df[df["segment"] == seg].copy() for seg in config.SEGMENTS}


def report_segment_positive_counts(df: pd.DataFrame):
    """Quick sanity check to run right after loading — flags a thin segment early."""
    df = derive_segment(df)
    summary = df.groupby("segment")[config.TARGET_COLUMN].agg(["count", "sum", "mean"])
    summary.columns = ["total_rows", "positive_count", "positive_rate"]
    print("[segment_builder] Segment / positive-count summary:")
    print(summary)
    for seg in config.SEGMENTS:
        if seg not in summary.index:
            continue
        if summary.loc[seg, "positive_count"] < 200:
            print(f"  WARNING: segment '{seg}' has only {int(summary.loc[seg, 'positive_count'])} "
                  f"positives — favor a simpler/more regularized model and treat CV metrics with "
                  f"extra caution.")
    return summary
