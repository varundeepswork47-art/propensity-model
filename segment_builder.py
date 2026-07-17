"""
segment_builder.py
--------------------
Derives which segment (health / non_health) each row currently belongs to,
by reading the existing "Segemnt" business column directly (e.g. values
like "Health - High Intent 260526" / "Non Health - High Intent 260526")
rather than inferring it from product codes.

Matching is done on normalized text (lowercased, punctuation stripped) so
it's resilient to the trailing campaign-date suffix changing every month,
and to minor spacing/casing differences (e.g. "Non-Health", "NON HEALTH").
"Non Health" is checked BEFORE "Health" since "health" is a substring of
"non health".
"""

import re
import numpy as np
import pandas as pd
import config


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z]+", " ", str(text).lower()).strip()


def derive_segment(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if config.SEGMENT_COLUMN_RAW not in df.columns:
        candidates = [c for c in df.columns if "segment" in c.lower() or "segemnt" in c.lower()]
        raise KeyError(
            f"Expected segment column '{config.SEGMENT_COLUMN_RAW}' not found. "
            f"Closest matches in this file: {candidates or 'none found'}. "
            f"Update config.SEGMENT_COLUMN_RAW to match the real column name."
        )

    normalized = df[config.SEGMENT_COLUMN_RAW].apply(_normalize)

    is_non_health = normalized.str.contains(r"non\s*health", regex=True)
    is_health = normalized.str.contains(r"health", regex=True) & ~is_non_health

    df["segment"] = np.select([is_non_health, is_health], ["non_health", "health"], default="unknown")

    unknown_count = (df["segment"] == "unknown").sum()
    if unknown_count:
        sample_values = df.loc[df["segment"] == "unknown", config.SEGMENT_COLUMN_RAW].unique()[:5]
        print(f"[segment_builder] WARNING: {unknown_count} rows had an unrecognized segment value. "
              f"Sample unrecognized values: {list(sample_values)}. These rows will be excluded "
              f"from training/scoring — review and extend the matching logic if this count is large.")

    return df


def split_by_segment(df: pd.DataFrame) -> dict:
    """Returns {'health': df_health, 'non_health': df_non_health}. Drops 'unknown' rows."""
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
