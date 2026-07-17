"""
data_loader.py
----------------
Loads the training extract, converts Excel-serial dates to real dates,
drops PII/metadata columns, and renames the target column.

NOTE ON SCALE: pandas.read_csv is fine for the 90k training sample. For
the full 1.4 crore scoring file, switch to chunked reads
(pd.read_csv(..., chunksize=500_000)) or Parquet — see the note at the
bottom of this file.
"""

import pandas as pd
import config


def convert_excel_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles THREE possible incoming shapes for each date column, without
    erroring on any of them:
      1. Already a real datetime (common when reading .xlsx directly —
         openpyxl/pandas often parses date-formatted cells automatically)
         -> left as-is.
      2. A numeric Excel serial number (e.g. 45160 — common when the same
         data has been round-tripped through a CSV export) -> converted
         using the 1899-12-30 origin.
      3. A text/string date (e.g. "12-02-2024", "2024-02-12") -> parsed
         directly. If most values in a string column fail to parse as a
         normal date, it's assumed to actually be a serial number stored
         as text and re-attempted that way instead.
    Unparseable individual values become NaT (missing) rather than raising.
    """
    df = df.copy()
    for col in config.EXCEL_SERIAL_DATE_COLUMNS:
        if col not in df.columns:
            continue

        series = df[col]

        if pd.api.types.is_datetime64_any_dtype(series):
            continue  # case 1 — already a real date, nothing to do

        if pd.api.types.is_numeric_dtype(series):
            df[col] = pd.to_datetime(series, origin="1899-12-30", unit="D", errors="coerce")
            continue  # case 2 — numeric serial number

        # case 3 — text/string column: try parsing as a normal date first
        parsed = pd.to_datetime(series, errors="coerce", dayfirst=True, format="mixed")
        if parsed.notna().mean() < 0.5:
            # Most values didn't parse as a normal date — likely a serial
            # number stored as text (e.g. "45160"). Retry that way.
            numeric_attempt = pd.to_numeric(series, errors="coerce")
            if numeric_attempt.notna().mean() > 0.5:
                parsed = pd.to_datetime(numeric_attempt, origin="1899-12-30", unit="D", errors="coerce")
        df[col] = parsed

    return df


def read_any(file_obj, filename: str) -> pd.DataFrame:
    """
    Reads either a CSV or an Excel file from a path OR a file-like object
    (e.g. Streamlit's file_uploader result), based on the filename
    extension. Shared by load_training_data() and app.py's upload widget
    so both go through identical logic.
    """
    filename_lower = filename.lower()
    if filename_lower.endswith((".xlsx", ".xls")):
        return pd.read_excel(file_obj, engine="openpyxl")
    return pd.read_csv(file_obj)


def load_training_data(path=config.TRAINING_FILE) -> pd.DataFrame:
    df = read_any(path, str(path))

    df = convert_excel_dates(df)

    # Drop PII and known-irrelevant columns if present (guarded, since the
    # 100%-missing PII columns in the sample WILL be populated in the real
    # user-uploaded file — they must never reach the feature matrix).
    cols_to_drop = [c for c in (config.PII_COLUMNS + config.DROP_COLUMNS) if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    df = df.rename(columns={config.TARGET_COLUMN_RAW: config.TARGET_COLUMN})

    return df


# ---------------------------------------------------------------------------
# SCALE NOTE (1.4 crore rows, scoring time)
# ---------------------------------------------------------------------------
# Training only ever touches the 90k labeled sample — that fits comfortably
# in memory as-is. The full 1.4 crore file is only relevant at BATCH
# SCORING time (applying the trained model to the whole base), and that
# should be done in chunks:
#
#   for chunk in pd.read_csv(full_file_path, chunksize=500_000):
#       chunk = convert_excel_dates(chunk)
#       ... engineer features, predict, write results incrementally ...
#
# Do not attempt to load all 1.4 crore rows into one DataFrame at once.
