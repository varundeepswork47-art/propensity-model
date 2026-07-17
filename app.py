"""
app.py
------
Streamlit dashboard — two-segment version (health / non_health).

For an uploaded lead list, each row's CURRENT segment is auto-detected
(via segment_builder), then routed to the matching trained model:
  - a "health" row -> scored by the health_model (predicts cross-sell
    INTO non-health)
  - a "non_health" row -> scored by the non_health_model (predicts
    cross-sell INTO health)

Manual weight-adjustment sliders let a user scenario-test feature
influence on top of the trained model, via SHAP contribution reweighting.

Run with: streamlit run app.py
"""

import logging

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

import config
import data_loader
import segment_builder
import feature_engineering

logger = logging.getLogger("propensity_app")
logging.basicConfig(level=logging.INFO)

DEFAULT_CUTOFF = 0.55  # default Yes/No probability cutoff, editable in the UI below

st.set_page_config(page_title="Cross-Sell Propensity", layout="wide")
st.title("Propensity to Cross-Sell Model")


@st.cache_resource
def load_model(segment: str):
    model_path = config.MODEL_DIR / f"{segment}_model.joblib"
    if not model_path.exists():
        return None
    bundle = joblib.load(model_path)
    return bundle["model"], bundle["feature_columns"], bundle["threshold"], bundle["category_maps"]


models = {seg: load_model(seg) for seg in config.SEGMENTS}
missing = [seg for seg, m in models.items() if m is None]
if missing:
    st.warning(f"No trained model found for: {missing}.")
    st.markdown("**Diagnostic info** — check this against your repo:")
    st.code(f"Looking in: {config.MODEL_DIR.resolve()}")
    if config.MODEL_DIR.exists():
        found_files = [f.name for f in config.MODEL_DIR.iterdir()]
        st.code(f"Files actually found in that folder: {found_files if found_files else '(empty)'}")
        expected = [f"{seg}_model.joblib" for seg in config.SEGMENTS]
        st.code(f"Expected filenames: {expected}")
    else:
        st.code("This folder does not exist at all in the deployed app. "
                 "Confirm 'models/' was committed and pushed to the repo, "
                 "and check for case-sensitivity (Linux is case-sensitive; "
                 "'Models' != 'models').")
    st.stop()

# ---------------------------------------------------------------------------
# Yes/No cutoff — fixed probability threshold, at the top of the sidebar.
# Applies the same way to both segments: a lead is "Yes" if its
# cross_sell_probability >= this value. This does NOT change the model or
# its scores in any way — it only decides where the Yes/No line is drawn
# on top of the probability the model already produced.
# ---------------------------------------------------------------------------
st.sidebar.markdown("### Yes/No probability cutoff")
cutoff = st.sidebar.number_input(
    "Cutoff",
    min_value=0.0, max_value=1.0, value=DEFAULT_CUTOFF, step=0.01, format="%.2f",
    help="A lead is marked 'Yes' if its cross_sell_probability is at or above this value. "
         "Doesn't retrain or change the model — only how scores get labeled.",
    label_visibility="collapsed",
)

st.sidebar.divider()

# ---------------------------------------------------------------------------
# Manual weight adjustment sliders (applied per segment's top features)
# ---------------------------------------------------------------------------
st.sidebar.markdown("### Adjust feature influence")
st.sidebar.caption("1.0 = model's learned weight, as-is. Move away from 1.0 to test scenarios.")

selected_segment_for_weights = st.sidebar.radio("Tune weights for", config.SEGMENTS)
_, feature_columns_for_weights, _, _ = models[selected_segment_for_weights]

# Segment-specific features are guaranteed to show first (these are what
# actually differ between Health and Non-Health), then filled up to 10
# total with common features. A plain [:10] slice would silently miss
# segment-specific features entirely, since COMMON_FEATURES (15 fields)
# is longer than 10 and always appears first in the column order.
segment_specific_present = [
    f for f in feature_columns_for_weights if f in config.SEGMENT_FEATURES[selected_segment_for_weights]
]
common_present = [f for f in feature_columns_for_weights if f not in segment_specific_present]
adjustable_features = segment_specific_present + common_present[: max(0, 10 - len(segment_specific_present))]

feature_weights = {
    feat: st.sidebar.slider(feat, 0.0, 2.0, 1.0, 0.1, key=f"{selected_segment_for_weights}_{feat}")
    for feat in adjustable_features
}

# ---------------------------------------------------------------------------
# Lead input
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload lead list (CSV or Excel)", type=["csv", "xlsx", "xls"])
if uploaded_file is None:
    st.stop()

raw_df = data_loader.read_any(uploaded_file, uploaded_file.name)
raw_df = data_loader.convert_excel_dates(raw_df)

try:
    raw_df = segment_builder.derive_segment(raw_df)
except KeyError as e:
    st.error(
        f"Couldn't detect segments in this file: {e}\n\n"
        f"This usually means the uploaded sheet's header for the product "
        f"code column doesn't exactly match what the app expects "
        f"(`{config.PRODUCT_CODE_COLUMN}`) — check for renamed, retyped, "
        f"or differently-cased column headers and re-upload."
    )
    st.stop()

st.write(f"Detected segments: {raw_df['segment'].value_counts().to_dict()}")

# ---------------------------------------------------------------------------
# Header check — logged to the backend console only (not shown in the UI)
# so a missing/renamed column is visible to whoever's monitoring the app's
# logs without surfacing noisy warnings to end users on every upload.
# ---------------------------------------------------------------------------
for segment in config.SEGMENTS:
    segment_df = raw_df[raw_df["segment"] == segment]
    if segment_df.empty:
        continue
    missing_cols = feature_engineering.missing_columns_report(segment_df, segment)
    if missing_cols:
        logger.warning(
            f"Segment '{segment}': uploaded file is missing expected columns "
            f"{missing_cols}. Treated as missing/empty for scoring (0 for "
            f"numeric fields, blank/unknown for categorical fields)."
        )


def apply_manual_weights(model, X: pd.DataFrame, weights: dict) -> np.ndarray:
    """Scenario-testing tool: reweights SHAP contributions, not a retrained model."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap_df = pd.DataFrame(shap_values, columns=X.columns, index=X.index)
    for feat, w in weights.items():
        if feat in shap_df.columns:
            shap_df[feat] = shap_df[feat] * w
    adjusted_log_odds = explainer.expected_value + shap_df.sum(axis=1)
    return 1 / (1 + np.exp(-adjusted_log_odds))


# ---------------------------------------------------------------------------
# Score each segment separately, then recombine
# ---------------------------------------------------------------------------
results = []
for segment in config.SEGMENTS:
    segment_df = raw_df[raw_df["segment"] == segment]
    if segment_df.empty:
        continue

    model, feature_columns, threshold, category_maps = models[segment]

    X, _, _, policy_status, _ = feature_engineering.build_feature_matrix(
        segment_df, segment, category_maps=category_maps
    )
    X = feature_engineering.align_to_model_columns(X, feature_columns, category_maps)
    # Missing NUMERIC columns are filled with 0. Missing CATEGORICAL columns
    # (entirely absent from this upload) are filled as missing/NaN using the
    # trained category set, so XGBoost treats them as "unknown" rather than
    # crashing on a dtype mismatch — see feature_engineering.align_to_model_columns.

    base_proba = model.predict_proba(X)[:, 1]

    weights_to_apply = feature_weights if segment == selected_segment_for_weights else None
    if weights_to_apply:
        try:
            proba = apply_manual_weights(model, X, weights_to_apply)
        except Exception as e:
            st.warning(f"Manual weight adjustment unavailable for '{segment}' ({e}). Using base model output.")
            proba = base_proba
    else:
        proba = base_proba

    # Confidence discount for Inactive policies — the model was trained
    # only on Active examples (see config.py note), so treat Inactive
    # predictions as extrapolation, not a fully learned pattern.
    if policy_status is not None:
        confidence_multiplier = policy_status.map(
            config.POLICY_STATUS_SCORING_CONFIDENCE_MULTIPLIER
        ).fillna(1.0).values
        proba = proba * confidence_multiplier

    # Pick the cutoff: the fixed probability cutoff set at the top of the
    # dashboard (default 0.55), applied the same way to every segment.
    active_threshold = cutoff
    st.caption(
        f"**{segment}**: Yes/No cutoff = {active_threshold:.2f} (fixed) "
        f"— model's originally trained top-{config.TOP_K_PERCENT_CAPACITY*100:.0f}%-capacity "
        f"threshold was {threshold:.4f}, shown here for reference only."
    )

    segment_result = segment_df.copy()
    segment_result["cross_sell_probability"] = proba
    segment_result["cross_sell_prediction"] = np.where(proba >= active_threshold, "Yes", "No")
    results.append(segment_result)

final_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()

st.subheader("Results")
st.dataframe(final_df)

if not final_df.empty:
    st.download_button(
        "Download results as CSV",
        final_df.to_csv(index=False).encode("utf-8"),
        file_name="cross_sell_predictions.csv",
    )
