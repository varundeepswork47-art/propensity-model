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

Yes/No cutoffs are driven by TOP-K% CAPACITY, computed live off the
probability distribution of the uploaded file — NOT a fixed number saved
at training time. Health defaults to top 3%, Non-Health to top 1%
(config.DASHBOARD_TOP_K_DEFAULT), with a toggle to switch either segment
to 5/10/15/20% capacity and see the cutoff probability score update
immediately.

Scoring only runs when the user clicks "Run model" — uploading a file by
itself does not trigger the (potentially expensive) prediction pass.

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
# Yes/No cutoff — TOP-K% CAPACITY, per segment.
#
# Each segment has its own default (Health = top 3%, Non-Health = top 1%),
# plus a toggle to widen the net to 5/10/15/20% for capacity what-ifs. This
# is NOT a fixed number carried over from training — it's recomputed live,
# below, off the probability distribution of whatever file gets uploaded
# and scored, so it always reflects "top K% of THIS batch". The actual
# cutoff probability score is shown once a file has been scored.
# ---------------------------------------------------------------------------
st.sidebar.markdown("### Yes/No cutoff — Top-K% capacity")

top_k_selection = {}
for segment in config.SEGMENTS:
    default_k = config.DASHBOARD_TOP_K_DEFAULT[segment]
    options = [default_k] + [o for o in config.DASHBOARD_TOP_K_TOGGLE_OPTIONS if o != default_k]
    top_k_selection[segment] = st.sidebar.radio(
        f"**{segment}** — flag top what % as 'Yes'?",
        options=options,
        index=0,
        format_func=lambda x, seg=segment: f"Top {x}%" + ("  (default)" if x == config.DASHBOARD_TOP_K_DEFAULT[seg] else ""),
        key=f"topk_{segment}",
        horizontal=True,
        help=f"Defaults to top {default_k}% for '{segment}'. Switch to see the cutoff probability "
             f"score and Yes/No split at 5/10/15/20% capacity instead. The underlying model scores "
             f"don't change — only where the Yes/No line is drawn and how many leads cross it.",
    )

# Filled in further down, once a file has been scored — shows the live
# cutoff probability score for the currently selected Top-K% on THIS file.
cutoff_score_placeholders = {segment: st.sidebar.empty() for segment in config.SEGMENTS}

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
# total with common features.
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
    st.session_state.pop("scored", None)
    st.session_state.pop("file_signature", None)
    st.stop()

# A brand-new file invalidates any previously scored results — forces a
# fresh "Run model" click rather than silently showing stale predictions
# from the last upload while a new one sits unscored underneath it.
file_signature = (uploaded_file.name, uploaded_file.size)
if st.session_state.get("file_signature") != file_signature:
    st.session_state.pop("scored", None)
    st.session_state["file_signature"] = file_signature

run_clicked = st.button("▶ Run model", type="primary")

if run_clicked:
    with st.spinner("Scoring leads..."):
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

        # Header check — logged to the backend console only, not shown in the
        # UI, so a missing/renamed column is visible to whoever's monitoring
        # the app's logs without surfacing noisy warnings on every upload.
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

        scored = {"segment_counts": raw_df["segment"].value_counts().to_dict()}
        for segment in config.SEGMENTS:
            segment_df = raw_df[raw_df["segment"] == segment]
            if segment_df.empty:
                continue

            model, feature_columns, _, category_maps = models[segment]

            X, _, _, policy_status, _ = feature_engineering.build_feature_matrix(
                segment_df, segment, category_maps=category_maps
            )
            X = feature_engineering.align_to_model_columns(X, feature_columns, category_maps)
            # Missing NUMERIC columns filled with 0. Missing CATEGORICAL
            # columns filled as missing/NaN using the trained category set —
            # see feature_engineering.align_to_model_columns.

            base_proba = model.predict_proba(X)[:, 1]

            scored[segment] = {
                "segment_df": segment_df,
                "X": X,
                "base_proba": base_proba,
                "policy_status": policy_status,
            }

        st.session_state["scored"] = scored

if "scored" not in st.session_state:
    st.info("File uploaded. Click **▶ Run model** above to score these leads.")
    st.stop()

scored = st.session_state["scored"]
st.write(f"Detected segments: {scored['segment_counts']}")


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


def compute_topk_threshold(proba: np.ndarray, top_k_percent: float) -> float:
    """
    Probability cutoff such that roughly top_k_percent of THIS scored batch
    lands 'Yes'. Mirrors train_model.select_threshold_for_top_k, but run
    live against whatever file was just uploaded — since the right cutoff
    score for "top K%" depends on that file's own score distribution, not
    the training holdout's.
    """
    if len(proba) == 0:
        return 0.0
    cutoff_index = int(len(proba) * (top_k_percent / 100.0))
    sorted_proba = np.sort(proba)[::-1]
    idx = min(cutoff_index, len(sorted_proba) - 1)
    return float(sorted_proba[idx])


# ---------------------------------------------------------------------------
# Apply weights + Inactive-policy confidence discount, then the live
# Top-K% cutoff, per segment. Cheap enough to redo on every rerun (e.g. the
# user flips a Top-K% toggle or a weight slider) using the cached
# base_proba/X from the last "Run model" click — no need to re-score.
# ---------------------------------------------------------------------------
results = []
for segment in config.SEGMENTS:
    if segment not in scored:
        continue

    segment_df = scored[segment]["segment_df"]
    X = scored[segment]["X"]
    base_proba = scored[segment]["base_proba"]
    policy_status = scored[segment]["policy_status"]
    model, _, _, _ = models[segment]

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

    top_k = top_k_selection[segment]
    active_threshold = compute_topk_threshold(proba, top_k)
    n_flagged = int((proba >= active_threshold).sum())

    cutoff_score_placeholders[segment].metric(
        f"{segment} cutoff @ top {top_k}%",
        f"{active_threshold:.4f}",
    )
    st.caption(
        f"**{segment}**: top {top_k}% cutoff on this file = **{active_threshold:.4f}** "
        f"→ {n_flagged} of {len(proba)} leads flagged 'Yes'."
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
