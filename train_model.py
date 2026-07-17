"""
train_model.py
----------------
Trains one model per segment ("health", "non_health") on a rare-event
target. Health and Non-Health have very different positive counts in the
real data — rather than assuming upfront how much model complexity each
segment can support, BOTH segments are run through the identical
candidate-hyperparameter search below, and whichever candidate wins for
that segment's own cross-validated data is used. This keeps the process
fair: the data decides each segment's settings, not a prior assumption
about which segment is "weaker."

Approach:
  1. For each candidate hyperparameter set in config.CANDIDATE_XGB_OVERRIDES,
     run cross-validation and record mean/std PR-AUC.
  2. Pick the candidate with the best (mean - std) — rewards both
     performance and stability, not just the highest average.
  3. Re-run full repeated CV with the winning candidate for an honest,
     final reported estimate.
  4. Retrain on all of the segment's data with the winning candidate and
     pick a Yes/No threshold based on business capacity.

Usage:
    python train_model.py --segment health
    python train_model.py --segment non_health
"""

import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import average_precision_score, classification_report
from xgboost import XGBClassifier

import config
import data_loader
import segment_builder
import feature_engineering


def compute_scale_pos_weight(y: pd.Series) -> float:
    pos = y.sum()
    neg = len(y) - pos
    return neg / max(pos, 1)


def run_cross_validation(X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series,
                          xgb_params: dict, n_repeats: int, label: str = "", verbose: bool = True):
    """Runs repeated stratified CV with a given hyperparameter set. Returns array of PR-AUC scores."""
    rskf = RepeatedStratifiedKFold(n_splits=config.N_CV_FOLDS, n_repeats=n_repeats,
                                    random_state=config.RANDOM_STATE)
    pr_auc_scores = []

    for train_idx, val_idx in rskf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        w_train = sample_weight.iloc[train_idx] if sample_weight is not None else None

        model = XGBClassifier(
            **xgb_params,
            scale_pos_weight=compute_scale_pos_weight(y_train),
            early_stopping_rounds=config.EARLY_STOPPING_ROUNDS,
        )
        model.fit(X_train, y_train, sample_weight=w_train, eval_set=[(X_val, y_val)], verbose=False)

        y_val_proba = model.predict_proba(X_val)[:, 1]
        pr_auc_scores.append(average_precision_score(y_val, y_val_proba))

    scores = np.array(pr_auc_scores)
    if verbose:
        print(f"  {label:<18} mean PR-AUC = {scores.mean():.4f}  std = {scores.std():.4f}  "
              f"range = [{scores.min():.4f}, {scores.max():.4f}]")
    return scores


def select_best_params(X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series, segment: str) -> dict:
    """
    Runs every candidate in config.CANDIDATE_XGB_OVERRIDES through the SAME
    search process and picks the winner by (mean - std) — this rewards a
    candidate for being both accurate AND stable, and penalizes a candidate
    that only looks good because of a lucky split. Both segments go through
    this identical process; nothing is assumed in advance.
    """
    print(f"\n[search] Segment '{segment}' — comparing {len(config.CANDIDATE_XGB_OVERRIDES)} "
          f"candidate configurations ({config.N_CV_FOLDS}-fold x {config.N_CV_REPEATS_FOR_SEARCH} "
          f"repeat{'s' if config.N_CV_REPEATS_FOR_SEARCH > 1 else ''} each)...")

    results = []
    for candidate in config.CANDIDATE_XGB_OVERRIDES:
        params = dict(config.XGB_PARAMS)
        params.update(candidate["params"])
        scores = run_cross_validation(X, y, sample_weight, params,
                                       n_repeats=config.N_CV_REPEATS_FOR_SEARCH,
                                       label=candidate["label"])
        results.append({
            "label": candidate["label"],
            "params": candidate["params"],
            "mean": scores.mean(),
            "std": scores.std(),
            "score": scores.mean() - scores.std(),  # conservative selection criterion
        })

    winner = max(results, key=lambda r: r["score"])
    print(f"  -> Selected: '{winner['label']}' (mean - std = {winner['score']:.4f})")

    final_params = dict(config.XGB_PARAMS)
    final_params.update(winner["params"])
    return final_params, winner


def select_threshold_for_top_k(y_true, y_proba, top_k_percent: float) -> float:
    """
    Picks the probability cutoff such that roughly top_k_percent of scored
    leads are flagged Yes — matches the threshold to sales capacity rather
    than defaulting to 0.5, which will rarely fire on a low base rate.
    """
    cutoff_index = int(len(y_proba) * top_k_percent)
    sorted_proba = np.sort(y_proba)[::-1]
    threshold = sorted_proba[min(cutoff_index, len(sorted_proba) - 1)]
    return float(threshold)


def evaluate_at_threshold(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    print(classification_report(y_true, y_pred, zero_division=0))


def train_final_model(X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series,
                       xgb_params: dict, segment: str):
    """
    Retrain on ALL of this segment's data (using the winning hyperparameters
    from select_best_params) with a small internal holdout only to drive
    early stopping — the headline performance estimate is the repeated-CV
    result reported just before this, not this single holdout.
    """
    skf = StratifiedKFold(n_splits=config.N_CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    train_idx, holdout_idx = next(skf.split(X, y))

    X_train, X_holdout = X.iloc[train_idx], X.iloc[holdout_idx]
    y_train, y_holdout = y.iloc[train_idx], y.iloc[holdout_idx]
    w_train = sample_weight.iloc[train_idx] if sample_weight is not None else None

    model = XGBClassifier(
        **xgb_params,
        scale_pos_weight=compute_scale_pos_weight(y_train),
        early_stopping_rounds=config.EARLY_STOPPING_ROUNDS,
    )
    model.fit(X_train, y_train, sample_weight=w_train, eval_set=[(X_holdout, y_holdout)], verbose=False)

    holdout_proba = model.predict_proba(X_holdout)[:, 1]
    threshold = select_threshold_for_top_k(y_holdout, holdout_proba, config.TOP_K_PERCENT_CAPACITY)

    print(f"\n[final] Segment '{segment}' — threshold for top "
          f"{config.TOP_K_PERCENT_CAPACITY*100:.0f}% capacity: {threshold:.4f}")
    evaluate_at_threshold(y_holdout, holdout_proba, threshold)

    return model, threshold


def train_for_segment(segment: str):
    print(f"\n{'='*70}\nTraining segment: {segment}\n{'='*70}")

    raw_df = data_loader.load_training_data()
    segment_builder.report_segment_positive_counts(raw_df)

    segment_df = segment_builder.split_by_segment(raw_df)[segment]
    X, y, sample_weight, _, category_maps = feature_engineering.build_feature_matrix(segment_df, segment)

    if y.sum() < 200:
        print(f"[train] NOTE: segment '{segment}' has under 200 positives — "
              f"results will be noisier; the hyperparameter search below will "
              f"favor a simpler configuration for THIS segment only if the "
              f"data itself shows that's more stable, not because of an assumption.")

    best_params, winner_info = select_best_params(X, y, sample_weight, segment)

    print(f"\n[cv-final] Segment '{segment}' — full repeated CV with the selected "
          f"'{winner_info['label']}' configuration "
          f"({config.N_CV_FOLDS}-fold x {config.N_CV_REPEATS} repeats):")
    final_scores = run_cross_validation(X, y, sample_weight, best_params,
                                         n_repeats=config.N_CV_REPEATS, label=winner_info["label"])
    relative_spread = final_scores.std() / max(final_scores.mean(), 1e-6)
    if relative_spread > 0.4:
        print(f"  CAUTION: std is {relative_spread*100:.0f}% of the mean even with the best "
              f"available configuration — this segment's estimate is still unstable. Treat as "
              f"directional, not production-ready, until more labeled positives are available.")

    model, threshold = train_final_model(X, y, sample_weight, best_params, segment)

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = config.MODEL_DIR / f"{segment}_model.joblib"
    joblib.dump(
        {"model": model, "feature_columns": list(X.columns), "threshold": threshold,
         "category_maps": category_maps, "selected_params_label": winner_info["label"]},
        model_path,
    )
    print(f"[train] Saved model to {model_path}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--segment", choices=config.SEGMENTS, required=True)
    args = parser.parse_args()
    train_for_segment(args.segment)
