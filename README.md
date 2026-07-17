
## File map

| File | Purpose |
|---|---|
| `config.py` | Paths, feature lists, segment logic, rare-event settings, PolicyStatus weighting — edit here first |
| `data_loader.py` | Loads the CSV, converts Excel-serial dates, strips PII, renames the label column |
| `segment_builder.py` | Reads the `Segemnt` column, classifies each row as `health` / `non_health` (handles casing/spacing/date-suffix variation), reports positive counts per segment |
| `feature_engineering.py` | Derived fields (tenure, presence flags), feature selection, missing-value handling, native categorical encoding |
| `train_model.py` | Stratified 5-fold CV, XGBoost with `scale_pos_weight` + early stopping, top-K threshold selection, saves model per segment |
| `app.py` | Streamlit dashboard — auto-routes leads to the right segment model, manual weight-adjustment sliders, Inactive-policy confidence discount |

## Handling the rare-event imbalance (~1.7% positive, thinner in Non-Health)

- **Stratified 5-fold CV**, not a single random split — keeps the ~1.7%
  positive rate consistent across folds so evaluation isn't skewed by an
  unlucky split.
- **`scale_pos_weight`** computed per fold from the actual imbalance
  ratio — tells XGBoost to penalize missing a rare positive much more
  than misclassifying a negative.
- **Shallow trees + early stopping** (`max_depth=4`, stops once
  validation PR-AUC stops improving) — with few hundred positives,
  a deep/long-trained model will memorize noise.
- **Threshold picked by business capacity** (top 10% of leads by
  default — change `TOP_K_PERCENT_CAPACITY`), not the default 0.5,
  which would almost never fire "Yes" at this base rate.
- **PR-AUC as the headline metric**, not accuracy — a model predicting
  "No" for everyone would still be ~98% accurate while useless.

## Handling Inactive policies

The training data is 100% `PolicyStatus == ACTIVE`, but the full 1.4
crore scoring base will include Inactive customers too. Since the model
has never seen a labeled Inactive example, it can't have genuinely
learned how that group behaves. Two things are wired in to handle this
responsibly:

1. `PolicyStatus` is included as a feature/sample-weight column, so the
   pipeline is ready to use it properly the moment Inactive-labeled
   training rows exist.
2. Until then, `app.py` applies a **conservative confidence discount**
   (`POLICY_STATUS_SCORING_CONFIDENCE_MULTIPLIER` in config.py, default
   0.7) to predictions on Inactive leads at scoring time — this flags
   that those scores are extrapolated, not fully learned, rather than
   presenting them with the same confidence as Active predictions.
   Revisit this multiplier (or remove it) once you have real Inactive
   outcome data to train on.

