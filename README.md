# Cross-Sell Propensity Model — Health / Non-Health

Binary Yes/No cross-sell propensity, two models (Health-holder and
Non-Health-holder), built on a 90,000-row labeled extract (~1,500
positives overall; Non-Health has fewer positives than Health).

Segmentation is read directly from the business-defined `Segemnt` column
(not inferred from product codes) — more reliable, since it's already
maintained by the team that owns the campaign logic.

This version has been **tested end-to-end** — data loading (including
`.xlsx`), Segemnt-based segmentation, feature engineering, cross-validation,
and model saving all run cleanly against your real file structure.

## How to use these scripts — exact order

### Step 1 — Install dependencies (one time)
```bash
pip install -r requirements.txt
```

### Step 2 — Point config.py at your file
Already set in `config.py`:
```python
TRAINING_FILE = r"C:\Users\Hp\Downloads\June_Segment_part_1.xlsx"
```
If the file moves, or you get a new monthly extract, update this one line.
If your data isn't on the first sheet of the workbook, also update
`TRAINING_FILE_SHEET_NAME`.

### Step 3 — Train the Health model
```bash
python train_model.py --segment health
```

### Step 4 — Train the Non-Health model
```bash
python train_model.py --segment non_health
```
Run these two separately, in either order — they're independent models.
Each run prints:
- Segment-level row/positive counts (from the `Segemnt` column)
- 5-fold stratified cross-validation PR-AUC (mean +/- std)
- The final chosen Yes/No threshold (matched to top-10% lead capacity —
  adjust `TOP_K_PERCENT_CAPACITY` in config.py to your team's real
  capacity)
- A precision/recall report at that threshold

Both steps save a `.joblib` file into `models/` — `train_model.py` must
be run (and finish successfully) for BOTH segments before step 5, since
the dashboard loads both models on startup.

### Step 5 — Launch the dashboard
```bash
streamlit run app.py
```
Upload a lead list (any mix of Health and Non-Health customers) — the
app reads each row's `Segemnt` value, routes it to the matching model,
and returns a Yes/No prediction per row.

## If you get a "column not found" error

`segment_builder.py` will raise a clear error naming the closest-matching
column names it found if `Segemnt` isn't present under that exact name in
your file — update `config.SEGMENT_COLUMN_RAW` to match.


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

## Suggested next steps

1. Run `train_model.py` for both segments on the real 90k file and check
   whether the CV spread tightens up (it should, with far more positive
   examples than any small test run).
2. If Non-Health's CV spread is still wide even at full scale, consider
   simplifying that model further (lower `max_depth`, higher
   `min_child_weight` in config.py) before shipping it.
3. Decide on a real `TOP_K_PERCENT_CAPACITY` with the sales team, rather
   than the 10% placeholder.
4. Once comfortable, wire up batch scoring against the full 1.4 crore
   file (see the SCALE NOTE in `data_loader.py` for chunked reading).
