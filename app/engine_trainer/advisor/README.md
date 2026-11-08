# Epoch Advisor ("Let Epoch decide")

A deterministic, offline micro-AI. It reads a plain-language description of the data and goal, the dataset
profile (`profile.profile_dataset(df)`) and optionally the DataFrame, then returns an `Advice`: task, target,
metric, split, effort, model style, columns to leave out, warnings, and a plain-language explanation for each
decision. When it is unsure it asks a question instead of guessing. No LLM, no network, no files, no global state.
Depends only on numpy / pandas / scipy and `engine_trainer.profile`.

## Public API

```python
from engine_trainer.advisor import advise, apply_answers, describe_advice, Advice

advice = advise(description, profile, df=None, hints=None)
text   = describe_advice(advice)
advice = apply_answers(advice, {"target": "price"})   # re-resolve after the user answers questions
advice.to_dict()                                      # JSON-serialisable
Advice.from_dict(d)                                   # (then pass profile=... to apply_answers)
```

`hints` keys: `task_type`, `target`, `data_kind`, `budget_minutes`, `priority`; plus `group_column` /
`time_column` (a column name or `"none"`) which `apply_answers` fills in.

### Advice fields (exactly these)

| field | values |
|---|---|
| `task_type` | `classification` / `regression` / `unknown` |
| `target` | column name or `None` (never silently the last column) |
| `target_candidates` | up to 5 of `{column, score, reasons[]}` (scores are relative, not probabilities) |
| `data_kind` | `tabular` |
| `metric` | `macro_f1`, `balanced_accuracy`, `roc_auc`, `pr_auc`, `rmse`, `mae`; `None` while the task is unknown |
| `split_strategy` | `stratified` / `random` / `group` / `temporal` |
| `group_column`, `time_column` | column name or `None` |
| `budget_mode` | `express` / `balanced` / `deep` |
| `optimisation_profile` | `balanced` / `accuracy_first` / `speed_first` / `lightweight` / `interpretable` |
| `exclude_columns` | ids, constants, free text, plus the group/time column (reasons are in `explanations`) |
| `warnings` | list of strings |
| `explanations` | one plain-language sentence per decision |
| `questions` | `{id, text, options[]}`, only when needed |
| `confidence` | `high` / `medium` / `low` |

Question ids and valid answers for `apply_answers`: `target` (a column), `task_type` (`classification` /
`regression`), `group_column` (a column or `none`), `time_column` (a column or `none`),
`confirm_many_classes` (`keep` / `choose_another`). An unknown id or unreadable task answer raises `ValueError`.
Metric is `None` and `task_type` is `unknown` until the target and task are resolved; consumers must handle that.

### Confidence

`low`: target or task unresolved. `medium`: a question is pending, the description was empty/unsure/unreadable,
the target was picked by name only, or the description conflicts with the data. `high`: the user (or an explicit
description such as "predict the price") fixed the target, the data is consistent, nothing is open.

## Examples

```python
advise("Customer records. I want to know who will leave.", profile, df)
# target=churn, classification, stratified split, metric roc_auc / balanced_accuracy by class balance, customer_id left out

advise("Each patient has several visits; predict the diagnosis", profile, df)
# split_strategy=group, group_column=patient_id (kept out of the features)

advise("Daily sales, forecast next month", profile, df)
# split_strategy=temporal, time_column=date

advise("", profile, df_with_unclear_columns)
# target=None, questions=[{id: "target", options: [...]}], confidence=low
```

## How it works (all heuristics)

* **Language** (`language.py`): lower-cases and strips accents, then scores weighted regex/keyword lexicons
  (weights hand-set like IDF values, repeats damped with `1 + ln(count)`) for regression vs classification,
  forecasting, repeated entities, speed, accuracy, small device and explainability wishes. Wishes preceded by
  "no need to / without / not" are ignored. Column mentions use exact, plural, small synonym-table
  (price~cost, churn~leave, ...) and fuzzy (`difflib`) matching; a mention counts as "the thing to predict" only
  next to a cue word such as predict / forecast / estimate / know whether. Input is capped (5000 chars, 600 tokens).
* **Target**: additive score from explicit mention, column-name patterns, profile hints, fit with the wording,
  missingness and a small last-column bonus (cannot decide alone). Decided only if score >= 4.5 and leads the runner-up
  by >= 2; otherwise a `target` question.
* **Task**: reuses `profile.infer_task` (profile-only fallback when no DataFrame). Categorical targets are
  classification; non-integer numbers regression; small-integer or mid-cardinality integer targets are asked
  about unless the wording decides; >50 categories asks `confirm_many_classes`. Data beats wording on conflict, with a warning.
* **Metric**: binary with minority share < 10% -> `pr_auc`; ratio > 3 -> `balanced_accuracy`; otherwise `roc_auc`;
  multiclass -> `macro_f1`; regression -> `rmse`, or `mae` if >= 2% of targets lie beyond 3 IQR.
* **Split**: temporal if forecast wording (score >= 3) plus a datetime or monotonic time-named column (or a hinted one);
  group if repeated-entity wording plus a repeated entity-named column, or a name-only proposal (which also asks for
  confirmation); stratified for classification if every class has >= 3 rows (the splitter's minimum); else random.
  Time wins if both apply (warned). Groups need >= 6 distinct values (splitter minimum).
* **Budget / style**: `budget_minutes` or "N minutes" in the text (<=5 express, <=30 balanced, else deep); words
  such as quick / best possible; > 300k rows -> express. Style from priority hint or wishes (phone/embedded ->
  lightweight, explain/regulator -> interpretable, best possible -> accuracy_first, quick -> speed_first).
* **Leakage** (warn only): names that sound post-outcome or contain the target's name; with a DataFrame, a single
  column that predicts the target almost perfectly (purity >= 0.99 over 20 bins, or rank correlation >= 0.995).

## Known limitations / failure modes

* Keyword understanding, not language understanding: English only, no real negation or sarcasm handling beyond a
  short "no need to ..." window; wording about several things at once can score close and produce no decision.
* Fuzzy/synonym matching can pick a wrong column when names are similar (`total_charges` vs `monthly_charges`):
  approximate matches are flagged in the explanation and low margins become a question.
* Name patterns assume common English column names; a target called `col7` is never found without a description or hint.
* The "group proposal from the column name alone" can be wrong (e.g. a repeated `user` column that is really a feature);
  it always asks for confirmation. Group detection ignores unnamed entity columns unless the description names the entity.
* A sorted date column with no forecast wording triggers a time question; datetimes not chosen as time are not used.
* Leakage checks cover single-column leaks only (no combinations), look at the first <= 60 columns and a 20k-row sample.
* Class balance, outliers and leakage need the DataFrame; with the profile alone the metric defaults to `macro_f1` / `rmse`.
* Thresholds (4.5 / margin 2, 3x imbalance, 10% minority, 2% outliers, 300k rows, minute cut-offs) are judgement calls,
  not tuned on any benchmark. **No accuracy figure is claimed for the Advisor.** The tests check designed behaviour on
  synthetic data only; they do not measure how often advice matches what an expert would choose.
