"""The Epoch Advisor: description + dataset profile (+ optional DataFrame) -> Advice.

Deterministic and offline. Every rule is a heuristic; each decision records a plain-language
explanation, and uncertain choices become questions instead of silent guesses.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from .. import profile as _profile
from . import datachecks as dc
from .language import LanguageSignals, analyse_description
from .models import Advice

AUTO_EXCLUDE_KINDS = {"id": "it looks like a row identifier (a different value on every row), so it cannot help predict new cases",
                      "constant": "it has the same value on every row, so it carries no information",
                      "text": "it is free text, which is not supported as input yet"}
METRIC_GLOSS = {
    "macro_f1": "macro F1 (a score that treats every category equally, so rare categories are not ignored)",
    "balanced_accuracy": "balanced accuracy (the average hit-rate per category, so a rare category counts as much as a common one)",
    "roc_auc": "ROC AUC (how well the model ranks a true case above a false one; 1.0 is perfect, 0.5 is guessing)",
    "pr_auc": "PR AUC (how well the model finds the rare cases without too many false alarms)",
    "rmse": "RMSE (typical size of the error, in the units of the number you predict, with big misses counted extra)",
    "mae": "MAE (average size of the error, in the units of the number you predict, with every miss counted equally)",
}
_TASK_ALIASES = {"classification": "classification", "classify": "classification", "category": "classification",
                 "categories": "classification", "label": "classification", "class": "classification",
                 "regression": "regression", "number": "regression", "quantity": "regression", "count": "regression",
                 "amount": "regression", "numeric": "regression"}
_PRIORITY_WORDS = (("light", "lightweight"), ("small", "lightweight"), ("device", "lightweight"), ("size", "lightweight"),
                   ("explain", "interpretable"), ("interpret", "interpretable"), ("accura", "accuracy_first"),
                   ("best", "accuracy_first"), ("speed", "speed_first"), ("fast", "speed_first"), ("quick", "speed_first"),
                   ("balanc", "balanced"))


def _profile_task(p: dict, n_rows: int) -> tuple[str, bool]:
    """(task, needs_confirmation) from the profile only; mirrors profile.infer_task."""
    nun = int(p.get("n_unique") or 0)
    if p.get("kind") in ("categorical", "boolean"):
        return "classification", nun > 50
    dtype = str(p.get("dtype", ""))
    ex = []
    for v in p.get("example_values", []):
        try:
            ex.append(float(v))
        except (TypeError, ValueError):
            pass
    if dtype.startswith("float") and any(v != round(v) for v in ex):
        return "regression", False
    if nun <= 10:
        return "classification", nun > 2
    if n_rows and (nun / n_rows > 0.2 or nun > 50):
        return "regression", nun <= 20
    return "classification", True


def _hint_priority(v: Any) -> str | None:
    s = str(v or "").lower()
    for key, prof in _PRIORITY_WORDS:
        if key in s:
            return prof
    return None


def advise(description: str, profile: dict, df: pd.DataFrame | None = None, hints: dict | None = None) -> Advice:
    """Choose task, target, metric, split, budget, profile and exclusions, with explanations.

    ``hints`` may contain ``task_type``, ``target``, ``data_kind``, ``budget_minutes``, ``priority`` and,
    for answered questions, ``group_column`` / ``time_column`` (a column name or ``"none"``).
    Never raises on odd text; a profile without columns yields a low-confidence Advice with a warning.
    """
    hints = dict(hints or {})
    state = {"description": description, "profile": profile, "hints": dict(hints), "df": df}
    warns: list[str] = []
    expl: list[str] = []
    questions: list[dict] = []
    soft: list[str] = []          # reasons confidence cannot be high
    profile = profile if isinstance(profile, dict) else {}
    cols = {str(c["name"]): c for c in profile.get("columns", [])}
    names = list(cols)
    n_rows = int(profile.get("n_rows") or (len(df) if df is not None else 0))
    if df is not None and not set(names) <= {str(c) for c in df.columns}:
        df = None   # profile and frame disagree; trust the profile only
    adv = Advice(_state=state)
    if not names:
        adv.warnings = ["The dataset profile has no columns, so there is nothing to advise on."]
        return adv

    # ---- data kind ---------------------------------------------------------------
    dk = str(hints.get("data_kind") or "tabular").lower()
    if dk not in ("tabular", "table", "csv", "spreadsheet", "auto"):
        warns.append(f"You said the data is '{dk}', but only table-shaped data is supported right now, so I treated it as a table.")

    elig = {n for n, p in cols.items() if p.get("kind") in dc.eligible_target_kinds()}
    lang = analyse_description(description, names, elig)
    if lang.empty:
        expl.append("You did not describe the data, so every choice below comes from looking at the data itself.")
    elif lang.unsure:
        expl.append("You said you are not sure what you want, so I relied on the data itself and kept my choices conservative.")
    elif not lang.understood:
        expl.append("I could not make much sense of the description (it may be in a language or style I do not read), "
                    "so every choice below comes from looking at the data itself.")
    if lang.empty or lang.unsure or not lang.understood:
        soft.append("no usable description")

    # ---- target ------------------------------------------------------------------
    target, cands, tsource = _resolve_target(cols, names, elig, lang, hints, df, warns, questions, expl)
    adv.target_candidates = cands
    adv.target = target

    auto_excl = {n: AUTO_EXCLUDE_KINDS[p["kind"]] for n, p in cols.items() if p.get("kind") in AUTO_EXCLUDE_KINDS and n != target}

    task = "unknown"
    stats: dict = {}
    if target is not None:
        task = _resolve_task(cols, target, n_rows, lang, hints, df, warns, questions, expl, soft)
        if tsource == "data" or tsource == "name":
            soft.append("target chosen without explicit instruction")
    else:
        soft.append("target unresolved")
    adv.task_type = task

    # ---- time and group -----------------------------------------------------------
    skip = set(auto_excl) | ({target} if target else set())
    time_col = _resolve_time(cols, names, df, lang, hints, skip, warns, questions, expl, soft)
    group_col = None
    if time_col:
        if lang.group >= 3 or hints.get("group_column") not in (None, "none"):
            warns.append("Your data seems to have repeated entities AND a time order. Only one can drive the split right now, "
                         "so I chose the time order (it is the stricter test for forecasting). Make sure the same entities "
                         "appearing in both past and future is what you expect.")
    else:
        group_col = _resolve_group(cols, n_rows, lang, hints, skip, warns, questions, expl, soft)
    adv.time_column, adv.group_column = time_col, group_col

    # ---- class balance ------------------------------------------------------------
    if target is not None and task == "classification" and df is not None and target in df.columns:
        stats = dc.class_stats(df[target])
    # ---- split --------------------------------------------------------------------
    if time_col:
        adv.split_strategy = "temporal"
        expl.append(f"Split: I will train on the earliest rows and test on the latest, ordered by '{time_col}', "
                    "because you want to predict forward in time and shuffling would let the model peek at the future.")
    elif group_col:
        adv.split_strategy = "group"
        expl.append(f"Split: all rows with the same '{group_col}' stay together (never some in training and some in testing), "
                    "otherwise the model could simply recognise the individual and the score would look better than it really is.")
    elif task == "classification" and not (stats and stats["min_count"] < 3):
        adv.split_strategy = "stratified"
        expl.append("Split: rows are shuffled but each category keeps its share in training, validation and test sets, so rare categories are present everywhere.")
    else:
        adv.split_strategy = "random"
        if task == "classification":
            warns.append(f"The rarest category has only {stats['min_count']} example(s); I used a plain random split because a fair "
                         "one needs at least 3. Consider merging or removing very rare categories.")
        if task != "unknown":
            expl.append("Split: rows are shuffled randomly into training, validation and test sets.")

    # ---- metric -------------------------------------------------------------------
    adv.metric = _choose_metric(task, target, df, stats, expl, warns)

    # ---- exclusions ---------------------------------------------------------------
    excl: list[str] = []
    for n, why in auto_excl.items():
        excl.append(n)
        expl.append(f"Left out '{n}': {why}.")
    for n, role in ((group_col, "only used to keep rows of the same entity together"), (time_col, "only used to put rows in time order")):
        if n and n not in excl:
            excl.append(n)
            expl.append(f"'{n}' is {role}, so it is not used as a feature.")
    adv.exclude_columns = excl

    # ---- leakage ------------------------------------------------------------------
    if target is not None and task != "unknown":
        for n, why in dc.leakage_suspects(df, cols, target, set(excl), task):
            warns.append(f"Possible answer leak: '{n}' - {why}. If this is only known after the outcome, remove it, "
                         "or the model will look great here and fail in real use.")

    # ---- budget & profile ---------------------------------------------------------
    adv.budget_mode = _choose_budget(n_rows, lang, hints, expl, warns)
    adv.optimisation_profile = _choose_profile(lang, hints, adv.budget_mode, expl)

    # ---- generic data warnings ----------------------------------------------------
    if 0 < n_rows < 50:
        warns.append(f"Only {n_rows} rows: any model trained on so little data will be unreliable. Treat results as a rough hint.")
    elif 0 < n_rows < 200:
        warns.append(f"Only {n_rows} rows: results will vary a lot from one split to another.")
    if n_rows and profile.get("n_duplicate_rows", 0) / n_rows > 0.05:
        warns.append(f"{profile['n_duplicate_rows']} rows are exact duplicates; duplicates can make scores look better than they are.")
    if n_rows and len(names) > n_rows:
        warns.append("There are more columns than rows, which makes over-fitting very likely.")
    if target is not None and cols[target].get("n_missing"):
        warns.append(f"{cols[target]['n_missing']} rows have no value for '{target}' and will be skipped.")
    if stats and stats["n_classes"] > 50:
        warns.append(f"'{target}' has {stats['n_classes']} categories; many will have very few examples.")
    if stats and stats["n_classes"] >= 2 and stats["min_count"] < 10 and stats["min_count"] >= 3:
        warns.append(f"The rarest category has only {stats['min_count']} examples; its results will be uncertain.")

    # ---- confidence ---------------------------------------------------------------
    unresolved = target is None or task == "unknown"
    if unresolved:
        conf = "low"
    elif questions or soft:
        conf = "medium"
    else:
        conf = "high"
    adv.warnings, adv.explanations, adv.questions, adv.confidence = warns, expl, questions, conf
    return adv


# ======================================================================================
def _resolve_target(cols, names, elig, lang: LanguageSignals, hints, df, warns, questions, expl):
    """Return (target|None, ranked candidates, source in {'hint','text','name','data',None})."""
    reject = {str(x) for x in hints.get("reject_targets", [])}
    prof_sugg = {s["column"]: s["score"] for s in profile_suggestions(cols, hints)}
    last = names[-1]
    text_task = lang.text_task
    ranked = []
    for i, n in enumerate(names):
        if n not in elig or n in reject:
            continue
        p = cols[n]
        score, why = 0.0, []
        m = lang.mentions.get(n)
        if m:
            if m.explicit:
                score += 8 * m.strength + (2 if m.strength >= 0.99 else 0)
                why.append("your description says this is what you want to predict")
            else:
                score += 1.5 * m.strength; why.append("mentioned in your description")
        ns, nw = dc.name_target_score(n)
        if ns:
            score += ns; why.append(nw)
        if n in prof_sugg:
            score += min(2.0, 0.4 * prof_sugg[n])
        nun, kind = int(p.get("n_unique") or 0), p.get("kind")
        if text_task == "regression":
            if kind == "numeric" and nun > 10:
                score += 1.5; why.append("numeric with many values, which fits predicting a number")
            elif kind in ("categorical", "boolean"):
                score -= 1
        elif text_task == "classification":
            if kind in ("categorical", "boolean") or nun <= 20:
                score += 1.5; why.append("a small set of values, which fits predicting a category")
            elif kind == "numeric" and nun > 20:
                score -= 1
        if p.get("pct_missing", 0) > 20:
            score -= 2; why.append("many missing values")
        elif p.get("pct_missing", 0) > 5:
            score -= 0.5
        if n == last:
            score += 0.5; why.append("last column (a weak hint only)")
        ranked.append((n, round(max(score, 0.0), 2), why, i))
    ranked.sort(key=lambda t: (-t[1], t[3]))
    cands = [{"column": n, "score": s, "reasons": w} for n, s, w, _ in ranked[:5]]

    ht = hints.get("target")
    if ht is not None:
        ht = str(ht)
        if ht in elig and ht not in reject:
            cands = [{"column": ht, "score": 10.0, "reasons": ["you chose this column"]}] + [c for c in cands if c["column"] != ht][:4]
            expl.append(f"Target: you told me to predict '{ht}'.")
            return ht, cands, "hint"
        why = "it is not in the data" if ht not in cols else f"its content ({cols[ht].get('kind')}) cannot be predicted"
        warns.append(f"You asked to predict '{ht}' but {why}, so I chose the target myself.")
    if not ranked:
        warns.append("No column looks usable as something to predict (all are identifiers, free text, dates or constant).")
        return None, cands, None
    top = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else -99
    m = lang.mentions.get(top[0])
    if top[1] >= 4.5 and top[1] - second >= 2:
        src = "text" if (m and m.explicit) else "name"
        if src == "text":
            expl.append(f"Target: I will predict '{top[0]}' because your description points to it"
                        f"{'' if m.kind == 'exact' else ' (matched approximately - please check it is right)'}.")
        else:
            expl.append(f"Target: I will predict '{top[0]}' because its name and position look like the outcome column. "
                        "Please check this is what you want.")
        return top[0], cands, src
    opts = [c["column"] for c in cands if c["score"] > 0.9] or [n for n in names if n in elig][:8]
    if len(opts) < 2 and len([n for n in names if n in elig]) > 1:
        opts = [n for n in names if n in elig][:8]
    expl.append("Target: I cannot tell for sure which column you want to predict, so I will ask instead of guessing.")
    questions.append({"id": "target", "text": "Which column do you want Epoch to predict?", "options": opts})
    return None, cands, None


def profile_suggestions(cols: dict, hints: dict) -> list[dict]:
    """Profile-style target suggestions recomputed from the profile alone (kind-based)."""
    out = []
    for n, p in cols.items():
        if p.get("kind") in ("id", "text", "datetime", "constant"):
            continue
        s = 0
        if _profile.TARGET_NAMES.match(n):
            s += 3
        if p.get("kind") in ("categorical", "boolean") and (p.get("n_unique") or 99) <= 20:
            s += 1
        if s:
            out.append({"column": n, "score": s})
    return out


def _resolve_task(cols, target, n_rows, lang, hints, df, warns, questions, expl, soft) -> str:
    p = cols[target]
    kind, nun = p.get("kind"), int(p.get("n_unique") or 0)
    text_task = lang.text_task
    if df is not None and target in df.columns:
        try:
            ts = _profile.infer_task(df, target)
            dtask, needs = ts.task_type, bool(ts.needs_confirmation)
            if kind == "numeric" and nun <= 2:
                needs = False
        except Exception:  # pragma: no cover - defensive
            dtask, needs = _profile_task(p, n_rows)
    else:
        dtask, needs = _profile_task(p, n_rows)
        if kind == "numeric" and nun <= 2:
            needs = False
    hint_task = _TASK_ALIASES.get(str(hints.get("task_type") or "").lower())
    categorical = kind in ("categorical", "boolean")

    if hint_task == "regression" and categorical:
        warns.append(f"You asked for a number prediction, but '{target}' holds categories, so I cannot do regression on it.")
        hint_task = None
    if hint_task:
        if hint_task == "classification" and kind == "numeric" and nun > 20 and dtask == "regression" and not needs:
            warns.append(f"'{target}' looks like a continuous number ({nun} distinct values). Treating it as categories is probably a mistake.")
            soft.append("task hint contradicts data")
        expl.append(f"Task: {_task_sentence(hint_task, target)} (as you requested).")
        return hint_task

    if categorical:
        if text_task == "regression":
            warns.append(f"Your description sounds like predicting a number, but '{target}' holds categories "
                         f"({nun} different values). I followed the data and will predict the category.")
            soft.append("description conflicts with data")
        if nun > 50:
            questions.append({"id": "confirm_many_classes",
                              "text": f"'{target}' has {nun} different values. Do you really want to predict which of {nun} categories a row belongs to?",
                              "options": ["keep", "choose_another"]})
            expl.append(f"Task: '{target}' has {nun} categories; that is unusual, so I will check with you.")
            soft.append("many classes")
        else:
            expl.append(f"Task: {_task_sentence('classification', target)} because it holds {nun} categories.")
        return "classification"

    if not needs:
        if text_task and text_task != dtask:
            what = "a number" if text_task == "regression" else "a category"
            warns.append(f"Your description sounds like predicting {what}, but the data in '{target}' says otherwise "
                         f"({'continuous numbers' if dtask == 'regression' else 'only a few distinct values'}). I followed the data.")
            soft.append("description conflicts with data")
        expl.append(f"Task: {_task_sentence(dtask, target)} because " +
                    (f"it holds numbers with many different values ({nun})." if dtask == "regression"
                     else f"it only takes {nun} distinct values." if nun else "of its values."))
        return dtask

    if text_task:
        expl.append(f"Task: '{target}' is a whole number with {nun} different values, which could be a category or a quantity; "
                    f"your description sounds like {'a number' if text_task == 'regression' else 'a category'}, so {_task_sentence(text_task, target)}.")
        soft.append("task decided from wording only")
        return text_task
    questions.append({"id": "task_type",
                      "text": (f"'{target}' holds whole numbers with {nun} different values. Are they labels, like a 1-5 rating or a class code "
                               "(answer 'classification': pick one of a few options), or a quantity to estimate, like a count or a price "
                               "(answer 'regression': predict a number)?"),
                      "options": ["classification", "regression"]})
    expl.append(f"Task: '{target}' could be either a category or a number, so I will ask.")
    return "unknown"


def _task_sentence(task: str, target: str) -> str:
    if task == "classification":
        return f"I will predict which category '{target}' falls into (classification)"
    return f"I will predict the number '{target}' (regression)"


def _resolve_time(cols, names, df, lang, hints, skip, warns, questions, expl, soft) -> str | None:
    ht = hints.get("time_column")
    if ht is not None and str(ht).lower() == "none":
        return None
    if ht is not None and str(ht) in cols and str(ht) not in skip:
        return str(ht)
    want = lang.temporal >= 3
    cands = dc.time_candidates(cols, df, skip)
    if want:
        if not cands:
            warns.append("You want to predict forward in time, but I found no date or time-ordered column, so I could not "
                         "use a time-based split. Results may be too optimistic for forecasting.")
            soft.append("forecast wording without time column")
            return None
        if len(cands) > 1 and cands[0]["score"] - cands[1]["score"] < 1:
            questions.append({"id": "time_column", "text": "Which column says when each row happened?",
                              "options": [c["column"] for c in cands[:6]] + ["none"]})
            expl.append("Time: several columns could be the time order, so I will ask which one to use.")
            return None
        return cands[0]["column"]
    if cands and df is not None and cands[0]["monotonic"] and ht is None:
        questions.append({"id": "time_column",
                          "text": (f"Your rows are already in order of '{cands[0]['column']}'. Do you want to predict the future from "
                                   "the past (then I test on the latest rows)? Answer with that column's name, or 'none' if row order does not matter."),
                          "options": [cands[0]["column"], "none"]})
        expl.append(f"Time: '{cands[0]['column']}' looks like a time order but you did not mention forecasting, so I will check.")
    return None


def _resolve_group(cols, n_rows, lang, hints, skip, warns, questions, expl, soft) -> str | None:
    hg = hints.get("group_column")
    if hg is not None and str(hg).lower() == "none":
        return None
    if hg is not None and str(hg) in cols and str(hg) not in skip:
        return _accept_group(str(hg), cols, warns, "you told me")
    want = lang.group >= 3
    cands = dc.group_candidates(cols, n_rows, skip, lang.entities, lang.mentions)
    if want:
        if cands:
            top = cands[0]
            if top["score"] >= 5 and (len(cands) == 1 or top["score"] - cands[1]["score"] >= 2):
                return _accept_group(top["column"], cols, warns, "your description says the same entity appears in several rows")
            questions.append({"id": "group_column", "text": "Which column identifies the same person/machine/customer across rows?",
                              "options": [c["column"] for c in cands[:6]] + ["none"]})
            expl.append("Groups: you said the same entity appears in several rows; I need you to pick which column identifies it.")
            soft.append("group column ambiguous")
            return None
        rep = dc.repeated_columns(cols, n_rows, skip)
        warns.append("You said the same person/machine/customer appears in several rows, but I could not find a column that "
                     "identifies them. Without one, the test score may be too optimistic because the model can recognise individuals.")
        soft.append("group wording without group column")
        if rep:
            questions.append({"id": "group_column", "text": "Is there a column that identifies the same person/machine/customer across rows?",
                              "options": rep + ["none"]})
        return None
    strong = [c for c in cands if c["score"] >= 2]
    if strong:
        top = strong[0]
        if len(strong) == 1 or top["score"] - strong[1]["score"] >= 2:
            g = _accept_group(top["column"], cols, warns, "its name and repeated values suggest it identifies the same entity in several rows")
            if g:
                questions.append({"id": "group_column", "text": f"'{g}' appears on several rows. Does it identify the same person/machine/customer? "
                                  "If yes, rows sharing it must stay on the same side of the train/test split.",
                                  "options": [g, "none"]})
                soft.append("group split is a proposal")
            return g
        questions.append({"id": "group_column", "text": "Several columns repeat across rows. Does one identify the same person/machine/customer?",
                          "options": [c["column"] for c in strong[:6]] + ["none"]})
        soft.append("group column ambiguous")
    return None


def _accept_group(col: str, cols: dict, warns: list, why: str) -> str | None:
    if int(cols[col].get("n_unique") or 0) < 6:
        warns.append(f"'{col}' has fewer than 6 distinct values, too few to keep groups apart across train, validation and test, so no group split was used.")
        return None
    return col


def _choose_metric(task, target, df, stats, expl, warns) -> str | None:
    if task == "unknown" or target is None:
        return None
    if task == "classification":
        if not stats:
            m, why = "macro_f1", "I could not check how balanced the categories are, so I chose a score that treats every category equally"
        elif stats["n_classes"] < 2:
            return "macro_f1"
        elif stats["n_classes"] == 2:
            r, share = stats["ratio"], stats["minority_share"]
            if share < 0.10:
                m, why = "pr_auc", (f"only {100 * share:.1f}% of rows are the rarer outcome ('{stats['minority']}'), so plain accuracy "
                                    "would look great even for a model that never finds it")
            elif r > 3:
                m, why = "balanced_accuracy", f"one outcome is {r:.1f}x more common than the other, so plain accuracy would be misleading"
            else:
                m, why = "roc_auc", "the two outcomes are reasonably balanced"
        else:
            r = stats["ratio"]
            m = "macro_f1"
            why = (f"the most common category is {r:.1f}x bigger than the rarest, so every category must count equally"
                   if r > 3 else "it counts every category equally")
        expl.append(f"Score: I will judge models by {METRIC_GLOSS[m]}, because {why}.")
        return m
    # regression
    m, why = "rmse", "it is the standard choice when big misses matter more than small ones; MAE will also be reported"
    if df is not None and target in df.columns:
        share = dc.heavy_outliers(df[target])
        if share >= 0.02:
            m, why = "mae", f"about {100 * share:.1f}% of values are extreme outliers, and MAE is less distorted by them (RMSE will also be reported)"
    expl.append(f"Score: I will judge models by {METRIC_GLOSS[m]}, because {why}.")
    return m


def _choose_budget(n_rows, lang, hints, expl, warns) -> str:
    mins = hints.get("budget_minutes")
    try:
        mins = float(mins) if mins is not None else None
    except (TypeError, ValueError):
        warns.append("I could not read your time budget, so I ignored it.")
        mins = None
    src = "you set a budget of %g minutes" % mins if mins is not None else None
    if mins is None and lang.minutes is not None:
        mins, src = lang.minutes, "your description mentions %g minutes" % lang.minutes
    if mins is not None:
        mode = "express" if mins <= 5 else "balanced" if mins <= 30 else "deep"
    elif lang.speed > 0 and lang.accuracy > 0:
        mode, src = "balanced", "you asked for both speed and the best result"
    elif lang.accuracy > 0:
        mode, src = "deep", "you asked for the best result you can get"
    elif lang.speed > 0:
        mode, src = "express", "you asked for a quick answer"
    elif n_rows > 300_000:
        mode, src = "express", f"the data is large ({n_rows:,} rows) and long searches would be slow"
    else:
        mode, src = "balanced", "the data is a moderate size and no time limit was given"
    expl.append(f"Effort: {mode} mode, because {src}. You can change this.")
    return mode


def _choose_profile(lang, hints, mode, expl) -> str:
    hp = _hint_priority(hints.get("priority"))
    if hp:
        prof, why = hp, "you set this priority"
    elif lang.light > 0:
        prof, why = "lightweight", "you need the model to run on a small device, so I prefer small, cheap models"
    elif lang.interpretable > 0:
        prof, why = "interpretable", "you need to be able to explain results, so I prefer simple models whose reasoning can be shown"
    elif lang.accuracy > 0 and not lang.speed > 0:
        prof, why = "accuracy_first", "you want the most accurate result, even if it takes longer or is harder to explain"
    elif lang.speed > 0 and not lang.accuracy > 0:
        prof, why = "speed_first", "you want an answer quickly"
    else:
        prof, why = "balanced", "no special requirement was found"
    expl.append(f"Model style: {prof.replace('_', ' ')}, because {why}.")
    return prof


# ======================================================================================
def apply_answers(advice: Advice, answers: dict, *, profile: dict | None = None, df: pd.DataFrame | None = None,
                  description: str | None = None) -> Advice:
    """Re-run the Advisor with the user's answers to ``advice.questions``.

    ``answers`` maps a question id to the chosen option (``target``, ``task_type``, ``group_column``,
    ``time_column``, ``confirm_many_classes``). Advice rebuilt via ``Advice.from_dict`` lacks the private
    inputs, so pass ``profile`` (and optionally ``df`` and ``description``) explicitly.
    Raises ValueError on an unknown question id or an unreadable answer.
    """
    st = advice._state or {}
    prof = profile if profile is not None else st.get("profile")
    if prof is None:
        raise ValueError("apply_answers needs the dataset profile: pass profile=... (this Advice has no stored inputs).")
    hints = dict(st.get("hints") or {})
    for qid, ans in (answers or {}).items():
        a = None if ans is None else str(ans)
        if qid == "target":
            hints["target"] = a
        elif qid == "task_type":
            t = _TASK_ALIASES.get((a or "").lower())
            if not t:
                raise ValueError(f"Unreadable answer for task_type: {ans!r} (use 'classification' or 'regression').")
            hints["task_type"] = t
        elif qid in ("group_column", "time_column"):
            hints[qid] = "none" if a is None or a.lower().startswith("none") else a
        elif qid == "confirm_many_classes":
            if (a or "").lower().startswith("keep"):
                hints["task_type"] = "classification"
            else:
                hints["reject_targets"] = list(hints.get("reject_targets", [])) + ([advice.target] if advice.target else [])
                hints.pop("target", None)
        else:
            raise ValueError(f"Unknown question id: {qid!r}")
    return advise(description if description is not None else st.get("description", ""), prof,
                  df if df is not None else st.get("df"), hints)


def describe_advice(advice: Advice) -> str:
    """A readable multi-line summary of an Advice (decisions, reasons, warnings, open questions)."""
    L = [f"Epoch Advisor - confidence: {advice.confidence}"]
    L.append(f"  Predict:   {advice.target or '(not decided yet)'}  [{advice.task_type}]")
    L.append(f"  Score by:  {advice.metric or '(decided once the task is known)'}")
    L.append(f"  Split:     {advice.split_strategy}" + (f" on '{advice.group_column}'" if advice.group_column else "")
             + (f" ordered by '{advice.time_column}'" if advice.time_column else ""))
    L.append(f"  Effort:    {advice.budget_mode}   Style: {advice.optimisation_profile}")
    if advice.exclude_columns:
        L.append("  Not used as inputs: " + ", ".join(advice.exclude_columns))
    if advice.explanations:
        L.append("\nWhy:")
        L += [f"  - {e}" for e in advice.explanations]
    if advice.warnings:
        L.append("\nHeads up:")
        L += [f"  ! {w}" for w in advice.warnings]
    if advice.questions:
        L.append("\nI need your help with:")
        for q in advice.questions:
            L.append(f"  ? {q['text']}  (options: {', '.join(q['options'])})")
    return "\n".join(L)
