"""Self-contained HTML evaluation report. Only measured values are shown; anything not measured says so."""
from __future__ import annotations

from html import escape as e

CSS = """body{font:15px/1.55 system-ui,Segoe UI,Arial,sans-serif;max-width:980px;margin:32px auto;padding:0 20px;color:#16202a;background:#fff}
h1{font-size:1.7rem;margin:0 0 4px}h2{font-size:1.15rem;margin:28px 0 8px;border-bottom:1px solid #d8dee4;padding-bottom:4px}
table{border-collapse:collapse;width:100%;font-size:.9rem}th,td{border:1px solid #d8dee4;padding:5px 9px;text-align:left}th{background:#f3f6f9}
.muted{color:#5b6773}.ok{color:#1a7f37}.bad{color:#b42318}.warn{color:#9a6700}.box{background:#f3f6f9;border-radius:8px;padding:12px 16px;margin:10px 0}
.num{text-align:right;font-variant-numeric:tabular-nums}.cm td{text-align:center}code{background:#eef2f6;padding:1px 5px;border-radius:4px}
.sev-high{border-left:4px solid #b42318}.sev-medium{border-left:4px solid #9a6700}.sev-low,.sev-info{border-left:4px solid #5b6773}.rec{margin:8px 0;padding:6px 12px;background:#fafbfc}"""


def _f(v, d=4):
    return "n/a" if v is None else (f"{v:.{d}f}" if isinstance(v, (int, float)) else e(str(v)))


def _cm(cm):
    labels, m = cm["labels"], cm["matrix"]
    mx = max((max(r) for r in m), default=1) or 1
    rows = "".join("<tr><th>" + e(str(labels[i])) + "</th>" + "".join(
        f'<td style="background:rgba(31,111,235,{0.08 + 0.7 * v / mx:.2f})">{v}</td>' for v in r) + "</tr>" for i, r in enumerate(m))
    return f'<table class="cm"><tr><th>true \\ predicted</th>{"".join("<th>" + e(str(l)) + "</th>" for l in labels)}</tr>{rows}</table>'


def build_report_html(exp: dict, summary: dict, selected_id: str, test: dict | None, test_count: int, env: dict, dataset_name: str = "") -> str:
    plan = exp["task"]
    task = plan["task_type"]
    metric = summary.get("selection_metric", "")
    cands = {c["cand_key"]: c for c in exp["candidates"]}
    sel = cands.get(selected_id, {})
    sv = (sel.get("val") or {})
    lower = metric in ("rmse", "mae", "log_loss")

    def disp(v):  # CV means are stored higher-is-better; show natural direction
        return None if v is None else (-v if lower else v)

    out = [f"<!doctype html><html lang='en'><meta charset='utf-8'><title>Epoch report: {e(dataset_name or exp['id'])}</title><style>{CSS}</style><body>"]
    out.append(f"<h1>Epoch evaluation report</h1><div class='muted'>Dataset: {e(dataset_name)} · experiment {e(exp['id'])} · created {e(exp['created_at'])}</div>")

    out.append("<h2>1. Result</h2><div class='box'>")
    out.append(f"<b>Task:</b> {e(task)} · <b>target:</b> {e(plan['target'])} · <b>selection metric:</b> {e(metric)}<br>")
    out.append(f"<b>Chosen model:</b> {e(selected_id)} — {e(summary.get('selection_reason', ''))}<br>")
    if sv.get("cv_mean") is not None:
        out.append(f"<b>Cross-validated {e(metric)}:</b> {_f(disp(sv['cv_mean']))} ± {_f(sv.get('cv_std'))} (mean ± std over folds of the development data)<br>")
    if summary.get("beats_baseline") is not None:
        out.append("<b>Beats the do-nothing baseline:</b> " + ("<span class='ok'>yes, by more than normal fold-to-fold variation</span>" if summary["beats_baseline"]
                                                              else "<span class='bad'>no — the model is not clearly better than guessing the most common answer or the average</span>") + "<br>")
    if test:
        keys = [k for k, v in test.items() if isinstance(v, (int, float)) and k not in ("n",)]
        out.append(f"<b>Locked test set</b> (n={test.get('n')}, used {test_count} time(s)): " + ", ".join(f"{e(k)} {_f(test[k])}" for k in keys[:6]) + "<br>")
    else:
        out.append("<b>Locked test set:</b> <span class='warn'>not evaluated yet. The numbers above are cross-validation estimates on development data.</span><br>")
    out.append("</div>")

    out.append("<h2>2. Leaderboard (cross-validation on development data)</h2><table><tr><th>Model</th><th>Status</th><th class='num'>CV " + e(metric) +
               "</th><th class='num'>± std</th><th>Tuned</th><th class='num'>Default CV</th><th class='num'>Fit s</th></tr>")
    rows = sorted(exp["candidates"], key=lambda c: -((c.get("val") or {}).get("cv_mean") if (c.get("val") or {}).get("cv_mean") is not None else -1e9))
    for c in rows:
        v = c.get("val") or {}
        mark = " ◀ chosen" if c["cand_key"] == selected_id else ""
        out.append(f"<tr><td>{e(c['cand_key'])}{mark}</td><td>{e(c['status'])}</td><td class='num'>{_f(disp(v.get('cv_mean')))}</td><td class='num'>{_f(v.get('cv_std'))}</td>"
                   f"<td>{'yes (' + str(v.get('n_trials')) + ' trials)' if v.get('tuned') else 'no' if v.get('tuned') is False else ''}</td><td class='num'>{_f(disp(v.get('default_cv')))}</td>"
                   f"<td class='num'>{_f(c.get('fit_seconds'), 2)}</td></tr>")
    out.append("</table><p class='muted'>Screened-only models were tried with default settings on a quick pass and not tuned. Skipped models did not fit the time budget.</p>")
    ens = summary.get("ensemble")
    if ens:
        out.append(f"<p><b>Ensemble:</b> members {e(', '.join(f'{m} ({w:.0%})' for m, w in ens['members']))}; gain over best single model {_f(ens['gain'])} "
                   f"(needs more than {_f(ens['margin'])}) — {'kept' if ens['accepted'] else 'not kept'}.</p>")

    if task == "classification" and sv.get("confusion_matrix"):
        out.append("<h2>3. Where the model is right and wrong (cross-validated predictions)</h2>" + _cm(sv["confusion_matrix"]))
        out.append("<table><tr><th>Class</th><th class='num'>Precision</th><th class='num'>Recall</th><th class='num'>F1</th><th class='num'>Support</th></tr>" + "".join(
            f"<tr><td>{e(str(c))}</td><td class='num'>{_f(m['precision'], 3)}</td><td class='num'>{_f(m['recall'], 3)}</td><td class='num'>{_f(m['f1'], 3)}</td><td class='num'>{m['support']}</td></tr>"
            for c, m in sv["per_class"].items()) + "</table>")
    elif task == "regression":
        out.append("<h2>3. Error summary (cross-validated predictions)</h2><table><tr><th>MAE</th><th>RMSE</th><th>R²</th><th>Median abs error</th></tr>"
                   f"<tr><td>{_f(sv.get('mae'))}</td><td>{_f(sv.get('rmse'))}</td><td>{_f(sv.get('r2'))}</td><td>{_f(sv.get('median_ae'))}</td></tr></table>")
    if test and task == "classification" and test.get("confusion_matrix"):
        out.append("<h3>Locked test set confusion matrix</h3>" + _cm(test["confusion_matrix"]))

    lc = summary.get("learning_curve") or []
    if lc:
        out.append("<h2>4. Learning curve</h2><table><tr><th>Training rows</th><th class='num'>CV score</th></tr>" +
                   "".join(f"<tr><td>{n}</td><td class='num'>{_f(disp(s))}</td></tr>" for n, s, _ in lc) + "</table>")

    recs = summary.get("recommendations") or []
    out.append("<h2>5. Recommendations</h2>")
    if not recs:
        out.append("<p class='muted'>No issues were detected by the automatic checks. That is not a guarantee of quality.</p>")
    for r in recs:
        out.append(f"<div class='rec sev-{e(r['severity'])}'><b>{e(r['title'])}</b> <span class='muted'>({e(r['severity'])})</span><br>{e(r['detail'])}<ul>" +
                   "".join(f"<li>{e(a)}</li>" for a in r["actions"]) + "</ul></div>")

    out.append("<h2>6. Data and protocol</h2><table>")
    sp = exp.get("split") or {}
    for k, v in [("Rows used", summary.get("n_rows_used")), ("Development rows (cross-validated)", summary.get("n_dev")),
                 ("Split strategy", summary.get("split_strategy", "stratified")), ("Cross-validation", summary.get("cv_scheme")),
                 ("Split sizes", sp.get("sizes")), ("Split hash", (sp.get("hash") or "")[:16]), ("Seed", exp["seed"]),
                 ("Time used (s)", summary.get("seconds_used")), ("Budget (s)", summary.get("budget_seconds"))]:
        out.append(f"<tr><th>{e(k)}</th><td>{e(str(v))}</td></tr>")
    out.append("</table>")
    fl = [f for f in exp.get("findings", []) if f["severity"] != "info"]
    if fl:
        out.append("<h3>Data warnings</h3><ul>" + "".join(f"<li>[{e(f['severity'])}] {e(f['message'])}</li>" for f in fl) + "</ul>")
    if summary.get("search_log"):
        out.append("<h3>Search notes</h3><ul>" + "".join(f"<li>{e(x)}</li>" for x in summary["search_log"]) + "</ul>")
    adv = summary.get("advice")
    if adv and adv.get("explanations"):
        out.append("<h3>Why Epoch chose these settings</h3><ul>" + "".join(f"<li>{e(x)}</li>" for x in adv["explanations"]) + "</ul>")

    out.append("<h2>7. Reproducibility</h2><table>" + "".join(f"<tr><th>{e(k)}</th><td>{e(str(v))}</td></tr>" for k, v in env.items()) + "</table>")
    out.append("<h2>8. Limits of this report</h2><ul><li>Scores come from one dataset and one random split; a different sample could give different numbers.</li>"
               "<li>Cross-validation estimates are for data like the data supplied. Epoch cannot know how the model behaves on data from a different source or period.</li>"
               "<li>Feature importance and recommendations describe what the model used, not what causes the outcome.</li>"
               "<li>Exported model files are pickle-based: load them only from sources you trust.</li></ul></body></html>")
    return "".join(out)
