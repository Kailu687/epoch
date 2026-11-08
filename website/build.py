"""Static site generator for the Epoch website.

Usage: python build.py            -> writes ./dist
Data sources: ../releases/releases.json, src/data/roadmap.json, src/pages/*.html, src/docs/*.md
Downloadable artifacts are only rendered as links after verification (see verify_artifact).
"""
import hashlib
import html
import json
import re
import shutil
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
DIST = ROOT / "dist"
REL_DIR = ROOT.parent / "releases"
CHANNELS = ["stable", "beta", "development"]
STATUSES = {"released", "prerelease", "in-development", "planned"}
esc = html.escape

NAV = [("index.html", "Home"), ("features.html", "Features"), ("how-it-works.html", "How it works"), ("downloads.html", "Downloads"),
       ("docs/index.html", "Docs"), ("changelog.html", "Changelog &amp; roadmap"), ("about.html", "About")]

# ------------------------------------------------------------ release manifest


def semver_key(v):
    m = re.match(r"^(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.-]+))?$", v)
    if not m:
        raise ValueError(f"invalid semantic version: {v!r}")
    major, minor, patch, pre = m.groups()
    return (int(major), int(minor), int(patch), 1 if pre is None else 0, pre or "")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_manifest(m):
    errs = []
    for r in m.get("releases", []):
        v = r.get("version", "?")
        try:
            semver_key(v)
        except ValueError as e:
            errs.append(str(e))
        if r.get("channel") not in CHANNELS:
            errs.append(f"{v}: bad channel {r.get('channel')!r}")
        if r.get("status") not in STATUSES:
            errs.append(f"{v}: bad status {r.get('status')!r}")
        for a in r.get("artifacts", []):
            for k in ("platform", "arch", "package", "name"):
                if not a.get(k):
                    errs.append(f"{v}: artifact missing {k}")
            if not (a.get("file") or a.get("url")):
                errs.append(f"{v}/{a.get('name')}: needs 'file' or 'url'")
    return errs


def verify_artifact(a, release_version):
    """Return (downloadable: bool, href: str|None, reason: str). Never guesses."""
    if a.get("file"):
        p = (REL_DIR / "artifacts" / a["file"]).resolve()
        if not str(p).startswith(str((REL_DIR / "artifacts").resolve())):
            return False, None, "artifact path escapes releases/artifacts"
        if not p.is_file():
            return False, None, "file not found at build time"
        if not a.get("sha256"):
            return False, None, "no SHA-256 recorded"
        actual = sha256_file(p)
        if actual.lower() != a["sha256"].lower():
            return False, None, "checksum mismatch"
        if a.get("size") is not None and p.stat().st_size != a["size"]:
            return False, None, "size mismatch"
        dest = DIST / "downloads" / release_version / p.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
        return True, f"downloads/{release_version}/{p.name}", "verified local file"
    if a.get("url") and a.get("url_verified_at"):
        return True, a["url"], f"url verified {a['url_verified_at']}"
    return False, None, "URL not verified"


def load_releases():
    m = json.loads((REL_DIR / "releases.json").read_text(encoding="utf-8"))
    errs = validate_manifest(m)
    if errs:
        raise SystemExit("releases.json invalid:\n  " + "\n  ".join(errs))
    rels = sorted(m["releases"], key=lambda r: semver_key(r["version"]), reverse=True)
    for r in rels:
        for a in r.get("artifacts", []):
            a["_ok"], a["_href"], a["_why"] = verify_artifact(a, r["version"])
    return rels


def latest_downloadable(rels, channels=("stable",)):
    for ch in channels:
        for r in rels:  # rels is sorted newest first
            if r["channel"] == ch and r["status"] in ("released", "prerelease"):
                oks = [a for a in r.get("artifacts", []) if a["_ok"]]
                if oks:
                    return r, oks
    return None, []


# ------------------------------------------------------------ layout


def brand_logo():
    """Logo supplied by the owner at src/assets/logo.(svg|png|webp), else the placeholder mark."""
    for ext in ("svg", "png", "webp"):
        if (SRC / "assets" / f"logo.{ext}").is_file():
            return f"assets/logo.{ext}"
    return "assets/favicon.svg"


def layout(title, desc, body, rel, depth=0, current=""):
    b = "../" * depth
    logo = brand_logo()
    nav = "".join(f'<li><a href="{b}{h}"{" aria-current=page" if h == current else ""}>{t}</a></li>' for h, t in NAV)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)}</title><meta name="description" content="{esc(desc)}">
<meta name="theme-color" content="#0b0e13"><link rel="icon" href="{b}assets/favicon.svg" type="image/svg+xml">
<link rel="stylesheet" href="{b}assets/style.css"></head>
<body><a class="skip" href="#main">Skip to content</a>
<header class="site"><div class="wrap"><a class="brand" href="{b}index.html"><img src="{b}{logo}" alt="" height="28" style="height:28px;width:auto"><span>Epoch</span></a>
<button class="menu-btn" aria-expanded="false" aria-controls="nav" id="mb">Menu</button>
<nav class="main" id="nav" aria-label="Main"><ul>{nav}</ul></nav></div></header>
<main id="main">{body}</main>
<footer class="site"><div class="wrap"><div class="cols">
<div><h4>Epoch</h4><p>A local-first machine learning training platform. Early development: see the roadmap for what works today.</p></div>
<div><h4>Product</h4><ul><li><a href="{b}downloads.html">Downloads &amp; releases</a></li><li><a href="{b}docs/index.html">Documentation</a></li><li><a href="{b}changelog.html">Changelog &amp; roadmap</a></li></ul></div>
<div><h4>Project</h4><ul><li>Source repository: not yet published</li><li>Support channel: none yet</li><li><a href="{b}about.html">About</a></li></ul></div>
</div><p class="cap">Site built {date.today().isoformat()} from the release manifest. Features marked planned are not available.</p></div></footer>
<script>var mb=document.getElementById('mb'),n=document.getElementById('nav');mb.addEventListener('click',function(){{var o=n.classList.toggle('open');mb.setAttribute('aria-expanded',o)}});</script>
</body></html>"""


def badge(status):
    cls = {"released": "b-ok", "prerelease": "b-warn", "in-development": "b-warn", "planned": "b-plan",
           "in-progress": "b-warn", "done": "b-ok", "implemented": "b-ok", "available": "b-ok", "partial": "b-warn"}.get(status, "b-muted")
    return f'<span class="badge {cls}">{esc(status)}</span>'


# ------------------------------------------------------------ pages


def fmt_size(n):
    if n is None:
        return "size unavailable"
    return f"{n / 1048576:.1f} MiB" if n >= 1048576 else f"{n / 1024:.1f} KiB"


def render_release(r, depth=0):
    rows = []
    for a in r.get("artifacts", []):
        sha = f'<span class="sha">{esc(a["sha256"])}</span>' if a.get("sha256") else '<span class="muted">unavailable</span>'
        if a["_ok"]:
            action = f'<a class="btn primary" href="{a["_href"]}" download>Download</a>'
        else:
            action = f'<span class="dl-na">Not downloadable ({esc(a["_why"])})</span>'
        rows.append(f"<tr><td>{esc(a['platform'])} {esc(a['arch'])}</td><td>{esc(a['package'])}</td><td>{esc(a['name'])}</td>"
                    f"<td>{fmt_size(a.get('size'))}</td><td>{sha}</td><td>{esc(a.get('min_os', 'unspecified'))}</td><td>{action}</td></tr>")
        if a.get("install_notes"):
            rows.append(f'<tr><td colspan="7" class="muted">Install: {esc(a["install_notes"])}</td></tr>')
    if rows:
        art = ('<div class="tbl"><table><thead><tr><th>Platform</th><th>Package</th><th>File</th><th>Size</th><th>SHA-256</th><th>Min OS</th><th></th></tr></thead><tbody>'
               + "".join(rows) + "</tbody></table></div>")
    else:
        art = '<div class="notice">No downloadable artifacts exist for this release. No installer has been built or verified.</div>'
    ms = r.get("min_system", {})
    req = "".join(f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in ms.items())
    li = lambda xs: "".join(f"<li>{esc(x)}</li>" for x in xs)
    pub = esc(r["published"]) if r.get("published") else "not yet published"
    return f"""<article class="rel card" id="v{esc(r['version'])}">
<div class="rel-head"><h3>Epoch {esc(r['version'])}</h3>{badge(r['status'])}<span class="badge b-muted">{esc(r['channel'])}</span><span class="muted">Released: {pub}</span></div>
<p>{esc(r.get('summary', ''))}</p>{art}
<div class="grid g2" style="margin-top:18px"><div><h3>Changelog</h3><ul>{li(r.get('changelog', []))}</ul><h3>Known issues</h3><ul>{li(r.get('known_issues', [])) or '<li>None recorded</li>'}</ul></div>
<div><h3>Minimum system requirements</h3><dl class="kv">{req or '<dt>—</dt><dd>unspecified</dd>'}</dl><h3 style="margin-top:14px">Compatibility</h3><p>{esc(r.get('compatibility_notes', 'None recorded'))}</p></div></div></article>"""


def page_downloads(rels):
    latest, oks = latest_downloadable(rels)
    dev, _ = latest_downloadable(rels, ("beta", "development"))
    if latest:
        top = f'<div class="notice info">Latest stable release: <strong>{esc(latest["version"])}</strong>. See its artifacts below.</div>'
    elif dev:
        top = (f'<div class="notice"><strong>There is no stable release yet.</strong> A {esc(dev["channel"])}-channel build '
               f'(<strong>{esc(dev["version"])}</strong>) is available below. It is unsigned, has only been tested on the machine that built it, '
               'and is meant for evaluation. See its known issues first.</div>')
    else:
        top = ('<div class="notice"><strong>No downloadable release exists yet.</strong> The first release is in development. '
               'Nothing on this page can be installed today. To try the current state, run it from source following the '
               '<a href="docs/installation.html">installation guide</a>.</div>')
    sections = []
    for ch in CHANNELS:
        rs = [r for r in rels if r["channel"] == ch]
        if rs:
            sections.append(f'<section><div class="wrap"><h2>{ch.capitalize()} channel</h2>' + "".join(render_release(r) for r in rs) + "</div></section>")
    body = f"""<section class="first"><div class="wrap"><div class="eyebrow">Downloads</div><h1>Releases</h1>
<p class="lead">Newest first. A download button appears only when the build verified that the file exists and matches its recorded SHA-256.</p>{top}
<h3>Verifying a download</h3><pre><code>Get-FileHash .\\Epoch-x.y.z-win-x64.zip -Algorithm SHA256</code></pre>
<p class="muted">Compare the output with the SHA-256 listed for the artifact. Release channels shown: only those with at least one release entry.</p></div></section>{''.join(sections)}"""
    return body


def page_changelog(rels, roadmap):
    ch = "".join(f'<div class="card" style="margin-bottom:14px"><div class="rel-head"><h3>{esc(r["version"])}</h3>{badge(r["status"])}<span class="muted">{esc(r["published"] or "unreleased")}</span></div><ul>'
                 + "".join(f"<li>{esc(x)}</li>" for x in r.get("changelog", [])) + "</ul></div>" for r in rels)
    road = ""
    for g in roadmap:
        rows = "".join(f"<tr><td>{esc(i['name'])}</td><td>{badge(i['status'])}</td><td class='muted'>{esc(i['note'])}</td></tr>" for i in g["items"])
        road += f'<h3 style="margin-top:24px">{esc(g["group"])}</h3><div class="tbl"><table><thead><tr><th>Feature</th><th>Status</th><th>Note</th></tr></thead><tbody>{rows}</tbody></table></div>'
    return f"""<section class="first"><div class="wrap"><div class="eyebrow">Changelog</div><h1>Changelog &amp; roadmap</h1>
<p class="lead">Status vocabulary: <b>in-progress</b> means implemented in source but not released; <b>planned</b> means not started.</p>
<h2>Changelog</h2>{ch}<h2 style="margin-top:40px">Roadmap</h2>{road}</div></section>"""


def md_to_html(md):
    out, lst, code, para = [], None, None, []

    def inl(t):
        t = esc(t)
        t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
        t = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", t)
        t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', t)
        return t

    def fp():
        nonlocal para
        if para:
            out.append("<p>" + inl(" ".join(para)) + "</p>")
            para = []

    def fl():
        nonlocal lst
        if lst:
            out.append(f"</{lst}>")
            lst = None

    for ln in md.splitlines():
        if ln.startswith("```"):
            if code is None:
                fp(); fl(); code = []
            else:
                out.append("<pre><code>" + esc("\n".join(code)) + "</code></pre>")
                code = None
            continue
        if code is not None:
            code.append(ln); continue
        m = re.match(r"^(#{2,3}) (.*)", ln)
        if m:
            fp(); fl(); n = len(m.group(1)); out.append(f"<h{n}>{inl(m.group(2))}</h{n}>"); continue
        m = re.match(r"^(- |\d+\. )(.*)", ln)
        if m:
            fp(); tag = "ul" if m.group(1) == "- " else "ol"
            if lst != tag:
                fl(); out.append(f"<{tag}>"); lst = tag
            out.append(f"<li>{inl(m.group(2))}</li>"); continue
        if not ln.strip():
            fp(); fl(); continue
        if ln.startswith("> "):
            fp(); fl(); out.append(f'<div class="notice">{inl(ln[2:])}</div>'); continue
        para.append(ln.strip())
    fp(); fl()
    return "\n".join(out)


def parse_front(text):
    meta = {}
    if text.startswith("---"):
        _, fm, text = text.split("---", 2)
        for ln in fm.strip().splitlines():
            k, _, v = ln.partition(":")
            meta[k.strip()] = v.strip()
    return meta, text


def build_docs(rels):
    docs = []
    for p in sorted((SRC / "docs").glob("*.md")):
        meta, body = parse_front(p.read_text(encoding="utf-8"))
        docs.append((int(meta.get("order", 99)), p.stem, meta, body))
    docs.sort()
    nav = '<nav class="docs-nav" aria-label="Documentation"><ul>{}</ul></nav>'
    for _, stem, meta, body in docs:
        items = "".join(f'<li><a href="{s}.html"{" aria-current=page" if s == stem else ""}>{esc(m["title"])}</a></li>' for _, s, m, _ in docs)
        status = meta.get("status", "draft")
        page = f"""<div class="wrap"><div class="docs-layout">{nav.format(items)}<article class="prose">
<h1>{esc(meta['title'])}</h1><p>{badge(status)} <span class="muted">{esc(meta.get('statusnote', ''))}</span></p>{md_to_html(body)}</article></div></div>"""
        write(DIST / "docs" / f"{stem}.html", layout(meta["title"] + " — Epoch docs", meta.get("desc", meta["title"]), page, rels, 1, "docs/index.html"))
    return docs


def write(p, s):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8")


def build():
    if DIST.exists():
        shutil.rmtree(DIST)
    DIST.mkdir(parents=True)
    shutil.copytree(SRC / "assets", DIST / "assets")
    rels = load_releases()
    roadmap = json.loads((SRC / "data" / "roadmap.json").read_text(encoding="utf-8"))
    latest, oks = latest_downloadable(rels)
    if latest:
        cta = f'<a class="btn primary" href="downloads.html">Download {esc(latest["version"])}</a>'
        dl_note = f"Latest stable release: {esc(latest['version'])}."
    else:
        dev, _ = latest_downloadable(rels, ("beta", "development"))
        if dev:
            cta = f'<a class="btn primary" href="downloads.html">Get the dev build {esc(dev["version"])}</a>'
            dl_note = "No stable release yet. An unsigned development build for Windows is available for evaluation."
        else:
            cta = '<a class="btn primary" href="downloads.html">Release status</a>'
            dl_note = "No installer has been released yet; the first release is in development."
    # static fragments
    for f in sorted((SRC / "pages").glob("*.html")):
        raw = f.read_text(encoding="utf-8")
        m = re.match(r"<!--title:(.*?)\|desc:(.*?)-->\n", raw)
        title, desc = m.group(1).strip(), m.group(2).strip()
        body = raw[m.end():].replace("{{cta}}", cta).replace("{{dl_note}}", dl_note)
        write(DIST / f.name, layout(title, desc, body, rels, 0, f.name))
    write(DIST / "downloads.html", layout("Downloads — Epoch", "Epoch releases, verified artifacts, checksums and changelogs.", page_downloads(rels), rels, 0, "downloads.html"))
    write(DIST / "changelog.html", layout("Changelog and roadmap — Epoch", "What is implemented, in progress and planned.", page_changelog(rels, roadmap), rels, 0, "changelog.html"))
    docs = build_docs(rels)
    cards = "".join(f'<a class="card" href="{s}.html" style="display:block;color:inherit"><h3>{esc(m["title"])}</h3><p class="muted">{esc(m.get("desc", ""))}</p>{badge(m.get("status", "draft"))}</a>' for _, s, m, _ in docs)
    idx = f'<div class="wrap" style="padding:48px 24px"><div class="eyebrow">Documentation</div><h1>Documentation</h1><div class="notice">Epoch is in early development. Each page is labelled with its status.</div><div class="grid g3">{cards}</div></div>'
    write(DIST / "docs" / "index.html", layout("Documentation — Epoch", "Installation, quick start, training workflow and troubleshooting.", idx, rels, 1, "docs/index.html"))
    write(DIST / "releases.json", (REL_DIR / "releases.json").read_text(encoding="utf-8"))
    n_ok = sum(1 for r in rels for a in r.get("artifacts", []) if a["_ok"])
    print(f"built {len(list(DIST.rglob('*.html')))} pages; releases={len(rels)}; downloadable artifacts={n_ok}")


if __name__ == "__main__":
    build()
