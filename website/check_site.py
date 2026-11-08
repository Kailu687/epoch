"""Check built site: internal links/assets resolve and every download link points at a real file."""
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse, unquote

DIST = Path(__file__).resolve().parent / "dist"


class P(HTMLParser):
    def __init__(self):
        super().__init__()
        self.refs, self.ids = [], set()

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if "id" in a:
            self.ids.add(a["id"])
        for k in ("href", "src"):
            if k in a:
                self.refs.append((tag, k, a[k], "download" in a))


def main():
    errs, n, dls = [], 0, 0
    parsed = {}
    for f in DIST.rglob("*.html"):
        p = P(); p.feed(f.read_text(encoding="utf-8")); parsed[f] = p
    for f, p in parsed.items():
        for tag, k, v, is_dl in p.refs:
            u = urlparse(v)
            if u.scheme in ("http", "https", "mailto"):
                continue
            n += 1
            target = f if not u.path else (f.parent / unquote(u.path)).resolve()
            if u.path and not target.exists():
                errs.append(f"{f.relative_to(DIST)}: broken {k}={v}")
            elif u.fragment and target in parsed and u.fragment not in parsed[target].ids:
                errs.append(f"{f.relative_to(DIST)}: missing anchor {v}")
            if is_dl:
                dls += 1
                if not target.is_file():
                    errs.append(f"{f.relative_to(DIST)}: download link to missing file {v}")
        if not re.search(r"<title>.+</title>", f.read_text(encoding="utf-8")):
            errs.append(f"{f.name}: no title")
    print(f"checked {len(parsed)} pages, {n} internal refs, {dls} download links")
    if errs:
        print("\n".join(errs)); sys.exit(1)
    print("OK")


main()
