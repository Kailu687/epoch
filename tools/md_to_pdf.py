"""Convert a restricted Markdown subset to a styled PDF with ReportLab.

Usage: python md_to_pdf.py input.md output.pdf

Supported: front matter (title/subtitle/version/date/status), #/##/### headings,
paragraphs, '-' bullets, '1.' lists, pipe tables, fenced code, **bold**, *italic*,
`code`, and `@@diagram <name>` directives for built-in diagrams.
"""
import os
import re
import sys
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, KeepTogether, PageBreak,
                                PageTemplate, Paragraph, Preformatted, Spacer,
                                Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon

FONT_DIR = r"C:\Windows\Fonts"
FONTS = {"ET": "segoeui.ttf", "ET-B": "segoeuib.ttf", "ET-I": "segoeuii.ttf",
         "ET-BI": "segoeuiz.ttf", "ET-M": "consola.ttf"}
BASE, BOLD, ITAL, MONO = "Helvetica", "Helvetica-Bold", "Helvetica-Oblique", "Courier"
if all(os.path.exists(os.path.join(FONT_DIR, f)) for f in FONTS.values()):
    for name, f in FONTS.items():
        pdfmetrics.registerFont(TTFont(name, os.path.join(FONT_DIR, f)))
    pdfmetrics.registerFontFamily("ET", normal="ET", bold="ET-B", italic="ET-I", boldItalic="ET-BI")
    BASE, BOLD, ITAL, MONO = "ET", "ET-B", "ET-I", "ET-M"

ACCENT = colors.HexColor("#1F6FEB")
INK = colors.HexColor("#1B1F24")
MUTED = colors.HexColor("#586069")
LINE = colors.HexColor("#D0D7DE")
HEAD_BG = colors.HexColor("#EAF1FB")

S = {
    "body": ParagraphStyle("body", fontName=BASE, fontSize=9.6, leading=13.6, textColor=INK, spaceAfter=5),
    "h1": ParagraphStyle("h1", fontName=BOLD, fontSize=17, leading=21, textColor=ACCENT, spaceBefore=6, spaceAfter=8),
    "h2": ParagraphStyle("h2", fontName=BOLD, fontSize=12.5, leading=16, textColor=INK, spaceBefore=10, spaceAfter=4),
    "h3": ParagraphStyle("h3", fontName=BOLD, fontSize=10.5, leading=14, textColor=MUTED, spaceBefore=8, spaceAfter=3),
    "bullet": ParagraphStyle("bullet", fontName=BASE, fontSize=9.6, leading=13.4, textColor=INK, leftIndent=14,
                             bulletIndent=3, spaceAfter=2),
    "cell": ParagraphStyle("cell", fontName=BASE, fontSize=8.3, leading=10.8, textColor=INK),
    "cellh": ParagraphStyle("cellh", fontName=BOLD, fontSize=8.3, leading=10.8, textColor=INK),
    "code": ParagraphStyle("code", fontName=MONO, fontSize=7.9, leading=10, textColor=INK, backColor=colors.HexColor("#F6F8FA"),
                           borderColor=LINE, borderWidth=0.5, borderPadding=5, leftIndent=4, spaceBefore=3, spaceAfter=8),
    "title": ParagraphStyle("title", fontName=BOLD, fontSize=30, leading=36, textColor=INK, spaceAfter=10),
    "subtitle": ParagraphStyle("subtitle", fontName=BASE, fontSize=14, leading=19, textColor=MUTED, spaceAfter=30),
    "meta": ParagraphStyle("meta", fontName=BASE, fontSize=10.5, leading=16, textColor=INK),
    "toc1": ParagraphStyle("toc1", fontName=BOLD, fontSize=10, leading=15, leftIndent=0),
    "toc2": ParagraphStyle("toc2", fontName=BASE, fontSize=9, leading=13, leftIndent=14),
}


def inline(text):
    t = escape(text)
    t = re.sub(r"`([^`]+)`", lambda m: f'<font name="{MONO}" size="8.4" color="#8250DF">{m.group(1)}</font>', t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<i>\1</i>", t)
    return t


# ---------------------------------------------------------------- diagrams
def _box(d, x, y, w, h, label, fill="#EAF1FB", stroke="#1F6FEB", size=7.5, bold=False):
    d.add(Rect(x, y, w, h, rx=3, ry=3, fillColor=colors.HexColor(fill), strokeColor=colors.HexColor(stroke), strokeWidth=0.8))
    lines = label.split("\n")
    y0 = y + h / 2 + (len(lines) - 1) * (size + 1) / 2 - size / 3
    for i, ln in enumerate(lines):
        d.add(String(x + w / 2, y0 - i * (size + 1), ln, fontName=BOLD if bold else BASE, fontSize=size,
                     textAnchor="middle", fillColor=INK))


def _arrow(d, x1, y1, x2, y2):
    d.add(Line(x1, y1, x2, y2, strokeColor=MUTED, strokeWidth=0.9))
    import math
    a = math.atan2(y2 - y1, x2 - x1)
    p = 4.5
    pts = [x2, y2, x2 - p * math.cos(a - 0.4), y2 - p * math.sin(a - 0.4), x2 - p * math.cos(a + 0.4), y2 - p * math.sin(a + 0.4)]
    d.add(Polygon(pts, fillColor=MUTED, strokeColor=MUTED))


def diagram_architecture():
    W, H = 480, 330
    d = Drawing(W, H)
    _box(d, 10, 285, 460, 36, "Desktop UI (PySide6) — screens, progress views, expert controls", "#DDF4E4", "#2DA44E", 8.5, True)
    _box(d, 10, 225, 460, 44, "Application Service Layer — workspace, jobs API, event bus (typed events)", "#FFF4D6", "#BF8700", 8.5, True)
    cols = [("Dataset adapters\n+ inspection", 10), ("Validation +\nprofiling", 102), ("Task +\nsplit protocol", 194),
            ("Preprocessing\npipelines", 286), ("Model\nproviders", 378)]
    for lbl, x in cols:
        _box(d, x, 160, 92, 46, lbl)
    for lbl, x in [("Training\norchestrator", 10), ("Resource\nmanager", 102), ("Evaluation", 194), ("Checkpoint\nmanager", 286), ("Plugin\nloader", 378)]:
        _box(d, x, 95, 92, 46, lbl, "#F3E8FF", "#8250DF")
    for lbl, x in [("Experiment DB\n(SQLite)", 10), ("Model registry", 102), ("Export +\ninference", 194), ("Logs +\ndiagnostics", 286), ("Workspace\nfiles", 378)]:
        _box(d, x, 20, 92, 46, lbl, "#F6F8FA", "#586069")
    _arrow(d, 240, 285, 240, 269)
    _arrow(d, 240, 225, 240, 206)
    _arrow(d, 240, 160, 240, 141)
    _arrow(d, 240, 95, 240, 66)
    d.add(String(244, 148, "worker process boundary", fontName=ITAL, fontSize=6.5, fillColor=MUTED))
    return d


def diagram_dataflow():
    W, H = 480, 200
    d = Drawing(W, H)
    steps = ["Import\nfile", "Sniff +\nprofile", "Findings\n(error/warn)", "Task +\nsplit plan", "Split\n(index sets)", "Fit pipeline\non TRAIN", "Train\ncandidates", "Select on\nVALIDATION"]
    bw, bh = 100, 40
    for i, s in enumerate(steps):
        row, col = divmod(i, 4)
        x = 10 + col * 118
        y = 140 - row * 70
        _box(d, x, y, bw, bh, s)
        if col < 3:
            _arrow(d, x + bw, y + bh / 2, x + 118, y + bh / 2)
    _arrow(d, 10 + 3 * 118 + bw / 2, 140, 10 + 3 * 118 + bw / 2, 100)
    d.add(Line(10 + 3 * 118 + bw / 2, 100, 60, 100, strokeColor=MUTED, strokeWidth=0.9))
    _arrow(d, 60, 100, 60, 90)
    _box(d, 10, 8, 460, 26, "Final model → ONE evaluation on sealed TEST set → register → export bundle", "#FFF4D6", "#BF8700", 8.5, True)
    _arrow(d, 10 + 3 * 118 + bw / 2, 70, 10 + 3 * 118 + bw / 2, 34)
    return d


def diagram_wireframe():
    W, H = 480, 220
    d = Drawing(W, H)
    d.add(Rect(5, 5, 470, 210, fillColor=colors.HexColor("#0D1117"), strokeColor=LINE))
    d.add(Rect(5, 190, 470, 25, fillColor=colors.HexColor("#161B22"), strokeColor=LINE))
    d.add(String(14, 198, "Epoch — workspace: Documents/Epoch", fontName=BASE, fontSize=7.5, fillColor=colors.white))
    d.add(Rect(5, 25, 90, 165, fillColor=colors.HexColor("#161B22"), strokeColor=LINE))
    for i, n in enumerate(["Data", "Task", "Models", "Train", "Results", "Registry", "Export", "Settings"]):
        d.add(String(14, 172 - i * 18, ("▸ " if i == 3 else "  ") + n, fontName=BASE, fontSize=7.5,
                     fillColor=colors.HexColor("#58A6FF") if i == 3 else colors.HexColor("#C9D1D9")))
    d.add(String(108, 175, "Active experiment  —  run 2026-10-19-001", fontName=BOLD, fontSize=8.5, fillColor=colors.white))
    rows = [("baseline (majority)", "done", 1.0), ("logistic regression", "done", 1.0), ("random forest", "running", 0.55), ("gradient boosting", "queued", 0.0)]
    for i, (n, st, p) in enumerate(rows):
        y = 148 - i * 22
        d.add(String(108, y + 4, n, fontName=BASE, fontSize=7.5, fillColor=colors.HexColor("#C9D1D9")))
        d.add(Rect(230, y, 150, 9, fillColor=colors.HexColor("#21262D"), strokeColor=None))
        d.add(Rect(230, y, 150 * p, 9, fillColor=colors.HexColor("#2DA44E" if st == "done" else "#1F6FEB"), strokeColor=None))
        d.add(String(388, y + 1, st, fontName=BASE, fontSize=7, fillColor=colors.HexColor("#8B949E")))
    d.add(Rect(108, 35, 360, 40, fillColor=colors.HexColor("#161B22"), strokeColor=LINE))
    d.add(String(116, 58, "CPU 62%   RAM 3.1 GB   GPU: not detected      [ Cancel run ]", fontName=BASE, fontSize=7.5, fillColor=colors.HexColor("#C9D1D9")))
    d.add(String(116, 43, "Illustrative wireframe only — values are placeholders, not measurements.", fontName=ITAL, fontSize=6.5, fillColor=colors.HexColor("#8B949E")))
    return d


DIAGRAMS = {"architecture": diagram_architecture, "dataflow": diagram_dataflow, "wireframe": diagram_wireframe}


# ---------------------------------------------------------------- doc
class Doc(BaseDocTemplate):
    def __init__(self, fn, **kw):
        super().__init__(fn, pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=20 * mm, bottomMargin=20 * mm, **kw)
        fr = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="f")
        self.addPageTemplates([PageTemplate(id="p", frames=[fr], onPage=self._footer)])
        self.short = ""

    def _footer(self, c, d):
        if d.page > 1:
            c.saveState()
            c.setFont(BASE, 8)
            c.setFillColor(MUTED)
            c.drawString(20 * mm, 11 * mm, self.short)
            c.drawRightString(A4[0] - 20 * mm, 11 * mm, f"Page {d.page}")
            c.setStrokeColor(LINE)
            c.line(20 * mm, 15 * mm, A4[0] - 20 * mm, 15 * mm)
            c.restoreState()

    def afterFlowable(self, f):
        if isinstance(f, Paragraph) and f.style.name in ("h1", "h2"):
            lvl = 0 if f.style.name == "h1" else 1
            text = f.getPlainText()
            key = "h%d-%s" % (lvl, self.seq.nextf("toc"))
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, level=lvl, closed=True)
            self.notify("TOCEntry", (lvl, text, self.page, key))


def parse_table(lines, width):
    rows = []
    for ln in lines:
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        rows.append(cells)
    hdr, body = rows[0], [r for r in rows[2:]]
    n = len(hdr)
    body = [r + [""] * (n - len(r)) if len(r) < n else r[:n] for r in body]
    weights = []
    for i in range(n):
        m = max([len(hdr[i])] + [len(r[i]) for r in body])
        weights.append(max(6, min(m, 70)) ** 0.8)
    tot = sum(weights)
    cw = [width * w / tot for w in weights]
    data = [[Paragraph(inline(c), S["cellh"]) for c in hdr]] + [[Paragraph(inline(c), S["cell"]) for c in r] for r in body]
    t = Table(data, colWidths=cw, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), HEAD_BG), ("GRID", (0, 0), (-1, -1), 0.4, LINE),
                           ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 4),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 4), ("TOPPADDING", (0, 0), (-1, -1), 3),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                           ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")])]))
    return [t, Spacer(1, 7)]


def build(md_path, pdf_path, toc=True, paper=False):
    text = open(md_path, encoding="utf-8").read()
    meta = {}
    if text.startswith("---"):
        _, fm, text = text.split("---", 2)
        for ln in fm.strip().splitlines():
            k, _, v = ln.partition(":")
            meta[k.strip()] = v.strip()
    doc = Doc(pdf_path, title=meta.get("title", ""), author="Epoch project")
    doc.short = meta.get("title", "")[:70] + ("…" if len(meta.get("title", "")) > 70 else "")
    if paper:  # journal-style: title block on page 1, sections flow without page breaks, no TOC
        tstyle = ParagraphStyle("ptitle", parent=S["title"], fontSize=17, leading=21, alignment=1, spaceAfter=6)
        sstyle = ParagraphStyle("psub", parent=S["subtitle"], fontSize=10, leading=14, alignment=1, spaceAfter=12)
        story = [Paragraph(escape(meta.get("title", "")), tstyle), Paragraph(escape(meta.get("subtitle", "")), sstyle)]
        toc = False
    else:
        story = [Spacer(1, 60 * mm), Paragraph(escape(meta.get("title", "")), S["title"]),
                 Paragraph(escape(meta.get("subtitle", "")), S["subtitle"])]
        for k in ("version", "date", "status"):
            if k in meta:
                story.append(Paragraph(f"<b>{k.capitalize()}:</b> {escape(meta[k])}", S["meta"]))
        story.append(PageBreak())
    if toc:
        story.append(Paragraph("Table of Contents", S["h1"]))
        t = TableOfContents()
        t.levelStyles = [S["toc1"], S["toc2"]]
        story += [t, PageBreak()]
    lines = text.strip("\n").splitlines()
    i, para, first_h1 = 0, [], True

    def flush():
        nonlocal para
        if para:
            story.append(Paragraph(inline(" ".join(para)), S["body"]))
            para = []

    while i < len(lines):
        ln = lines[i]
        if ln.startswith("```"):
            flush()
            code = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                code.append(lines[i].replace("\t", "    "))
                i += 1
            story.append(Preformatted("\n".join(code), S["code"]))
        elif ln.startswith("@@diagram"):
            flush()
            story.append(KeepTogether([DIAGRAMS[ln.split()[1]](), Spacer(1, 6)]))
        elif ln.startswith("|"):
            flush()
            blk = []
            while i < len(lines) and lines[i].startswith("|"):
                blk.append(lines[i])
                i += 1
            story += parse_table(blk, doc.width)
            continue
        elif ln.startswith("# "):
            flush()
            if not first_h1 and not paper:
                story.append(PageBreak())
            first_h1 = False
            story.append(Paragraph(inline(ln[2:]), S["h1"]))
        elif ln.startswith("## "):
            flush()
            story.append(Paragraph(inline(ln[3:]), S["h2"]))
        elif ln.startswith("### "):
            flush()
            story.append(Paragraph(inline(ln[4:]), S["h3"]))
        elif re.match(r"^- ", ln):
            flush()
            story.append(Paragraph(inline(ln[2:]), S["bullet"], bulletText="•"))
        elif re.match(r"^\d+\. ", ln):
            flush()
            n, rest = ln.split(". ", 1)
            story.append(Paragraph(inline(rest), S["bullet"], bulletText=n + "."))
        elif ln.strip() == "":
            flush()
        else:
            para.append(ln.strip())
        i += 1
    flush()
    doc.multiBuild(story)
    print("wrote", pdf_path)


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2], toc=("--no-toc" not in sys.argv), paper=("--paper" in sys.argv))

