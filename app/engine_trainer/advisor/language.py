"""Lightweight, deterministic reading of a plain-language dataset description.

No model, no network: a weighted keyword/regex scorer (weights play the role of
hand-set IDF values; repeats are damped with ``1 + ln(count)``) plus fuzzy matching
of words against column names. Everything here is a heuristic.
"""
from __future__ import annotations

import difflib
import math
import re
import unicodedata
from dataclasses import dataclass, field

MAX_CHARS = 5000
MAX_TOKENS = 600

_SENT_SPLIT = re.compile(r"[.!?;\n\r]+")
_TOKEN = re.compile(r"[^\W_]+(?:'[^\W_]+)?", re.UNICODE)

ENTITY = (r"(?:patient|speaker|customer|machine|user|person|people|subject|participant|device|client|household|"
          r"student|driver|session|store|sensor|animal|player|employee|member|account|family|visitor|child|children)s?")

# ---- intent lexicons: (regex, weight) -------------------------------------------------
REGRESSION = [
    (r"\bhow much\b", 3), (r"\bhow many\b", 2),
    (r"\b(?:price|prices|pricing|cost|costs|salary|salaries|wage|wages|income|revenue|profit|rent|fare|worth)\b", 2),
    (r"\b(?:estimate|estimating|estimation)\b", 1.5),
    (r"\b(?:number of|total|average|temperature|weight|height|duration|quantity|demand|volume|amount)\b", 1),
    (r"\b(?:continuous|numeric|numerical)\b", 1.5), (r"\bregress\w*\b", 4), (r"\b(?:rmse|mae)\b", 2),
]
CLASSIFICATION = [
    (r"\byes\b.{0,6}\bno\b", 3), (r"\bwhether\b", 2), (r"\bclassif\w*\b", 4), (r"\bcategor\w*\b", 2),
    (r"\b(?:label|labels|labelled|labeled|class|classes)\b", 2), (r"\b(?:type of|kind of|which one)\b", 1.5),
    (r"\b(?:fraud\w*|churn\w*|spam|default\w*|diagnos\w*|disease|cancer|leave|leaving|quit|cancel\w*|attrition|"
     r"surviv\w*|approv\w*|reject\w*|detect\w*|anomal\w*|fail\w*|risk|convert\w*|purchase|subscribe\w*)\b", 2),
    (r"\bwho will\b", 2), (r"\b(?:will|likely|probab\w*|chance)\b", 1), (r"\b(?:binary|multiclass|positive|negative)\b", 1),
]
TEMPORAL = [
    (r"\bforecast\w*\b", 4), (r"\bnext (?:day|week|month|quarter|year|hour|season)s?\b", 3), (r"\btime[- ]series\b", 3),
    (r"\b(?:future|upcoming|going forward|tomorrow|seasonal\w*|chronolog\w*|time[- ]ordered)\b", 2),
    (r"\b(?:over time|trend\w*|historical)\b", 1.5),
]
GROUP = [
    (rf"\bsame {ENTITY}\b", 4),
    (r"\b(?:multiple|several|many|repeated|more than one)\b[^.]{0,25}\b(?:rows?|records?|recordings?|samples?|measurements?|"
     r"visits?|readings?|observations?|entries|transactions?|times)\b", 3),
    (rf"\b(?:rows?|records?|samples?|recordings?|measurements?|visits?|readings?)\b[^.]{{0,25}}\b(?:per|for each|of each|from each|of the same)\b[^.]{{0,15}}\b{ENTITY}\b", 4),
    (rf"\b{ENTITY}\b[^.]{{0,25}}\b(?:appears?|appearing|shows? up|repeats?|repeated|multiple times|many times|several times|more than once|multiple rows)\b", 4),
    (r"\bmore than once\b", 3), (r"\b(?:multiple|several|many) times\b", 2),
]
SPEED = [(r"\b(?:quick\w*|fast|rapid\w*|asap|in a hurry|just try|rough|prototype|draft|few minutes|short on time|speed\w*)\b", 2)]
ACCURACY = [(r"\b(?:best possible|most accurate|as accurate as possible|highest accuracy|top accuracy|maximum accuracy|best model|"
             r"best results?|as good as possible|overnight|no matter how long|take your time|thorough\w*|best you can)\b", 3)]
LIGHT = [(r"\b(?:phone|mobile|android|ios|embedded|microcontroller|arduino|raspberry|esp32|edge device|small device|tiny|"
          r"low[- ]power|low memory|limited memory|small model|lightweight|light weight|wearable|iot)\b", 3)]
INTERP = [(r"\b(?:explain\w*|interpret\w*|transparent\w*|understand why|understand how|justif\w*|regulat\w*|audit\w*|compliance|"
           r"clinician|stakeholders?|show why)\b", 2)]
UNSURE = re.compile(r"\b(?:i )?(?:don't|dont|do not) know\b|\bno idea\b|\bnot sure\b|\bidk\b|\bdunno\b|\bunknown\b|\bunsure\b|"
                    r"\bwhatever\b|\byou decide\b|\blet epoch decide\b|\bsurprise me\b|\bno clue\b")
_NEG = re.compile(r"(?:no need to|no need for|dont need to|don't need to|do not need to|not need to|without|"
                  r"dont care about|don't care about|not interested in|not)\s+(?:\w+\s+){0,2}$")
_MINUTES = re.compile(r"(\d+(?:\.\d+)?)\s*(minutes?|mins?|hours?|hrs?|h)\b")

STOP = frozenset("""about above after again all also and any are because been before being both but can could did does doing done
each few for from further had has have having her here him his how into its just like made make many more most much must
need not now off once only other our out over own same she should some such than that the their them then there these they
this those through too under until very want wants was were what when where which while who whom why will with would you your
data dataset file rows row columns column table records record predict know thing things find help trying""".split())

CUES = ("predict", "forecast", "estimat", "target", "determin", "guess", "anticipat", "detect", "classif", "identif",
        "flag", "know", "tell", "figur", "whether", "find", "label", "decid", "spot", "recogni", "diagnos", "much", "many")
FILLER = frozenset("is are was be what that which i we you want need wanna to am trying would like hope the thing column field "
                   "variable value my our it".split())
GENERIC = frozenset("id num no number code flag value count name date type total score".split())

_SYNONYM_GROUPS = [
    {"price", "cost", "fare", "rent", "worth", "pricing"},
    {"churn", "leave", "leaving", "left", "quit", "cancel", "cancellation", "attrition", "unsubscribe"},
    {"fraud", "fraudulent", "scam", "fake", "cheat"},
    {"salary", "wage", "pay", "income", "earning"},
    {"sale", "revenue", "turnover", "demand"},
    {"disease", "diagnosis", "illness", "sick"},
    {"default", "repay", "repayment"},
    {"spam", "junk"},
    {"survive", "survival", "survived", "alive", "died", "death"},
]


def normalise(text) -> str:
    """Lower-case, strip accents, cap length. Never raises (non-str becomes '')."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = unicodedata.normalize("NFKD", text[:MAX_CHARS])
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.lower().replace("’", "'")


def sentences(text) -> list[list[str]]:
    """Tokenised sentences (list of token lists), capped at MAX_TOKENS tokens in total."""
    out, total = [], 0
    for part in _SENT_SPLIT.split(normalise(text)):
        toks = _TOKEN.findall(part)[: max(0, MAX_TOKENS - total)]
        if toks:
            out.append(toks)
            total += len(toks)
    return out


def stem(w: str) -> str:
    """Tiny plural stemmer (good enough for column-name matching)."""
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def split_name(name: str) -> list[str]:
    """Split a column name into lower-case words (snake, kebab, camelCase, spaces)."""
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(name))
    return [w for w in re.split(r"[^A-Za-z0-9]+", normalise(s)) if w]


def _score(text: str, patterns, negatable: bool = False) -> float:
    total = 0.0
    for pat, w in patterns:
        n = 0
        for m in re.finditer(pat, text):
            if negatable and _NEG.search(text[max(0, m.start() - 35):m.start()]):
                continue
            n += 1
        if n:
            total += w * (1 + math.log(n))
    return total


def parse_minutes(text: str) -> float | None:
    """First 'N minutes/hours' phrase in the text, as minutes."""
    m = _MINUTES.search(normalise(text))
    if not m:
        return None
    v = float(m.group(1))
    return v * 60 if m.group(2).startswith("h") else v


@dataclass
class Mention:
    """A column the description seems to talk about."""
    column: str
    strength: float          # 0..1, how sure the word-to-column match is
    explicit: bool           # described as the thing to predict (e.g. 'predict the price')
    kind: str                # exact | fuzzy | synonym


@dataclass
class LanguageSignals:
    empty: bool = True
    unsure: bool = False
    understood: bool = False
    regression: float = 0.0
    classification: float = 0.0
    temporal: float = 0.0
    group: float = 0.0
    speed: float = 0.0
    accuracy: float = 0.0
    light: float = 0.0
    interpretable: float = 0.0
    minutes: float | None = None
    entities: set = field(default_factory=set)
    mentions: dict = field(default_factory=dict)   # column -> Mention

    @property
    def text_task(self) -> str | None:
        d = self.regression - self.classification
        if d >= 2:
            return "regression"
        if -d >= 2:
            return "classification"
        return None


def _entities(text: str) -> set:
    """Entity nouns (patient, speaker...) named anywhere in a text that talks about repetition."""
    ents = set()
    if not any(re.search(pat, text) for pat, _ in GROUP):
        return ents
    for e in re.finditer(r"\b" + ENTITY + r"\b", text):
        w = e.group(0)
        ents.add(stem({"people": "person", "children": "child"}.get(w, w)))
    return ents


def find_mentions(sents: list[list[str]], columns: list[str], eligible: set | None = None) -> dict:
    """Map column -> Mention by exact, plural, synonym and fuzzy word matching.

    ``eligible`` (columns that could be the target) are the only ones allowed to
    'block' another column from counting as explicit.
    """
    eligible = set(columns) if eligible is None else eligible
    col_words, vocab = {}, {}
    for c in columns:
        ws = [stem(w) for w in split_name(c)]
        col_words[c] = sorted(set(ws))
        for w in set(ws):
            vocab.setdefault(w, set()).add(c)
    if not vocab:
        return {}
    syn_of = {}
    for gi, g in enumerate(_SYNONYM_GROUPS):
        for w in g:
            syn_of[stem(w)] = gi
    syn_vocab = {}
    for w in vocab:
        if w in syn_of:
            syn_vocab.setdefault(syn_of[w], []).append(w)
    vocab_list = sorted(vocab)
    fuzzy_ok = len(vocab_list) <= 1500
    best: dict = {}
    for toks in sents:
        stems = [stem(t) for t in toks]
        hits: dict = {}      # col -> {colword: (pos, kind, strength)}
        for pos, t in enumerate(stems):
            if t.isdigit() or (t in STOP and t not in vocab):
                continue
            matched = []
            if t in vocab:
                matched.append((t, "exact", 1.0))
            else:
                if t in syn_of:
                    matched += [(w, "synonym", 0.7) for w in syn_vocab.get(syn_of[t], [])]
                if not matched and fuzzy_ok and len(t) >= 4:
                    cutoff = 0.9 if len(t) == 4 else 0.8
                    for w in difflib.get_close_matches(t, [v for v in vocab_list if len(v) >= 4 and v[0] == t[0]], n=3, cutoff=cutoff):
                        matched.append((w, "fuzzy", 0.85))
            for w, kind, st in matched:
                for c in vocab[w]:
                    hits.setdefault(c, {}).setdefault(w, (pos, kind, st))
        cands = []   # (col, pos, strength, kind)
        for c, ws in hits.items():
            allw = col_words[c]
            need = max(1, math.ceil(0.5 * len(allw)))
            if len(ws) < need or (len(allw) > 1 and all(w in GENERIC for w in ws)):
                continue
            cov = len(ws) / len(allw)
            st = (sum(v[2] for v in ws.values()) / len(ws)) * (0.6 + 0.4 * cov)
            kinds = {v[1] for v in ws.values()}
            kind = "exact" if kinds == {"exact"} else ("fuzzy" if "fuzzy" in kinds else "synonym")
            cands.append((c, min(v[0] for v in ws.values()), st, kind))
        cue_pos = [i for i, t in enumerate(toks) if t.startswith(CUES)]
        elig_pos = sorted((p, c) for c, p, _, _ in cands if c in eligible)
        first_elig = elig_pos[0] if elig_pos else None
        for c, pos, st, kind in cands:
            explicit = False
            for k in cue_pos:
                if 0 < pos - k <= 7 and not any(k < p < pos and cc != c for p, cc in elig_pos):
                    explicit = True
                    break
            if not explicit and first_elig and first_elig == (pos, c):
                later = [k for k in cue_pos if 0 < k - pos <= 6]
                if later and all(t in FILLER for t in toks[pos + 1:later[0]]):
                    explicit = True
            cur = best.get(c)
            if cur is None or (explicit, st) > (cur.explicit, cur.strength):
                best[c] = Mention(c, round(st, 3), explicit, kind)
    return best


def analyse_description(description, columns: list[str], eligible: set | None = None) -> LanguageSignals:
    """Read a (possibly empty, garbled or non-English) description. Never raises."""
    text = normalise(description)
    sig = LanguageSignals()
    sig.empty = not text.strip()
    if sig.empty:
        return sig
    sig.unsure = bool(UNSURE.search(text))
    sig.regression = _score(text, REGRESSION)
    sig.classification = _score(text, CLASSIFICATION)
    sig.temporal = _score(text, TEMPORAL)
    sig.group = _score(text, GROUP)
    sig.speed = _score(text, SPEED, True)
    sig.accuracy = _score(text, ACCURACY, True)
    sig.light = _score(text, LIGHT, True)
    sig.interpretable = _score(text, INTERP, True)
    sig.minutes = parse_minutes(text)
    sig.entities = _entities(text)
    sig.mentions = find_mentions(sentences(text), columns, eligible)
    sig.understood = bool(sig.mentions) or sig.minutes is not None or any(
        v > 0 for v in (sig.regression, sig.classification, sig.temporal, sig.group, sig.speed, sig.accuracy, sig.light, sig.interpretable))
    return sig
