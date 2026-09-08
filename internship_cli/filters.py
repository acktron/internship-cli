"""Post filtering: recency, hiring intent, funding mentions, company allowlist.

Everything here works on text that LinkedIn actually rendered. Nothing in this
module infers facts that were not present in the post.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

# --------------------------------------------------------------------------
# Relative date parsing
# --------------------------------------------------------------------------

# LinkedIn writes "m" for minutes and "mo" for months, so longer tokens must be
# tried first. Values are whole seconds; dividing once at the end keeps the
# result exact, where multiplying by a precomputed fraction of a day would not.
_SECONDS_PER_DAY = 86_400
_UNIT_SECONDS: dict[str, int] = {
    "second": 1, "sec": 1, "s": 1,
    "minute": 60, "min": 60, "m": 60,
    "hour": 3_600, "hr": 3_600, "h": 3_600,
    "day": 86_400, "d": 86_400,
    "week": 604_800, "w": 604_800,
    "month": 2_630_016, "mo": 2_630_016, "mos": 2_630_016,   # 30.44 days
    "year": 31_557_600, "yr": 31_557_600, "y": 31_557_600,   # 365.25 days
}

# Longest-first so "month" beats "mo" beats "m".
_UNIT_PATTERN = "|".join(sorted(_UNIT_SECONDS, key=len, reverse=True))
_RELATIVE_RE = re.compile(
    rf"(\d+)\s*({_UNIT_PATTERN})s?\b(?:\s+ago)?",
    re.IGNORECASE,
)

_NOW_WORDS = ("just now", "just posted", "now", "today")


def parse_relative_date(text: str) -> float | None:
    """Convert a LinkedIn relative timestamp to an age in days.

    Handles "2d", "3 weeks ago", "1mo", "5 hours ago", "yesterday", "just now".
    Returns None when nothing timestamp-like is present.
    """
    if not text:
        return None

    lowered = text.strip().lower()

    if any(word in lowered for word in _NOW_WORDS):
        # "today" and friends still lose to an explicit "3d" elsewhere in the string.
        if not _RELATIVE_RE.search(lowered):
            return 0.0
    if "yesterday" in lowered:
        return 1.0

    match = _RELATIVE_RE.search(lowered)
    if not match:
        return None

    amount = int(match.group(1))
    seconds = _UNIT_SECONDS.get(match.group(2).lower())
    if seconds is None:
        return None
    return amount * seconds / _SECONDS_PER_DAY


def parse_datetime_attr(value: str) -> float | None:
    """Age in days from an ISO timestamp (a `<time datetime=...>` attribute)."""
    if not value:
        return None
    raw = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    delta = datetime.now(timezone.utc) - parsed
    return max(0.0, delta.total_seconds() / 86_400)


def linkedin_date_filter(max_age_days: float) -> str | None:
    """Closest LinkedIn `datePosted` bucket that does NOT cut inside the window.

    LinkedIn only offers 24h / week / month. Anything between 7 and 31 days must
    use `past-month` and be trimmed client-side, otherwise posts the user asked
    to keep get dropped by the server.
    """
    if max_age_days <= 1:
        return "past-24h"
    if max_age_days <= 7:
        return "past-week"
    if max_age_days <= 31:
        return "past-month"
    return None


def humanise_age(days: float | None) -> str:
    if days is None:
        return "unknown"
    if days < 1 / 24:
        return f"{int(days * 1440)}m ago"
    if days < 1:
        return f"{int(days * 24)}h ago"
    if days < 7:
        return f"{int(days)}d ago"
    if days < 30:
        return f"{int(days / 7)}w ago"
    return f"{int(days / 30.44)}mo ago"


# --------------------------------------------------------------------------
# Hiring intent matching
# --------------------------------------------------------------------------

# Three independent signals. --match all keeps a post only when ALL three fire
# in the post text. --match loose keeps hiring + (role OR domain).
DEFAULT_HIRING_TERMS = (
    "hiring", "we're hiring", "we are hiring", "now hiring", "looking for",
    "seeking", "recruiting", "join our team", "join us", "apply now", "apply",
    "open role", "open position", "opening", "openings", "vacancy",
    "drop your resume", "share your resume", "dm me", "send your cv",
    "applications open", "we're looking", "we are looking",
)

DEFAULT_ROLE_TERMS = (
    "intern", "interns", "internship", "internships", "trainee", "apprentice",
)

DEFAULT_DOMAIN_TERMS = (
    "ai", "a.i.", "ml", "ai/ml", "machine learning", "artificial intelligence",
    "deep learning", "nlp", "natural language processing", "llm", "llms",
    "computer vision", "genai", "gen ai", "generative ai", "data science",
    "data scientist", "data engineer", "software", "software engineer",
    "software engineering", "backend", "frontend", "full stack", "fullstack",
    "neural network", "transformers", "pytorch", "tensorflow",
)

MATCH_MODES = ("all", "loose")


def _term_regex(term: str) -> re.Pattern[str]:
    """Word-boundary matcher.

    Short tokens are the whole reason this exists: a naive substring test makes
    "ml" match "html" and "ai" match "said" or "trained".
    """
    escaped = re.escape(term.strip().lower())
    # Allow flexible whitespace inside multi-word terms.
    escaped = escaped.replace(r"\ ", r"\s+")
    return re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE)


@dataclass(frozen=True)
class MatchResult:
    matched: bool
    reasons: list[str]

    def why(self) -> str:
        return "; ".join(self.reasons)


class HiringMatcher:
    """Decides whether a post is advertising an AI/ML-ish internship.

    Modes (CLI: --match):
      all   — keep only when ALL three signals appear in the post text:
                (a) hiring intent  — hiring / looking for / apply …
                (b) internship     — intern / internship / trainee …
                (c) domain         — ai / ml / data science / software …
      loose — keep when hiring intent is present AND (role OR domain).

    A university career-fair post or an HR internship that only has (a)+(b)
    fails under --match all. That is intentional.
    """

    def __init__(
        self,
        hiring_terms: Sequence[str] = DEFAULT_HIRING_TERMS,
        role_terms: Sequence[str] = DEFAULT_ROLE_TERMS,
        domain_terms: Sequence[str] = DEFAULT_DOMAIN_TERMS,
        mode: str = "all",
    ) -> None:
        if mode not in MATCH_MODES:
            raise ValueError(f"Unknown match mode {mode!r}; choose one of {MATCH_MODES}")
        self.mode = mode
        self._groups = {
            "hiring": [(t, _term_regex(t)) for t in hiring_terms if t.strip()],
            "role": [(t, _term_regex(t)) for t in role_terms if t.strip()],
            "domain": [(t, _term_regex(t)) for t in domain_terms if t.strip()],
        }

    def _hits(self, group: str, text: str) -> list[str]:
        return [term for term, pattern in self._groups[group] if pattern.search(text)]

    def match(self, text: str) -> MatchResult:
        if not text:
            return MatchResult(False, ["no post text"])

        hiring = self._hits("hiring", text)
        role = self._hits("role", text)
        domain = self._hits("domain", text)

        # Explicit branches — never fall through to a weaker rule.
        if self.mode == "all":
            matched = bool(hiring) and bool(role) and bool(domain)
        elif self.mode == "loose":
            matched = bool(hiring) and bool(role or domain)
        else:  # pragma: no cover - guarded in __init__
            raise ValueError(f"Unknown match mode {self.mode!r}")

        reasons: list[str] = []
        for label, hits in (("hiring", hiring), ("role", role), ("domain", domain)):
            if hits:
                # Comma-joined, since terms such as "ai/ml" already contain a slash.
                reasons.append(f"{label}:{', '.join(hits[:3])}")
            else:
                reasons.append(f"{label}:none")

        return MatchResult(matched, reasons)


# --------------------------------------------------------------------------
# Funding mentions quoted from the post itself
# --------------------------------------------------------------------------

# These extract what the POST SAYS. They never assert anything about a company
# that the author did not write down.
_FUNDING_PATTERNS = (
    re.compile(r"\bpre[-\s]?seed\b", re.IGNORECASE),
    re.compile(r"\bseed[-\s]?(?:funded|stage|round)\b", re.IGNORECASE),
    re.compile(r"\bseries\s+[a-j]\b", re.IGNORECASE),
    re.compile(r"\by[-\s]?combinator\b", re.IGNORECASE),
    re.compile(r"\byc\s?[swf]\d{2}\b", re.IGNORECASE),
    re.compile(r"\b(?:vc|venture)[-\s]?backed\b", re.IGNORECASE),
    # Capitalised tokens only, so this captures an investor name and stops at the
    # sentence boundary instead of running on into the next clause.
    re.compile(r"\bbacked by\s+[A-Z][\w&]*(?:[\s,]+(?:and\s+)?[A-Z][\w&]*){0,3}"),
    re.compile(r"\braised\s+(?:\$|usd|inr|₹|rs\.?)\s?[\d.,]+\s?(?:k|m|mn|bn|cr|crore|million|billion)?\b",
               re.IGNORECASE),
    re.compile(r"\bfunded\s+startup\b", re.IGNORECASE),
    re.compile(r"\b(?:unicorn|soonicorn)\b", re.IGNORECASE),
)


def extract_funding_mentions(text: str) -> list[str]:
    """Return funding phrases verbatim from the post text.

    Patterns overlap by design ("pre-seed stage" satisfies both the pre-seed and
    the seed-stage rule), so overlapping spans are resolved to a single phrase —
    earliest start wins, then longest — rather than reported twice.
    """
    if not text:
        return []

    spans: list[tuple[int, int, str]] = []
    for pattern in _FUNDING_PATTERNS:
        for match in pattern.finditer(text):
            phrase = re.sub(r"\s+", " ", match.group(0)).strip(" .,;:")
            if phrase:
                spans.append((match.start(), match.end(), phrase))

    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))

    found: list[str] = []
    seen_lower: set[str] = set()
    last_end = -1
    for start, end, phrase in spans:
        if start < last_end:
            continue
        if phrase.lower() in seen_lower:
            continue
        found.append(phrase)
        seen_lower.add(phrase.lower())
        last_end = end
    return found


# --------------------------------------------------------------------------
# Company allowlist
# --------------------------------------------------------------------------

_NORMALISE_RE = re.compile(r"[^a-z0-9]+")
_COMPANY_SUFFIXES = (
    "inc", "llc", "ltd", "limited", "pvt", "private", "plc", "corp",
    "corporation", "co", "company", "technologies", "technology", "tech",
    "labs", "lab", "solutions", "systems", "software", "services", "global",
    "india", "group", "holdings", "ventures", "studio", "studios", "ai",
)


def normalise_company(name: str) -> str:
    """Fold a company name for comparison: lowercase, strip punctuation/suffixes."""
    if not name:
        return ""
    folded = _NORMALISE_RE.sub(" ", name.lower()).strip()
    tokens = [t for t in folded.split() if t]
    while len(tokens) > 1 and tokens[-1] in _COMPANY_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


def load_allowlist(path: str | Path) -> set[str]:
    """Read a company allowlist. One name per line; `#` comments allowed.

    Also accepts a CSV export: the first column of each row is used.
    """
    file_path = Path(path).expanduser()
    if not file_path.exists():
        raise FileNotFoundError(f"Allowlist not found: {file_path}")

    names: set[str] = set()
    for line in file_path.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        if "," in entry:
            entry = entry.split(",", 1)[0].strip().strip('"')
        normalised = normalise_company(entry)
        if normalised:
            names.add(normalised)
    return names


def company_in_allowlist(company: str, allowlist: Iterable[str]) -> bool:
    """True when the company matches an allowlist entry.

    Matches on the normalised name, plus containment either way so that
    "Acme" on the list still matches a scraped "Acme Technologies India".
    """
    target = normalise_company(company)
    if not target:
        return False
    for entry in allowlist:
        if not entry:
            continue
        if target == entry:
            return True
        if f" {entry} " in f" {target} " or f" {target} " in f" {entry} ":
            return True
    return False


def load_companies(path: str | Path) -> list[str]:
    """Read target companies for company-posts. One name per line; `#` comments.

    Preserves original spelling/order (needed for LinkedIn keyword queries).
    A CSV export also works — the first column of each row is used.
    """
    file_path = Path(path).expanduser()
    if not file_path.exists():
        raise FileNotFoundError(f"Companies file not found: {file_path}")

    companies: list[str] = []
    seen: set[str] = set()
    for line in file_path.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        if "," in entry:
            entry = entry.split(",", 1)[0].strip().strip('"')
        if not entry:
            continue
        key = normalise_company(entry) or entry.lower()
        if key in seen:
            continue
        seen.add(key)
        companies.append(entry)
    return companies


def text_mentions_company(text: str, company: str) -> bool:
    """True when `company` appears in `text` (word-boundary, suffix-tolerant)."""
    if not text or not company:
        return False

    if _term_regex(company).search(text):
        return True

    core = normalise_company(company)
    if core and core != company.strip().lower():
        if _term_regex(core).search(text):
            return True

    if core:
        folded = _NORMALISE_RE.sub(" ", text.lower()).strip()
        if f" {core} " in f" {folded} ":
            return True
    return False


# --------------------------------------------------------------------------
# Company-posts STRONG match (proximity-aware)
# --------------------------------------------------------------------------

# Cues that the post is offering a role, not narrating / advertising a product.
# Deliberately omits bare "hiring" (matches "campus hiring", "Hiring Managers").
# Prefer "we're hiring" / "hiring:" / apply / role: / DM / looking for …
DEFAULT_JOB_OFFER_CUES = (
    "we're hiring", "we are hiring", "now hiring", "hiring:",
    "apply now", "apply", "applications open",
    "open role", "open position", "role:", "position",
    "join our team", "join us",
    "dm me", "dm", "send your cv", "send your resume",
    "drop your resume", "share your resume",
    "looking for", "we are looking", "we're looking",
)

# Hiring + intern must sit in the same sentence or within this many words.
_PROXIMITY_WORDS = 15

# "looking for" / "seeking" only count as open-role language when the intern
# term comes AFTER them ("looking for an AI intern"), not before
# ("internship module. Looking for learners").
_FORWARD_HIRING_TERMS = frozenset({
    "looking for", "seeking", "we're looking", "we are looking",
    "recruiting",
})

# Internship mentions that are narrative / program copy, not an open role.
_INCIDENTAL_INTERN_RE = re.compile(
    r"(?:"
    r"\bas an intern\b"
    r"|\bmy (?:first|second|third|\d+(?:st|nd|rd|th))? ?(?:real )?task as an intern\b"
    r"|\b(?:my|our|college|campus)\s+internship\b"
    r"|\binternship\s+(?:program|module|course|coordinator|journey|story|diary|experience)\b"
    r"|\bintern(?:s)?\s+(?:program|module|course)\b"
    r"|\bstudent intern\b"
    r"|\bformer intern\b"
    r"|\bex-?intern\b"
    r")",
    re.IGNORECASE,
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD_SPAN_RE = re.compile(r"\w+(?:['’./]\w+)*", re.UNICODE)


def _term_spans(text: str, terms: Sequence[str]) -> list[tuple[int, int, str]]:
    """Character spans `(start, end, term)` for every term hit in `text`."""
    spans: list[tuple[int, int, str]] = []
    # Longer terms first so "we're hiring" wins over bare "hiring" when both match.
    ordered = sorted((t for t in terms if t and t.strip()), key=len, reverse=True)
    for term in ordered:
        for match in _term_regex(term).finditer(text):
            spans.append((match.start(), match.end(), term))
    return spans


def _incidental_intern_spans(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in _INCIDENTAL_INTERN_RE.finditer(text or "")]


def _span_inside(span: tuple[int, int], covers: Sequence[tuple[int, int]]) -> bool:
    start, end = span
    for c_start, c_end in covers:
        if c_start <= start and end <= c_end:
            return True
        # Role token overlaps an incidental phrase (e.g. "intern" inside "as an intern").
        if start < c_end and end > c_start:
            return True
    return False


def _role_spans_open(text: str, role_spans: Sequence[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    """Drop internship hits that only appear in narrative / program phrasing."""
    incidental = _incidental_intern_spans(text)
    if not incidental:
        return list(role_spans)
    return [
        span for span in role_spans
        if not _span_inside((span[0], span[1]), incidental)
    ]


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    if not text:
        return []
    spans: list[tuple[int, int]] = []
    start = 0
    for match in _SENTENCE_SPLIT_RE.finditer(text):
        end = match.start()
        if end > start and text[start:end].strip():
            spans.append((start, end))
        start = match.end()
    if start < len(text) and text[start:].strip():
        spans.append((start, len(text)))
    return spans or [(0, len(text))]


def _same_sentence(text: str, a: tuple[int, int], b: tuple[int, int]) -> bool:
    for sent_start, sent_end in _sentence_spans(text):
        if sent_start <= a[0] < sent_end and sent_start <= b[0] < sent_end:
            return True
    return False


def _word_distance(text: str, a: tuple[int, int], b: tuple[int, int]) -> int:
    """Words strictly between two character spans (0 if overlapping/adjacent)."""
    if a[1] <= b[0]:
        gap = text[a[1]:b[0]]
    elif b[1] <= a[0]:
        gap = text[b[1]:a[0]]
    else:
        return 0
    return len(_WORD_SPAN_RE.findall(gap))


def _near(
    text: str,
    a: tuple[int, int],
    b: tuple[int, int],
    max_words: int = _PROXIMITY_WORDS,
) -> bool:
    return _same_sentence(text, a, b) or _word_distance(text, a, b) <= max_words


def _hire_role_near(
    text: str,
    hiring_spans: Sequence[tuple[int, int, str]],
    role_spans: Sequence[tuple[int, int, str]],
    max_words: int = _PROXIMITY_WORDS,
) -> tuple[str, str] | None:
    """Hiring + intern co-occurring as an open-role phrase."""
    for h_start, h_end, h_term in hiring_spans:
        for r_start, r_end, r_term in role_spans:
            if not _near(text, (h_start, h_end), (r_start, r_end), max_words=max_words):
                continue
            # "looking for motivated learners" after "internship module" is not a hire.
            if h_term.lower() in _FORWARD_HIRING_TERMS and r_start < h_start:
                continue
            return h_term, r_term
    return None


def _cue_near_role(
    text: str,
    cue_spans: Sequence[tuple[int, int, str]],
    role_box: tuple[int, int],
    max_words: int,
) -> list[tuple[int, int, str]]:
    near: list[tuple[int, int, str]] = []
    for c_start, c_end, cue in cue_spans:
        if not _near(text, role_box, (c_start, c_end), max_words=max_words):
            continue
        if cue.lower() in _FORWARD_HIRING_TERMS and role_box[0] < c_start:
            continue
        near.append((c_start, c_end, cue))
    return near


def match_strong_company(
    text: str,
    company: str,
    matcher: HiringMatcher | None = None,
    *,
    company_text: str | None = None,
    job_offer_cues: Sequence[str] = DEFAULT_JOB_OFFER_CUES,
    proximity_words: int = _PROXIMITY_WORDS,
) -> MatchResult:
    """STRONG match for company-posts — bag-of-words is not enough.

    Requires ALL of:
      1. company mentioned (in `company_text` if given, else `text`)
      2. a hiring term AND an internship term in the same sentence or within
         `proximity_words` of each other (an open-role phrase, not scattered hits)
      3. a job-offer cue (we're hiring / apply / role: / DM / …) near the
         internship term, AND a domain term near that same internship term
      4. domain = AI/ML/data/software

    Narrative intern stories, newsletters, and "internship program" course ads
    are rejected even when the keywords appear somewhere in the post.

    Proximity always runs on the post body (`text`). `company_text` only widens
    where the company name may appear (poster / company fields).
    """
    empty = MatchResult(
        False,
        ["company:none", "hiring:none", "role:none", "domain:none", "proximity:none"],
    )
    if not (text or "").strip():
        return empty

    matcher = matcher or HiringMatcher(mode="all")
    mention_haystack = company_text if company_text is not None else text
    mentioned = text_mentions_company(mention_haystack, company)

    hiring_terms = [t for t, _ in matcher._groups["hiring"]]
    role_terms = [t for t, _ in matcher._groups["role"]]
    domain_terms = [t for t, _ in matcher._groups["domain"]]

    hiring_spans = _term_spans(text, hiring_terms)
    role_spans_all = _term_spans(text, role_terms)
    role_spans = _role_spans_open(text, role_spans_all)
    domain_spans = _term_spans(text, domain_terms)
    cue_spans = _term_spans(text, job_offer_cues)

    hire_role = _hire_role_near(
        text, hiring_spans, role_spans, max_words=proximity_words,
    )

    # Job-offer cue + domain must both sit near the SAME open internship mention.
    offer_role_domain: tuple[str, str, str] | None = None
    for r_start, r_end, role_term in role_spans:
        role_box = (r_start, r_end)
        near_cues = _cue_near_role(text, cue_spans, role_box, proximity_words)
        near_domains = [
            (d_start, d_end, dom)
            for d_start, d_end, dom in domain_spans
            if _near(text, role_box, (d_start, d_end), max_words=proximity_words)
        ]
        if near_cues and near_domains:
            offer_role_domain = (near_cues[0][2], role_term, near_domains[0][2])
            break

    matched = bool(mentioned and hire_role and offer_role_domain)

    reasons: list[str] = [
        f"company:{company}" if mentioned else "company:none",
    ]
    if hire_role:
        reasons.append(f"hiring:{hire_role[0]}")
        reasons.append(f"role:{hire_role[1]}")
    else:
        # Still surface bag hits so drops are diagnosable.
        bag_hiring = sorted({t for _, _, t in hiring_spans})[:3]
        bag_role = sorted({t for _, _, t in role_spans_all})[:3]
        reasons.append(f"hiring:{', '.join(bag_hiring)}" if bag_hiring else "hiring:none")
        reasons.append(f"role:{', '.join(bag_role)}" if bag_role else "role:none")

    if offer_role_domain:
        reasons.append(f"domain:{offer_role_domain[2]}")
        reasons.append(
            f"proximity:hire+intern≤{proximity_words}w; "
            f"offer:{offer_role_domain[0]}~{offer_role_domain[1]}~{offer_role_domain[2]}"
        )
    else:
        bag_domain = sorted({t for _, _, t in domain_spans})[:3]
        reasons.append(f"domain:{', '.join(bag_domain)}" if bag_domain else "domain:none")
        if not role_spans and role_spans_all:
            reasons.append("proximity:intern only in narrative/program phrasing")
        elif hire_role:
            reasons.append("proximity:no job-offer cue near intern+domain")
        elif hiring_spans and role_spans_all:
            reasons.append(
                f"proximity:hiring/intern >{proximity_words}w apart (or different sentences)"
            )
        else:
            reasons.append("proximity:none")

    return MatchResult(matched, reasons)


# --------------------------------------------------------------------------
# Source legitimacy (mention vs first-party posting)
# --------------------------------------------------------------------------

_HASHTAG_RE = re.compile(r"(?:^|\s)#\w+")
_DM_REFERRAL_RE = re.compile(
    r"\b(?:dm|message|ping)\s+me\b.{0,80}\b(?:referral|refer)\b"
    r"|\b(?:referral|refer)\b.{0,80}\b(?:dm|message)\s+me\b",
    re.IGNORECASE | re.DOTALL,
)
_COMMUNITY_INVITE_RE = re.compile(
    r"\bjoin\s+(?:my|our)\s+(?:whatsapp|telegram|discord|community)\b"
    r"|\bwhatsapp\s+group\b|\btelegram\s+(?:group|channel)\b",
    re.IGNORECASE,
)
_GENERIC_DM_HIRE_RE = re.compile(
    r"\bi['’]m hiring\b.{0,80}\bdm me\b|\bdm me\b.{0,80}\b(?:hiring|intern)",
    re.IGNORECASE | re.DOTALL,
)
_MULTI_ROLE_DUMP_RE = re.compile(
    r"(?:(?:^|\n)\s*(?:\d+[\).]|[-•])\s*.{0,60}(?:intern|opening|hiring).*){3,}",
    re.IGNORECASE,
)
_CREATOR_FARMER_RE = re.compile(
    r"\b(?:referral farmer|job aggregator|daily job alert|jobscan)\b"
    r"|\bfollow\s+for\s+(?:more\s+)?(?:jobs|openings)\b",
    re.IGNORECASE,
)


def count_hashtags(text: str) -> int:
    return len(_HASHTAG_RE.findall(text or ""))


def source_spam_signals(
    text: str,
    *,
    poster: str = "",
    company: str = "",
    is_company_page: bool = False,
    link_kind: str = "none",
) -> list[str]:
    """Cheap REJECT cues for aggregator / farmer / invite-only posts."""
    signals: list[str] = []
    body = text or ""
    tags = count_hashtags(body)
    if tags > 8:
        signals.append(f"hashtags:{tags}")
    if _DM_REFERRAL_RE.search(body):
        signals.append("dm-for-referral")
    if link_kind == "junk":
        signals.append("group-invite")
    elif _COMMUNITY_INVITE_RE.search(body):
        signals.append("group-invite")
    if _GENERIC_DM_HIRE_RE.search(body) and link_kind not in {"careers"}:
        signals.append("generic-dm-hire")
    if _MULTI_ROLE_DUMP_RE.search(body):
        signals.append("multi-role-dump")
    if _CREATOR_FARMER_RE.search(body):
        signals.append("aggregator-copy")
    if (
        not is_company_page
        and poster
        and company
        and poster.strip().lower() != company.strip().lower()
        and link_kind not in {"careers"}
        and _GENERIC_DM_HIRE_RE.search(body)
    ):
        signals.append("not-hiring-company")
    return signals


def heuristic_source_reject(
    text: str,
    *,
    poster: str = "",
    company: str = "",
    is_company_page: bool = False,
    link_kind: str = "none",
) -> tuple[bool, str]:
    """Hard-drop obvious junk without an LLM. Return (reject, reason).

    Careers/ATS links and a LinkedIn company-page actor are NOT enough to keep
    a post — aggregators wrap those all the time.
    """
    del is_company_page  # affiliation is decided by classify_first_party
    signals = source_spam_signals(
        text,
        poster=poster,
        company=company,
        is_company_page=False,
        link_kind=link_kind,
    )
    if link_kind == "junk":
        return True, "junk-link:" + (",".join(signals) if signals else "invite-or-chat")
    # DM-for-referral / generic "DM me" are apply-path cues, not automatic junk.
    hard_prefixes = {
        "hashtags",
        "group-invite",
        "multi-role-dump",
        "aggregator-copy",
    }
    hit = [s for s in signals if s.split(":")[0] in hard_prefixes]
    if hit:
        return True, "source-spam:" + ",".join(hit)
    return False, ""


# --------------------------------------------------------------------------
# Opening gate (real role, not poster identity)
# --------------------------------------------------------------------------

KEEP_SOURCE_TYPES = frozenset({"company", "employee", "recruiter", "creator"})
FIRST_PARTY_SOURCE_TYPES = KEEP_SOURCE_TYPES  # backward-compatible alias
SOURCE_TYPES = (
    "company", "employee", "recruiter", "creator", "aggregator", "spam",
)
HIGH_CRED_FOLLOWERS = 10_000

_AGGREGATOR_POSTER_RE = re.compile(
    r"edutech|edu[\s\-]?tech|job.?alert|jobs?\s+daily|hiring\s+alert|"
    r"job\s*board|career\s*(?:radar|digest|update)|frontlines|jobright|"
    r"jobscan|daily\s+jobs|jobs\s+in\s+the|vacancy\s*alert|fresher\s*jobs|"
    r"cybersecurity jobs|hiring opportunities",
    re.IGNORECASE,
)
_COMMUNITY_FARM_RE = re.compile(
    r"\bjoin my community\b|\bwhatsapp (?:group|community)\b",
    re.IGNORECASE,
)
_CREATOR_BAIT_RE = re.compile(
    r"most juniors ask|internship kah[aā] se|"
    r"your internship search shouldn|internship calendar|"
    r"save this post|found something you should save",
    re.IGNORECASE,
)
_EXPERIENCED_ROLE_RE = re.compile(
    r"\b(?:[2-9]|\d{2})\+?\s*years?\s+(?:of\s+)?experience\b",
    re.IGNORECASE,
)
_MULTI_COMPANY_RE = re.compile(
    r"\b(?:ibm|google|amazon|meta|apple|infosys|tcs|wipro|oracle|salesforce)"
    r"\b.{0,50}\b(?:and|&|\/)\b.{0,50}\b(?:microsoft|google|amazon|ibm)\b"
    r"|\b(?:ibm|google|amazon).{0,30}(?:microsoft|are hiring)",
    re.IGNORECASE,
)
_STYLIZED_RE = re.compile(r"[\U0001D400-\U0001D7FF]{12,}")
_RECRUITER_HEADLINE_RE = re.compile(
    r"\b(?:recruiter|talent acquisition|\bta\b|hiring manager|sourcer|"
    r"university recruiter|campus recruiter)\b",
    re.IGNORECASE,
)
_BODY_COMPANY_FIELD_RE = re.compile(
    r"(?:^|\n)\s*Company:\s*([^\n|]+)",
    re.IGNORECASE,
)


def company_field_from_post(body: str) -> str:
    """Extract a ``Company: …`` line from structured job-post copy."""
    if not body:
        return ""
    match = _BODY_COMPANY_FIELD_RE.search(body)
    return match.group(1).strip() if match else ""


_URL_IN_TEXT_RE = re.compile(r"https?://|\blnkd\.in/|\bwww\.", re.IGNORECASE)
_APPLY_PATH_RE = re.compile(
    r"\bapply\s+(?:now|here|at|via|on)?\b|\bapplications?\s+open\b"
    r"|\bcareers?(?:\s+page)?\b"
    r"|\b(?:dm|message|ping|comment)\b.{0,60}\b(?:referral|refer|cv|resume|apply|link)\b"
    r"|\b(?:referral|refer)\b.{0,60}\b(?:dm|comment|message|ping)\b"
    r"|\blink\s+in\s+(?:the\s+)?(?:bio|comments?)\b"
    r"|\bcomment\s+(?:for|to)\s+(?:a\s+)?referral\b",
    re.IGNORECASE | re.DOTALL,
)
_SPECIFIC_ROLE_RE = re.compile(
    r"\b(?:software(?:\s+engineering)?|swe|sde|ai|a\.i\.|ml|"
    r"machine learning|data(?:\s+scien(?:ce|tist))?|backend|frontend|"
    r"full[\s-]?stack|research|security|cybersecurity|devops|"
    r"engineering|developer|analyst|scientist|product)\b"
    r".{0,48}\b(?:intern|internship|interns)\b"
    r"|"
    r"\b(?:intern|internship|interns)\b.{0,48}\b"
    r"(?:software|swe|sde|ai|ml|machine learning|data|backend|frontend|"
    r"full[\s-]?stack|research|security|engineering|developer|analyst)\b"
    r"|"
    r"\brole:\s*\S[^\n]{2,80}"
    r"|"
    r"\b(?:intern|internship)\s+(?:opening|position|role)s?\b",
    re.IGNORECASE | re.DOTALL,
)
_FOLLOWERS_WITH_WORD_RE = re.compile(
    r"(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*([kKmM])?\+?\s+"
    r"(?:linkedin\s+)?followers?\b",
    re.IGNORECASE,
)
_FOLLOWERS_K_LINKEDIN_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*([kKmM])\+?(?:\s*\+)?(?:\s*linkedin)\b",
    re.IGNORECASE,
)
_VERIFIED_RE = re.compile(
    r"\bverified\b|\bverification badge\b",
    re.IGNORECASE,
)


def parse_follower_count(text: str) -> int | None:
    """Best-effort follower count from a headline/profile line ("31k+ LinkedIn")."""
    if not text:
        return None
    match = _FOLLOWERS_WITH_WORD_RE.search(text) or _FOLLOWERS_K_LINKEDIN_RE.search(text)
    if not match:
        return None
    raw, suffix = match.group(1), (match.group(2) or "")
    try:
        amount = float(raw.replace(",", ""))
    except ValueError:
        return None
    factor = {"": 1, "k": 1_000, "K": 1_000, "m": 1_000_000, "M": 1_000_000}[suffix]
    return int(amount * factor)


def format_follower_count(count: int) -> str:
    if count >= 1_000_000:
        value = count / 1_000_000
        return f"{int(value)}m" if value == int(value) else f"{value:.1f}m"
    if count >= 1_000:
        value = count / 1_000
        return f"{int(value)}k" if value == int(value) else f"{value:.1f}k"
    return str(count)


def poster_is_verified(text: str) -> bool:
    return bool(_VERIFIED_RE.search(text or ""))


def opening_audit_reasons(
    *,
    poster: str = "",
    headline: str = "",
    source_type: str = "",
    reason: str = "",
    text: str = "",
) -> list[str]:
    """Tokens written into why_matched so a run can be audited."""
    hay = f"{poster}\n{headline}\n{text}"
    followers = parse_follower_count(hay)
    tokens = [f"poster:{poster or '(unknown)'}"]
    if followers is not None:
        tokens.append(f"followers:{format_follower_count(followers)}")
    if poster_is_verified(hay):
        tokens.append("verified:yes")
    if source_type:
        tokens.append(
            f"source:{source_type} ({reason})" if reason else f"source:{source_type}"
        )
    elif reason:
        tokens.append(reason)
    return tokens


def _poster_is_target(poster: str, target: str) -> bool:
    if not poster or not target:
        return False
    if company_in_allowlist(poster, {normalise_company(target)}):
        return True
    if company_in_allowlist(target, {normalise_company(poster)}):
        return True
    return False


def _has_specific_role(body: str) -> bool:
    return bool(_SPECIFIC_ROLE_RE.search(body or ""))


def _has_apply_path(body: str, apply_links: Sequence[str] = ()) -> bool:
    if any(str(url).strip() for url in apply_links):
        return True
    return bool(_URL_IN_TEXT_RE.search(body or "") or _APPLY_PATH_RE.search(body or ""))


def _hiring_org_is_target(
    target: str,
    *,
    poster: str,
    headline: str,
    body: str,
) -> tuple[bool, str]:
    """Whether the target is the organisation actually hiring."""
    if not target:
        return False, "no target company"
    listed = _BODY_COMPANY_FIELD_RE.search(body)
    if listed:
        listed_name = listed.group(1).strip()
        if listed_name and (
            _poster_is_target(listed_name, target)
            or text_mentions_company(listed_name, target)
        ):
            return True, f"Company:{listed_name}"
        if listed_name:
            return False, f"hiring company in post is {listed_name!r}, not {target!r}"
    if _poster_is_target(poster, target):
        return True, "poster is the target company page"
    company_re = _term_regex(target)
    hiring_subj = re.compile(
        rf"(?:{company_re.pattern})\s+is\s+hiring"
        rf"|(?:intern(?:ship)?|role|opening|position).{{0,40}}(?:at|with|for)\s+(?:{company_re.pattern})"
        rf"|join\s+(?:{company_re.pattern})"
        rf"|(?:{company_re.pattern}).{{0,50}}(?:intern(?:ship)?)",
        re.IGNORECASE | re.DOTALL,
    )
    if company_re.search(body) and hiring_subj.search(body):
        return True, "target named as hiring org"
    if text_mentions_company(headline, target):
        return True, "headline employer is the target"
    if text_mentions_company(body, target):
        return False, "target mentioned but not as the hiring org"
    return False, "target not named as the hiring org"


def _infer_source_type(
    target: str,
    *,
    poster: str,
    headline: str,
    body: str,
) -> str:
    hay = f"{poster}\n{headline}\n{body}"
    if _AGGREGATOR_POSTER_RE.search(poster) or _AGGREGATOR_POSTER_RE.search(headline):
        return "aggregator"
    if _COMMUNITY_FARM_RE.search(hay) and not _has_specific_role(body):
        return "spam"
    if target and _poster_is_target(poster, target):
        return "company"
    if target and text_mentions_company(headline, target):
        if _RECRUITER_HEADLINE_RE.search(headline):
            return "recruiter"
        return "employee"
    if (
        _AGGREGATOR_POSTER_RE.search(body[:400])
        or _MULTI_COMPANY_RE.search(body)
        or _MULTI_ROLE_DUMP_RE.search(body)
    ):
        return "aggregator"
    if poster:
        return "creator"
    return "spam"


def classify_first_party(
    target_company: str,
    *,
    poster: str = "",
    headline: str = "",
    text: str = "",
    apply_links: Sequence[str] = (),
    strict: bool = True,
) -> tuple[str, bool, str]:
    """Return (source_type, keep, reason) for a company-posts candidate.

    Keep when the post is a concrete opening (specific role + target as hiring
    org + apply path). Poster affiliation is a scored signal, not a filter:
    company, employee, recruiter, and creator all pass.
    """
    target = (target_company or "").strip()
    poster = (poster or "").strip()
    headline = (headline or "").strip()
    body = text or ""
    hay = f"{poster}\n{headline}\n{body}"
    source_type = _infer_source_type(
        target, poster=poster, headline=headline, body=body
    )
    followers = parse_follower_count(hay)
    verified = poster_is_verified(hay)
    high_cred = verified or (
        followers is not None and followers >= HIGH_CRED_FOLLOWERS
    )

    if source_type == "aggregator":
        reason = (
            f"aggregator account ({poster or 'unknown'})"
            if _AGGREGATOR_POSTER_RE.search(poster)
            or _AGGREGATOR_POSTER_RE.search(headline)
            else "job-aggregator / multi-company roundup"
        )
        if not strict:
            return "aggregator", True, "loose mode: company mentioned, poster unverified"
        return "aggregator", False, reason
    if source_type == "spam" and _COMMUNITY_FARM_RE.search(hay):
        return "spam", False, "community / paid-service spam, no real role"

    if _EXPERIENCED_ROLE_RE.search(body):
        return source_type or "spam", False, "experienced role, not an internship opening"

    hiring_ok, hiring_why = _hiring_org_is_target(
        target, poster=poster, headline=headline, body=body
    )
    has_role = _has_specific_role(body)
    has_apply = _has_apply_path(body, apply_links)
    bait = bool(_CREATOR_BAIT_RE.search(body) or _STYLIZED_RE.search(body))

    if not hiring_ok:
        return source_type, False, f"off-topic: {hiring_why}"
    if bait and not (has_role and has_apply):
        return "creator", False, "no concrete opening (engagement-bait / no role+apply path)"
    if not has_role or not has_apply:
        signals = sum((has_role, hiring_ok, has_apply))
        if high_cred and signals >= 2 and source_type in KEEP_SOURCE_TYPES:
            cred = (
                f"{format_follower_count(followers)} followers"
                if followers is not None
                else "verified"
            )
            return (
                source_type if source_type in KEEP_SOURCE_TYPES else "creator",
                True,
                f"borderline opening rescued by poster credibility ({cred})",
            )
        missing = []
        if not has_role:
            missing.append("specific role")
        if not has_apply:
            missing.append("apply path")
        return (
            source_type,
            False,
            "no concrete opening (missing " + " + ".join(missing) + ")",
        )

    if source_type not in KEEP_SOURCE_TYPES:
        if not strict:
            return source_type, True, "loose mode: concrete opening"
        return source_type, False, f"rejected source_type {source_type}"

    return source_type, True, f"real opening ({hiring_why}; role+apply path)"
