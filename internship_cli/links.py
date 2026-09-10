"""Outbound-link extraction, redirect resolution, and ATS vs junk classification.

Resolution is HTTP-only (HEAD, then GET) with a per-run cache and a small delay.
A browser tab is not opened here — that is far more expensive than following a
redirect, and LinkedIn volume is already the bottleneck.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


# --------------------------------------------------------------------------
# Config — extend these tuples rather than scattering host checks
# --------------------------------------------------------------------------

# Final hosts that count as a real application destination.
ATS_HOST_SUFFIXES: tuple[str, ...] = (
    "greenhouse.io",
    "boards.greenhouse.io",
    "job-boards.greenhouse.io",
    "lever.co",
    "jobs.lever.co",
    "ashbyhq.com",
    "jobs.ashbyhq.com",
    "myworkdayjobs.com",
    "workday.com",
    "smartrecruiters.com",
    "jobs.smartrecruiters.com",
    "jobvite.com",
    "jobs.jobvite.com",
    "icims.com",
    "wellfound.com",
    "angel.co",  # Wellfound legacy
)

# Chat / community / self-promo aggregators — never treat as an apply link.
JUNK_HOST_SUFFIXES: tuple[str, ...] = (
    "chat.whatsapp.com",
    "wa.me",
    "api.whatsapp.com",
    "whatsapp.com",
    "t.me",
    "telegram.me",
    "telegram.org",
    "discord.gg",
    "discord.com",
    "jobscans.in",
    "linktr.ee",
    "bio.link",
)

# Path tokens on an otherwise-unknown host that still look like careers.
_CAREERS_PATH_RE = re.compile(
    r"/(?:careers?|jobs?|job-openings?|openings?|internships?|join-us|joinus)"
    r"(?:/|$)",
    re.IGNORECASE,
)

_LINKEDIN_JOBS_RE = re.compile(r"/jobs/(?:view|collections)/", re.IGNORECASE)
_LINKEDIN_GROUP_RE = re.compile(r"/groups?/", re.IGNORECASE)
_INVITE_RE = re.compile(r"invite|join.?group|community", re.IGNORECASE)

_BARE_URL_RE = re.compile(
    r"(?:https?://|www\.)[^\s<>\"')\]]+",
    re.IGNORECASE,
)
_LNKD_RE = re.compile(r"(?:https?://)?(?:www\.)?lnkd\.in/[A-Za-z0-9_-]+", re.IGNORECASE)

_USER_AGENT = "internship-radar/0.1 (+local personal use; redirect resolve only)"

LogFn = Callable[[str], None]


def _host(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower().rstrip(".")
    except Exception:  # noqa: BLE001
        return ""


def _host_matches(host: str, suffixes: Sequence[str]) -> bool:
    if not host:
        return False
    for suffix in suffixes:
        suffix = suffix.lower().lstrip(".")
        if host == suffix or host.endswith("." + suffix):
            return True
    return False


def _normalise_href(href: str) -> str:
    href = (href or "").strip()
    if not href or href.startswith("#") or href.lower().startswith("javascript:"):
        return ""
    if href.startswith("//"):
        href = "https:" + href
    elif href.startswith("/"):
        href = "https://www.linkedin.com" + href
    elif href.lower().startswith("www."):
        href = "https://" + href
    return href.split()[0].rstrip(").,;")


def extract_urls_from_text(text: str) -> list[str]:
    """Pull http(s) and lnkd.in URLs out of post body text."""
    if not text:
        return []
    found: list[str] = []
    seen: set[str] = set()
    for match in list(_BARE_URL_RE.finditer(text)) + list(_LNKD_RE.finditer(text)):
        url = _normalise_href(match.group(0))
        if not url:
            continue
        key = url.rstrip("/").lower()
        if key in seen:
            continue
        seen.add(key)
        found.append(url)
    return found


def is_shortener(url: str) -> bool:
    host = _host(url)
    return host in {"lnkd.in", "lnkd.in."} or host.endswith(".lnkd.in") or host == "lnkd.in"


def is_linkedin_internal(url: str) -> bool:
    host = _host(url)
    return host == "linkedin.com" or host.endswith(".linkedin.com")


def is_outbound_or_shortener(url: str) -> bool:
    """Keep apply-relevant hrefs: off-LinkedIn, lnkd.in, or LinkedIn jobs."""
    if not url:
        return False
    if is_shortener(url):
        return True
    if is_linkedin_internal(url):
        path = urlparse(url).path or ""
        return bool(_LINKEDIN_JOBS_RE.search(path) or _LINKEDIN_GROUP_RE.search(path))
    return urlparse(url).scheme in {"http", "https"}


def _company_domain_hit(host: str, company: str) -> bool:
    """True when the host looks like the target company's own site."""
    if not host or not company:
        return False
    from .filters import normalise_company

    tokens = [t for t in normalise_company(company).split() if len(t) >= 3]
    if not tokens:
        slug = re.sub(r"[^a-z0-9]+", "", (company or "").lower())
        tokens = [slug] if len(slug) >= 3 else []
    labels = host.replace("-", "")
    for token in tokens:
        if token in labels:
            return True
    return False


def classify_final_url(url: str, company: str = "") -> str:
    """Return 'careers', 'junk', or 'unknown' for an already-resolved URL."""
    if not url:
        return "unknown"
    host = _host(url)
    path = urlparse(url).path or ""
    blob = f"{host}{path}"

    if _host_matches(host, JUNK_HOST_SUFFIXES):
        return "junk"
    if is_linkedin_internal(url) and _LINKEDIN_GROUP_RE.search(path):
        return "junk"
    if _INVITE_RE.search(blob) and not _CAREERS_PATH_RE.search(path):
        # group-invite lnkd.in destinations, "join my community" landing pages
        if is_linkedin_internal(url) or "community" in blob.lower():
            return "junk"

    if is_linkedin_internal(url) and _LINKEDIN_JOBS_RE.search(path):
        return "careers"
    if _host_matches(host, ATS_HOST_SUFFIXES) and not is_linkedin_internal(url):
        return "careers"
    if _CAREERS_PATH_RE.search(path):
        return "careers"
    if _company_domain_hit(host, company):
        return "careers"
    return "unknown"


@dataclass
class LinkVerdict:
    kind: str  # careers | junk | unknown | none
    urls: list[str] = field(default_factory=list)
    resolved: dict[str, str] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)

    @property
    def reject(self) -> bool:
        return self.kind == "junk"


class RedirectResolver:
    """Follow redirects with a cache and throttle. One instance per scrape run."""

    def __init__(
        self,
        *,
        delay: float = 0.35,
        timeout: float = 8.0,
        log: LogFn = lambda _m: None,
    ) -> None:
        self.delay = max(0.0, delay)
        self.timeout = timeout
        self.log = log
        self._cache: dict[str, str] = {}
        self._last_request_at = 0.0

    def resolve(self, url: str) -> str:
        key = (url or "").strip()
        if not key:
            return ""
        if key in self._cache:
            return self._cache[key]
        if not is_shortener(key) and not _looks_like_redirect_host(key):
            # Direct destinations do not need a round-trip.
            self._cache[key] = key
            return key

        self._throttle()
        final = _http_follow(key, timeout=self.timeout) or key
        self._cache[key] = final
        return final

    def _throttle(self) -> None:
        if self.delay <= 0:
            return
        wait = self.delay - (time.monotonic() - self._last_request_at)
        if wait > 0:
            time.sleep(wait)
        self._last_request_at = time.monotonic()


_REDIRECT_HOSTS = frozenset({
    "lnkd.in", "bit.ly", "t.co", "tinyurl.com", "ow.ly", "buff.ly",
})
_SOCIAL_OR_CHAT_HOST_SUFFIXES = (
    "chat.whatsapp.com", "wa.me", "api.whatsapp.com", "whatsapp.com",
    "t.me", "telegram.me", "telegram.org", "discord.gg", "discord.com",
)
_LINK_HUB_OR_JUNK_HOST_SUFFIXES = ("linktr.ee", "bio.link", "jobscans.in")
_PORTFOLIO_HOST_SUFFIXES = (
    "github.com", "gitlab.com", "bitbucket.org", "behance.net", "dribbble.com",
)


def _looks_like_redirect_host(url: str) -> bool:
    host = _host(url)
    return host in _REDIRECT_HOSTS or host.endswith(".lnkd.in")


def _http_follow(url: str, timeout: float = 8.0) -> str:
    """HEAD first; GET if HEAD is refused. Returns the final URL or ''."""
    headers = {"User-Agent": _USER_AGENT, "Accept": "*/*"}
    for method in ("HEAD", "GET"):
        request = Request(url, method=method, headers=headers)
        try:
            with urlopen(request, timeout=timeout) as response:
                return str(getattr(response, "url", "") or url)
        except HTTPError as exc:
            # Some hosts 405 HEAD; fall through to GET. Other HTTP codes still
            # often expose the redirected URL on the error.
            final = str(getattr(exc, "url", "") or "")
            if final and final != url:
                return final
            if method == "HEAD" and exc.code in {403, 404, 405, 501}:
                continue
            return final or url
        except (URLError, TimeoutError, OSError, ValueError):
            if method == "HEAD":
                continue
            return ""
    return ""


def classify_links(
    urls: Iterable[str],
    *,
    company: str = "",
    resolver: RedirectResolver | None = None,
) -> LinkVerdict:
    """Resolve shorteners and classify the set of outbound links on a post."""
    unique: list[str] = []
    seen: set[str] = set()
    for raw in urls:
        url = _normalise_href(str(raw))
        if not url or not is_outbound_or_shortener(url):
            continue
        key = url.rstrip("/").lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(url)

    if not unique:
        return LinkVerdict(kind="none")

    resolver = resolver or RedirectResolver()
    resolved: dict[str, str] = {}
    kinds: list[str] = []
    reasons: list[str] = []
    for url in unique:
        final = resolver.resolve(url)
        resolved[url] = final
        kind = classify_final_url(final, company=company)
        kinds.append(kind)
        host = _host(final) or _host(url)
        reasons.append(f"link:{kind}:{host}")

    if "careers" in kinds:
        kind = "careers"
    elif "junk" in kinds:
        kind = "junk"
    else:
        kind = "unknown"
    return LinkVerdict(kind=kind, urls=unique, resolved=resolved, reasons=reasons)


def classify_link_categories(
    urls: Iterable[str],
    *,
    company: str = "",
    resolver: RedirectResolver | None = None,
) -> tuple[LinkVerdict, list[str]]:
    """Resolve URLs once and return categories suitable for semantic gating."""
    verdict = classify_links(urls, company=company, resolver=resolver)
    categories: list[str] = []
    for raw in verdict.urls:
        final = verdict.resolved.get(raw, raw)
        host = _host(final) or _host(raw)
        if _host_matches(host, _SOCIAL_OR_CHAT_HOST_SUFFIXES):
            category = "social_or_chat"
        elif _host_matches(host, _LINK_HUB_OR_JUNK_HOST_SUFFIXES):
            category = "link_hub_or_junk"
        elif _host_matches(host, _PORTFOLIO_HOST_SUFFIXES):
            category = "poster_portfolio_or_github"
        elif _company_domain_hit(host, company):
            category = "company_domain"
        elif classify_final_url(final, company=company) == "careers":
            category = "ats_or_careers"
        elif verdict.kind == "junk":
            category = "link_hub_or_junk"
        else:
            category = "unknown"
        if category not in categories:
            categories.append(category)
    return verdict, categories
