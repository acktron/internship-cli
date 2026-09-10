"""Scrape LinkedIn content (post) search results.

Unlike the jobs tab, content search is an infinite-scroll feed, so pagination is
"scroll and let more load" rather than a `start=` offset. Post bodies are also
truncated behind a "…see more" toggle, which must be expanded before any text
matching, or the hiring filter reads half a sentence.
"""

from __future__ import annotations

import re
import sys
import time
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, quote, unquote, urlencode, urlparse, urlunparse

from .debug import dump_debug
from .gate import gate_company_posts_semantic
from .filters import (
    HiringMatcher,
    author_affiliation_matches_target,
    classify_first_party,
    company_field_from_post,
    extract_funding_mentions,
    heuristic_source_reject,
    humanise_age,
    linkedin_date_filter,
    match_company_candidate_net,
    match_strong_company,
    opening_audit_reasons,
    parse_datetime_attr,
    parse_follower_count,
    parse_relative_date,
    poster_is_verified,
    poster_employer_from_headline,
)
from .links import (
    RedirectResolver,
    classify_link_categories,
    extract_urls_from_text,
    is_shortener,
)
from .output import make_snippet
from .scraper import _clean_link, _dedupe_text
from .llm import LLMClient

CONTENT_SEARCH_URL = "https://www.linkedin.com/search/results/content/"

# LinkedIn's content search now renders through server-driven UI
# (data-sdui-screen="...search.SearchResultsContent"). Every CSS class on that
# page is a hashed build artefact such as "_1e5f23a7", so class selectors are
# worthless there and the legacy ones below are kept only for older surfaces.
# The SDUI hooks — role, componentkey, data-testid, view-name — are the anchors.
_POST_SELECTORS = (
    "[role='listitem'][componentkey*='FLAGSHIP_SEARCH']",
    "[role='listitem']:has([data-testid='expandable-text-box'])",
    "[role='listitem'][componentkey*='SearchResult']",
    "[role='listitem'][componentkey*='Update']",
    "[data-view-name='feed-full-update']",
    "[data-view-name='search-entity-result']",
    "[data-finite-scroll-hotkey-item]",
    "div.feed-shared-update-v2",
    "div.fie-impression-container",
    "li.reusable-search__result-container",
    "div[data-urn*='activity']",
    "div.update-components-actor",
)

# Empty / blocked search states — distinct from "selectors drifted".
_EMPTY_RESULT_SELECTORS = (
    "text=/No results found/i",
    "text=/No matching results/i",
    "text=/We didn’t find any results/i",
    "text=/We didn't find any results/i",
    "[data-testid='search-no-results']",
    ".search-no-results",
    ".search-reusable-search-no-results",
)

_POSTER_SELECTORS = (
    "span.update-components-actor__title span[aria-hidden='true']",
    ".update-components-actor__title",
    ".update-components-actor__name",
    ".feed-shared-actor__name",
)

_SUBDESC_SELECTORS = (
    ".update-components-actor__sub-description span[aria-hidden='true']",
    ".update-components-actor__sub-description",
    ".feed-shared-actor__sub-description",
)

_TEXT_SELECTORS = (
    "[data-testid='expandable-text-box']",
    ".update-components-text",
    ".feed-shared-inline-show-more-text",
    ".update-components-update-v2__commentary",
    ".feed-shared-update-v2__description",
)

_SEE_MORE_SELECTORS = (
    "[data-testid='expandable-text-button']",
    "button.feed-shared-inline-show-more-text__see-more-less-toggle",
    "button.inline-show-more-text__button",
    ".feed-shared-inline-show-more-text button",
    "button.see-more",
    "button[aria-label*='see more' i]",
    "[role='button'][aria-label*='see more' i]",
    "span.feed-shared-inline-show-more-text__see-more-less-toggle",
    "button:has-text('see more')",
    "button:has-text('…more')",
    "button:has-text('...more')",
)

_SEE_MORE_LABEL_RE = re.compile(
    r"^\s*(?:…|\.{2,3})?\s*(?:see\s+)?more\s*$",
    re.IGNORECASE,
)
_TRUNCATED_TAIL_RE = re.compile(
    r"(?:…|\.{3})\s*(?:more)?\s*$",
    re.IGNORECASE,
)

_EXPAND_CARD_JS = r"""
el => {
  const isMore = (node) => {
    const t = (node.innerText || node.textContent || '').replace(/\s+/g, ' ').trim();
    const aria = (node.getAttribute('aria-label') || '').toLowerCase();
    const testid = (node.getAttribute('data-testid') || '');
    if (testid === 'expandable-text-button') return true;
    if (aria.includes('see more')) return true;
    if (/^(…|\.\.\.)?see more$/i.test(t)) return true;
    if (/^(…|\.\.\.)more$/i.test(t)) return true;
    return false;
  };
  let n = 0;
  for (const node of el.querySelectorAll('button, [role="button"], span, a')) {
    if (!isMore(node)) continue;
    try { node.click(); n += 1; } catch (e) {}
  }
  return n;
}
"""

_ENTITY_SELECTORS = (
    ".update-components-entity",
    ".update-components-job-card",
    ".update-components-article",
    ".feed-shared-mini-job",
)

_ENTITY_TITLE_SELECTORS = (
    ".update-components-entity__title span[aria-hidden='true']",
    ".update-components-entity__title",
    ".update-components-article__title",
)

_ENTITY_SUBTITLE_SELECTORS = (
    ".update-components-entity__subtitle span[aria-hidden='true']",
    ".update-components-entity__subtitle",
    ".update-components-article__subtitle",
)

_ENTITY_CAPTION_SELECTORS = (
    ".update-components-entity__secondary-subtitle span[aria-hidden='true']",
    ".update-components-entity__secondary-subtitle",
    ".update-components-entity__caption",
)

_SCROLL_CONTAINER_SELECTORS = (
    "[data-testid='lazy-column']",
    "[data-component-type='LazyColumn']",
    "div.scaffold-finite-scroll",
)

_SHOW_MORE_RESULTS_SELECTORS = (
    "button.scaffold-finite-scroll__load-button",
    "button:has-text('Show more results')",
)

# The poster's name is only reliably available in this aria-label.
_POSTER_MENU_RE = re.compile(r"Open control menu for post by\s+(.+)", re.IGNORECASE)

# "Architect @ TransUnion | AI Data Cloud" -> "TransUnion"
_HEADLINE_COMPANY_RE = re.compile(
    r"(?:@|\bat\b)\s+([A-Z][\w&.\-]*(?:\s+[A-Z][\w&.\-]*){0,3})"
)

_ACTIVITY_RE = re.compile(r"urn:li:activity:(\d+)", re.IGNORECASE)

# Every candidate list, so --debug can report which ones still match anything.
SELECTOR_GROUPS = {
    "post containers": _POST_SELECTORS,
    "poster name": _POSTER_SELECTORS,
    "timestamp / sub-description": _SUBDESC_SELECTORS,
    "post text": _TEXT_SELECTORS,
    "see-more toggle": _SEE_MORE_SELECTORS,
    "attached entity card": _ENTITY_SELECTORS,
    "entity title": _ENTITY_TITLE_SELECTORS,
    "entity subtitle": _ENTITY_SUBTITLE_SELECTORS,
}


@dataclass
class Post:
    poster: str = ""
    company: str = ""
    role: str = ""
    location: str = ""
    post_date: str = ""
    job_link: str = ""
    why_matched: str = ""
    funding_signal: str = ""
    company_class: str = ""
    post_url: str = ""
    post_text: str = ""
    activity_urn: str = ""
    outbound_links: list[str] = field(default_factory=list)
    link_class: str = ""
    link_categories: list[str] = field(default_factory=list)
    source_type: str = ""
    poster_headline: str = ""
    follower_count: int | None = None
    poster_verified: bool = False
    engagement: dict[str, int] = field(default_factory=dict)
    days_old: float | None = None
    _reasons: list[str] = field(default_factory=list, repr=False)
    # Used by --resolve-links to reopen the poster's feed when the search card
    # click does not navigate to an exact permalink.
    _actor_profile_url: str = field(default="", repr=False)
    _card_index: int = field(default=-1, repr=False)


def build_content_url(
    query: str,
    max_age_days: float = 14,
    sort_by_date: bool = True,
    location: str = "",
) -> str:
    """Build the content-search URL, including LinkedIn's own date filter."""
    params: list[tuple[str, str]] = [("keywords", query)]

    if location:
        # Content search has no true location facet; folding it into the keywords
        # is the only lever available, and it is a soft one.
        params[0] = ("keywords", f"{query} {location}".strip())

    bucket = linkedin_date_filter(max_age_days)
    query_string = urlencode(params)
    if bucket:
        query_string += "&datePosted=" + quote(f'"{bucket}"', safe="")
    if sort_by_date:
        query_string += "&sortBy=" + quote('"date_posted"', safe="")

    return f"{CONTENT_SEARCH_URL}?{query_string}"


def _first_text(scope, selectors) -> str:
    for selector in selectors:
        try:
            node = scope.locator(selector).first
            if node.count() == 0:
                continue
            text = _dedupe_text(node.inner_text(timeout=1_500))
            if text:
                return text
        except Exception:  # noqa: BLE001 - selector drift; try the next candidate
            continue
    return ""


def is_see_more_label(text: str) -> bool:
    """True for LinkedIn 'see more' / '…more' toggle labels."""
    return bool(_SEE_MORE_LABEL_RE.match((text or "").replace("\xa0", " ").strip()))


def looks_truncated(text: str) -> bool:
    """True when the body still ends in an ellipsis / '…more' clamp."""
    return bool(_TRUNCATED_TAIL_RE.search((text or "").rstrip()))


def _expand_card_see_more(card, page=None) -> int:
    """Click every see-more toggle *inside this card*, then settle."""
    expanded = 0
    try:
        expanded = int(card.evaluate(_EXPAND_CARD_JS) or 0)
    except Exception:  # noqa: BLE001
        expanded = 0

    for selector in _SEE_MORE_SELECTORS:
        try:
            buttons = card.locator(selector)
            count = buttons.count()
        except Exception:  # noqa: BLE001
            continue
        for index in range(count):
            button = buttons.nth(index)
            try:
                button.click(timeout=900, force=True)
                expanded += 1
            except Exception:  # noqa: BLE001
                continue

    if expanded:
        try:
            if page is not None:
                page.wait_for_timeout(220)
            else:
                time.sleep(0.12)
        except Exception:  # noqa: BLE001
            pass
    return expanded


def expanded_post_text(card, page=None) -> str:
    """Expand '…more' on this card, then re-query the body (longest wins)."""
    _expand_card_see_more(card, page)
    return _post_body_text(card)


def _expand_see_more(page, log: Callable[[str], None]) -> int:
    """Click every "…see more" toggle so post bodies are complete."""
    expanded = 0
    for selector in _SEE_MORE_SELECTORS:
        try:
            buttons = page.locator(selector)
            count = buttons.count()
        except Exception:  # noqa: BLE001
            continue
        for index in range(count):
            button = buttons.nth(index)
            try:
                button.click(timeout=1_200)
                expanded += 1
                page.wait_for_timeout(120)
                continue
            except Exception:  # noqa: BLE001 - fall through to a forced click
                pass
            try:
                # The SDUI toggle sets pointer-events:none and aria-hidden, so a
                # normal click is never actionable; force past the check.
                button.click(timeout=1_200, force=True)
                expanded += 1
                page.wait_for_timeout(120)
            except Exception:  # noqa: BLE001 - a stale/hidden toggle is not fatal
                continue
    if expanded:
        log(f"expanded {expanded} truncated post(s)")
    return expanded


def _wait_for_posts(page, timeout_ms: int, log: Callable[[str], None]) -> bool:
    """Block until at least one post container exists, or the timeout expires."""
    combined = ", ".join(_POST_SELECTORS)
    try:
        page.wait_for_selector(combined, state="attached", timeout=timeout_ms)
        return True
    except Exception:  # noqa: BLE001
        log(f"no post container appeared within {timeout_ms / 1000:.0f}s")
        return False


def _scroll_container(page):
    for selector in _SCROLL_CONTAINER_SELECTORS:
        try:
            node = page.locator(selector).first
            if node.count():
                return node
        except Exception:  # noqa: BLE001
            continue
    return None


def _scroll_once(page, pause: float) -> None:
    """One infinite-scroll nudge + optional 'show more results' click."""
    container = _scroll_container(page)
    scrolled = False
    if container is not None:
        try:
            container.evaluate(
                "el => el.scrollBy(0, (el.clientHeight || 800) * 2)"
            )
            scrolled = True
        except Exception:  # noqa: BLE001
            scrolled = False
    if not scrolled:
        try:
            page.mouse.wheel(0, 2_400)
        except Exception:  # noqa: BLE001
            pass
    page.wait_for_timeout(int(pause * 1000))

    # Lazily-loaded results arrive over the network, so settle before the
    # next scroll rather than racing ahead of the response.
    try:
        page.wait_for_load_state("networkidle", timeout=5_000)
    except Exception:  # noqa: BLE001
        pass

    for selector in _SHOW_MORE_RESULTS_SELECTORS:
        try:
            button = page.locator(selector).first
            if button.count() and button.is_visible(timeout=500):
                button.click(timeout=2_000)
                page.wait_for_timeout(int(pause * 1000))
                break
        except Exception:  # noqa: BLE001
            continue


def _load_more(page, scrolls: int, pause: float, log: Callable[[str], None]) -> None:
    """Drive the infinite-scroll feed to materialise more results."""
    for round_index in range(max(0, scrolls)):
        _scroll_once(page, pause)
        log(f"scroll {round_index + 1}/{scrolls}")


def _peek_post_ages(page) -> list[float]:
    """Ages (days) of currently loaded cards that have a parseable timestamp."""
    cards = _find_posts(page)
    if cards is None:
        return []
    ages: list[float] = []
    for index in range(cards.count()):
        card = cards.nth(index)
        try:
            stamp, header = _header_parts(card)
            sub = stamp or _first_text(card, _SUBDESC_SELECTORS) or header
            age = _post_age_days(card, sub)
        except Exception:  # noqa: BLE001
            continue
        if age is not None:
            ages.append(age)
    return ages


def _load_more_until_age(
    page,
    max_age_days: float,
    max_scrolls: int,
    pause: float,
    log: Callable[[str], None],
) -> None:
    """Scroll a date-sorted feed until a post older than `max_age_days` appears.

    Stops early when the oldest dated card exceeds the window, when scrolling
    stops adding cards, or when `max_scrolls` is hit.
    """
    prev_count = 0
    stagnant = 0
    limit = max(1, max_scrolls)

    for round_index in range(limit):
        ages = _peek_post_ages(page)
        cards = _find_posts(page)
        count = cards.count() if cards is not None else 0

        if ages:
            newest, oldest = min(ages), max(ages)
            log(
                f"loaded {count} card(s); newest {humanise_age(newest)}, "
                f"oldest {humanise_age(oldest)}"
            )
            if oldest > max_age_days:
                log(
                    f"oldest post exceeds --max-age-days ({max_age_days:g}d); "
                    "stopping scroll"
                )
                return
        else:
            log(f"loaded {count} card(s); no parseable ages yet")

        if count > 0 and count == prev_count:
            stagnant += 1
            if stagnant >= 2:
                log("no new posts after scrolling; stopping")
                return
        else:
            stagnant = 0
        prev_count = count

        _scroll_once(page, pause)
        log(f"scroll {round_index + 1}/{limit} (until age > {max_age_days:g}d)")

    ages = _peek_post_ages(page)
    if ages and max(ages) <= max_age_days:
        log(
            f"hit max scrolls ({limit}) before covering {max_age_days:g}d "
            f"(oldest still {humanise_age(max(ages))}); raise --scrolls"
        )


# Tags the outermost element carrying an activity-like URN. Class names get
# renamed in every redesign; the URN attribute is the one anchor that has
# survived them, so this is the fallback when no known selector matches.
_TAG_POSTS_JS = r"""
() => {
  const urnAttrs = [
    'data-urn', 'data-id', 'data-entity-urn', 'data-activity-urn',
    'data-chameleon-result-urn', 'data-finite-scroll-hotkey-item'
  ];
  const hasUrn = (el) => urnAttrs.some(a => {
    const v = el.getAttribute(a);
    return v && /activity|ugcPost|share/i.test(v);
  });
  const looksLikeCard = (el) => {
    if (!el || el.nodeType !== 1) return false;
    const ck = (el.getAttribute('componentkey') || el.getAttribute('componentKey') || '');
    const view = el.getAttribute('data-view-name') || '';
    if (/FLAGSHIP_SEARCH|SearchResult|Update|feed-full-update|search-entity/i.test(ck + ' ' + view)) {
      return true;
    }
    if (el.getAttribute('role') === 'listitem' && el.querySelector('[data-testid="expandable-text-box"]')) {
      return true;
    }
    return hasUrn(el);
  };

  document.querySelectorAll('[data-ir-post]').forEach(el => el.removeAttribute('data-ir-post'));

  let tagged = 0;
  for (const el of Array.from(document.querySelectorAll('*')).filter(looksLikeCard)) {
    // Only tag the outermost carrier, or nested URNs would double-count.
    let ancestor = el.parentElement, nested = false;
    while (ancestor) {
      if (looksLikeCard(ancestor)) { nested = true; break; }
      ancestor = ancestor.parentElement;
    }
    if (!nested) { el.setAttribute('data-ir-post', '1'); tagged++; }
  }
  return tagged;
}
"""


def _page_reports_no_results(page) -> bool:
    """True when LinkedIn shows an empty-results state (not selector drift)."""
    for selector in _EMPTY_RESULT_SELECTORS:
        try:
            if page.locator(selector).first.count() > 0:
                return True
        except Exception:  # noqa: BLE001
            continue
    return False


def _find_posts(page, log: Callable[[str], None] = lambda _m: None):
    best = None
    best_count = 0
    for selector in _POST_SELECTORS:
        try:
            locator = page.locator(selector)
            count = locator.count()
        except Exception:  # noqa: BLE001
            continue
        if count > best_count:
            best, best_count = locator, count

    if best is not None and best_count > 0:
        return best

    try:
        tagged = page.evaluate(_TAG_POSTS_JS)
    except Exception as exc:  # noqa: BLE001
        log(f"structural fallback failed: {exc}")
        return None

    if tagged:
        log(f"no known selector matched; found {tagged} post(s) by structure instead")
        return page.locator("[data-ir-post='1']")
    return None


# Patterns that identify a SINGLE post, as opposed to a company/profile feed.
_EXACT_POST_PATH_RE = re.compile(
    r"(?:/feed/update/|/posts/[^/?#]+(?:-activity-|_)|/pulse/)",
    re.IGNORECASE,
)
_COMPANY_FEED_RE = re.compile(r"/company/[^/]+/posts/?$", re.IGNORECASE)
_PROFILE_FEED_RE = re.compile(r"/in/[^/]+/(?:posts|recent-activity)/?$", re.IGNORECASE)
_ACTIVITY_QUERY_RE = re.compile(
    r"(?:urn:li:activity:|urn%3Ali%3Aactivity%3A)(\d+)",
    re.IGNORECASE,
)
_EMBED_URN_RE = re.compile(r"urn:li:(share|activity|ugcPost):(\d+)", re.IGNORECASE)
_POST_SLUG_URN_RE = re.compile(r"(activity|ugcPost|share)[_-](\d+)", re.IGNORECASE)

def _plausible_urn_id(urn_id: str) -> bool:
    """True for real 19-digit LinkedIn post urn ids."""
    digits = re.fullmatch(r"(\d+)", (urn_id or "").strip())
    if not digits:
        return False
    d = digits.group(1)
    return len(d) == 19 and not d.startswith("1500")


def _plausible_share_id(share_id: str) -> bool:
    return _plausible_urn_id(share_id)


def permalink_from_embed_blob(blob: str) -> str:
    """Build /feed/update/… from an embed iframe src or dialog textarea."""
    if not blob:
        return ""
    decoded = unquote(blob.replace("&amp;", "&"))
    match = _EMBED_URN_RE.search(decoded)
    if not match:
        return ""
    kind, urn_id = match.group(1), match.group(2)
    if not _plausible_urn_id(urn_id):
        return ""
    if kind.lower() == "ugcpost":
        kind = "ugcPost"
    return f"https://www.linkedin.com/feed/update/urn:li:{kind}:{urn_id}/"


def canonical_post_permalink(url: str) -> str:
    """Normalize copy-link / lnkd.in / posts slug URLs to /feed/update/urn:li:…"""
    if not url:
        return ""
    decoded = unquote((url or "").strip().split()[0].replace("&amp;", "&"))
    if _is_search_results_url(decoded):
        return ""
    from_embed = permalink_from_embed_blob(decoded)
    if from_embed:
        return from_embed
    slug = _POST_SLUG_URN_RE.search(decoded)
    if slug:
        kind_raw = slug.group(1).lower()
        kind = {"ugcpost": "ugcPost", "share": "share", "activity": "activity"}[kind_raw]
        urn_id = slug.group(2)
        if _plausible_urn_id(urn_id):
            return f"https://www.linkedin.com/feed/update/urn:li:{kind}:{urn_id}/"
    if "/feed/update/" in decoded and _is_exact_post_url(decoded):
        parsed = urlparse(decoded)
        return urlunparse(("https", "www.linkedin.com", parsed.path.rstrip("/") + "/", "", "", ""))
    if "/posts/" in decoded:
        parsed = urlparse(decoded)
        if _is_exact_post_url(decoded):
            return urlunparse(("https", "www.linkedin.com", parsed.path.rstrip("/") + "/", "", "", ""))
    return ""


def _plausible_activity_id(activity_id: str) -> bool:
    """True only for real 19-digit activity ids (reject SDUI proto leftovers)."""
    return _plausible_urn_id(activity_id)


def extract_activity_id_from_href(href: str) -> str:
    """Activity id from a post permalink anchor href only."""
    if not href:
        return ""
    decoded = unquote(href)
    match = _ACTIVITY_RE.search(decoded) or _ACTIVITY_QUERY_RE.search(decoded)
    if not match:
        match = re.search(r"activity[_-](\d+)", decoded, re.I)
    if match and _plausible_activity_id(match.group(1)):
        return match.group(1)
    return ""


def extract_activity_id_from_html(html: str) -> str:
    """First activity id from an <a href=…> in card markup (tests / fixtures)."""
    if not html:
        return ""
    for href in re.findall(r"""<a[^>]+href=["']([^"']+)["']""", html, re.I):
        found = extract_activity_id_from_href(href)
        if found:
            return found
    return ""


# Timestamp ("3h") and author-title anchors carry the real post permalink.
_POST_ANCHOR_HREFS_JS = r"""
el => {
  const out = [];
  const seen = new Set();
  const isPostHref = (h) => {
    if (!h) return false;
    const u = h.toLowerCase();
    if (u.includes('/search/results/')) return false;
    if (u.includes('/feed/update/') || u.includes('urn:li:activity')) return true;
    if (u.includes('/posts/') && u.includes('activity')) return true;
    return false;
  };
  const isProfileHref = (h) => {
    if (!h) return false;
    const u = h.toLowerCase();
    return u.includes('/in/') || /\/company\/[^/]+\/?$/.test(u.split('?')[0]);
  };
  const add = (h) => {
    h = (h || '').trim();
    if (!h || seen.has(h)) return;
    seen.add(h);
    out.push(h);
  };

  const body = el.querySelector('[data-testid="expandable-text-box"]');
  const inHeader = (node) => body ? !body.contains(node) : true;

  const subSelectors = [
    '.update-components-actor__sub-description',
    '.feed-shared-actor__sub-description',
    '.update-components-actor__meta',
  ];
  for (const sel of subSelectors) {
    for (const node of el.querySelectorAll(sel)) {
      for (const a of node.querySelectorAll('a[href]')) {
        if (inHeader(a)) add(a.getAttribute('href'));
      }
    }
  }

  const timeEl = el.querySelector('time');
  if (timeEl) {
    const link = timeEl.closest('a[href]');
    if (link) add(link.getAttribute('href'));
  }

  const authorSelectors = [
    '.update-components-actor__title a[href]',
    '.feed-shared-actor__title a[href]',
    'a.update-components-actor__meta-link[href]',
    '.update-components-actor__image a[href]',
  ];
  for (const sel of authorSelectors) {
    for (const a of el.querySelectorAll(sel)) {
      if (inHeader(a)) add(a.getAttribute('href'));
    }
  }

  // SDUI + classic: any post permalink anchor in the card header.
  for (const a of el.querySelectorAll('a[href]')) {
    if (!inHeader(a)) continue;
    const href = a.getAttribute('href') || '';
    if (!isPostHref(href)) continue;
    add(href);
  }

  // Timestamp text wrapped in an anchor (relative age label).
  const timeRe = /^\s*\d+\s*(s|m|h|d|w|mo|yr|sec|min|hour|day|week|month|year)/i;
  for (const a of el.querySelectorAll('a[href]')) {
    if (!inHeader(a)) continue;
    const href = a.getAttribute('href') || '';
    if (isProfileHref(href) || href.includes('/search/results/')) continue;
    const label = (a.innerText || a.textContent || '').replace(/\s+/g, ' ').trim();
    if (timeRe.test(label)) add(href);
  }

  return out;
}
"""

_OUTBOUND_HREFS_JS = r"""
el => Array.from(el.querySelectorAll('a[href]'))
  .map(a => a.getAttribute('href') || '')
  .filter(Boolean)
"""


def activity_permalink(activity_id: str) -> str:
    aid = re.search(r"(\d+)", activity_id or "")
    if not aid or not _plausible_activity_id(aid.group(1)):
        return ""
    return f"https://www.linkedin.com/feed/update/urn:li:activity:{aid.group(1)}/"


_DEBUG_CARD_PATH = Path("debug_card.html")
_debug_card_written = False


def _absolute_post_url(href: str) -> str:
    """Make a LinkedIn post href absolute and drop query/fragment."""
    raw = unquote((href or "").strip())
    if not raw:
        return ""
    if raw.startswith("//"):
        raw = "https:" + raw
    elif raw.startswith("/"):
        raw = "https://www.linkedin.com" + raw
    parsed = urlparse(raw)
    if "linkedin.com" not in (parsed.netloc or ""):
        return ""
    path = parsed.path or "/"
    if not path.endswith("/"):
        path += "/"
    return urlunparse(("https", "www.linkedin.com", path, "", "", ""))


def permalink_from_href(href: str) -> tuple[str, str]:
    """Return (absolute_permalink, activity_id) from a post anchor href."""
    aid = extract_activity_id_from_href(href)
    if not aid:
        return "", ""
    absolute = _absolute_post_url(href)
    if not absolute or not _is_exact_post_url(absolute):
        return "", ""
    return absolute, aid


def _post_anchor_hrefs(card) -> list[str]:
    try:
        hrefs = card.evaluate(_POST_ANCHOR_HREFS_JS) or []
    except Exception:  # noqa: BLE001
        hrefs = []
    return [str(h).strip() for h in hrefs if h]


def _permalink_from_card_anchors(card) -> tuple[str, str, str]:
    """Return (url, kind, activity_id) from timestamp / author anchor hrefs."""
    for href in _post_anchor_hrefs(card):
        url, aid = permalink_from_href(href)
        if url:
            return url, "exact", aid
    return "", "", ""


def _dump_debug_card(card, log: Callable[[str], None]) -> None:
    """Write one kept card's outerHTML when no valid permalink anchor was found."""
    global _debug_card_written
    if _debug_card_written:
        return
    try:
        html = str(card.evaluate("el => el.outerHTML || ''") or "")
    except Exception as exc:  # noqa: BLE001
        html = f"<!-- evaluate failed: {exc} -->\n"
    try:
        _DEBUG_CARD_PATH.write_text(html, encoding="utf-8")
        _debug_card_written = True
        log(f"wrote {_DEBUG_CARD_PATH} (no valid post anchor href)")
    except Exception as exc:  # noqa: BLE001
        log(f"failed to write {_DEBUG_CARD_PATH}: {exc}")


def _outbound_hrefs(card) -> list[str]:
    try:
        hrefs = card.evaluate(_OUTBOUND_HREFS_JS) or []
    except Exception:  # noqa: BLE001
        hrefs = []
    urls: list[str] = []
    for href in hrefs:
        resolved = _unwrap_safety_link(str(href))
        cleaned = _clean_link(resolved) if resolved else ""
        if cleaned:
            urls.append(cleaned)
        elif resolved:
            urls.append(resolved)
    return urls

def _normalise_activity_url(href: str) -> str:
    """Turn a highlightedUpdateUrn / raw activity reference into /feed/update/…"""
    if not href:
        return ""
    match = _ACTIVITY_RE.search(href) or _ACTIVITY_QUERY_RE.search(href)
    if not match:
        match = re.search(r"activity[_-](\d+)", href, re.I)
    if match and _plausible_activity_id(match.group(1)):
        return f"https://www.linkedin.com/feed/update/urn:li:activity:{match.group(1)}/"
    return _clean_link(href)


def _is_exact_post_url(href: str) -> bool:
    if not href:
        return False
    path = urlparse(href).path.rstrip("/")
    if path.endswith("/search/results/content") or "/search/results/content" in path:
        return False
    if "urn:li:activity:" in href or "urn%3Ali%3Aactivity" in href.lower():
        match = _ACTIVITY_RE.search(href) or _ACTIVITY_QUERY_RE.search(href)
        return bool(match) and _plausible_activity_id(match.group(1))
    if "urn:li:share:" in href.lower() or "urn%3ali%3ashare" in href.lower():
        match = re.search(r"urn:li:share:(\d+)", href, re.I) or re.search(
            r"urn%3Ali%3Ashare%3A(\d+)", href, re.I
        )
        return bool(match) and _plausible_urn_id(match.group(1))
    if "urn:li:ugcpost:" in href.lower() or "urn%3ali%3augcpost" in href.lower():
        match = re.search(r"urn:li:ugcPost:(\d+)", href, re.I) or re.search(
            r"urn%3Ali%3AugcPost%3A(\d+)", href, re.I
        )
        return bool(match) and _plausible_urn_id(match.group(1))
    if "/feed/update/" in path:
        share_match = re.search(r"urn:li:share:(\d+)", href, re.I)
        if share_match and _plausible_urn_id(share_match.group(1)):
            return True
        ugc_match = re.search(r"urn:li:ugcPost:(\d+)", href, re.I)
        if ugc_match and _plausible_urn_id(ugc_match.group(1)):
            return True
        match = _ACTIVITY_RE.search(href) or _ACTIVITY_QUERY_RE.search(href)
        return bool(match) and _plausible_activity_id(match.group(1))
    if _COMPANY_FEED_RE.search(path) or _PROFILE_FEED_RE.search(path):
        return False
    # /posts/<slug-with-activity-id> is a specific post; bare /posts/ feeds are not.
    if "/posts/" in path:
        slug = path.split("/posts/", 1)[-1]
        if not slug:
            return False
        match = _POST_SLUG_URN_RE.search(slug)
        if match:
            return _plausible_urn_id(match.group(2))
        match = re.search(r"activity[_-](\d+)", slug, re.I)
        return bool(match) and _plausible_urn_id(match.group(1))
    return bool(_EXACT_POST_PATH_RE.search(path))


def _is_feed_fallback_url(href: str) -> bool:
    if not href:
        return False
    path = urlparse(href).path.rstrip("/")
    return bool(_COMPANY_FEED_RE.search(path) or _PROFILE_FEED_RE.search(path)
                or re.search(r"/company/[^/]+/?$", path)
                or re.search(r"/in/[^/]+/?$", path))


def _actor_hrefs(card) -> list[str]:
    hrefs: list[str] = []
    for selector in ("a[href*='/company/']", "a[href*='/in/']"):
        try:
            nodes = card.locator(selector)
            for index in range(min(nodes.count(), 6)):
                href = nodes.nth(index).get_attribute("href", timeout=800) or ""
                if href:
                    hrefs.append(_clean_link(href))
        except Exception:  # noqa: BLE001
            continue
    return hrefs


def _is_search_results_url(href: str) -> bool:
    if not href:
        return False
    path = urlparse(href).path.rstrip("/")
    return path.endswith("/search/results/content") or "/search/results/content/" in path


def _post_permalink(card, poster: str = "", text: str = "", page=None) -> tuple[str, str, str]:
    """Return (url, kind, activity_id) from timestamp / author anchor hrefs."""
    del poster, text, page
    return _permalink_from_card_anchors(card)


def _unwrap_safety_link(href: str) -> str:
    """LinkedIn wraps outbound links as /safety/go/?url=<encoded>. Unwrap them."""
    if "/safety/go/" not in href:
        return href
    try:
        target = parse_qs(urlparse(href.replace("&amp;", "&")).query).get("url", [""])[0]
    except Exception:  # noqa: BLE001
        return href
    return unquote(target) if target else href


def _job_link(card) -> str:
    for selector in (
        "a[href*='/jobs/view/']",
        "a[href*='/jobs/']",
        "a[href*='lnkd.in']",
        "a[href*='/safety/go/']",
    ):
        try:
            node = card.locator(selector).first
            if node.count() == 0:
                continue
            href = node.get_attribute("href", timeout=1_000) or ""
            if not href:
                continue
            resolved = _unwrap_safety_link(href)
            # The global nav's bare /jobs/ link is not a posting.
            if resolved.rstrip("/").endswith("linkedin.com/jobs"):
                continue
            return _clean_link(resolved)
        except Exception:  # noqa: BLE001
            continue
    return ""


def _poster_name(card) -> str:
    """Read the poster from the control-menu aria-label, the one stable source."""
    for selector in ("[aria-label*='control menu for post by' i]",
                     "[aria-label*=\"'s profile\" i]", "[aria-label*='’s profile' i]"):
        try:
            node = card.locator(selector).first
            if node.count() == 0:
                continue
            label = node.get_attribute("aria-label", timeout=1_000) or ""
        except Exception:  # noqa: BLE001
            continue
        match = _POSTER_MENU_RE.search(label)
        if match:
            return _clean_poster(match.group(1))
        match = re.search(r"View\s+(.+?)[’']s profile", label)
        if match:
            return _clean_poster(match.group(1))
    return ""


# The timestamp must be read from the post header only. The body routinely
# contains date-like phrases ("5 Day Training Program", "3rd August"), and
# scanning the whole card would happily parse those as the post's age.
_HEADER_TEXT_JS = r"""
el => {
  // Narrow: the visibility globe sits immediately after the timestamp.
  let stamp = '';
  const vis = el.querySelector('svg[aria-label^="Visibility" i]');
  if (vis) {
    const host = vis.closest('span, p, div');
    stamp = host && host.textContent ? host.textContent.trim() : '';
  }

  // Wide: everything before the post body, which carries name and headline.
  const box = el.querySelector('[data-testid="expandable-text-box"]');
  const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
  const parts = [];
  while (walker.nextNode()) {
    const node = walker.currentNode;
    if (box && box.contains(node)) break;
    const t = (node.textContent || '').trim();
    if (t) parts.push(t);
  }

  return { timestamp: stamp, header: parts.join(' ') };
}
"""


def _header_parts(card) -> tuple[str, str]:
    """(timestamp text, full header text) for a card."""
    try:
        data = card.evaluate(_HEADER_TEXT_JS) or {}
    except Exception:  # noqa: BLE001
        return "", ""
    return (data.get("timestamp") or "").strip(), (data.get("header") or "").strip()


def _post_body_text(card) -> str:
    """Full post body after any see-more click.

    CSS line-clamps the body, so innerText can come back visually truncated
    while textContent holds everything; whichever is longer is the real text.
    Every candidate selector is tried — the first hit is often still clamped.
    """
    best = ""
    for selector in _TEXT_SELECTORS:
        try:
            node = card.locator(selector).first
            if node.count() == 0:
                continue
            for getter in (node.inner_text, node.text_content):
                try:
                    candidate = _dedupe_text(getter(timeout=1_500) or "")
                except Exception:  # noqa: BLE001
                    continue
                if len(candidate) > len(best):
                    best = candidate
        except Exception:  # noqa: BLE001
            continue
    return best


def _company_from_headline(header: str) -> str:
    """Best-effort company from the poster's headline ("Architect @ TransUnion")."""
    return poster_employer_from_headline(header)


_ENGAGEMENT_RE = re.compile(
    r"(?P<count>[\d,]+)\s*(?P<label>reactions?|likes?|comments?|reposts?|shares?)\b",
    re.IGNORECASE,
)


def _engagement_from_card(card) -> dict[str, int]:
    """Best-effort rendered engagement; lack of a count is not an error."""
    try:
        text = card.inner_text(timeout=1_500) or ""
    except Exception:  # noqa: BLE001
        return {}
    counts = {"likes": 0, "comments": 0, "reposts": 0}
    saw_count = False
    for match in _ENGAGEMENT_RE.finditer(text):
        try:
            count = int(match.group("count").replace(",", ""))
        except ValueError:
            continue
        saw_count = True
        label = match.group("label").lower()
        key = "likes" if label.startswith(("reaction", "like")) else (
            "comments" if label.startswith("comment") else "reposts"
        )
        counts[key] = max(counts[key], count)
    return counts if saw_count else {}


def _engagement_spam_signal(engagement: dict[str, int], links: list[str]) -> str:
    """One negative-only engagement heuristic for semantic-gate context."""
    if (
        engagement
        and not any(engagement.values())
        and sum(1 for link in links if is_shortener(link)) >= 2
    ):
        return "zero_engagement_multiple_shorteners"
    return ""


def _post_age_days(card, sub_description: str) -> float | None:
    """Prefer a machine-readable timestamp, fall back to the relative label."""
    try:
        time_node = card.locator("time").first
        if time_node.count():
            attr = time_node.get_attribute("datetime", timeout=800) or ""
            age = parse_datetime_attr(attr)
            if age is not None:
                return age
            age = parse_relative_date(time_node.inner_text(timeout=800))
            if age is not None:
                return age
    except Exception:  # noqa: BLE001
        pass
    return parse_relative_date(sub_description)


# "Asha Ramanathan 3rd+ • 2d • Edited" -> "Asha Ramanathan". The actor line runs
# name, connection degree, then timestamp, all inline, so a raw innerText read
# drags the metadata into the name.
_DEGREE_RE = re.compile(r"\s*\b(?:1st|2nd|3rd\+?|\d+(?:st|nd|rd|th))\b\s*\+?\s*$", re.IGNORECASE)


def _clean_poster(raw: str) -> str:
    name = _dedupe_text(raw).split("•")[0].strip()
    previous = None
    while name != previous:
        previous = name
        name = _DEGREE_RE.sub("", name).strip(" ·-–—•")
    return name


def _card_lines(card) -> list[str]:
    try:
        raw = card.inner_text(timeout=2_000) or ""
    except Exception:  # noqa: BLE001
        return []
    return [line.strip() for line in raw.splitlines() if line.strip()]


def _extract_post(card, page=None) -> Post | None:
    poster = _poster_name(card) or _clean_poster(_first_text(card, _POSTER_SELECTORS))
    stamp, header = _header_parts(card)
    sub_description = stamp or _first_text(card, _SUBDESC_SELECTORS) or header
    text = expanded_post_text(card, page)

    # Last resort when both the classes and the SDUI hooks miss.
    if not (poster and text):
        lines = _card_lines(card)
        if lines:
            if not poster:
                poster = _clean_poster(lines[0])
            if not text:
                text = " ".join(lines)
            if not sub_description:
                sub_description = " ".join(lines[:3])

    if not (poster or text):
        return None

    role = company = location = ""
    company_source = ""
    for selector in _ENTITY_SELECTORS:
        try:
            entity = card.locator(selector).first
            if entity.count() == 0:
                continue
        except Exception:  # noqa: BLE001
            continue
        role = _first_text(entity, _ENTITY_TITLE_SELECTORS)
        company = _first_text(entity, _ENTITY_SUBTITLE_SELECTORS)
        location = _first_text(entity, _ENTITY_CAPTION_SELECTORS)
        if role or company:
            company_source = "job-card"
            break

    actor_hrefs = _actor_hrefs(card)
    actor_profile = next(
        (h for h in actor_hrefs if "/company/" in h or "/in/" in h),
        "",
    )

    # Most posts have no attached job card, so company has to be inferred. When a
    # company page authored the post the poster IS the company; otherwise the
    # poster's headline is a weaker clue (their employer, not necessarily the
    # hiring company), so the provenance is recorded rather than asserted.
    # Detect company authorship from the actor link, NOT from post_url.
    if not company:
        from_body = company_field_from_post(text)
        if from_body:
            company, company_source = from_body, "post-body"
        elif any("/company/" in href for href in actor_hrefs):
            company, company_source = poster, "company-page"
        else:
            derived = _company_from_headline(header)
            if derived:
                company, company_source = derived, "poster-headline"

    permalink, permalink_kind, activity_id = _post_permalink(
        card, poster=poster, text=text, page=page
    )
    if permalink and _is_search_results_url(permalink):
        permalink, permalink_kind = "", ""

    hrefs = _outbound_hrefs(card)
    outbound = list(dict.fromkeys(hrefs + extract_urls_from_text(text)))

    age = _post_age_days(card, sub_description)

    post = Post(
        poster=poster,
        company=company,
        role=role,
        location=location,
        post_date=humanise_age(age) if age is not None else (sub_description[:40] or "unknown"),
        job_link=_job_link(card),
        post_url=permalink,
        post_text=text,
        activity_urn=f"urn:li:activity:{activity_id}" if activity_id else "",
        outbound_links=outbound,
        poster_headline=header,
        follower_count=parse_follower_count(f"{header}\n{poster}"),
        poster_verified=poster_is_verified(f"{header}\n{poster}"),
        engagement=_engagement_from_card(card),
        days_old=age,
        _actor_profile_url=actor_profile,
    )
    if company_source:
        post._reasons.append(f"company:{company_source}")
    _set_url_reason(post, "exact" if permalink_kind == "exact" else "")
    return post


def _apply_filters(
    post: Post,
    matcher: HiringMatcher,
    max_age_days: float,
    keep_undated: bool,
    *,
    candidate_company: str = "",
) -> tuple[bool, str]:
    """Return (keep, reason). Reason explains a drop or documents a match."""
    if post.days_old is None:
        if not keep_undated:
            return False, "no parseable date"
        post._reasons.append("recency:undated(kept)")
    elif post.days_old > max_age_days:
        return False, f"older than {max_age_days:g}d ({humanise_age(post.days_old)})"
    else:
        post._reasons.append(f"recency:{humanise_age(post.days_old)}")

    if candidate_company:
        company_text = " ".join(
            part for part in (post.post_text, post.company, post.poster, post.poster_headline) if part
        )
        result = match_company_candidate_net(
            post.post_text, candidate_company, matcher, company_text=company_text,
        )
    else:
        result = matcher.match(post.post_text)
    if not result.matched:
        return False, f"not a candidate ({result.why()})" if candidate_company else f"not a hiring post ({result.why()})"
    post._reasons.extend(result.reasons)

    return True, ""


def _apply_link_and_source_gates(
    post: Post,
    resolver: RedirectResolver,
    company: str = "",
    *,
    require_opening: bool = True,
) -> tuple[bool, str]:
    """Classify outbound links and hard-drop junk / invite-only sources.

    Returns (reject, reason).
    """
    urls = list(post.outbound_links or [])
    if post.job_link:
        urls.append(post.job_link)
    verdict, categories = classify_link_categories(
        urls, company=company or post.company, resolver=resolver,
    )
    post.link_class = verdict.kind
    post.link_categories = categories
    post._reasons.extend(verdict.reasons[:4])
    if verdict.kind == "careers":
        post._reasons.append("link:careers")
    elif verdict.kind == "junk":
        post._reasons.append("link:junk")

    is_company_page = any(r == "company:company-page" for r in post._reasons)
    reject, why = heuristic_source_reject(
        post.post_text,
        poster=post.poster,
        company=company or post.company,
        is_company_page=is_company_page,
        link_kind=verdict.kind,
    )
    if reject:
        return True, why

    if company and require_opening:
        apply_links = list(post.outbound_links or [])
        if post.job_link:
            apply_links.append(post.job_link)
        source_type, keep, reason = classify_first_party(
            company,
            poster=post.poster,
            headline=post.poster_headline,
            text=post.post_text,
            apply_links=apply_links,
            strict=True,
        )
        post.source_type = source_type
        post._reasons.extend(
            opening_audit_reasons(
                poster=post.poster,
                headline=post.poster_headline,
                source_type=source_type,
                reason=reason,
                text=post.post_text,
            )
        )
        if not keep:
            return True, f"not-opening:{source_type}:{reason}"
    elif is_company_page:
        post.source_type = post.source_type or "company"
    return False, ""


def _snippet_tokens(text: str, n: int = 6) -> list[str]:
    words = re.findall(r"[\w']{3,}", (text or "").replace("*", " "))
    skip = {
        "the", "and", "for", "are", "you", "your", "our", "with", "this", "that",
        "hiring", "looking", "intern", "internship", "apply", "join",
    }
    out: list[str] = []
    for word in words:
        if word.lower() in skip:
            continue
        out.append(word)
        if len(out) >= n:
            break
    return out or words[:n]


def _tag_card(card, index: int) -> None:
    try:
        card.evaluate("(el, i) => el.setAttribute('data-ir-card', String(i))", index)
    except Exception:  # noqa: BLE001
        pass


def clean_permalink(url: str) -> str:
    """Keep a post permalink; drop search listings and tracking query params."""
    raw = (url or "").strip()
    if not raw:
        return ""
    candidate = raw.split()[0].strip()
    if _is_search_results_url(candidate):
        return ""
    from_embed = permalink_from_embed_blob(candidate)
    if from_embed and _is_exact_post_url(from_embed):
        return from_embed
    canonical = canonical_post_permalink(candidate)
    if canonical and _is_exact_post_url(canonical):
        return canonical
    normalised = _normalise_activity_url(candidate)
    if _is_exact_post_url(normalised):
        parsed = urlparse(normalised)
        return urlunparse(parsed._replace(query="", fragment=""))
    cleaned = _clean_link(candidate)
    return cleaned if _is_exact_post_url(cleaned) else ""


def _permalink_cache_key(post: Post) -> str:
    if post.activity_urn:
        return post.activity_urn.lower()
    if post.post_url and _is_exact_post_url(post.post_url):
        return post.post_url.rstrip("/").lower()
    return f"{post.poster}|{(post.post_text or '')[:120]}".lower()


def _find_card_for_post(page, post: Post):
    """Re-locate a search-result card by tag, then poster + text snippet."""
    if getattr(post, "_card_index", -1) >= 0:
        try:
            tagged = page.locator(f"[data-ir-card='{post._card_index}']")
            if tagged.count() > 0:
                return tagged.first
        except Exception:  # noqa: BLE001
            pass
    cards = _find_posts(page)
    if cards is None:
        return None
    tokens = [t.lower() for t in _snippet_tokens(post.post_text)]
    poster = (post.poster or "").lower()
    for index in range(cards.count()):
        card = cards.nth(index)
        try:
            blob = (card.inner_text(timeout=1_500) or "").lower()
        except Exception:  # noqa: BLE001
            continue
        if poster and poster not in blob:
            compact_poster = re.sub(r"\s+", "", poster)
            compact_blob = re.sub(r"\s+", "", blob)
            if compact_poster not in compact_blob:
                continue
        if tokens and sum(1 for t in tokens if t in blob) < min(2, len(tokens)):
            continue
        return card
    return None


def _close_open_menu(page) -> None:
    try:
        page.keyboard.press("Escape")
        page.wait_for_timeout(200)
    except Exception:  # noqa: BLE001
        pass


def _open_post_control_menu(card, post: Post):
    selectors = [
        "button[aria-label*='Open control menu for post by' i]",
        "button[aria-label*='control menu for post' i]",
        "button[aria-label*='Open control menu' i]",
        "button[aria-label*='More actions' i]",
        "button[aria-label*='More options' i]",
        "[aria-label*='Open control menu' i]",
        "[aria-label*='More actions' i]",
    ]
    if post.poster:
        selectors.insert(
            0,
            f"button[aria-label='Open control menu for post by {post.poster}']",
        )
    for selector in selectors:
        try:
            button = card.locator(selector).first
            if button.count() == 0:
                continue
            button.scroll_into_view_if_needed(timeout=1_500)
            button.click(timeout=2_500)
            return button
        except Exception:  # noqa: BLE001
            try:
                card.locator(selector).first.click(timeout=2_500, force=True)
                return card.locator(selector).first
            except Exception:  # noqa: BLE001
                continue
    try:
        hit = card.evaluate(
            """el => {
              const btns = Array.from(el.querySelectorAll('button,[role="button"]'));
              const btn = btns.find(b => {
                const a = (b.getAttribute('aria-label') || '').toLowerCase();
                return /control menu|more actions|more options/.test(a);
              });
              if (!btn) return false;
              btn.click();
              return true;
            }"""
        )
        if hit:
            return card
    except Exception:  # noqa: BLE001
        pass
    return None


def _click_menu_item_matching(page, pattern: str) -> bool:
    """Click a visible overflow-menu row whose label matches `pattern` (regex)."""
    candidates = [
        page.get_by_role("menuitem", name=re.compile(pattern, re.I)),
        page.get_by_role("button", name=re.compile(pattern, re.I)),
        page.get_by_role("option", name=re.compile(pattern, re.I)),
        page.locator('[role="menuitem"], [role="option"], li, button, div[role="button"]').filter(
            has_text=re.compile(pattern, re.I)
        ),
    ]
    for locator in candidates:
        try:
            target = locator.first
            if target.count() == 0:
                continue
            target.wait_for(state="visible", timeout=2_500)
            target.click(timeout=2_500)
            return True
        except Exception:  # noqa: BLE001
            continue
    try:
        clicked = page.evaluate(
            r"""(pattern) => {
              const re = new RegExp(pattern, 'i');
              const roots = Array.from(document.querySelectorAll(
                '[role="menu"], .artdeco-dropdown__content, .artdeco-dropdown__content-inner, [data-test-modal]'
              ));
              roots.push(document.body);
              const seen = new Set();
              for (const root of roots) {
                const nodes = root.querySelectorAll
                  ? root.querySelectorAll('[role="menuitem"], [role="option"], li, button, a, span, div')
                  : [];
                for (const n of nodes) {
                  if (seen.has(n)) continue;
                  seen.add(n);
                  const t = (n.innerText || n.textContent || '').replace(/\s+/g, ' ').trim();
                  if (!t || t.length > 120 || !re.test(t)) continue;
                  const hit = n.closest('[role="menuitem"], [role="option"], button, a, li') || n;
                  try { hit.click(); return t; } catch (e) {}
                }
              }
              return '';
            }""",
            pattern,
        )
        return bool(clicked)
    except Exception:  # noqa: BLE001
        return False


def _click_embed_menu_item(page) -> bool:
    for pattern in (
        r"embed this post",
        r"^embed post$",
        r"embed",
    ):
        if _click_menu_item_matching(page, pattern):
            return True
    return False


_READ_EMBED_DIALOG_JS = r"""
() => {
  const dialog = document.querySelector(
    '[role="dialog"], [aria-modal="true"], .artdeco-modal'
  );
  if (!dialog) return '';
  const iframe = dialog.querySelector(
    'iframe[src*="linkedin.com/embed"], iframe[src*="urn:li"]'
  );
  if (iframe && iframe.src) return iframe.src;
  const ta = dialog.querySelector('textarea');
  if (ta && (ta.value || ta.textContent)) return ta.value || ta.textContent;
  for (const inp of dialog.querySelectorAll('input')) {
    const v = inp.value || '';
    if (/urn:li:(share|activity):/i.test(v)) return v;
  }
  for (const a of dialog.querySelectorAll('a[href*="urn:li"]')) {
    const h = a.getAttribute('href') || '';
    if (h) return h;
  }
  return dialog.innerText || '';
}
"""


def _read_embed_dialog_permalink(page) -> str:
    try:
        page.locator('[role="dialog"], [aria-modal="true"], .artdeco-modal').first.wait_for(
            state="visible", timeout=4_000
        )
    except Exception:  # noqa: BLE001
        pass
    try:
        blob = str(page.evaluate(_READ_EMBED_DIALOG_JS) or "").strip()
    except Exception:  # noqa: BLE001
        blob = ""
    return blob


def _overflow_menu_labels(page) -> str:
    try:
        return str(page.evaluate(
            r"""() => Array.from(document.querySelectorAll(
              '[role="menuitem"], [role="option"], .artdeco-dropdown__content *'
            )).map(n => (n.innerText || '').trim()).filter(Boolean).join(' | ')"""
        ) or "")
    except Exception:  # noqa: BLE001
        return ""


def _read_clipboard_via_paste(page) -> str:
    """Paste from OS clipboard into a scratch textarea (works when readText() is blocked)."""
    paste_key = "Meta+v" if sys.platform == "darwin" else "Control+v"
    try:
        page.evaluate(
            r"""() => {
              let ta = document.getElementById('ir-clipboard-scratch');
              if (!ta) {
                ta = document.createElement('textarea');
                ta.id = 'ir-clipboard-scratch';
                ta.setAttribute('aria-hidden', 'true');
                ta.style.position = 'fixed';
                ta.style.left = '-9999px';
                ta.style.top = '0';
                document.body.appendChild(ta);
              }
              ta.value = '';
              ta.focus();
            }"""
        )
        page.keyboard.press(paste_key)
        page.wait_for_timeout(250)
        raw = str(page.evaluate(
            "() => document.getElementById('ir-clipboard-scratch')?.value || ''"
        ) or "").strip()
        return raw
    except Exception:  # noqa: BLE001
        return ""


def _normalize_copied_url(raw: str, resolver: RedirectResolver) -> str:
    raw = (raw or "").strip()
    if not raw:
        return ""
    candidate = raw.split()[0].strip()
    if is_shortener(candidate) or "lnkd.in" in candidate:
        try:
            candidate = resolver.resolve(candidate)
        except Exception:  # noqa: BLE001
            pass
    return canonical_post_permalink(candidate)


def _read_copied_permalink(
    page,
    resolver: RedirectResolver,
    log: Callable[[str], None] | None = None,
) -> str:
    """Read a permalink after 'Copy link to post' from toast and/or clipboard."""
    for _ in range(4):
        exact = _read_toast_permalink(page)
        if exact:
            return exact
        for raw in (_read_clipboard(page), _read_clipboard_via_paste(page)):
            if not raw:
                continue
            exact = _normalize_copied_url(raw, resolver)
            if exact:
                return exact
            if log and raw:
                log(f"  copy-link raw (rejected): {raw[:120]}")
        page.wait_for_timeout(500)
    try:
        blob = str(page.evaluate(
            r"""() => {
              const toast = document.querySelector(
                '.artdeco-toast-item, [data-test-artdeco-toast], [role="alert"]'
              );
              if (toast) return toast.innerText || '';
              for (const a of document.querySelectorAll('a[href]')) {
                const h = a.href || '';
                if (/linkedin\\.com\\/(feed\\/update|posts)\\//i.test(h)) return h;
              }
              return '';
            }"""
        ) or "")
    except Exception:  # noqa: BLE001
        blob = ""
    return clean_permalink(blob) or canonical_post_permalink(blob)


def _permalink_from_overflow_menu(
    card,
    page,
    post: Post,
    log: Callable[[str], None],
    resolver: RedirectResolver,
) -> tuple[str, str]:
    """Open … menu once; embed dialog first, else copy-link."""
    try:
        page.bring_to_front()
    except Exception:  # noqa: BLE001
        pass
    try:
        page.context.grant_permissions(
            ["clipboard-read", "clipboard-write"],
            origin="https://www.linkedin.com",
        )
    except Exception:  # noqa: BLE001
        pass
    _close_open_menu(page)
    try:
        page.evaluate("() => navigator.clipboard.writeText('')")
    except Exception:  # noqa: BLE001
        pass
    try:
        card.scroll_into_view_if_needed(timeout=2_000)
    except Exception:  # noqa: BLE001
        pass
    if _open_post_control_menu(card, post) is None:
        log("  control-menu button not found")
        return "", ""
    page.wait_for_timeout(650)
    labels = _overflow_menu_labels(page)

    if re.search(r"embed\s+(this\s+)?post", labels, re.I):
        if _click_embed_menu_item(page):
            page.wait_for_timeout(900)
            blob = _read_embed_dialog_permalink(page)
            url = permalink_from_embed_blob(blob)
            _close_open_menu(page)
            if url and _is_exact_post_url(url):
                return url, "embed"
            log(f"  embed dialog had no usable URN{f' ({blob[:80]}…)' if blob else ''}")
            _close_open_menu(page)
            if _open_post_control_menu(card, post) is None:
                return "", ""
            page.wait_for_timeout(650)
        else:
            log(f"  embed menu click failed ({labels})")

    if re.search(r"copy link", labels, re.I):
        if _click_copy_link_menu_item(page):
            page.wait_for_timeout(700)
            exact = _read_copied_permalink(page, resolver, log=log)
            _close_open_menu(page)
            if exact and _is_exact_post_url(exact):
                return exact, "copy"
            log("  copy-link returned no valid permalink")
        else:
            log("  copy-link menu click failed")
    else:
        log(f"  no embed/copy menu items ({labels})")

    _close_open_menu(page)
    return "", ""


def _embed_link_from_card(card, page, post: Post, log: Callable[[str], None]) -> str:
    """Deprecated wrapper — use _permalink_from_overflow_menu."""
    url, method = _permalink_from_overflow_menu(card, page, post, log)
    return url if method == "embed" else ""


def _resolve_permalink_from_card(
    card,
    page,
    post: Post,
    log: Callable[[str], None],
    resolver: RedirectResolver,
) -> tuple[str, str]:
    """Return (permalink, method) where method is anchor|embed|copy|''."""
    for href in _post_anchor_hrefs(card):
        url, _aid = permalink_from_href(href)
        if url:
            return url, "anchor"
    return _permalink_from_overflow_menu(card, page, post, log, resolver)


def _click_copy_link_menu_item(page) -> bool:
    for pattern in (
        r"copy link to post",
        r"^copy link$",
        r"copy link",
    ):
        if _click_menu_item_matching(page, pattern):
            return True
    return False


def _read_clipboard(page) -> str:
    try:
        text = page.evaluate("() => navigator.clipboard.readText()")
    except Exception:  # noqa: BLE001
        return ""
    return (text or "").strip()


def _read_toast_permalink(page) -> str:
    selectors = (
        ".artdeco-toast-item",
        "[data-test-artdeco-toast]",
        ".artdeco-toast",
        "[role='alert']",
        ".artdeco-toasts",
    )
    for selector in selectors:
        try:
            node = page.locator(selector).first
            if node.count() == 0:
                continue
            href = ""
            try:
                href = node.locator("a[href]").first.get_attribute("href", timeout=400) or ""
            except Exception:  # noqa: BLE001
                href = ""
            blob = href or (node.inner_text(timeout=400) or "")
            cleaned = clean_permalink(blob)
            if cleaned:
                return cleaned
            found = extract_activity_id_from_href(blob)
            if found:
                return activity_permalink(found)
        except Exception:  # noqa: BLE001
            continue
    return ""


def _copy_link_from_card(
    card,
    page,
    post: Post,
    log: Callable[[str], None],
    resolver: RedirectResolver | None = None,
) -> str:
    """… → Copy link to post (clipboard / toast)."""
    url, method = _permalink_from_overflow_menu(
        card, page, post, log, resolver or RedirectResolver(delay=0.25, log=log),
    )
    return url if method == "copy" else ""


def _set_url_reason(post: Post, status: str) -> None:
    """Replace every post_url:* token with a single status (or drop them)."""
    post._reasons = [r for r in post._reasons if not str(r).startswith("post_url:")]
    if status:
        post._reasons.append(f"post_url:{status}")
    post.why_matched = "; ".join(post._reasons)


def _clear_post_url(post: Post) -> None:
    """Empty post_url without writing unresolved/copy-failed into why_matched."""
    if _is_search_results_url(post.post_url) or not _is_exact_post_url(post.post_url):
        post.post_url = ""
    _set_url_reason(post, "")


def _mark_resolved(post: Post, url: str) -> None:
    cleaned = clean_permalink(url)
    if not cleaned:
        _clear_post_url(post)
        return
    post.post_url = cleaned
    embed = _EMBED_URN_RE.search(cleaned)
    if embed and not post.activity_urn:
        kind = embed.group(1)
        if kind.lower() == "ugcpost":
            kind = "ugcPost"
        post.activity_urn = f"urn:li:{kind}:{embed.group(2)}"
    else:
        match = _ACTIVITY_RE.search(cleaned) or _ACTIVITY_QUERY_RE.search(cleaned)
        if match and not post.activity_urn:
            post.activity_urn = f"urn:li:activity:{match.group(1)}"
    _set_url_reason(post, "exact")


def resolve_post_links(
    page,
    posts: list[Post],
    search_url: str,
    max_resolve: int = 25,
    resolve_delay: float = 1.2,
    log: Callable[[str], None] = print,
    cache: dict[str, str] | None = None,
    link_resolver: RedirectResolver | None = None,
) -> dict[str, int]:
    """Fill permalinks for kept posts via embed dialog (then copy-link fallback)."""
    tally = {
        "attempted": 0, "copied": 0, "embedded": 0, "failed": 0,
        "skipped_exact": 0, "cached": 0,
    }
    if not posts:
        return tally

    cache = cache if cache is not None else {}
    resolver = link_resolver or RedirectResolver(delay=0.25, log=log)

    try:
        page.context.grant_permissions(
            ["clipboard-read", "clipboard-write"],
            origin="https://www.linkedin.com",
        )
    except Exception:  # noqa: BLE001
        pass

    for post in posts:
        if post.post_url and not _is_exact_post_url(post.post_url):
            _clear_post_url(post)
        key = _permalink_cache_key(post)
        if key and key in cache:
            _mark_resolved(post, cache[key])
            tally["cached"] += 1
            continue
        if _is_exact_post_url(post.post_url):
            tally["skipped_exact"] += 1
            if key:
                cache[key] = post.post_url
            continue
        if tally["attempted"] >= max(0, max_resolve):
            _clear_post_url(post)
            continue

        tally["attempted"] += 1
        card = _find_card_for_post(page, post)
        if card is None:
            log(f"permalink: card not found for {post.poster or post.company or 'post'}")
            _clear_post_url(post)
            tally["failed"] += 1
            continue
        try:
            card.scroll_into_view_if_needed(timeout=2_000)
        except Exception:  # noqa: BLE001
            pass
        exact, method = _resolve_permalink_from_card(card, page, post, log, resolver)
        if exact and _is_exact_post_url(exact):
            _mark_resolved(post, exact)
            if key:
                cache[key] = exact
            if method == "embed":
                tally["embedded"] += 1
            else:
                tally["copied"] += 1
            log(f"  -> permalink ({method}): {exact}")
        else:
            _dump_debug_card(card, log)
            _clear_post_url(post)
            tally["failed"] += 1
        if resolve_delay > 0:
            time.sleep(resolve_delay)

    del search_url
    return tally


def scrape_posts(
    page,
    query: str,
    max_age_days: float = 14,
    scrolls: int = 3,
    limit: int | None = None,
    delay: float = 2.0,
    location: str = "",
    matcher: HiringMatcher | None = None,
    keep_undated: bool = False,
    sort_by_date: bool = True,
    scroll_until_max_age: bool = False,
    debug: bool = False,
    debug_dir: str = ".",
    network_idle_ms: int = 15_000,
    wait_for_posts_ms: int = 20_000,
    log: Callable[[str], None] = print,
    link_resolver: RedirectResolver | None = None,
    target_company: str = "",
    candidate_net: bool = False,
    debug_funnel: bool = False,
    defer_filters: bool = False,
) -> tuple[list[Post], dict[str, Any], str]:
    """Scrape and filter LinkedIn post search results.

    Returns (kept posts, drop tally, search_url). The search URL is needed by
    --resolve-links so the second pass can return to the results page.

    When `scroll_until_max_age` is set (company-posts), `scrolls` is a ceiling
    and scrolling continues until a loaded post is older than `max_age_days`.
    """
    matcher = matcher or HiringMatcher()
    global _debug_card_written
    _debug_card_written = False
    resolver = link_resolver or RedirectResolver(delay=0.35, log=log)
    stats: dict[str, Any] = {
        "seen": 0, "kept": 0, "too_old": 0, "not_hiring": 0, "undated": 0,
        "unparsed": 0, "junk_link": 0, "source_spam": 0,
        "drop_reasons": {},
    }

    def dropped(stage: str, reason: str, post: Post | None = None) -> None:
        if "company:none" in reason:
            label = "no_company_mention"
        elif "celebration:obvious" in reason:
            label = "celebration"
        elif "hiring:none" in reason and "role:none" in reason:
            label = "no_intern_or_hiring_term"
        elif reason.startswith("older"):
            label = "too_old"
        else:
            label = reason.split(" (", 1)[0]
        key = f"{stage}:{label}"
        stats["drop_reasons"][key] = stats["drop_reasons"].get(key, 0) + 1
        if debug_funnel:
            snippet = ((post.post_text if post else "") or "").replace("\n", " ")[:120]
            log(f"    funnel→drop {stage}: {post.poster if post else '?'} — {reason} | {snippet!r}")

    url = build_content_url(query, max_age_days=max_age_days, sort_by_date=sort_by_date,
                            location=location)
    log(f"Content search: {url}")

    raw_html = ""
    try:
        _resp = page.goto(url, wait_until="domcontentloaded", timeout=45_000)
        if _resp is not None:
            raw_html = _resp.text()
    except Exception as exc:  # noqa: BLE001
        log(f"Could not load the search page: {exc}")
        return [], stats, url

    # Build URN lookup from the raw server HTML (before hydration strips them).
    # Direct href anchors: href="/feed/update/urn:li:…/"
    _href_urns: list[str] = re.findall(
        r'/feed/update/(urn:li:(?:share|activity|ugcPost):\d+)/', raw_html
    )

    try:
        page.wait_for_load_state("networkidle", timeout=network_idle_ms)
    except Exception:  # noqa: BLE001 - a busy page never goes fully idle; carry on
        log("page never reached network idle; continuing")

    page.wait_for_timeout(1_500)

    if "/login" in (page.url or "") or "/authwall" in (page.url or ""):
        log("LinkedIn redirected to a login wall — the session expired.")
        return [], stats, url

    if not _wait_for_posts(page, wait_for_posts_ms, log):
        log("continuing anyway; --debug will capture what did render")

    if scroll_until_max_age:
        _load_more_until_age(
            page,
            max_age_days=max_age_days,
            max_scrolls=scrolls,
            pause=delay,
            log=log,
        )
    else:
        _load_more(page, scrolls, delay, log)
    _expand_see_more(page, log)

    # Dumped before the early return below, since a zero-match run is exactly
    # the case worth capturing.
    if debug:
        dump_debug(page, SELECTOR_GROUPS, out_dir=debug_dir, log=log)

    cards = _find_posts(page, log)

    if cards is None:
        if _page_reports_no_results(page):
            log("LinkedIn reports no results for this query (empty page, not a selector miss).")
        else:
            log("No post containers matched the known selectors.")
            if not debug:
                log("Re-run with --debug to dump the page and find the current selector.")
        return [], stats, url

    total = cards.count()
    log(f"found {total} post container(s) on the page")

    kept: list[Post] = []
    seen_keys: set[str] = set()

    for index in range(total):
        try:
            card = cards.nth(index)
            _tag_card(card, index)
            post = _extract_post(card, page)
            if post is not None:
                post._card_index = index
        except Exception:  # noqa: BLE001 - one bad card should not kill the run
            post = None

        if post is None:
            stats["unparsed"] += 1
            continue

        key = post.post_url or post.activity_urn or f"{post.poster}|{post.post_text[:120]}"
        if key in seen_keys:
            continue
        seen_keys.add(key)
        stats["seen"] += 1

        # company-posts dedupes across every query variation before any filter.
        # Return raw extracted cards so its coordinator owns that single funnel.
        if defer_filters:
            kept.append(post)
            continue

        keep, reason = _apply_filters(
            post, matcher, max_age_days, keep_undated,
            candidate_company=target_company if candidate_net else "",
        )
        if not keep:
            if reason.startswith("older"):
                stats["too_old"] += 1
            elif reason.startswith("no parseable"):
                stats["undated"] += 1
            else:
                stats["not_hiring"] += 1
            dropped("prefilter", reason, post)
            continue

        reject, why = _apply_link_and_source_gates(
            post, resolver, company=target_company or post.company,
            require_opening=not candidate_net,
        )
        if reject:
            if why.startswith("junk-link"):
                stats["junk_link"] += 1
            else:
                stats["source_spam"] += 1
            dropped("source", why, post)
            continue

        mentions = extract_funding_mentions(post.post_text)
        if mentions:
            post.funding_signal = "; ".join(mentions)
            post._reasons.append(f"funding-in-post:{mentions[0]}")

        post.why_matched = "; ".join(post._reasons)
        if post.post_url and not _is_exact_post_url(post.post_url):
            _clear_post_url(post)

        kept.append(post)
        stats["kept"] += 1

        if limit is not None and len(kept) >= limit:
            log(f"reached limit of {limit}")
            break

        if delay:
            time.sleep(min(delay, 0.2))

    if defer_filters:
        return kept, stats, url

    # Apply the cheap raw-html href fallback if exactly one unresolved card
    unresolved_posts = [p for p in kept if not p.post_url]
    if len(unresolved_posts) == 1 and len(_href_urns) == 1:
        unresolved_posts[0].post_url = f"https://www.linkedin.com/feed/update/{_href_urns[0]}/"
        log("  -> permalink (href-fallback): " + unresolved_posts[0].post_url)

    # Add notes for any still unresolved posts
    for p in kept:
        if not p.post_url:
            p.post_url = "link unavailable — LinkedIn did not expose a permalink for this post (reshare/company/external)"

    resolved = sum(1 for p in kept if p.post_url and p.post_url.startswith("http"))
    log(f"post_url resolved {resolved} / {len(kept)} kept posts")
    return kept, stats, url


DEFAULT_MAX_QUERIES_PER_COMPANY = 7
# Absolute ceiling for CLI / agent / env overrides (keeps runaway volume bounded).
MAX_QUERIES_PER_COMPANY = 7

_MAX_QUERY_CHARS = 160
_MAX_QUERY_WORDS = 22


def company_search_query(company: str, role_query: str = "intern") -> str:
    """One structured LinkedIn keyword query — never a scraped post body."""
    name = (company or "").strip()
    intern = "(intern OR internship)"
    hiring = '(hiring OR "we\'re hiring" OR openings)'
    role = (role_query or "intern").strip() or "intern"
    if role.lower() in {"intern", "internship", "intern or internship"}:
        return f'"{name}" {intern} {hiring}'
    return f'"{name}" ({role}) {intern} {hiring}'


def is_post_body_query(query: str) -> bool:
    """True when `query` looks like scraped post text rather than keywords."""
    q = (query or "").strip()
    if not q:
        return False
    if "\n" in q:
        return True
    if len(q) > _MAX_QUERY_CHARS:
        return True
    if len(q.split()) > _MAX_QUERY_WORDS:
        return True
    if q.count(".") >= 2 and len(q) > 80:
        return True
    return False


def sanitize_company_queries(
    queries: list[str],
    company: str,
    role_query: str = "intern",
    *,
    max_queries: int = DEFAULT_MAX_QUERIES_PER_COMPANY,
) -> list[str]:
    """Drop post-body pastes; fall back to structured defaults if nothing remains."""
    kept: list[str] = []
    seen: set[str] = set()
    for raw in queries:
        q = str(raw).replace("{company}", company).strip()
        if not q or is_post_body_query(q):
            continue
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        kept.append(q)
        if len(kept) >= max(1, max_queries):
            break
    if not kept:
        return default_company_queries(
            company, role_query=role_query, max_queries=max_queries
        )
    return kept


def default_company_queries(
    company: str,
    role_query: str = "intern",
    *,
    max_queries: int = DEFAULT_MAX_QUERIES_PER_COMPANY,
) -> list[str]:
    """Diverse LinkedIn-style boolean queries for one company.

    Built from components (quoted name + role cluster + intern/hiring cues),
    not from scraped post text. Example:

      '"Sarvam AI" (intern OR internship) (hiring OR "we\'re hiring" OR openings)'
    """
    name = (company or "").strip()
    if not name:
        return []
    quoted = f'"{name}"'
    intern = "(intern OR internship)"
    hiring = '(hiring OR "we\'re hiring" OR openings)'
    role = (role_query or "intern").strip() or "intern"
    role_token = role.split(" OR ")[0].split()[0] if role else "intern"
    if role_token.lower() in {"or", "and"}:
        role_token = "intern"

    variations = [
        f"{quoted} intern",
        f"{quoted} {intern} {hiring}",
        f"{quoted} (SDE OR backend OR software) {intern}",
        f'{quoted} (ML OR data OR "machine learning") {intern} hiring',
        f"{quoted} intern hiring",
        f'{quoted} "we\'re hiring" {intern}',
        f"{quoted} {intern} openings",
    ]
    if role.lower() not in {"intern", "internship"}:
        variations.insert(1, f"{quoted} ({role}) {intern} {hiring}")

    out: list[str] = []
    seen: set[str] = set()
    for query in variations:
        key = query.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(query)
        if len(out) >= max(1, max_queries):
            break
    return out


def _post_dedupe_key(post: Post) -> str:
    if post.post_url and not _is_search_results_url(post.post_url):
        return post.post_url.rstrip("/").lower()
    normalized = re.sub(r"\s+", " ", post.post_text or "").strip().lower()
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return f"text:{(post.poster or '').strip().lower()}:{digest}"


def dedupe_posts_by_identity(posts: list[Post], seen: set[str] | None = None) -> list[Post]:
    """Keep one post per permalink, else per poster plus normalized-text hash."""
    keys = seen if seen is not None else set()
    unique: list[Post] = []
    for post in posts:
        key = _post_dedupe_key(post)
        if key in keys:
            continue
        keys.add(key)
        unique.append(post)
    return unique


def funnel_counts_are_monotonic(funnel: dict[str, Any]) -> bool:
    """Guard the four public funnel counters against instrumentation regressions."""
    return (
        int(funnel.get("seen", 0)) >= int(funnel.get("passed_prefilter", 0))
        >= int(funnel.get("sent_to_gate", 0))
        >= int(funnel.get("kept_by_gate", 0))
    )


def _strong_hits_for_company(
    found: list[Post],
    company: str,
    matcher: HiringMatcher,
    totals: dict[str, Any],
    *,
    strict_source: bool = True,
    prefilter: str = "strict",
    debug_strong: bool = False,
    debug_funnel: bool = False,
    drop_tally: dict[str, int] | None = None,
    log: Callable[[str], None] | None = None,
) -> list[tuple[Post, list[str]]]:
    """Apply strict opening checks or the broad semantic-gate candidate net."""
    kept: list[tuple[Post, list[str]]] = []
    for post in found:
        mention_text = " ".join(
            part
            for part in (
                post.post_text,
                post.company,
                post.poster,
                post.poster_headline,
            )
            if part
        )
        strong = (
            match_company_candidate_net(
                post.post_text or "", company, matcher=matcher, company_text=mention_text,
            )
            if prefilter == "loose"
            else match_strong_company(
                post.post_text or "", company, matcher=matcher, company_text=mention_text,
            )
        )
        if not strong.matched:
            totals["weak"] += 1
            reason = strong.why()
            totals.setdefault("prefilter_drops", {})[reason] = totals.setdefault("prefilter_drops", {}).get(reason, 0) + 1
            if drop_tally is not None:
                drop_tally[f"prefilter:{reason}"] = drop_tally.get(f"prefilter:{reason}", 0) + 1
            if (debug_strong or debug_funnel) and log:
                snippet = (post.post_text or "").replace("\n", " ")[:140]
                log(
                    f"    funnel→drop prefilter: {post.poster or '?'} — {reason} "
                    f"| {snippet!r}"
                )
            continue
        # In loose mode scraper has already performed the cheap source spam
        # rejection. Do not demand first-party proof here: that is semantic
        # gate territory. Keep a source label only for explainable output.
        if prefilter == "loose":
            source_type, _keep, reason = classify_first_party(
                company, poster=post.poster, headline=post.poster_headline,
                text=post.post_text, apply_links=list(post.outbound_links or []), strict=False,
            )
            post.source_type = source_type
            kept.append((post, list(strong.reasons) + [f"candidate-net:{reason}"]))
            continue
        apply_links = list(post.outbound_links or [])
        if post.job_link:
            apply_links.append(post.job_link)
        source_type, keep, reason = classify_first_party(
            company,
            poster=post.poster,
            headline=post.poster_headline,
            text=post.post_text,
            apply_links=apply_links,
            strict=strict_source,
        )
        post.source_type = source_type
        extra = opening_audit_reasons(
            poster=post.poster,
            headline=post.poster_headline,
            source_type=source_type,
            reason=reason,
            text=post.post_text,
        )
        if not keep:
            totals["not_first_party"] = totals.get("not_first_party", 0) + 1
            totals.setdefault("prefilter_drops", {})[f"source:{reason}"] = totals.setdefault("prefilter_drops", {}).get(f"source:{reason}", 0) + 1
            if drop_tally is not None:
                key = f"prefilter:source:{reason}"
                drop_tally[key] = drop_tally.get(key, 0) + 1
            if (debug_strong or debug_funnel) and log:
                log(
                    f"    funnel→drop prefilter: {post.poster or '?'} — "
                    f"{source_type}: {reason}"
                )
            continue
        kept.append((post, list(strong.reasons) + extra))
    return kept


def search_company_internship_posts(
    page,
    companies: list[str],
    *,
    max_age_days: float = 7.0,
    scrolls: int = 15,
    limit_per_company: int | None = None,
    delay: float = 2.0,
    company_delay: float = 5.0,
    matcher: HiringMatcher | None = None,
    role_query: str = "intern",
    queries: list[str] | None = None,
    queries_by_company: dict[str, list[str]] | None = None,
    max_queries_per_company: int = DEFAULT_MAX_QUERIES_PER_COMPANY,
    resolve_links: bool = True,
    max_resolve: int = 25,
    resolve_delay: float = 1.2,
    keep_undated: bool = False,
    network_idle_ms: int = 15_000,
    wait_for_posts_ms: int = 20_000,
    strict_source: bool = True,
    debug_strong: bool = False,
    semantic_client: LLMClient | None = None,
    prefilter: str | None = None,
    adaptive_gate: bool = True,
    gate_history_path: str = ".gate_history.jsonl",
    debug_funnel: bool = False,
    log: Callable[[str], None] = print,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Walk companies with several query variations; merge STRONG matches.

    For each company, runs up to `max_queries_per_company` LinkedIn searches
    (narrow → broader). A variation that returns 0 cards advances to the next.
    Results are deduped by permalink, or poster plus normalized-text hash when
    no permalink is available, before any prefilter or funnel accounting.

    `queries` applies the same variation list to every company (with the company
    name substituted when a pattern contains ``{company}``). Prefer
    `queries_by_company` when each firm needs its own list.
    """
    matcher = matcher or HiringMatcher(mode="all")
    if prefilter is None:
        prefilter = "loose" if semantic_client is not None else "strict"
    if prefilter not in {"strict", "loose"}:
        raise ValueError("prefilter must be 'strict' or 'loose'")
    max_q = max(
        1,
        min(
            MAX_QUERIES_PER_COMPANY,
            int(max_queries_per_company or DEFAULT_MAX_QUERIES_PER_COMPANY),
        ),
    )
    results: list[dict[str, str]] = []
    totals = {
        "searched": 0,
        "seen": 0,
        "strong": 0,
        "weak": 0,
        "queries_tried": 0,
        "resolve_attempted": 0,
        "resolve_copied": 0,
        "junk_link": 0,
        "source_spam": 0,
        "not_first_party": 0,
        "semantic_before": 0,
        "semantic_kept": 0,
        "drop_reasons": {},
        "funnel_by_company": {},
    }
    resolver = RedirectResolver(delay=0.35, log=log)
    permalink_cache: dict[str, str] = {}

    for index, company in enumerate(companies):
        if index > 0 and company_delay > 0:
            log(f"waiting {company_delay:g}s before next company…")
            time.sleep(company_delay)

        variations: list[str] = []
        if queries_by_company and company in queries_by_company:
            variations = [str(q).strip() for q in queries_by_company[company] if str(q).strip()]
        elif queries:
            for raw in queries:
                q = str(raw).strip()
                if not q:
                    continue
                variations.append(q.replace("{company}", company))
        variations = sanitize_company_queries(
            variations, company, role_query=role_query, max_queries=max_q
        )

        log(
            f"[{index + 1}/{len(companies)}] {company} — "
            f"{len(variations)} query variation(s) (last {max_age_days:g} days)"
        )
        totals["searched"] += 1

        merged: list[tuple[Post, list[str], str]] = []  # post, reasons, query
        identity_keys: set[str] = set()
        last_search_url = ""
        company_seen = 0
        company_hiring_pass = 0
        funnel = {
            "seen": 0, "passed_prefilter": 0, "sent_to_gate": 0,
            "kept_by_gate": 0, "drop_reasons": {},
        }

        for q_index, query in enumerate(variations):
            log(f"  query {q_index + 1}/{len(variations)}: {query!r}")
            totals["queries_tried"] += 1

            found, stats, search_url = scrape_posts(
                page,
                query=query,
                max_age_days=max_age_days,
                scrolls=scrolls,
                limit=None,
                delay=delay,
                matcher=matcher,
                keep_undated=keep_undated,
                scroll_until_max_age=True,
                network_idle_ms=network_idle_ms,
                wait_for_posts_ms=wait_for_posts_ms,
                log=log,
                link_resolver=resolver,
                target_company=company,
                candidate_net=prefilter == "loose",
                debug_funnel=debug_funnel,
                defer_filters=True,
            )
            last_search_url = search_url or last_search_url

            # Cross-query identity dedupe happens before prefiltering, source
            # checks, counters, link resolution, and the semantic gate.
            raw_count = len(found)
            unique_raw: list[Post] = []
            found = dedupe_posts_by_identity(found, identity_keys)
            if debug_funnel and len(found) != raw_count:
                log(f"    funnel→dedupe: collapsed {raw_count - len(found)} repeated post(s)")
            for post in found:
                company_seen += 1
                totals["seen"] += 1
                funnel["seen"] += 1

                keep, reason = _apply_filters(
                    post, matcher, max_age_days, keep_undated,
                    candidate_company=company if prefilter == "loose" else "",
                )
                if not keep:
                    label = "too_old" if reason.startswith("older") else (
                        "no_company_mention" if "company:none" in reason else "not_candidate"
                    )
                    key = f"prefilter:{label}"
                    funnel["drop_reasons"][key] = funnel["drop_reasons"].get(key, 0) + 1
                    if debug_funnel:
                        log(f"    funnel→drop prefilter: {post.poster or '?'} — {reason}")
                    continue

                reject, why = _apply_link_and_source_gates(
                    post, resolver, company=company or post.company,
                    require_opening=prefilter != "loose",
                )
                if reject:
                    label = f"source:{why}"
                    funnel["drop_reasons"][label] = funnel["drop_reasons"].get(label, 0) + 1
                    if why.startswith("junk-link"):
                        totals["junk_link"] += 1
                    else:
                        totals["source_spam"] += 1
                    if debug_funnel:
                        log(f"    funnel→drop source: {post.poster or '?'} — {why}")
                    continue

                mentions = extract_funding_mentions(post.post_text)
                if mentions:
                    post.funding_signal = "; ".join(mentions)
                    post._reasons.append(f"funding-in-post:{mentions[0]}")
                post.why_matched = "; ".join(post._reasons)
                if post.post_url and not _is_exact_post_url(post.post_url):
                    _clear_post_url(post)
                unique_raw.append(post)

            found = unique_raw
            company_hiring_pass += len(found)

            if not found:
                log("  → 0 posts; trying next broader variation…")
                continue

            strong_drops: dict[str, int] = {}
            strong_here = _strong_hits_for_company(
                found,
                company,
                matcher,
                totals,
                strict_source=strict_source,
                prefilter=prefilter,
                debug_strong=debug_strong,
                debug_funnel=debug_funnel,
                drop_tally=strong_drops,
                log=log,
            )
            for reason, count in strong_drops.items():
                funnel["drop_reasons"][reason] = funnel["drop_reasons"].get(reason, 0) + count
            added = 0
            new_strong: list[Post] = []
            for post, reasons in strong_here:
                merged.append((post, reasons, query))
                new_strong.append(post)
                added += 1
                if limit_per_company is not None and len(merged) >= limit_per_company:
                    break

            if resolve_links and new_strong:
                tally = resolve_post_links(
                    page,
                    new_strong,
                    search_url,
                    max_resolve=max_resolve,
                    resolve_delay=resolve_delay,
                    log=log,
                    cache=permalink_cache,
                )
                totals["resolve_attempted"] += tally.get("attempted", 0)
                totals["resolve_copied"] += tally.get("copied", 0)

            log(f"  → hiring-pass {len(found)} • new strong {added} • merged {len(merged)}")
            if limit_per_company is not None and len(merged) >= limit_per_company:
                break

        funnel["passed_prefilter"] = len(merged)
        for post, strong_reasons, query in merged:
            reasons: list[str] = []
            for reason in post._reasons:
                if reason.startswith("recency:") or reason.startswith("post_url:"):
                    reasons.append(reason)
            reasons.extend(strong_reasons)
            for reason in post._reasons:
                if reason.startswith("funding-in-post:") or reason.startswith("link:"):
                    reasons.append(reason)
            reasons.append(f"query:{query}")
            why = "; ".join(r for r in reasons if r)
            results.append(
                {
                    "company": company,
                    "poster": post.poster,
                    "poster_headline": post.poster_headline,
                    "snippet": make_snippet(post.post_text),
                    # Semantic gating consumes full text; serializers ignore this
                    # internal field because it is not in COMPANY_POST_FIELDS.
                    "post_text": post.post_text,
                    "post_url": post.post_url,
                    "why_matched": why,
                    "source_type": post.source_type,
                    "link_class": post.link_class,
                    "link_categories": post.link_categories,
                    "outbound_links": post.outbound_links,
                    "poster_employer": poster_employer_from_headline(post.poster_headline),
                    "author_affiliation_matches_target": author_affiliation_matches_target(
                        post.poster_headline, company
                    ),
                    "engagement": post.engagement,
                    "engagement_spam_signal": _engagement_spam_signal(
                        post.engagement, post.outbound_links
                    ),
                    "activity_urn": post.activity_urn,
                }
            )
            totals["strong"] += 1
        totals["funnel_by_company"][company] = funnel

        log(
            f"{company}: seen {company_seen} • hiring-pass {company_hiring_pass} • "
            f"prefilter {len(merged)} across {len(variations)} variation(s)"
        )

    if semantic_client is not None and results:
        totals["semantic_before"] = len(results)
        for row in results:
            totals["funnel_by_company"][row["company"]]["sent_to_gate"] += 1
        gate_drops: dict[str, dict[str, int]] = {}
        results = gate_company_posts_semantic(
            semantic_client,
            results,
            adaptive=adaptive_gate,
            history_path=gate_history_path,
            drop_tally=gate_drops,
            debug_funnel=debug_funnel,
            log=log,
        )
        totals["semantic_kept"] = len(results)
        for row in results:
            totals["funnel_by_company"][row["company"]]["kept_by_gate"] += 1
        for company, funnel in totals["funnel_by_company"].items():
            funnel["drop_reasons"].update({
                reason: funnel["drop_reasons"].get(reason, 0) + count
                for reason, count in gate_drops.get(company, {}).items()
            })
            log(
                f"funnel {company}: seen {funnel['seen']} → passed_prefilter "
                f"{funnel['passed_prefilter']} → sent_to_gate {funnel['sent_to_gate']} "
                f"→ kept_by_gate {funnel['kept_by_gate']}"
            )
            if funnel["drop_reasons"]:
                detail = ", ".join(
                    f"{reason}={count}"
                    for reason, count in sorted(funnel["drop_reasons"].items())
                )
                log(f"funnel {company} drops: {detail}")
    else:
        for company, funnel in totals["funnel_by_company"].items():
            # Keep the public funnel monotonic even when no LLM is configured:
            # the strict deterministic prefilter is the terminal decision stage.
            funnel["sent_to_gate"] = funnel["passed_prefilter"]
            funnel["kept_by_gate"] = funnel["passed_prefilter"]
            log(
                f"funnel {company}: seen {funnel['seen']} → passed_prefilter "
                f"{funnel['passed_prefilter']} → sent_to_gate {funnel['sent_to_gate']} "
                f"(semantic skipped) → kept_by_gate {funnel['kept_by_gate']}"
            )
            if funnel["drop_reasons"]:
                detail = ", ".join(
                    f"{reason}={count}"
                    for reason, count in sorted(funnel["drop_reasons"].items())
                )
                log(f"funnel {company} drops: {detail}")

    return results, totals
