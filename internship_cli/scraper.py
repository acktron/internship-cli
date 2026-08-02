"""LinkedIn job search result extraction.

LinkedIn's class names change frequently, so every field is resolved against a
list of candidate selectors and falls back to text heuristics.
"""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass
from typing import Callable, Iterable
from urllib.parse import urlencode, urlparse, urlunparse

SEARCH_URL = "https://www.linkedin.com/jobs/search/"
RESULTS_PER_PAGE = 25

_CARD_SELECTORS = (
    "div.job-card-container",
    "li.scaffold-layout__list-item",
    "li.jobs-search-results__list-item",
    "div[data-job-id]",
    "li[data-occludable-job-id]",
)

_TITLE_SELECTORS = (
    "a.job-card-container__link",
    "a.job-card-list__title--link",
    "a.job-card-list__title",
    ".job-card-list__title",
    ".artdeco-entity-lockup__title",
    "a[href*='/jobs/view/']",
)

_COMPANY_SELECTORS = (
    ".artdeco-entity-lockup__subtitle",
    ".job-card-container__primary-description",
    ".job-card-container__company-name",
    "[class*='subtitle'] span",
)

_LOCATION_SELECTORS = (
    ".job-card-container__metadata-wrapper li",
    "ul.job-card-container__metadata-wrapper li span",
    ".artdeco-entity-lockup__caption li",
    ".artdeco-entity-lockup__caption",
    ".job-card-container__metadata-item",
)

_SCROLL_CONTAINERS = (
    "div.jobs-search-results-list",
    "div.scaffold-layout__list-detail-inner div.scaffold-layout__list",
    "div.scaffold-layout__list",
    "main",
)

_NO_RESULTS_MARKERS = (
    "no matching jobs found",
    "no results found",
    "we couldn't find",
)


@dataclass(frozen=True)
class Job:
    title: str
    company: str
    location: str
    link: str

    def as_dict(self) -> dict:
        return asdict(self)


def build_search_url(
    query: str,
    location: str = "",
    start: int = 0,
    remote_only: bool = False,
    posted_within_days: int | None = None,
) -> str:
    params: dict[str, str] = {"keywords": query}
    if location:
        params["location"] = location
    if start:
        params["start"] = str(start)
    if remote_only:
        params["f_WT"] = "2"  # LinkedIn's "Remote" workplace-type filter
    if posted_within_days:
        params["f_TPR"] = f"r{int(posted_within_days) * 86_400}"

    parsed = urlparse(SEARCH_URL)
    return urlunparse(parsed._replace(query=urlencode(params)))


def _dedupe_text(raw: str) -> str:
    """Collapse whitespace and drop LinkedIn's duplicated accessibility text."""
    if not raw:
        return ""

    lines = [re.sub(r"\s+", " ", line).strip() for line in raw.splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return ""

    seen: list[str] = []
    for line in lines:
        if line not in seen:
            seen.append(line)
    text = " ".join(seen).strip(" ·-–—")

    # "Acme Corp Acme Corp" -> "Acme Corp"
    halves = text.split()
    if len(halves) % 2 == 0:
        mid = len(halves) // 2
        if halves[:mid] == halves[mid:]:
            text = " ".join(halves[:mid])

    return text.strip()


def _first_text(card, selectors: Iterable[str]) -> str:
    for selector in selectors:
        try:
            node = card.locator(selector).first
            if node.count() == 0:
                continue
            text = _dedupe_text(node.inner_text(timeout=1_500))
            if text:
                return text
        except Exception:  # noqa: BLE001 - selector drift, try the next candidate
            continue
    return ""


def _clean_link(href: str) -> str:
    if not href:
        return ""
    if href.startswith("/"):
        href = "https://www.linkedin.com" + href
    parsed = urlparse(href)
    # Strip tracking query params; the /jobs/view/<id>/ path is the stable identity.
    return urlunparse(parsed._replace(query="", fragment=""))


def _job_id(link: str) -> str:
    match = re.search(r"/jobs/view/(\d+)", link or "")
    return match.group(1) if match else (link or "")


def _find_cards(page):
    for selector in _CARD_SELECTORS:
        try:
            locator = page.locator(selector)
            if locator.count() > 0:
                return locator
        except Exception:  # noqa: BLE001
            continue
    return None


def _scroll_results(page, pause_ms: int = 450, rounds: int = 8) -> None:
    """Nudge the virtualised results list so every card renders."""
    container = None
    for selector in _SCROLL_CONTAINERS:
        try:
            candidate = page.locator(selector).first
            if candidate.count() > 0:
                container = candidate
                break
        except Exception:  # noqa: BLE001
            continue

    for step in range(rounds):
        try:
            if container is not None:
                container.evaluate(
                    "el => el.scrollBy(0, el.clientHeight || 600)"
                )
            else:
                page.mouse.wheel(0, 800)
        except Exception:  # noqa: BLE001 - scrolling is best-effort
            page.mouse.wheel(0, 800)
        page.wait_for_timeout(pause_ms)

        if step == rounds - 1:
            try:
                page.keyboard.press("End")
            except Exception:  # noqa: BLE001
                pass


def _extract_cards(page) -> list[Job]:
    cards = _find_cards(page)
    if cards is None:
        return []

    jobs: list[Job] = []
    for index in range(cards.count()):
        card = cards.nth(index)
        try:
            link_node = card.locator("a[href*='/jobs/view/']").first
            href = link_node.get_attribute("href", timeout=1_500) if link_node.count() else ""
        except Exception:  # noqa: BLE001
            href = ""

        link = _clean_link(href or "")
        title = _first_text(card, _TITLE_SELECTORS)
        company = _first_text(card, _COMPANY_SELECTORS)
        location = _first_text(card, _LOCATION_SELECTORS)

        if not (title or link):
            continue

        jobs.append(
            Job(
                title=title or "(unknown title)",
                company=company or "(unknown company)",
                location=location or "",
                link=link,
            )
        )

    return jobs


def _page_says_no_results(page) -> bool:
    try:
        body = (page.locator("body").inner_text(timeout=3_000) or "").lower()
    except Exception:  # noqa: BLE001
        return False
    return any(marker in body for marker in _NO_RESULTS_MARKERS)


def scrape_jobs(
    page,
    query: str,
    location: str = "",
    pages: int = 1,
    limit: int | None = None,
    delay: float = 2.5,
    remote_only: bool = False,
    posted_within_days: int | None = None,
    log: Callable[[str], None] = print,
) -> list[Job]:
    """Page through LinkedIn job search results and return deduplicated jobs.

    `delay` throttles navigation between pages. Keep it at a couple of seconds:
    hammering the site is both rude and the fastest way to get flagged.
    """
    collected: list[Job] = []
    seen: set[str] = set()

    for page_index in range(max(1, pages)):
        url = build_search_url(
            query,
            location=location,
            start=page_index * RESULTS_PER_PAGE,
            remote_only=remote_only,
            posted_within_days=posted_within_days,
        )
        log(f"Page {page_index + 1}/{pages}: {url}")

        try:
            page.goto(url, wait_until="domcontentloaded", timeout=45_000)
        except Exception as exc:  # noqa: BLE001
            log(f"  Could not load page {page_index + 1}: {exc}")
            break

        page.wait_for_timeout(2_000)

        if "/login" in (page.url or "") or "/authwall" in (page.url or ""):
            log("  LinkedIn redirected to a login wall — the session expired.")
            break

        _scroll_results(page)
        found = _extract_cards(page)

        if not found:
            if _page_says_no_results(page):
                log("  LinkedIn reported no matching jobs.")
            else:
                log("  No job cards matched the known selectors on this page.")
            break

        new_count = 0
        for job in found:
            key = _job_id(job.link) or f"{job.title}|{job.company}|{job.location}"
            if key in seen:
                continue
            seen.add(key)
            collected.append(job)
            new_count += 1
            if limit is not None and len(collected) >= limit:
                log(f"  Collected {new_count} new (limit of {limit} reached).")
                return collected

        log(f"  Collected {new_count} new job(s), {len(collected)} total.")

        if page_index < pages - 1 and delay > 0:
            time.sleep(delay)

    return collected
