"""Diagnostics for when LinkedIn changes its markup and selectors stop matching.

`--debug` dumps three artefacts:
  debug_page.html   the full rendered DOM
  debug_report.json a compact, structured summary (read this first)
  debug_page.png    a full-page screenshot

The report matters more than the HTML. A LinkedIn page is megabytes of minified
markup, so rather than requiring someone to read it, the discovery pass below
looks for elements that carry an activity URN — the one thing LinkedIn has kept
stable across redesigns — and reports their tags, classes and a truncated
sample. That usually names the current post-container selector outright, even
when every hardcoded guess returns zero.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

# Runs in the page. Finds post-like elements structurally instead of trusting
# any class name, then reports what they actually look like.
_DISCOVERY_JS = r"""
() => {
  const trunc = (s, n) => {
    s = s || '';
    return s.length > n ? s.slice(0, n) + '\u2026[truncated]' : s;
  };
  const classOf = (el) => {
    if (!el) return '';
    const c = el.className;
    return typeof c === 'string' ? c : (c && c.baseVal) || '';
  };

  const urnAttrs = [
    'data-urn', 'data-id', 'data-entity-urn', 'data-activity-urn',
    'data-chameleon-result-urn', 'data-finite-scroll-hotkey-item'
  ];
  const urnOf = (el) => {
    for (const a of urnAttrs) {
      const v = el.getAttribute(a);
      if (v && /activity|ugcPost|share/i.test(v)) return a + '=' + v;
    }
    return null;
  };

  const all = Array.from(document.querySelectorAll('*'));

  // 1. Elements carrying an activity URN: the most reliable structural anchor.
  const carriers = [];
  for (const el of all) {
    const urn = urnOf(el);
    if (urn) carriers.push({ el, urn });
  }

  const carrierClassFreq = {};
  const carrierAttrFreq = {};
  for (const { el } of carriers) {
    for (const t of classOf(el).split(/\s+/)) {
      if (t) carrierClassFreq[t] = (carrierClassFreq[t] || 0) + 1;
    }
    for (const a of Array.from(el.attributes)) {
      carrierAttrFreq[a.name] = (carrierAttrFreq[a.name] || 0) + 1;
    }
  }

  // 2. Class tokens that look feed/post related, across the whole document.
  const interesting = /feed|update|post|search-result|occludable|entity|actor|commentary|scaffold/i;
  const classFreq = {};
  for (const el of all) {
    for (const t of classOf(el).split(/\s+/)) {
      if (t && interesting.test(t)) classFreq[t] = (classFreq[t] || 0) + 1;
    }
  }

  // 3. Any element whose text mentions hiring, to prove posts really rendered.
  let hiringTextNodes = 0;
  for (const el of all) {
    if (el.children.length === 0 && /hiring|intern/i.test(el.textContent || '')) {
      hiringTextNodes++;
    }
  }

  const sortTop = (obj, n) =>
    Object.entries(obj).sort((a, b) => b[1] - a[1]).slice(0, n)
      .map(([k, v]) => ({ name: k, count: v }));

  const samples = carriers.slice(0, 3).map(({ el, urn }) => ({
    urn: urn,
    tag: el.tagName.toLowerCase(),
    classes: classOf(el),
    attributes: Array.from(el.attributes).map(a => a.name),
    parent_tag: el.parentElement ? el.parentElement.tagName.toLowerCase() : null,
    parent_classes: classOf(el.parentElement),
    text_preview: trunc((el.innerText || '').replace(/\s+/g, ' ').trim(), 300),
    outer_html: trunc(el.outerHTML, 6000)
  }));

  return {
    url: location.href,
    title: document.title,
    total_elements: all.length,
    body_text_length: (document.body ? (document.body.innerText || '').length : 0),
    urn_carrier_count: carriers.length,
    urn_carrier_classes: sortTop(carrierClassFreq, 25),
    urn_carrier_attributes: sortTop(carrierAttrFreq, 25),
    interesting_class_frequency: sortTop(classFreq, 40),
    elements_mentioning_hiring: hiringTextNodes,
    samples: samples
  };
}
"""


def count_candidates(page, groups: Mapping[str, Sequence[str]]) -> dict[str, list[dict]]:
    """Count how many elements each candidate selector matches, group by group."""
    counts: dict[str, list[dict]] = {}
    for group, selectors in groups.items():
        rows = []
        for selector in selectors:
            try:
                rows.append({"selector": selector, "count": page.locator(selector).count()})
            except Exception as exc:  # noqa: BLE001 - an invalid selector is itself a finding
                rows.append({"selector": selector, "count": -1, "error": str(exc)[:200]})
        counts[group] = rows
    return counts


def dump_debug(
    page,
    groups: Mapping[str, Sequence[str]],
    out_dir: str | Path = ".",
    log: Callable[[str], None] = print,
) -> dict:
    """Write the debug artefacts and return the report dict."""
    directory = Path(out_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)

    html_path = directory / "debug_page.html"
    report_path = directory / "debug_report.json"
    shot_path = directory / "debug_page.png"

    log("--- debug: selector match counts ---")
    counts = count_candidates(page, groups)
    for group, rows in counts.items():
        total = sum(r["count"] for r in rows if r["count"] > 0)
        log(f"  [{group}] {total} total match(es)")
        for row in rows:
            marker = "  " if row["count"] > 0 else "x "
            log(f"    {marker}{row['count']:>4}  {row['selector']}")

    try:
        discovery = page.evaluate(_DISCOVERY_JS)
    except Exception as exc:  # noqa: BLE001
        discovery = {"error": f"discovery script failed: {exc}"}
        log(f"  discovery script failed: {exc}")

    try:
        html = page.content()
        html_path.write_text(html, encoding="utf-8")
        log(f"  wrote {html_path} ({len(html) / 1024:.0f} KB)")
    except Exception as exc:  # noqa: BLE001
        log(f"  could not save HTML: {exc}")

    try:
        page.screenshot(path=str(shot_path), full_page=True)
        log(f"  wrote {shot_path}")
    except Exception as exc:  # noqa: BLE001
        log(f"  could not save screenshot: {exc}")

    report = {"selector_counts": counts, "discovery": discovery}
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"  wrote {report_path}")

    if isinstance(discovery, dict) and "error" not in discovery:
        log("--- debug: what the page actually contains ---")
        log(f"  final url            : {discovery.get('url')}")
        log(f"  page title           : {discovery.get('title')}")
        log(f"  total elements       : {discovery.get('total_elements')}")
        log(f"  activity-urn elements: {discovery.get('urn_carrier_count')}")
        log(f"  nodes saying hiring  : {discovery.get('elements_mentioning_hiring')}")

        carriers = discovery.get("urn_carrier_classes") or []
        if carriers:
            log("  most common classes on activity-urn elements:")
            for row in carriers[:8]:
                log(f"    {row['count']:>4}  .{row['name']}")
        else:
            log("  no elements carried an activity URN — the feed may not have rendered.")
            top = discovery.get("interesting_class_frequency") or []
            if top:
                log("  most common feed-ish classes seen instead:")
                for row in top[:10]:
                    log(f"    {row['count']:>4}  .{row['name']}")

    return report
