"""LLM relevance + source-legitimacy gate after keyword pre-filter."""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from .filters import (
    KEEP_SOURCE_TYPES,
    SOURCE_TYPES,
    format_follower_count,
    parse_follower_count,
)
from .llm import LLMClient, LLMError, Message
from .output import make_snippet

LogFn = Callable[[str], None]

_SOURCE_ALIASES = {"farmer": "spam"}

GATE_PROMPT = """You are filtering LinkedIn posts for an internship-search tool.

For EACH post, decide if it describes a CONCRETE, REAL internship opening the
reader can act on RIGHT NOW. Poster identity is NOT a keep/reject filter.

KEEP (keep=true) when ALL of these are true:
- a specific role/title (e.g. Software Engineering Intern, AI Intern)
- the TARGET company is named as the hiring organisation
- an apply path exists: a URL, careers/ATS link, or "DM/comment for referral"
Accepted posters: company page, employee, recruiter, OR a credible individual
creator (high follower count, verified, established job-sharing account).

REJECT (keep=false) only for true junk:
- off-topic: role or company does not match the search (cybersecurity USA
  roundups when searching Microsoft internships; "2+ years experience" when
  we want interns; a different company is actually hiring)
- no concrete opening: motivational/engagement-bait/question posts with no
  specific role + apply path ("most juniors ask me one question…", generic
  "DM me")
- obvious scam / paid-service spam with no real role
- job-aggregator roundups and multi-company listicles

Do NOT reject an employee referral or a large-following creator just because
they are not the company page. High credibility can rescue a borderline post;
very low credibility + a vague post should be rejected.

source_type MUST be one of:
  company | employee | recruiter | creator | aggregator | spam

Return JSON only, shaped as:
{"results":[{"id":0,"keep":true,"reason":"short","source_type":"company"}]}

Posts:
"""


def _strip_fences(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _normalise_source_type(value: Any) -> str:
    raw = str(value or "").strip().lower()
    raw = _SOURCE_ALIASES.get(raw, raw)
    return raw if raw in SOURCE_TYPES else ""


def _parse_verdicts(text: str, n: int) -> dict[int, dict[str, Any]]:
    try:
        parsed = json.loads(_strip_fences(text))
    except json.JSONDecodeError:
        return {}

    rows = parsed.get("results", []) if isinstance(parsed, dict) else parsed
    if not isinstance(rows, list):
        return {}

    by_id: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            index = int(row.get("id"))
        except (TypeError, ValueError):
            continue
        if index < 0 or index >= n:
            continue
        if "keep" in row:
            keep = bool(row.get("keep"))
        else:
            keep = bool(row.get("hiring_now"))
        source_type = _normalise_source_type(row.get("source_type"))
        if keep and not source_type:
            if row.get("is_company_role"):
                source_type = "company"
            else:
                source_type = "employee"
        if not keep and not source_type:
            source_type = "spam"
        by_id[index] = {
            "keep": keep,
            "reason": str(row.get("reason") or "").strip(),
            "source_type": source_type,
        }
    return by_id


def _post_block(index: int, post: dict[str, Any]) -> str:
    company = post.get("company") or ""
    poster = post.get("poster") or ""
    headline = post.get("poster_headline") or post.get("headline") or ""
    role = post.get("role") or ""
    snippet = post.get("snippet") or make_snippet(str(post.get("post_text") or ""), 400)
    link_class = post.get("link_class") or ""
    links = post.get("outbound_links") or []
    if isinstance(links, str):
        links = [links]
    job_link = post.get("job_link") or ""
    extra_links = ", ".join(str(u) for u in list(links)[:6] if u)
    if job_link and job_link not in extra_links:
        extra_links = f"{job_link}; {extra_links}".strip("; ")
    return (
        f"[{index}] target_company={company!r} poster={poster!r} "
        f"headline={headline!r} role={role!r} link_class={link_class!r}\n"
        f"links: {extra_links or '(none)'}\n"
        f"text: {snippet}"
    )


def gate_hiring_posts(
    client: LLMClient,
    posts: list[dict[str, Any]],
    *,
    log: LogFn = print,
) -> list[dict[str, Any]]:
    """One batched LLM call; keep concrete openings (incl. creators)."""
    if not posts:
        return []

    log(f"Relevance gate: scoring {len(posts)} candidate(s) with {client.model}…")
    prompt = GATE_PROMPT + "\n\n".join(_post_block(i, p) for i, p in enumerate(posts))

    try:
        completion = client.complete(
            [Message(role="user", content=prompt)],
            temperature=0.0,
            json_mode=True,
        )
    except LLMError as exc:
        log(f"Relevance gate failed ({exc}); keeping keyword-filtered results.")
        return list(posts)

    by_id = _parse_verdicts(completion.content or "", len(posts))
    if not by_id:
        log("Relevance gate returned unparseable JSON; keeping keyword results.")
        return list(posts)

    kept: list[dict[str, Any]] = []
    dropped = 0
    for index, post in enumerate(posts):
        verdict = by_id.get(index)
        if verdict is None or not verdict["keep"]:
            dropped += 1
            continue

        source_type = verdict["source_type"] or "creator"
        if source_type not in KEEP_SOURCE_TYPES:
            dropped += 1
            continue

        enriched = dict(post)
        reason = verdict["reason"]
        prior = str(enriched.get("why_matched") or "").strip()
        poster = str(enriched.get("poster") or "").strip()
        headline = str(
            enriched.get("poster_headline") or enriched.get("headline") or ""
        )
        followers = parse_follower_count(f"{poster}\n{headline}\n{enriched.get('snippet') or ''}")
        gate_note = f"gate:keep,source:{source_type}"
        if poster:
            gate_note += f"; poster:{poster}"
        if followers is not None:
            gate_note += f"; followers:{format_follower_count(followers)}"
        if reason:
            gate_note += f" ({reason})"
        enriched["why_matched"] = f"{prior}; {gate_note}" if prior else gate_note
        enriched["source_type"] = source_type
        enriched["gate_reason"] = reason
        enriched["hiring_now"] = True
        enriched["is_company_role"] = source_type in KEEP_SOURCE_TYPES
        kept.append(enriched)

    rank = {"company": 0, "employee": 1, "recruiter": 2, "creator": 3}
    kept.sort(key=lambda row: rank.get(str(row.get("source_type") or ""), 9))

    log(f"Relevance gate: kept {len(kept)} / {len(posts)} (dropped {dropped}).")
    return kept


def normalize_agent_row(row: dict[str, Any]) -> dict[str, str]:
    """Compact fields written to results.txt / printed in the terminal."""
    company = str(row.get("company") or row.get("poster") or "").strip()
    role = str(row.get("role") or "").strip()
    if not role:
        snippet = str(row.get("snippet") or "").strip()
        role = make_snippet(snippet, 80) if snippet else ""
    source_type = str(row.get("source_type") or "").strip()
    why = str(row.get("why_matched") or "").strip()
    gate_reason = str(row.get("gate_reason") or "").strip()
    if gate_reason and gate_reason not in why:
        why = f"{why}; gate_reason:{gate_reason}" if why else f"gate_reason:{gate_reason}"
    return {
        "company": company,
        "role": role,
        "why_matched": why,
        "post_url": str(row.get("post_url") or "").strip(),
        "source_type": source_type,
    }


def format_result_blocks(rows: list[dict[str, str]]) -> str:
    lines: list[str] = []
    fields = ("company", "role", "why_matched", "post_url", "source_type")
    for index, row in enumerate(rows, start=1):
        lines.append(f"--- {index} ---")
        for field in fields:
            if field == "source_type" and not row.get(field):
                continue
            lines.append(f"{field}: {row.get(field, '')}")
        lines.append("")
    return "\n".join(lines).rstrip() + ("\n" if rows else "")
