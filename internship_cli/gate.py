"""LLM relevance + source-legitimacy gate after keyword pre-filter."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from .filters import (
    KEEP_SOURCE_TYPES,
    SOURCE_TYPES,
    format_follower_count,
    is_structural_roundup,
    normalise_company,
    parse_follower_count,
)
from .llm import LLMClient, LLMError, Message
from .output import make_snippet

LogFn = Callable[[str], None]

_SOURCE_ALIASES = {"farmer": "spam"}

SEMANTIC_GATE_BATCH_SIZE = 5
DEFAULT_GATE_HISTORY_PATH = ".gate_history.jsonl"
MAX_PROFILE_EXAMPLES = 4
_SEMANTIC_INTENTS = frozenset({
    "company_hiring",
    "candidate_seeking",
    "project_showcase",
    "offer_celebration",
    "referral_or_roundup",
    "mention_only",
    "other",
})
_OBVIOUS_CANDIDATE_SEEKING_RE = re.compile(
    r"\b(?:i['’]?m|i am|we are)\s+(?:actively\s+)?looking\s+for\b"
    r"|\bopen\s+to\s+(?:work|opportunities|internships?)\b"
    r"|\bplease\s+(?:refer|refer me)\b"
    r"|\b(?:seeking|looking for)\s+(?:an?\s+)?(?:internship|intern\s+role|opportunity)\b",
    re.IGNORECASE,
)

SEMANTIC_GATE_PROMPT = """You are a strict semantic relevance gate for LinkedIn
internship results. Assess each post against its target_company. Do not infer a
hiring offer from a project, hackathon, Buildathon, portfolio, or a candidate
asking for work.

For EACH post, return exactly one JSON row with:
- id: the supplied integer
- author_is_hirer: true only if the poster/headline shows a target-company
  recruiter, employee, official page, or explicit referrer. A student,
  job-seeker, or person at another company is false.
- is_offer_not_request: true only if the post offers a role. First-person
  singular job seeking ("I'm looking for", "open to work") is false.
- cta_is_application: true only for an inbound application path: careers/ATS,
  apply here, or sending a CV to the hiring side. A GitHub, portfolio, demo, or
  no CTA is false.
- role_named: true only when a specific internship role is named.
- intent: exactly one of company_hiring, candidate_seeking, project_showcase,
  offer_celebration, referral_or_roundup, mention_only, other.
- reason: one concise sentence.

Structured priors are authoritative evidence, not decoration:
- link_categories containing ats_or_careers or company_domain strongly support
  cta_is_application. poster_portfolio_or_github, social_or_chat, and
  link_hub_or_junk strongly indicate a non-application CTA.
- a non-empty poster_employer that does not match the target strongly weighs
  against author_is_hirer. A student/no employer is not a positive affiliation.
- engagement is NEVER evidence to keep. It may only support a spam concern when
  zero engagement appears with multiple shortener/junk links.
- profile_prior=trusted_poster is a conservative local hint from repeated good
  history, not permission to keep a post that fails the required judgments.
- A named, single-company role with an ATS/careers/company application path is
  still a useful lead when posted by a third party. Do not require that poster
  to work for the target company. Only use author affiliation as proof for
  referral-only posts with no independent application path.

Return JSON only:
{"results":[{"id":0,"author_is_hirer":true,"is_offer_not_request":true,
 "cta_is_application":true,"role_named":true,"intent":"company_hiring",
 "reason":"Recruiter at target company advertises a named role with careers link."}]}

Posts:
"""

_GENERIC_FEW_SHOTS = """Generic examples:
- GOOD: Target-company recruiter posts “We are hiring a Software Engineering
  Intern” with an ATS link. intent=company_hiring.
- BAD: Student shares a Buildathon project and asks to be considered for an
  internship. intent=candidate_seeking or project_showcase.
"""

_JSON_RETRY_PROMPT = """
Your previous response was not a complete valid JSON verdict for every supplied
post. Return ONLY a JSON object with a `results` array. No markdown, prose, or
code fences. Include every id exactly once and use real JSON booleans.
"""

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
    snippet = make_snippet(
        str(post.get("post_text") or post.get("snippet") or ""),
        600,
    )
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
        f"link_categories: {post.get('link_categories') or []}\n"
        f"poster_employer: {post.get('poster_employer') or '(unknown)'}; "
        f"author_affiliation_matches_target: "
        f"{post.get('author_affiliation_matches_target', False)!r}\n"
        f"engagement: {post.get('engagement') or {}}\n"
        f"engagement_spam_signal: {post.get('engagement_spam_signal') or '(none)'}\n"
        f"profile_prior: {post.get('profile_prior') or '(none)'}; "
        f"profile_opening_domains: {post.get('profile_opening_domains') or []}\n"
        f"links: {extra_links or '(none)'}\n"
        f"text: {snippet}"
    )


def _history_key(row: dict[str, Any]) -> str:
    url = str(row.get("post_url") or "").strip()
    if url:
        return url
    return "|".join(
        str(row.get(key) or "").strip().lower()
        for key in ("company", "poster", "gate_reason")
    )


def load_gate_history(path: str | Path = DEFAULT_GATE_HISTORY_PATH) -> list[dict[str, Any]]:
    """Read local JSONL history; invalid/missing rows never interrupt a run."""
    file_path = Path(path).expanduser()
    if not file_path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    try:
        lines = file_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def write_gate_history(
    rows: list[dict[str, Any]],
    path: str | Path = DEFAULT_GATE_HISTORY_PATH,
) -> None:
    """Upsert personal gate rows by permalink while retaining rows without one."""
    file_path = Path(path).expanduser()
    existing = load_gate_history(file_path)
    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row in existing + rows:
        key = _history_key(row)
        if key not in merged:
            order.append(key)
        merged[key] = row
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(
            "".join(json.dumps(merged[key], ensure_ascii=False) + "\n" for key in order),
            encoding="utf-8",
        )
    except OSError:
        return


def apply_gate_labels(
    labels: list[dict[str, Any]],
    path: str | Path = DEFAULT_GATE_HISTORY_PATH,
) -> int:
    """Attach explicit good/bad feedback to existing history rows by permalink."""
    wanted = {
        str(row.get("post_url") or "").strip(): str(row.get("label") or "").lower()
        for row in labels
        if str(row.get("label") or "").lower() in {"good", "bad"}
        and str(row.get("post_url") or "").strip()
    }
    if not wanted:
        return 0
    rows = load_gate_history(path)
    updated = 0
    for row in rows:
        label = wanted.get(str(row.get("post_url") or "").strip())
        if not label:
            continue
        row["user_label"] = label
        row["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        updated += 1
    if updated:
        write_gate_history(rows, path)
    return updated


def _link_domains(post: dict[str, Any]) -> list[str]:
    domains: list[str] = []
    for raw in post.get("outbound_links") or []:
        host = (urlparse(str(raw)).hostname or "").lower()
        if host and host not in domains:
            domains.append(host)
    return domains


def _history_row(
    post: dict[str, Any],
    *,
    kept: bool,
    verdict: dict[str, Any] | None = None,
    reason: str = "",
) -> dict[str, Any]:
    verdict = verdict or {}
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "company": str(post.get("company") or ""),
        "poster": str(post.get("poster") or ""),
        "poster_employer": str(post.get("poster_employer") or ""),
        "post_url": str(post.get("post_url") or ""),
        "intent": str(verdict.get("intent") or "other"),
        "link_class": str(post.get("link_class") or ""),
        "link_domains": _link_domains(post),
        "kept": bool(kept),
        "gate_reason": reason or str(verdict.get("reason") or ""),
        "author_is_hirer": bool(verdict.get("author_is_hirer")),
        "is_offer_not_request": bool(verdict.get("is_offer_not_request")),
        "cta_is_application": bool(verdict.get("cta_is_application")),
        "role_named": bool(verdict.get("role_named")),
    }


def company_gate_profile(
    history: list[dict[str, Any]],
    company: str,
) -> dict[str, set[str] | list[dict[str, Any]]]:
    """Derive conservative local priors for one target company."""
    target = normalise_company(company)
    rows = [row for row in history if normalise_company(str(row.get("company") or "")) == target]
    positives: dict[str, int] = {}
    negatives: dict[str, int] = {}
    labelled_good: set[str] = set()
    labelled_bad: set[str] = set()
    domains: set[str] = set()
    labels: list[dict[str, Any]] = []
    for row in rows:
        poster = str(row.get("poster") or "").strip().lower()
        label = str(row.get("user_label") or "").strip().lower()
        positive = bool(row.get("kept")) or label == "good"
        negative = not bool(row.get("kept")) or label == "bad"
        if poster:
            positives[poster] = positives.get(poster, 0) + int(positive)
            negatives[poster] = negatives.get(poster, 0) + int(negative)
            if label == "good":
                labelled_good.add(poster)
            elif label == "bad":
                labelled_bad.add(poster)
        if positive:
            domains.update(str(value).lower() for value in row.get("link_domains") or [] if value)
        if label in {"good", "bad"}:
            labels.append(row)
    trusted = {
        poster for poster, count in positives.items()
        if count >= 2 or poster in labelled_good
    }
    blocked = {
        poster for poster, count in negatives.items()
        if count >= 2 or poster in labelled_bad
    }
    return {
        "trusted_posters": trusted - blocked,
        "blocked_posters": blocked,
        "opening_domains": domains,
        "labeled_examples": sorted(
            labels, key=lambda row: str(row.get("timestamp") or ""), reverse=True
        )[:MAX_PROFILE_EXAMPLES],
    }


def _few_shots(profile: dict[str, set[str] | list[dict[str, Any]]]) -> str:
    examples = profile.get("labeled_examples") or []
    if not examples:
        return _GENERIC_FEW_SHOTS
    lines = ["User-labelled examples for this target company:"]
    for row in examples:
        lines.append(
            f"- {str(row.get('user_label')).upper()}: poster={row.get('poster')!r}; "
            f"intent={row.get('intent')!r}; reason={row.get('gate_reason')!r}"
        )
    return "\n".join(lines)


def semantic_prefilter_reason(post: dict[str, Any]) -> str:
    """Return a cheap rejection reason for obvious candidate-seeking copy."""
    text = "\n".join(
        str(post.get(key) or "")
        for key in ("snippet", "post_text", "role")
    )
    if _OBVIOUS_CANDIDATE_SEEKING_RE.search(text):
        return "obvious candidate-seeking language"
    return ""


def _parse_semantic_verdicts(text: str, n: int) -> dict[int, dict[str, Any]]:
    """Parse only complete, schema-valid semantic verdicts."""
    try:
        parsed = json.loads(_strip_fences(text))
    except json.JSONDecodeError:
        return {}
    rows = parsed.get("results", []) if isinstance(parsed, dict) else None
    if not isinstance(rows, list):
        return {}

    verdicts: dict[int, dict[str, Any]] = {}
    required = (
        "author_is_hirer", "is_offer_not_request", "cta_is_application",
        "role_named", "intent", "reason",
    )
    for row in rows:
        if not isinstance(row, dict) or any(key not in row for key in required):
            continue
        try:
            index = int(row.get("id"))
        except (TypeError, ValueError):
            continue
        intent = str(row["intent"]).strip().lower()
        # bool(...) would accept strings such as "false"; require actual JSON booleans.
        if (
            index < 0 or index >= n
            or intent not in _SEMANTIC_INTENTS
            or any(not isinstance(row[key], bool) for key in required[:4])
        ):
            continue
        verdicts[index] = {
            "author_is_hirer": row["author_is_hirer"],
            "is_offer_not_request": row["is_offer_not_request"],
            "cta_is_application": row["cta_is_application"],
            "role_named": row["role_named"],
            "intent": intent,
            "reason": str(row["reason"]).strip(),
        }
    return verdicts


def _salvage_semantic_verdicts(text: str, n: int) -> dict[int, dict[str, Any]]:
    """Best-effort extraction of a JSON object embedded in a chatty response."""
    cleaned = _strip_fences(text)
    for start, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            parsed, _end = json.JSONDecoder().raw_decode(cleaned[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return _parse_semantic_verdicts(json.dumps(parsed), n)
    return {}


def semantic_keep(verdict: dict[str, Any], post: dict[str, Any] | None = None) -> bool:
    """Keep independently applyable openings; gate referral-only posts by author.

    This deliberately separates lead usefulness from poster provenance.  A
    third party can faithfully share a target-company ATS opening, while a bare
    referral request needs a target-affiliated hirer to be trustworthy.
    """
    if not (
        verdict.get("intent") == "company_hiring"
        and verdict.get("role_named")
    ):
        return False
    if not post:
        return bool(verdict.get("cta_is_application") or verdict.get("author_is_hirer"))

    text = "\n".join(str(post.get(key) or "") for key in ("post_text", "snippet"))
    if is_structural_roundup(text):
        return False
    categories = {str(value) for value in (post.get("link_categories") or [])}
    application = {"ats_or_careers", "company_domain"}
    non_application = {
        "poster_portfolio_or_github", "social_or_chat", "link_hub_or_junk",
    }
    link_class = str(post.get("link_class") or "").lower()
    explicit_apply = bool(re.search(
        r"\b(?:apply\s+(?:here|now|via|using)|application\s+(?:form|link)|"
        r"(?:share|send)\s+(?:your\s+)?(?:resume|cv)\s+to\s+\w+)",
        text,
        re.IGNORECASE,
    ))
    has_application_path = bool(
        categories.intersection(application)
        or link_class in {"ats_or_careers", "company_domain", "careers"}
        or explicit_apply
    )
    if has_application_path:
        # A self-promo/portfolio-only CTA is already excluded above; an actual
        # application path wins even if the post also links to social context.
        return True
    if categories.intersection(non_application):
        return False
    employer = str(post.get("poster_employer") or "").strip()
    return bool(
        verdict.get("author_is_hirer")
        and verdict.get("is_offer_not_request")
        and (not employer or post.get("author_affiliation_matches_target"))
    )


def semantic_source_type(post: dict[str, Any], verdict: dict[str, Any]) -> str:
    """Expose whether a kept lead came from the target or a share with apply link."""
    existing = str(post.get("source_type") or "").lower()
    if existing in {"company", "company_page"}:
        return "company_page"
    if existing in {"employee", "recruiter"}:
        return existing
    if verdict.get("author_is_hirer") and post.get("author_affiliation_matches_target"):
        headline = str(post.get("poster_headline") or "")
        return "recruiter" if re.search(r"\b(?:recruiter|talent|sourcer)\b", headline, re.I) else "employee"
    return "third_party_with_apply_link"


def gate_company_posts_semantic(
    client: LLMClient,
    posts: list[dict[str, Any]],
    *,
    log: LogFn = print,
    batch_size: int = SEMANTIC_GATE_BATCH_SIZE,
    adaptive: bool = True,
    history_path: str | Path | None = None,
    drop_tally: dict[str, dict[str, int]] | None = None,
    debug_funnel: bool = False,
) -> list[dict[str, Any]]:
    """Fail-closed semantic gate for deterministic company-post matches.

    An unavailable backend is an operational skip: callers retain their
    deterministic result set. A response that is malformed or incomplete is a
    model verdict failure and drops that batch rather than accidentally keeping
    a false positive.
    """
    if not posts:
        return []

    history = load_gate_history(history_path) if adaptive and history_path else []
    candidates_by_company: dict[str, list[dict[str, Any]]] = {}
    history_rows: list[dict[str, Any]] = []

    def dropped(post: dict[str, Any], reason: str) -> None:
        if drop_tally is not None:
            company = str(post.get("company") or "")
            company_tally = drop_tally.setdefault(company, {})
            company_tally[reason] = company_tally.get(reason, 0) + 1
        if debug_funnel:
            log(
                f"    funnel→drop gate: {post.get('poster') or '?'} — {reason}"
            )

    for post in posts:
        reason = semantic_prefilter_reason(post)
        if reason:
            log(f"Semantic gate: dropped {post.get('company') or 'candidate'} ({reason}).")
            dropped(post, "gate:candidate_seeking")
            history_rows.append(_history_row(post, kept=False, reason=reason))
            continue
        company = str(post.get("company") or "")
        candidates_by_company.setdefault(company, []).append(post)

    kept: list[dict[str, Any]] = []
    size = max(1, min(SEMANTIC_GATE_BATCH_SIZE, int(batch_size)))
    for company, candidates in candidates_by_company.items():
        profile = company_gate_profile(history, company) if adaptive else {
            "trusted_posters": set(), "blocked_posters": set(),
            "opening_domains": set(), "labeled_examples": [],
        }
        trusted = profile["trusted_posters"]
        blocked = profile["blocked_posters"]
        examples = profile["labeled_examples"]
        if adaptive:
            log(
                f"loaded profile for {company}: {len(trusted)} trusted posters, "
                f"{len(blocked)} blocked, {len(examples)} labeled examples."
            )
        eligible: list[dict[str, Any]] = []
        for post in candidates:
            poster = str(post.get("poster") or "").strip().lower()
            if poster and poster in blocked:
                reason = "blocked by conservative local profile"
                log(f"Semantic gate: dropped {post.get('company') or 'candidate'} ({reason}).")
                dropped(post, "gate:blocked_profile")
                history_rows.append(_history_row(post, kept=False, reason=reason))
                continue
            enriched = dict(post)
            if poster in trusted:
                enriched["profile_prior"] = "trusted_poster"
            if profile["opening_domains"]:
                enriched["profile_opening_domains"] = sorted(profile["opening_domains"])
            eligible.append(enriched)

        for start in range(0, len(eligible), size):
            batch = eligible[start:start + size]
            prompt = (
                SEMANTIC_GATE_PROMPT + _few_shots(profile) + "\n\n"
                + "\n\n".join(_post_block(index, post) for index, post in enumerate(batch))
            )
            try:
                completion = client.complete(
                    [Message(role="user", content=prompt)],
                    temperature=0.0,
                    json_mode=True,
                )
            except LLMError as exc:
                log(f"Semantic gate unavailable ({exc}); skipping semantic gate.")
                return list(posts)

            verdicts = _parse_semantic_verdicts(completion.content or "", len(batch))
            if len(verdicts) != len(batch):
                salvaged = _salvage_semantic_verdicts(completion.content or "", len(batch))
                if len(salvaged) == len(batch):
                    verdicts = salvaged
                    log("Semantic gate: salvaged JSON embedded in a non-JSON response.")
                else:
                    log(
                        "WARNING: semantic gate returned invalid/incomplete JSON; "
                        "retrying once with a strict JSON-only reprompt."
                    )
                    try:
                        retry = client.complete(
                            [Message(role="user", content=prompt + _JSON_RETRY_PROMPT)],
                            temperature=0.0,
                            json_mode=True,
                        )
                    except LLMError as exc:
                        log(f"Semantic gate unavailable on JSON retry ({exc}); skipping semantic gate.")
                        return list(posts)
                    verdicts = _parse_semantic_verdicts(retry.content or "", len(batch))
                    if len(verdicts) != len(batch):
                        verdicts = _salvage_semantic_verdicts(retry.content or "", len(batch))
                    if len(verdicts) != len(batch):
                        log("WARNING: semantic gate JSON retry also failed; dropping that batch fail-closed.")
            for index, post in enumerate(batch):
                verdict = verdicts.get(index)
                if verdict is None:
                    reason = "missing or invalid JSON verdict"
                    log(f"Semantic gate: dropped post with {reason}.")
                    dropped(post, "gate:invalid_json")
                    history_rows.append(_history_row(post, kept=False, reason=reason))
                    continue
                if not semantic_keep(verdict, post):
                    log(
                        f"Semantic gate: dropped {post.get('company') or 'candidate'} "
                        f"({verdict['intent']}: {verdict['reason']})."
                    )
                    employer = str(post.get("poster_employer") or "").strip()
                    categories = set(post.get("link_categories") or [])
                    text = "\n".join(str(post.get(key) or "") for key in ("post_text", "snippet"))
                    has_apply = bool(
                        categories.intersection({"ats_or_careers", "company_domain"})
                        or str(post.get("link_class") or "").lower() in {"ats_or_careers", "company_domain", "careers"}
                        or re.search(r"\b(?:apply\s+(?:here|now|via|using)|application\s+(?:form|link))", text, re.I)
                    )
                    if verdict.get("intent") == "candidate_seeking":
                        drop_reason = "gate:candidate_seeking"
                    elif is_structural_roundup(text):
                        drop_reason = "gate:roundup"
                    elif employer and not post.get("author_affiliation_matches_target") and not has_apply:
                        drop_reason = "gate:employer_mismatch"
                    elif categories.intersection({"poster_portfolio_or_github", "social_or_chat", "link_hub_or_junk"}):
                        drop_reason = "gate:no_application_cta"
                    elif not verdict.get("author_is_hirer"):
                        drop_reason = "gate:not_hirer"
                    else:
                        drop_reason = "gate:rejected"
                    dropped(post, drop_reason)
                    history_rows.append(_history_row(post, kept=False, verdict=verdict))
                    continue
                enriched = dict(post)
                decision = (
                    "semantic:company_hiring"
                    f"; author_is_hirer:{str(verdict['author_is_hirer']).lower()}"
                    f"; offer_not_request:{str(verdict['is_offer_not_request']).lower()}"
                    f"; cta_is_application:{str(verdict['cta_is_application']).lower()}"
                    f"; role_named:{str(verdict['role_named']).lower()}"
                )
                prior = str(enriched.get("why_matched") or "").strip()
                enriched["why_matched"] = f"{prior}; {decision}" if prior else decision
                enriched["gate_reason"] = verdict["reason"]
                enriched["source_type"] = semantic_source_type(enriched, verdict)
                kept.append(enriched)
                history_rows.append(_history_row(enriched, kept=True, verdict=verdict))

    if history_path:
        write_gate_history(history_rows, history_path)
    log(f"Semantic gate: kept {len(kept)} / {len(posts)} deterministic candidate(s).")
    return kept


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
