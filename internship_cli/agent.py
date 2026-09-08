"""Interactive LLM agent that drives LinkedIn scrapers via tool calls."""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ContextManager, Iterator, Optional

from .browser import BrowserSessionError, ensure_logged_in, persistent_context
from .filters import (
    DEFAULT_DOMAIN_TERMS,
    DEFAULT_HIRING_TERMS,
    DEFAULT_ROLE_TERMS,
    MATCH_MODES,
    HiringMatcher,
    load_companies,
)
from .gate import gate_hiring_posts, normalize_agent_row
from .llm import LLMClient, LLMError, Message, ToolCall, get_backend, get_llm_client
from .output import make_snippet
from .posts import (
    DEFAULT_MAX_QUERIES_PER_COMPANY,
    MAX_QUERIES_PER_COMPANY,
    resolve_post_links,
    scrape_posts,
    search_company_internship_posts,
)

DEFAULT_MAX_TOOL_CALLS = 3
DEFAULT_MAX_COMPANIES = 5
DEFAULT_AGENT_SCROLLS = 5
DEFAULT_AGENT_LIMIT = 10
AGENT_RESULT_FIELDS = ("company", "role", "why_matched", "post_url", "source_type")

LogFn = Callable[[str], None]
BrowserFactory = Callable[[], ContextManager[Any]]


SYSTEM_PROMPT = """You are internship-radar's hiring-search agent.

You help the user find recent LinkedIn internship / hiring posts.
Parse each request for: role/keywords, location, recency (days), match strictness,
depth, and any company list. Then call a scraper tool with explicit parameters
(do not rely on hidden defaults when the user stated a preference).

When summarising matches, list each hit with:
- company (or poster if company is missing)
- why it matched (short)
- post link (post_url)

Tool results are already relevance-gated: only posts the local model judged as
hiring_now (open internship roles) remain. Prefer company/recruiter hits when
summarising. Do not re-include dropped celebrations or placement chatter.

Tool choice:
- Prefer scrape_posts for open keyword searches ("AI intern in Bengaluru, last 7 days").
- Prefer company_posts when the user names specific companies or a companies file.

Company multi-query (important):
- Correct obvious misspellings before searching (Servam→Sarvam, Infosis→Infosys).
- For company_posts, ALWAYS pass queries: about 10 distinct LinkedIn-style
  BOOLEAN keyword queries per company. Build them from components — quoted
  company name + role/domain terms (SDE, backend, ML, data, software) +
  intern/hiring cues (intern, internship, hiring, "we're hiring", openings).
  Vary the role cluster and the hiring cue so the set is diverse, not
  near-duplicates. Example for Sarvam AI:
  ['"Sarvam AI" (intern OR internship) (hiring OR "we\'re hiring" OR openings)',
   '"Sarvam AI" (SDE OR backend OR software) (intern OR internship)',
   '"Sarvam AI" (ML OR data OR "machine learning") (intern OR internship) hiring',
   '"Sarvam AI" intern hiring',
   '"Sarvam AI" "we\'re hiring" (intern OR internship)'].
- NEVER paste a scraped post body (or a long sentence from a post) back in as
  a query. Keywords only.
- Or pass queries_by_company when firms need different lists.
- The tool tries each variation, merges/dedupes hits, and advances automatically
  when a variation returns 0 posts. Do not rely on a single rigid query.

How to tune parameters from the request:
- max_age_days: map "last week"→7, "last 2 weeks"/"fortnight"→14, "last month"→30.
- scrolls / limit: if the user wants more results or "dig deeper", raise scrolls
  (and limit). If they want a quick sample, keep scrolls low (1–3).
- match_mode: default "all" (strict: hiring + intern role + domain). Use
  match_mode="loose" when the user says results are too strict, wants looser
  matches, or HR/general internship posts without an AI/ML keyword.
- query / role_query: refine LinkedIn keywords. If they ask for broader roles,
  widen wording (e.g. "AI intern" → "AI OR ML OR data science intern hiring").
- role_terms / domain_terms / hiring_terms: optional overrides for the text
  matcher when the user names specific signals to require or drop.
- resolve_links=true only when they ask for exact / permalink / copy-link URLs
  (slower; needs a visible browser).

Retry policy:
- If the first tool call returns count=0, you MAY retry once with adjusted
  parameters (looser match_mode, broader/corrected queries, higher scrolls, or
  larger max_age_days) before summarising. Stay within the tool-call budget.
- Keep LinkedIn load reasonable: few companies; ~7 query variations each is the
  default (hard cap {max_q}). Prefer raising company_delay over cutting queries.
- Do not invent posts, companies, or URLs that were not in tool results.
- After you have usable results (or a failed retry), give the final answer.
""".replace("{max_q}", str(MAX_QUERIES_PER_COMPANY))


def _tool(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str],
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


_MATCH_MODE_PROP = {
    "type": "string",
    "enum": list(MATCH_MODES),
    "description": (
        'Filter strictness. "all" = hiring AND internship role AND domain '
        '(default, precise). "loose" = hiring AND (role OR domain) — use when '
        "the user says matches are too strict or wants broader hits."
    ),
}

_SCROLLS_PROP = {
    "type": "integer",
    "description": (
        "Infinite-scroll rounds / search depth. Raise when the user wants more "
        "results or to dig deeper; keep small (1–3) for a quick sample. Default 5."
    ),
}

_RESOLVE_LINKS_PROP = {
    "type": "boolean",
    "description": (
        "If true, second-pass copy-link for exact /feed/update/… permalinks. "
        "Use only when the user asks for exact/permalink URLs. Slower."
    ),
}

_TERM_LIST_PROP = {
    "type": "array",
    "items": {"type": "string"},
    "description": "Optional override list (or pass a comma-separated string).",
}

AGENT_TOOLS: list[dict[str, Any]] = [
    _tool(
        "scrape_posts",
        "Keyword search of LinkedIn content/posts for internship hiring signals. "
        "Use for open-ended requests (role + optional location + recency). "
        "Set query yourself — broaden keywords when the user asks for wider roles "
        '(e.g. "AI OR ML OR data intern hiring"). Increase scrolls for more results; '
        'use match_mode="loose" if too few/strict; set resolve_links when they want '
        "exact permalinks.",
        {
            "query": {
                "type": "string",
                "description": (
                    "LinkedIn content-search keywords. Refine/broaden from the user "
                    "request (OR-combine related roles when they ask for broader)."
                ),
            },
            "location": {
                "type": "string",
                "description": "Optional location folded into the keywords.",
            },
            "max_age_days": {
                "type": "number",
                "description": (
                    "Recency window in days (keep newer posts). "
                    "Map last week→7, 2 weeks→14, month→30. Default 14."
                ),
            },
            "match_mode": _MATCH_MODE_PROP,
            "scrolls": _SCROLLS_PROP,
            "limit": {
                "type": "integer",
                "description": "Max posts to return after filtering. Default 10.",
            },
            "hiring_terms": {
                **_TERM_LIST_PROP,
                "description": (
                    "Optional hiring-intent terms for the matcher "
                    f"(default built-ins include {', '.join(DEFAULT_HIRING_TERMS[:4])}…)."
                ),
            },
            "role_terms": {
                **_TERM_LIST_PROP,
                "description": (
                    "Optional internship role terms to require/match in post text "
                    f"(defaults include {', '.join(DEFAULT_ROLE_TERMS[:4])})."
                ),
            },
            "domain_terms": {
                **_TERM_LIST_PROP,
                "description": (
                    "Optional domain terms (AI/ML/software…) for the matcher "
                    f"(defaults include {', '.join(DEFAULT_DOMAIN_TERMS[:6])}…)."
                ),
            },
            "resolve_links": _RESOLVE_LINKS_PROP,
            "max_resolve": {
                "type": "integer",
                "description": "Max posts to copy-link when resolve_links=true. Default 10.",
            },
        },
        ["query"],
    ),
    _tool(
        "company_posts",
        "Company-first LinkedIn search with several query variations per company "
        "(narrow → broader), merge/dedupe STRONG internship hiring matches. "
        "Correct misspellings in company names. Always pass queries (~7) when you "
        "can; if a variation returns 0 posts the tool tries the next broader one. "
        "Use when the user names companies or a companies file.",
        {
            "companies": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Company names to search (corrected spellings, e.g. Sarvam AI "
                    "not Servam AI)."
                ),
            },
            "companies_file": {
                "type": "string",
                "description": "Optional path to a companies.txt / CSV (one name per line).",
            },
            "queries": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "About 10 structured LinkedIn keyword queries per company. "
                    "Compose from quoted company + role terms (SDE/backend/ML/data/"
                    "software) + intern/hiring cues with quoting and OR. "
                    "Do NOT paste scraped post text. Example: "
                    '["\\"Sarvam AI\\" (intern OR internship) (hiring OR \\"we\'re hiring\\" OR openings)", '
                    '"\\"Sarvam AI\\" (SDE OR backend OR software) (intern OR internship)"]. '
                    "If omitted, structured defaults are generated from the company + role_query."
                ),
            },
            "queries_by_company": {
                "type": "object",
                "additionalProperties": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "description": (
                    "Optional per-company query lists (same rules as queries). "
                    "Overrides `queries` for named companies."
                ),
            },
            "max_queries_per_company": {
                "type": "integer",
                "description": (
                    "Hard cap on LinkedIn searches per company "
                    f"(default {DEFAULT_MAX_QUERIES_PER_COMPANY}, "
                    f"max {MAX_QUERIES_PER_COMPANY})."
                ),
            },
            "role_query": {
                "type": "string",
                "description": (
                    "Used only when generating default query variations "
                    '(default "intern").'
                ),
            },
            "max_age_days": {
                "type": "number",
                "description": (
                    "Recency window in days. Map last week→7, 2 weeks→14, month→30. "
                    "Default 7."
                ),
            },
            "match_mode": _MATCH_MODE_PROP,
            "scrolls": _SCROLLS_PROP,
            "limit_per_company": {
                "type": "integer",
                "description": "Max strong matches kept per company. Default 3.",
            },
            "max_companies": {
                "type": "integer",
                "description": (
                    f"Hard cap on companies searched this call. "
                    f"Default {DEFAULT_MAX_COMPANIES}."
                ),
            },
            "hiring_terms": {
                **_TERM_LIST_PROP,
                "description": "Optional hiring-intent term overrides for the matcher.",
            },
            "role_terms": {
                **_TERM_LIST_PROP,
                "description": "Optional internship role term overrides for the matcher.",
            },
            "domain_terms": {
                **_TERM_LIST_PROP,
                "description": "Optional domain term overrides for the matcher.",
            },
            "resolve_links": _RESOLVE_LINKS_PROP,
            "max_resolve": {
                "type": "integer",
                "description": "Max posts to copy-link when resolve_links=true. Default 10.",
            },
        },
        [],
    ),
]


def _parse_args(raw: str) -> dict[str, Any]:
    if not raw or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _as_int(value: Any, default: int, *, lo: int = 1, hi: int = 50) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, number))


def _as_float(value: Any, default: float, *, lo: float = 0.5, hi: float = 60.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, number))


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _as_terms(value: Any, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        return tuple(parts) if parts else default
    if isinstance(value, (list, tuple)):
        parts = [str(item).strip() for item in value if str(item).strip()]
        return tuple(parts) if parts else default
    return default


def _as_match_mode(value: Any, default: str = "all") -> str:
    mode = str(value or default).strip().lower() or default
    return mode if mode in MATCH_MODES else default


def _build_matcher(args: dict[str, Any], *, default_mode: str = "all") -> HiringMatcher:
    return HiringMatcher(
        hiring_terms=_as_terms(args.get("hiring_terms"), DEFAULT_HIRING_TERMS),
        role_terms=_as_terms(args.get("role_terms"), DEFAULT_ROLE_TERMS),
        domain_terms=_as_terms(args.get("domain_terms"), DEFAULT_DOMAIN_TERMS),
        mode=_as_match_mode(args.get("match_mode"), default_mode),
    )


def _as_query_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _as_queries_by_company(value: Any) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, list[str]] = {}
    for key, raw in value.items():
        name = str(key).strip()
        if not name:
            continue
        queries = _as_query_list(raw)
        if queries:
            out[name] = queries
    return out


def make_browser_factory(
    user_data_dir: str,
    *,
    headless: bool = False,
    log: LogFn = print,
) -> BrowserFactory:
    """Return a context manager that opens Chromium, then fully closes it."""

    @contextmanager
    def factory() -> Iterator[Any]:
        log("Opening browser for scraping…")
        with persistent_context(user_data_dir, headless=headless) as context:
            if not ensure_logged_in(context, headless=headless):
                raise BrowserSessionError(
                    "Not logged into LinkedIn. Run `internship-radar login` first."
                )
            page = context.pages[0] if context.pages else context.new_page()
            yield page
        log("Browser closed — freeing memory for the LLM.")

    return factory


@dataclass
class AgentRuntime:
    """Tool implementations. Browser is opened only while tools run."""

    page: Any = None
    browser_factory: Optional[BrowserFactory] = None
    log: LogFn = print
    delay: float = 2.0
    company_delay: float = 4.0
    max_companies: int = DEFAULT_MAX_COMPANIES
    max_queries_per_company: int = DEFAULT_MAX_QUERIES_PER_COMPANY

    @contextmanager
    def browser_session(self) -> Iterator[None]:
        """Open the browser for scraping; guarantee it is closed on exit."""
        if self.browser_factory is None:
            # Tests / pre-injected page: nothing to open or close.
            yield
            return

        with self.browser_factory() as page:
            self.page = page
            try:
                yield
            finally:
                self.page = None

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if self.page is None and self.browser_factory is not None:
            return {"error": "Browser is not open; cannot scrape."}
        if name == "scrape_posts":
            return self._scrape_posts(arguments)
        if name == "company_posts":
            return self._company_posts(arguments)
        return {"error": f"Unknown tool {name!r}. Available: scrape_posts, company_posts."}

    def _scrape_posts(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query") or "").strip()
        if not query:
            return {"error": "scrape_posts requires a non-empty query."}

        location = str(args.get("location") or "").strip()
        max_age_days = _as_float(args.get("max_age_days"), 14.0)
        limit = _as_int(args.get("limit"), DEFAULT_AGENT_LIMIT, hi=25)
        scrolls = _as_int(args.get("scrolls"), DEFAULT_AGENT_SCROLLS, hi=15)
        matcher = _build_matcher(args)
        resolve_links = _as_bool(args.get("resolve_links"), True)
        max_resolve = _as_int(args.get("max_resolve"), 10, hi=25)

        self.log(
            f"tool scrape_posts: query={query!r} location={location!r} "
            f"max_age_days={max_age_days:g} match_mode={matcher.mode} "
            f"limit={limit} scrolls={scrolls} resolve_links={resolve_links}"
        )
        found, stats, search_url = scrape_posts(
            self.page,
            query=query,
            max_age_days=max_age_days,
            scrolls=scrolls,
            limit=limit,
            delay=self.delay,
            location=location,
            matcher=matcher,
            log=lambda message: self.log(f"  {message}"),
        )

        resolve_tally: dict[str, int] = {}
        if found:
            resolve_tally = resolve_post_links(
                self.page,
                found,
                search_url,
                max_resolve=max_resolve,
                resolve_delay=max(self.delay, 1.2),
                log=lambda message: self.log(f"  {message}"),
            )

        results = [
            {
                "company": post.company or post.poster or "",
                "poster": post.poster,
                "poster_headline": post.poster_headline,
                "role": post.role,
                "location": post.location,
                "post_date": post.post_date,
                "snippet": make_snippet(post.post_text),
                "why_matched": post.why_matched,
                "post_url": post.post_url,
                "source_type": post.source_type,
                "link_class": post.link_class,
                "outbound_links": list(post.outbound_links),
                "job_link": post.job_link,
            }
            for post in found
        ]
        return {
            "tool": "scrape_posts",
            "query": query,
            "location": location,
            "max_age_days": max_age_days,
            "match_mode": matcher.mode,
            "scrolls": scrolls,
            "limit": limit,
            "resolve_links": resolve_links,
            "search_url": search_url,
            "stats": stats,
            "resolve_tally": resolve_tally or None,
            "count": len(results),
            "results": results,
        }

    def _company_posts(self, args: dict[str, Any]) -> dict[str, Any]:
        companies: list[str] = []
        raw_list = args.get("companies") or []
        if isinstance(raw_list, str):
            companies.extend(part.strip() for part in raw_list.split(",") if part.strip())
        elif isinstance(raw_list, list):
            companies.extend(str(item).strip() for item in raw_list if str(item).strip())

        companies_file = str(args.get("companies_file") or "").strip()
        if companies_file:
            path = Path(companies_file).expanduser()
            if not path.is_file():
                return {"error": f"companies_file not found: {companies_file}"}
            companies.extend(load_companies(path))

        # Preserve order, drop empties/dupes.
        companies = list(dict.fromkeys(c for c in companies if c))
        if not companies:
            return {
                "error": "company_posts needs companies=[] and/or companies_file=...",
            }

        cap = _as_int(
            args.get("max_companies"),
            self.max_companies,
            lo=1,
            hi=self.max_companies,
        )
        truncated = companies[cap:]
        companies = companies[:cap]

        max_age_days = _as_float(args.get("max_age_days"), 7.0)
        limit_per_company = _as_int(args.get("limit_per_company"), 3, hi=10)
        scrolls = _as_int(args.get("scrolls"), DEFAULT_AGENT_SCROLLS, hi=15)
        role_query = str(args.get("role_query") or "intern").strip() or "intern"
        max_queries = _as_int(
            args.get("max_queries_per_company"),
            getattr(self, "max_queries_per_company", DEFAULT_MAX_QUERIES_PER_COMPANY),
            hi=MAX_QUERIES_PER_COMPANY,
        )
        queries = _as_query_list(args.get("queries"))
        queries_by_company = _as_queries_by_company(args.get("queries_by_company"))
        matcher = _build_matcher(args)
        resolve_links = _as_bool(args.get("resolve_links"), True)
        max_resolve = _as_int(args.get("max_resolve"), 10, hi=25)

        self.log(
            f"tool company_posts: {len(companies)} company(ies), "
            f"role_query={role_query!r} max_age_days={max_age_days:g} "
            f"match_mode={matcher.mode} scrolls={scrolls} "
            f"max_queries_per_company={max_queries} resolve_links={resolve_links}"
        )
        if queries:
            self.log(f"  shared query variations: {queries!r}")
        results, totals = search_company_internship_posts(
            self.page,
            companies,
            max_age_days=max_age_days,
            scrolls=scrolls,
            limit_per_company=limit_per_company,
            delay=self.delay,
            company_delay=self.company_delay,
            matcher=matcher,
            role_query=role_query,
            queries=queries or None,
            queries_by_company=queries_by_company or None,
            max_queries_per_company=max_queries,
            resolve_links=resolve_links,
            max_resolve=max_resolve,
            resolve_delay=max(self.delay, 1.2),
            strict_source=True,
            log=lambda message: self.log(f"  {message}"),
        )
        payload: dict[str, Any] = {
            "tool": "company_posts",
            "companies": companies,
            "role_query": role_query,
            "queries": queries or None,
            "queries_by_company": queries_by_company or None,
            "max_queries_per_company": max_queries,
            "max_age_days": max_age_days,
            "match_mode": matcher.mode,
            "scrolls": scrolls,
            "resolve_links": resolve_links,
            "stats": totals,
            "count": len(results),
            "results": results,
        }
        if truncated:
            payload["warning"] = (
                f"Capped at {cap} companies; skipped: {', '.join(truncated[:10])}"
                + ("…" if len(truncated) > 10 else "")
            )
        return payload


@dataclass
class Agent:
    """One conversation turn (or REPL session) over the shared LLM + tools.

    Memory-aware sequencing: LLM calls never overlap a live Chromium session.
    Browser opens only for tool execution, then is fully closed before the
    relevance gate and summary LLM calls.
    """

    client: LLMClient
    runtime: AgentRuntime
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS
    history: list[Message] = field(default_factory=list)
    log: LogFn = print
    last_results: list[dict[str, str]] = field(default_factory=list)
    last_query: str = ""

    def __post_init__(self) -> None:
        if not self.history:
            self.history = [Message(role="system", content=SYSTEM_PROMPT)]

    def handle(self, user_text: str) -> str:
        text = (user_text or "").strip()
        if not text:
            return "Say what internship search you want (role, location, recency, companies)."

        self.last_query = text
        self.last_results = []
        self.history.append(Message(role="user", content=text))
        tool_calls_used = 0

        # Allow one extra LLM round after the cap so it can summarise.
        for _ in range(self.max_tool_calls + 2):
            allow_tools = tool_calls_used < self.max_tool_calls
            # After scraping, Chromium must already be closed (see browser_session).
            if tool_calls_used > 0:
                if allow_tools:
                    self.log(
                        f"reviewing results with {self.client.model} "
                        "(may retry with adjusted params or summarise)..."
                    )
                else:
                    self.log(f"summarizing with {self.client.model}...")

            try:
                completion = self.client.complete(
                    self.history,
                    tools=AGENT_TOOLS if allow_tools else None,
                    temperature=0.2,
                )
            except LLMError:
                if self.history and self.history[-1].role == "user":
                    self.history.pop()
                raise

            if completion.tool_calls and allow_tools:
                pending = completion.tool_calls
                remaining = self.max_tool_calls - tool_calls_used
                if len(pending) > remaining:
                    pending = pending[:remaining]

                self.history.append(
                    Message(
                        role="assistant",
                        content=completion.content or "",
                        tool_calls=pending,
                    )
                )
                # Scrape with browser, then close it before any Ollama work.
                raw_results: list[tuple[ToolCall, dict[str, Any]]] = []
                with self.runtime.browser_session():
                    for call in pending:
                        self.log(f"→ {call.name}({call.arguments})")
                        raw_results.append((call, self._run_tool(call)))
                        tool_calls_used += 1

                for call, result in raw_results:
                    gated = self._apply_relevance_gate(result)
                    self.history.append(
                        Message(
                            role="tool",
                            name=call.name,
                            tool_call_id=call.id,
                            content=json.dumps(gated, ensure_ascii=False),
                        )
                    )

                if tool_calls_used >= self.max_tool_calls:
                    self.log(
                        f"Tool-call cap reached ({self.max_tool_calls}); "
                        "asking the model to summarise."
                    )
                continue

            answer = (completion.content or "").strip()
            if not answer and tool_calls_used:
                answer = (
                    "I ran the search tools but could not produce a summary. "
                    "Check the tool output above for raw matches."
                )
            if not answer:
                answer = "I could not parse that request. Try naming a role, location, or companies."
            self.history.append(Message(role="assistant", content=answer))
            return answer

        fallback = (
            f"Stopped after {tool_calls_used} tool call(s) "
            f"(cap {self.max_tool_calls}) without a final answer."
        )
        self.history.append(Message(role="assistant", content=fallback))
        return fallback

    def _run_tool(self, call: ToolCall) -> dict[str, Any]:
        args = _parse_args(call.arguments)
        try:
            return self.runtime.execute(call.name, args)
        except Exception as exc:  # noqa: BLE001 - surface to the model, keep REPL alive
            return {"error": f"{type(exc).__name__}: {exc}"}

    def _apply_relevance_gate(self, result: dict[str, Any]) -> dict[str, Any]:
        """Filter tool results with one batched Ollama call (browser already closed)."""
        if result.get("error"):
            return result
        posts = result.get("results")
        if not isinstance(posts, list) or not posts:
            result = dict(result)
            result["count"] = 0
            result["results"] = []
            result["gated"] = True
            return result

        candidates = [p for p in posts if isinstance(p, dict)]
        before = len(candidates)
        kept = gate_hiring_posts(self.client, candidates, log=self.log)
        normalized = [normalize_agent_row(row) for row in kept]
        # Accumulate for --out / terminal listing (latest turn).
        self.last_results.extend(normalized)

        gated = dict(result)
        gated["results"] = kept
        gated["count"] = len(kept)
        gated["gated"] = True
        gated["gate_stats"] = {"before": before, "after": len(kept)}
        return gated


def build_agent(
    *,
    user_data_dir: Optional[str] = None,
    headless: bool = False,
    page: Any = None,
    model: Optional[str] = None,
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
    max_companies: int = DEFAULT_MAX_COMPANIES,
    max_queries_per_company: int = DEFAULT_MAX_QUERIES_PER_COMPANY,
    delay: float = 2.0,
    company_delay: float = 4.0,
    log: LogFn = print,
) -> Agent:
    """Build an agent.

    Prefer `user_data_dir` so the browser opens only during tool calls.
    Passing `page` directly is for tests / advanced callers.
    """
    client = get_llm_client(model)
    browser_factory: Optional[BrowserFactory] = None
    if user_data_dir is not None:
        browser_factory = make_browser_factory(
            user_data_dir, headless=headless, log=log
        )
    runtime = AgentRuntime(
        page=page,
        browser_factory=browser_factory,
        log=log,
        delay=delay,
        company_delay=company_delay,
        max_companies=max_companies,
        max_queries_per_company=max(
            1, min(MAX_QUERIES_PER_COMPANY, int(max_queries_per_company))
        ),
    )
    return Agent(
        client=client,
        runtime=runtime,
        max_tool_calls=max(1, max_tool_calls),
        log=log,
    )


def describe_backend(client: LLMClient) -> str:
    return f"{get_backend()} / {client.model}"
