"""Command line interface for internship-radar."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from . import __version__
from .agent import (
    AGENT_RESULT_FIELDS,
    DEFAULT_MAX_COMPANIES,
    DEFAULT_MAX_TOOL_CALLS,
    build_agent,
    describe_backend,
)
from .browser import (
    DEFAULT_USER_DATA_DIR,
    BrowserSessionError,
    ensure_logged_in,
    persistent_context,
    profile_hint,
)
from .gate import format_result_blocks
from .classify import ClassificationError, CompanyClassifier, is_startup
from .filters import (
    DEFAULT_DOMAIN_TERMS,
    DEFAULT_HIRING_TERMS,
    DEFAULT_ROLE_TERMS,
    MATCH_MODES,
    HiringMatcher,
    company_in_allowlist,
    load_allowlist,
    load_companies,
)
from .llm import LLMError, load_project_env
from .output import (
    COMPANY_POST_FIELDS,
    FORMATS,
    JOB_FIELDS,
    POST_FIELDS,
    TXT_FIELDS,
    infer_format,
    write_results,
)
from .posts import resolve_post_links, scrape_posts, search_company_internship_posts, DEFAULT_MAX_QUERIES_PER_COMPANY, MAX_QUERIES_PER_COMPANY
from .scraper import scrape_jobs

WARNING = (
    "Automated scraping of LinkedIn violates their User Agreement and can get "
    "your account restricted or banned. Use at your own risk, on your own account, "
    "and keep the volume low."
)


def _profile_option(func):
    return click.option(
        "--user-data-dir",
        default=DEFAULT_USER_DATA_DIR,
        show_default=True,
        help="Directory holding the persistent Chromium profile (your LinkedIn session).",
    )(func)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="internship-radar")
def main() -> None:
    """Scrape LinkedIn job search results from your own logged-in browser.

    WARNING: scraping LinkedIn violates their Terms of Service and may result in
    account restriction. Use at your own risk.

    LLM settings are read from a project-root `.env` (LLM_BACKEND, OLLAMA_MODEL,
    GEMINI_API_KEY) so you do not need to export them in every terminal.
    """
    load_project_env()


@main.command()
@_profile_option
def login(user_data_dir: str) -> None:
    """Open a browser window so you can sign into LinkedIn once."""
    click.secho(profile_hint(user_data_dir), fg="cyan")
    try:
        with persistent_context(user_data_dir, headless=False) as context:
            if ensure_logged_in(context, headless=False):
                click.secho("You are ready to run `internship-radar search`.", fg="green")
            else:
                sys.exit(1)
    except BrowserSessionError as exc:
        click.secho(str(exc), fg="red")
        sys.exit(1)


@main.command()
@click.argument("query")
@click.option("--location", "-l", default="", help="Location filter, e.g. \"Bengaluru, India\".")
@click.option("--out", "-o", "out_path", default="results.json", show_default=True,
              help="Output file path.")
@click.option("--format", "-f", "fmt", type=click.Choice(FORMATS), default=None,
              help="Output format. Defaults to the extension of --out (json if unknown).")
@click.option("--pages", "-p", default=1, show_default=True,
              help="How many result pages to walk (25 jobs per page).")
@click.option("--limit", "-n", default=None, type=int, help="Stop after this many jobs.")
@click.option("--delay", default=2.5, show_default=True,
              help="Seconds to wait between page loads. Please do not set this to 0.")
@click.option("--remote", "remote_only", is_flag=True, help="Only remote roles.")
@click.option("--posted-within", "posted_within_days", default=None, type=int,
              help="Only jobs posted within the last N days.")
@click.option("--headless", is_flag=True,
              help="Run without a visible window (requires an existing logged-in profile).")
@_profile_option
def search(
    query: str,
    location: str,
    out_path: str,
    fmt: str | None,
    pages: int,
    limit: int | None,
    delay: float,
    remote_only: bool,
    posted_within_days: int | None,
    headless: bool,
    user_data_dir: str,
) -> None:
    """Search LinkedIn jobs for QUERY and write the results to a file.

    Example:

        internship-radar search "backend internship" --location "Remote" --out jobs.csv
    """
    click.secho(f"WARNING: {WARNING}", fg="yellow")
    click.secho(profile_hint(user_data_dir), fg="cyan")

    if delay < 1:
        click.secho(
            "Note: a delay below 1s hits LinkedIn harder than a human would and "
            "raises your odds of being flagged.",
            fg="yellow",
        )

    fmt = infer_format(out_path, fmt)

    try:
        with persistent_context(user_data_dir, headless=headless) as context:
            if not ensure_logged_in(context, headless=headless):
                sys.exit(1)

            page = context.pages[0] if context.pages else context.new_page()

            click.echo()
            click.secho(f"Searching: {query!r}" + (f" in {location!r}" if location else ""), bold=True)

            jobs = scrape_jobs(
                page,
                query=query,
                location=location,
                pages=pages,
                limit=limit,
                delay=delay,
                remote_only=remote_only,
                posted_within_days=posted_within_days,
                log=lambda message: click.echo(f"  {message}"),
            )
    except BrowserSessionError as exc:
        click.secho(str(exc), fg="red")
        sys.exit(1)
    except KeyboardInterrupt:
        click.echo()
        click.secho("Interrupted.", fg="red")
        sys.exit(130)

    if not jobs:
        click.secho(
            "\nNo jobs collected. Common causes: the search genuinely has no results, "
            "the session expired (run `internship-radar login`), or LinkedIn changed "
            "their markup. Re-run without --headless to watch what happens.",
            fg="red",
        )
        sys.exit(2)

    written = write_results(
        jobs,
        out_path,
        fmt=fmt,
        fields=JOB_FIELDS,
        title=f"LinkedIn job results — {query}",
        meta={"query": query, "location": location},
    )

    click.echo()
    for job in jobs[:5]:
        click.echo(f"  • {job.company} — {job.title}" + (f" ({job.location})" if job.location else ""))
    if len(jobs) > 5:
        click.echo(f"  … and {len(jobs) - 5} more")

    click.secho(f"\nWrote {len(jobs)} job(s) to {written} as {fmt}.", fg="green")


def _split_terms(raw: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    if not raw:
        return default
    return tuple(term.strip() for term in raw.split(",") if term.strip())


@main.command()
@click.argument("query")
@click.option("--location", "-l", default="",
              help="Folded into the keywords; content search has no real location facet.")
@click.option("--out", "-o", "out_path", default="posts.json", show_default=True,
              help="Output file path.")
@click.option("--format", "-f", "fmt", type=click.Choice(FORMATS), default=None,
              help="Output format. Defaults to the extension of --out (json if unknown).")
@click.option("--max-age-days", default=14.0, show_default=True,
              help="Drop posts older than this many days.")
@click.option("--scrolls", "-s", default=5, show_default=True,
              help="Infinite-scroll rounds; each one loads more posts.")
@click.option("--network-idle-ms", default=15000, show_default=True,
              help="How long to wait for network idle after the search page loads.")
@click.option("--wait-for-posts-ms", default=20000, show_default=True,
              help="How long to wait for post containers to appear before scraping.")
@click.option("--limit", "-n", default=None, type=int, help="Stop after this many kept posts.")
@click.option("--delay", default=2.0, show_default=True,
              help="Seconds to wait between scrolls. Please do not set this to 0.")
@click.option("--match", "match_mode", type=click.Choice(MATCH_MODES), default="all",
              show_default=True,
              help="'all' requires hiring intent AND internship AND AI/ML/software domain "
                   "in the post text; 'loose' needs hiring plus either role or domain.")
@click.option("--hiring-terms", default=None, help="Comma-separated override for hiring terms.")
@click.option("--role-terms", default=None, help="Comma-separated override for role terms.")
@click.option("--domain-terms", default=None, help="Comma-separated override for domain terms.")
@click.option("--keep-undated", is_flag=True,
              help="Keep posts whose timestamp could not be parsed (default: drop them).")
@click.option("--allowlist", "allowlist_path", default=None,
              type=click.Path(exists=True, dir_okay=False),
              help="Only keep companies on this list. One name per line, or a CSV's first column.")
@click.option("--classify-llm", is_flag=True,
              help="Annotate companies with an LLM guess (LLM_BACKEND=gemini|ollama). "
                   "Gemini needs GEMINI_API_KEY; ollama uses local OLLAMA_MODEL. Never verified.")
@click.option("--llm-filter", is_flag=True,
              help="Also DROP companies the LLM does not call a startup. Filters on a guess.")
@click.option("--gemini-model", default="gemini-2.0-flash", show_default=True,
              help="Gemini model when LLM_BACKEND=gemini (ignored for ollama; use OLLAMA_MODEL).")
@click.option("--include-text", is_flag=True, help="Add the full post text as a column.")
@click.option("--resolve-links", is_flag=True,
              help="Kept posts already get a copy-link pass. This flag also "
                   "forces a visible browser (ignores --headless) so you can "
                   "watch the … → Copy link clicks.")
@click.option("--max-resolve", default=10, show_default=True,
              help="Max kept posts to copy-link (URN hits are skipped).")
@click.option("--resolve-delay", default=1.2, show_default=True,
              help="Seconds to wait between copy-link attempts on kept posts.")
@click.option("--debug", is_flag=True,
              help="Dump debug_page.html, debug_report.json and debug_page.png after "
                   "scrolling, and log how many elements each candidate selector matches.")
@click.option("--debug-dir", default=".", show_default=True,
              help="Where to write the --debug artefacts.")
@click.option("--headless", is_flag=True,
              help="Run without a visible window. Off by default: posts runs VISIBLE so "
                   "you can watch what LinkedIn actually renders.")
@_profile_option
def posts(
    query: str,
    location: str,
    out_path: str,
    fmt: str | None,
    max_age_days: float,
    scrolls: int,
    network_idle_ms: int,
    wait_for_posts_ms: int,
    limit: int | None,
    delay: float,
    match_mode: str,
    hiring_terms: str | None,
    role_terms: str | None,
    domain_terms: str | None,
    keep_undated: bool,
    allowlist_path: str | None,
    classify_llm: bool,
    llm_filter: bool,
    gemini_model: str,
    include_text: bool,
    resolve_links: bool,
    max_resolve: int,
    resolve_delay: float,
    debug: bool,
    debug_dir: str,
    headless: bool,
    user_data_dir: str,
) -> None:
    """Search LinkedIn POSTS for QUERY and keep the recent hiring ones.

    Example:

        internship-radar posts "AI intern hiring" --max-age-days 14 -o posts.csv
    """
    click.secho(f"WARNING: {WARNING}", fg="yellow")
    click.secho(profile_hint(user_data_dir), fg="cyan")

    if delay < 1:
        click.secho(
            "Note: a delay below 1s hits LinkedIn harder than a human would and "
            "raises your odds of being flagged.",
            fg="yellow",
        )

    if llm_filter and not classify_llm:
        classify_llm = True
        click.secho("--llm-filter implies --classify-llm.", fg="yellow")

    if resolve_links and headless:
        click.secho(
            "--resolve-links needs a visible browser so you can watch the "
            "control-menu clicks. Ignoring --headless.",
            fg="yellow",
        )
        headless = False

    if resolve_delay < 1:
        click.secho(
            "Note: --resolve-delay below 1s fires copy-link faster than a human "
            "would and raises your odds of being flagged.",
            fg="yellow",
        )

    allowlist: set[str] = set()
    if allowlist_path:
        allowlist = load_allowlist(allowlist_path)
        click.secho(f"Allowlist: {len(allowlist)} company name(s) from {allowlist_path}", fg="cyan")

    fmt = infer_format(out_path, fmt)
    matcher = HiringMatcher(
        hiring_terms=_split_terms(hiring_terms, DEFAULT_HIRING_TERMS),
        role_terms=_split_terms(role_terms, DEFAULT_ROLE_TERMS),
        domain_terms=_split_terms(domain_terms, DEFAULT_DOMAIN_TERMS),
        mode=match_mode,
    )

    resolve_tally: dict[str, int] = {}

    try:
        with persistent_context(user_data_dir, headless=headless) as context:
            if not ensure_logged_in(context, headless=headless):
                sys.exit(1)

            page = context.pages[0] if context.pages else context.new_page()

            click.echo()
            click.secho(
                f"Searching posts: {query!r} (last {max_age_days:g} days)", bold=True
            )

            found, stats, search_url = scrape_posts(
                page,
                query=query,
                max_age_days=max_age_days,
                scrolls=scrolls,
                limit=limit,
                delay=delay,
                location=location,
                matcher=matcher,
                keep_undated=keep_undated,
                debug=debug,
                debug_dir=debug_dir,
                network_idle_ms=network_idle_ms,
                wait_for_posts_ms=wait_for_posts_ms,
                log=lambda message: click.echo(f"  {message}"),
            )

            click.echo()
            click.secho(
                f"Seen {stats['seen']} • kept {stats['kept']} • too old {stats['too_old']} "
                f"• not hiring {stats['not_hiring']} • undated {stats['undated']}",
                fg="cyan",
            )

            # Allowlist before resolve so we don't burn opens on posts we'll drop.
            if allowlist and found:
                before = len(found)
                found = [
                    p for p in found
                    if company_in_allowlist(p.company or p.poster, allowlist)
                ]
                for post in found:
                    post.why_matched += "; company:allowlist"
                    post._reasons.append("company:allowlist")
                click.secho(f"Allowlist filter: {before} -> {len(found)}", fg="cyan")

            if found:
                click.echo()
                resolve_tally = resolve_post_links(
                    page,
                    found,
                    search_url=search_url,
                    max_resolve=max_resolve,
                    resolve_delay=resolve_delay,
                    log=lambda message: click.echo(f"  {message}"),
                )
                click.secho(
                    f"Permalinks: attempted {resolve_tally.get('attempted', 0)} • "
                    f"from-DOM {resolve_tally.get('copied', 0)} • "
                    f"failed {resolve_tally.get('failed', 0)} • "
                    f"skipped(already exact) {resolve_tally.get('skipped_exact', 0)}",
                    fg="cyan",
                )
    except BrowserSessionError as exc:
        click.secho(str(exc), fg="red")
        sys.exit(1)
    except KeyboardInterrupt:
        click.echo()
        click.secho("Interrupted.", fg="red")
        sys.exit(130)

    if classify_llm and found:
        try:
            classifier = CompanyClassifier(model=gemini_model)
            names = [p.company or p.poster for p in found]
            verdicts = classifier.classify(names)

            click.secho(
                "LLM labels are UNVERIFIED GUESSES from a model with a training "
                "cutoff. Treat them as triage, not funding data.",
                fg="yellow",
            )

            for post in found:
                verdict = verdicts.get(post.company or post.poster)
                if verdict:
                    post.company_class = verdict.label()

            if llm_filter:
                before = len(found)
                found = [
                    p for p in found
                    if is_startup(verdicts.get(p.company or p.poster))
                ]
                click.secho(
                    f"LLM filter: {before} -> {len(found)} "
                    "(companies it did not recognise were dropped)",
                    fg="yellow",
                )
        except ClassificationError as exc:
            click.secho(f"Classification skipped: {exc}", fg="red")

    if not found:
        if stats["seen"] == 0:
            click.secho(
                "\nNo posts were found at all, which points at the selectors rather "
                "than the filters.",
                fg="red",
            )
            if debug:
                target = Path(debug_dir).resolve()
                click.secho(
                    f"Debug artefacts written to {target}:\n"
                    "  debug_report.json  <- structured summary, start here\n"
                    "  debug_page.html    <- full rendered DOM\n"
                    "  debug_page.png     <- screenshot of what the browser saw",
                    fg="yellow",
                )
            else:
                click.secho("Re-run the same command with --debug to capture the page.", fg="yellow")
        else:
            click.secho(
                "\nPosts were found but all were filtered out. Try --match loose, a "
                "larger --max-age-days, or more --scrolls. If they were dropped as "
                "'not hiring', the terms do not fit how these posts are worded.",
                fg="red",
            )
        sys.exit(2)

    if fmt == "txt":
        fields = list(TXT_FIELDS)
    else:
        fields = list(POST_FIELDS) + (["post_text"] if include_text else [])
    written = write_results(
        found,
        out_path,
        fmt=fmt,
        fields=fields,
        title=f"LinkedIn hiring posts — {query}",
        meta={
            "query": query,
            "location": location,
            "max_age_days": max_age_days,
            "match_mode": match_mode,
            "allowlist_used": bool(allowlist),
            "llm_classified": bool(classify_llm),
            "resolve_links": bool(resolve_links),
            "resolve_tally": resolve_tally or None,
        },
    )

    click.echo()
    for post in found[:5]:
        who = post.company or post.poster or "(unknown)"
        role = f" — {post.role}" if post.role else ""
        click.echo(f"  • {who}{role} ({post.post_date})")
    if len(found) > 5:
        click.echo(f"  … and {len(found) - 5} more")

    click.secho(f"\nWrote {len(found)} post(s) to {written} as {fmt}.", fg="green")


@main.command("company-posts")
@click.option(
    "--companies",
    "companies_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Target companies file. One company per line (~20). See companies.example.txt.",
)
@click.option("--out", "-o", "out_path", default="results.txt", show_default=True,
              help="Output file path.")
@click.option("--format", "-f", "fmt", type=click.Choice(FORMATS), default=None,
              help="Output format. Defaults to the extension of --out (txt if unknown).")
@click.option("--max-age-days", default=7.0, show_default=True,
              help="Drop posts older than this many days.")
@click.option("--scrolls", "-s", default=15, show_default=True,
              help="Max infinite-scroll rounds per company. Stops earlier once a "
                   "loaded post is older than --max-age-days.")
@click.option("--network-idle-ms", default=15000, show_default=True,
              help="How long to wait for network idle after the search page loads.")
@click.option("--wait-for-posts-ms", default=20000, show_default=True,
              help="How long to wait for post containers to appear before scraping.")
@click.option("--limit", "-n", default=None, type=int,
              help="Max kept posts per company.")
@click.option("--delay", default=2.0, show_default=True,
              help="Seconds to wait between scrolls within a company search.")
@click.option("--company-delay", default=5.0, show_default=True,
              help="Seconds to wait between companies. Please keep this >= 3.")
@click.option(
    "--max-queries-per-company",
    default=DEFAULT_MAX_QUERIES_PER_COMPANY,
    show_default=True,
    type=click.IntRange(1, MAX_QUERIES_PER_COMPANY),
    envvar="MAX_QUERIES_PER_COMPANY",
    show_envvar=True,
    help=(
        "Max LinkedIn keyword variations to try per company "
        f"(hard cap {MAX_QUERIES_PER_COMPANY}). More queries = more LinkedIn load."
    ),
)
@click.option("--hiring-terms", default=None, help="Comma-separated override for hiring terms.")
@click.option("--role-terms", default=None, help="Comma-separated override for role terms.")
@click.option("--domain-terms", default=None, help="Comma-separated override for domain terms.")
@click.option("--keep-undated", is_flag=True,
              help="Keep posts whose timestamp could not be parsed (default: drop them).")
@click.option(
    "--strict-source/--loose-source",
    default=True,
    show_default=True,
    help="Strict (default): keep concrete openings from the company page, "
         "employees, recruiters, or credible creators. Loose: also allow "
         "aggregator mentions of the company.",
)
@click.option("--headless", is_flag=True,
              help="Run without a visible window (requires an existing logged-in profile).")
@click.option(
    "--debug-strong",
    is_flag=True,
    help="Log why each hiring-pass post fails the STRONG gate.",
)
@_profile_option
def company_posts(
    companies_path: str,
    out_path: str,
    fmt: str | None,
    max_age_days: float,
    scrolls: int,
    network_idle_ms: int,
    wait_for_posts_ms: int,
    limit: int | None,
    delay: float,
    company_delay: float,
    max_queries_per_company: int,
    hiring_terms: str | None,
    role_terms: str | None,
    domain_terms: str | None,
    keep_undated: bool,
    strict_source: bool,
    debug_strong: bool,
    headless: bool,
    user_data_dir: str,
) -> None:
    """Search LinkedIn posts per target company; keep STRONG internship hiring matches.

    For each company in COMPANIES, runs a content search for '"Company" intern',
    scrolls until posts age past --max-age-days, then keeps only STRONG matches:
    company mention + hiring/intern co-occurring near a job-offer cue and a
    domain term (AI/ML/data/software). Product posts and newsletters are dropped.

    Example:

        internship-radar company-posts --companies companies.txt --max-age-days 7 -o results.txt
    """
    click.secho(f"WARNING: {WARNING}", fg="yellow")
    click.secho(profile_hint(user_data_dir), fg="cyan")

    if delay < 1:
        click.secho(
            "Note: a delay below 1s hits LinkedIn harder than a human would and "
            "raises your odds of being flagged.",
            fg="yellow",
        )
    if company_delay < 3:
        click.secho(
            "Note: --company-delay below 3s fires ~20 company searches back-to-back "
            "and raises your odds of being flagged.",
            fg="yellow",
        )

    try:
        companies = load_companies(companies_path)
    except FileNotFoundError as exc:
        click.secho(str(exc), fg="red")
        sys.exit(1)

    if not companies:
        click.secho(f"No companies found in {companies_path}.", fg="red")
        sys.exit(1)

    click.secho(
        f"Company-first mode: {len(companies)} company(ies) from {companies_path}",
        fg="cyan",
    )

    # Unknown extensions default to txt for this command (results.txt).
    if fmt is None and Path(out_path).suffix.lower() not in {
        ".json", ".csv", ".md", ".markdown", ".txt",
    }:
        fmt = "txt"
    fmt = infer_format(out_path, fmt)

    matcher = HiringMatcher(
        hiring_terms=_split_terms(hiring_terms, DEFAULT_HIRING_TERMS),
        role_terms=_split_terms(role_terms, DEFAULT_ROLE_TERMS),
        domain_terms=_split_terms(domain_terms, DEFAULT_DOMAIN_TERMS),
        mode="all",
    )

    try:
        with persistent_context(user_data_dir, headless=headless) as context:
            if not ensure_logged_in(context, headless=headless):
                sys.exit(1)

            page = context.pages[0] if context.pages else context.new_page()
            results, totals = search_company_internship_posts(
                page,
                companies,
                max_age_days=max_age_days,
                scrolls=scrolls,
                limit_per_company=limit,
                delay=delay,
                company_delay=company_delay,
                max_queries_per_company=max_queries_per_company,
                matcher=matcher,
                keep_undated=keep_undated,
                strict_source=strict_source,
                debug_strong=debug_strong,
                network_idle_ms=network_idle_ms,
                wait_for_posts_ms=wait_for_posts_ms,
                log=lambda message: click.echo(f"  {message}"),
            )
    except BrowserSessionError as exc:
        click.secho(str(exc), fg="red")
        sys.exit(1)
    except KeyboardInterrupt:
        click.echo()
        click.secho("Interrupted.", fg="red")
        sys.exit(130)

    if not results:
        click.secho(
            "\nNo STRONG matches. Need company mention + hiring/intern co-occurring "
            "near a job-offer cue and an AI/ML/data/software term. "
            "Try a larger --max-age-days or raise --scrolls.",
            fg="red",
        )
        sys.exit(2)

    written = write_results(
        results,
        out_path,
        fmt=fmt,
        fields=list(COMPANY_POST_FIELDS),
        title="LinkedIn company-first internship posts",
        meta={
            "companies_file": companies_path,
            "companies": len(companies),
            "max_age_days": max_age_days,
            "match": "strong (company + hire~intern proximity + offer cue~domain)",
            "searched": totals["searched"],
            "group_by": "company",
        },
    )

    click.echo()
    by_company: dict[str, int] = {}
    for row in results:
        by_company[row["company"]] = by_company.get(row["company"], 0) + 1
    for company, count in by_company.items():
        click.echo(f"  • {company}: {count} post(s)")

    click.secho(
        f"\nWrote {len(results)} strong match(es) across {len(by_company)} "
        f"company(ies) to {written} as {fmt}.",
        fg="green",
    )


@main.command("agent")
@click.option(
    "--once",
    "once_request",
    default=None,
    help="Run a single natural-language request and exit (non-interactive).",
)
@click.option(
    "--out",
    "-o",
    "out_path",
    default="results.txt",
    show_default=True,
    help="Write gated results (company/role/why_matched/post_url) to this txt file.",
)
@click.option(
    "--max-tool-calls",
    default=DEFAULT_MAX_TOOL_CALLS,
    show_default=True,
    help="Max LinkedIn tool calls per user request (runaway guard).",
)
@click.option(
    "--max-companies",
    default=DEFAULT_MAX_COMPANIES,
    show_default=True,
    help="Hard cap on companies searched in one company_posts tool call.",
)
@click.option(
    "--max-queries-per-company",
    default=DEFAULT_MAX_QUERIES_PER_COMPANY,
    show_default=True,
    type=click.IntRange(1, MAX_QUERIES_PER_COMPANY),
    envvar="MAX_QUERIES_PER_COMPANY",
    show_envvar=True,
    help=(
        "Default max query variations per company for company_posts tool calls "
        f"(hard cap {MAX_QUERIES_PER_COMPANY})."
    ),
)
@click.option("--delay", default=2.0, show_default=True,
              help="Seconds to wait between scrolls inside a tool call.")
@click.option("--company-delay", default=4.0, show_default=True,
              help="Seconds to wait between companies inside company_posts.")
@click.option("--headless", is_flag=True,
              help="Run the browser without a visible window.")
@_profile_option
def agent(
    once_request: str | None,
    out_path: str,
    max_tool_calls: int,
    max_companies: int,
    max_queries_per_company: int,
    delay: float,
    company_delay: float,
    headless: bool,
    user_data_dir: str,
) -> None:
    """Interactive REPL: ask in natural language; the LLM calls scrape tools.

    Uses LLM_BACKEND (gemini|ollama) and OLLAMA_MODEL / GEMINI_API_KEY from .env.
    After scraping, a local-Ollama relevance gate keeps only open hiring posts.
    Gated hits are printed and written to --out (default results.txt).

    Examples:

        internship-radar agent

        internship-radar agent --once "AI intern roles in Bengaluru, last 7 days"
    """
    click.secho(f"WARNING: {WARNING}", fg="yellow")
    click.secho(profile_hint(user_data_dir), fg="cyan")

    # Probe login once, then close Chromium so it is not sitting open during
    # Ollama tool-selection / summary (they fight for RAM on small Macs).
    try:
        with persistent_context(user_data_dir, headless=headless) as context:
            if not ensure_logged_in(context, headless=headless):
                sys.exit(1)
    except BrowserSessionError as exc:
        click.secho(str(exc), fg="red")
        sys.exit(1)

    try:
        bot = build_agent(
            user_data_dir=user_data_dir,
            headless=headless,
            max_tool_calls=max_tool_calls,
            max_companies=max_companies,
            max_queries_per_company=max_queries_per_company,
            delay=delay,
            company_delay=company_delay,
            log=lambda message: click.echo(message),
        )
    except LLMError as exc:
        click.secho(str(exc), fg="red")
        sys.exit(1)

    click.secho(
        f"Agent ready ({describe_backend(bot.client)}; "
        f"max {max_tool_calls} tool call(s)/request).",
        fg="green",
    )
    click.echo("Tools: scrape_posts, company_posts (+ Ollama relevance gate)")
    click.echo(f"Gated results → {out_path}")
    click.echo(
        "Browser opens only while scraping, then closes before the LLM gate/summary.\n"
        "Type a request, or 'exit' / 'quit' to leave.\n"
    )

    try:
        if once_request is not None:
            _agent_turn(bot, once_request, out_path=out_path)
            return

        while True:
            try:
                line = click.prompt("you", prompt_suffix="> ")
            except (EOFError, KeyboardInterrupt):
                click.echo()
                click.secho("Bye.", fg="cyan")
                return
            if line.strip().lower() in {"exit", "quit", "q", ":q"}:
                click.secho("Bye.", fg="cyan")
                return
            _agent_turn(bot, line, out_path=out_path)
    except KeyboardInterrupt:
        click.echo()
        click.secho("Interrupted.", fg="red")
        sys.exit(130)


def _agent_turn(bot, text: str, *, out_path: str) -> None:
    try:
        answer = bot.handle(text)
    except LLMError as exc:
        click.secho(f"LLM error: {exc}", fg="red")
        cause = exc.__cause__ or exc.__context__
        if cause is not None and str(cause) not in str(exc):
            click.secho(f"  underlying: {type(cause).__name__}: {cause}", fg="red")
        return
    except BrowserSessionError as exc:
        click.secho(str(exc), fg="red")
        return

    rows = list(bot.last_results)
    if rows:
        click.echo()
        click.secho(f"Gated results ({len(rows)}):", fg="cyan", bold=True)
        click.echo(format_result_blocks(rows).rstrip())
        written = write_results(
            rows,
            out_path,
            fmt="txt",
            fields=list(AGENT_RESULT_FIELDS),
            title="internship-radar agent — gated hiring posts",
            meta={
                "query": bot.last_query,
                "backend": describe_backend(bot.client),
                "gated": True,
            },
        )
        click.secho(f"\nWrote {len(rows)} result(s) to {written}", fg="green")
    else:
        click.secho("\nNo gated hiring posts to write.", fg="yellow")

    click.echo()
    click.secho("agent>", fg="cyan", nl=False)
    click.echo(f" {answer}")


if __name__ == "__main__":  # pragma: no cover
    main()
