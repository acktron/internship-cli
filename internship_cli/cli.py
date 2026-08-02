"""Command line interface for internship-radar."""

from __future__ import annotations

import sys

import click

from . import __version__
from .browser import (
    DEFAULT_USER_DATA_DIR,
    BrowserSessionError,
    ensure_logged_in,
    persistent_context,
    profile_hint,
)
from .output import FORMATS, infer_format, write_results
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
    """


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

    written = write_results(jobs, out_path, fmt=fmt, query=query, location=location)

    click.echo()
    for job in jobs[:5]:
        click.echo(f"  • {job.company} — {job.title}" + (f" ({job.location})" if job.location else ""))
    if len(jobs) > 5:
        click.echo(f"  … and {len(jobs) - 5} more")

    click.secho(f"\nWrote {len(jobs)} job(s) to {written} as {fmt}.", fg="green")


if __name__ == "__main__":  # pragma: no cover
    main()
