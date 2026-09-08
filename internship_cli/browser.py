"""Persistent Chromium session management.

The browser profile lives in a local directory so the user logs into LinkedIn
once, in a real browser window, and the session cookies persist across runs.
This tool never sees or stores the user's credentials.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import click

DEFAULT_USER_DATA_DIR = ".linkedin-profile"

FEED_URL = "https://www.linkedin.com/feed/"
LOGIN_URL = "https://www.linkedin.com/login"

# URL fragments that mean "you are not (fully) signed in".
_LOGGED_OUT_MARKERS = ("/login", "/authwall", "/uas/login", "/checkpoint", "/signup")

# Elements that only render for an authenticated member.
_SIGNED_IN_SELECTORS = (
    "nav.global-nav",
    "#global-nav",
    "img.global-nav__me-photo",
    "[data-control-name='identity_welcome_message']",
)


class BrowserSessionError(RuntimeError):
    """Raised when Chromium cannot be launched."""


@contextmanager
def persistent_context(
    user_data_dir: str | Path,
    headless: bool = False,
    slow_mo: int = 0,
) -> Iterator["object"]:
    """Yield a Playwright persistent browser context backed by `user_data_dir`."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover - import guard
        raise BrowserSessionError(
            "Playwright is not installed. Run: pip install . && playwright install chromium"
        ) from exc

    profile_dir = Path(user_data_dir).expanduser().resolve()
    profile_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        try:
            context = playwright.chromium.launch_persistent_context(
                user_data_dir=str(profile_dir),
                headless=headless,
                slow_mo=slow_mo,
                viewport={"width": 1440, "height": 900},
                locale="en-US",
            )
        except Exception as exc:  # noqa: BLE001 - surface a friendly hint
            raise BrowserSessionError(
                f"Could not launch Chromium ({exc}).\n"
                "If the browser is missing, run: playwright install chromium\n"
                "If a previous run is still open, close it — a profile dir allows only one session."
            ) from exc

        try:
            context.grant_permissions(
                ["clipboard-read", "clipboard-write"],
                origin="https://www.linkedin.com",
            )
        except Exception:  # noqa: BLE001
            pass

        try:
            yield context
        finally:
            context.close()


def is_logged_in(page, timeout_ms: int = 20_000) -> bool:
    """Navigate to the feed and report whether LinkedIn considers us signed in."""
    try:
        page.goto(FEED_URL, wait_until="domcontentloaded", timeout=timeout_ms)
    except Exception:  # noqa: BLE001 - a failed nav is treated as "unknown, assume logged out"
        return False

    page.wait_for_timeout(1_500)

    current = (page.url or "").lower()
    if any(marker in current for marker in _LOGGED_OUT_MARKERS):
        return False

    for selector in _SIGNED_IN_SELECTORS:
        try:
            if page.locator(selector).first.is_visible(timeout=2_000):
                return True
        except Exception:  # noqa: BLE001 - selector drift is expected, try the next one
            continue

    # No known nav element matched, but we were not bounced to a login wall either.
    return "linkedin.com" in current and "/login" not in current


def ensure_logged_in(context, headless: bool) -> bool:
    """Check the session and, if needed, pause so the user can sign in by hand.

    Returns True once the session looks authenticated.
    """
    page = context.pages[0] if context.pages else context.new_page()

    if is_logged_in(page):
        click.secho("Existing LinkedIn session found.", fg="green")
        return True

    if headless:
        click.secho(
            "Not logged in, and --headless was requested.\n"
            "Run `internship-radar login` once in a visible window first.",
            fg="red",
        )
        return False

    click.echo()
    click.secho("=" * 68, fg="yellow")
    click.secho("  LinkedIn login required", fg="yellow", bold=True)
    click.secho("=" * 68, fg="yellow")
    click.echo(
        "A Chromium window is open. Please:\n"
        "  1. Sign into LinkedIn in that window (including any 2FA prompt).\n"
        "  2. Wait until your feed loads.\n"
        "  3. Come back here and press Enter.\n\n"
        "Your credentials are typed directly into LinkedIn — this tool never\n"
        "reads or stores them. Only the browser profile is kept locally."
    )

    try:
        page.goto(LOGIN_URL, wait_until="domcontentloaded", timeout=30_000)
    except Exception:  # noqa: BLE001 - the user can navigate manually if this fails
        pass

    try:
        click.pause(info="Press Enter once you are logged in... ")
    except (EOFError, KeyboardInterrupt):
        click.echo()
        click.secho("Login cancelled.", fg="red")
        return False

    if is_logged_in(page):
        click.secho("Login confirmed. Session saved to the browser profile.", fg="green")
        return True

    click.secho(
        "Still could not confirm a signed-in session. "
        "Try `internship-radar login` again and make sure your feed loads.",
        fg="red",
    )
    return False


def profile_hint(user_data_dir: str | Path) -> str:
    """Human-readable note about where the session is stored."""
    path = Path(user_data_dir).expanduser().resolve()
    exists = "existing" if path.exists() else "new"
    return f"Browser profile ({exists}): {path}"


def eprint(message: str) -> None:
    print(message, file=sys.stderr)
