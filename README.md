# internship-cli

A small command line tool that drives **your own** logged-in Chromium browser to collect
LinkedIn job search results (company, title, location, link) into a JSON, CSV, or Markdown file.

You sign into LinkedIn once, in a real browser window. The session is kept in a local
Chromium profile directory, so subsequent runs reuse it. The tool never asks for, reads,
or stores your password.

---

## ⚠️ Read this before you install

**Automated scraping of LinkedIn violates the [LinkedIn User Agreement](https://www.linkedin.com/legal/user-agreement)**
(section 8.2 prohibits scraping, crawling, and using bots or automated methods on the service).

Concretely, this means:

- **Your account can be restricted, suspended, or permanently banned.** LinkedIn actively
  detects automation, and there is no appeal guarantee. If your account matters to your job
  search, that risk is real and it is yours.
- **This tool makes no attempt to hide that it is automation.** It does not spoof
  fingerprints, rotate proxies, solve CAPTCHAs, or evade bot detection. It is a plain browser
  being driven at a throttled pace.
- **It is for personal, small-scale use on your own account only.** Do not use it to build a
  dataset, resell listings, or scrape on someone else's behalf.
- **Keep the volume low.** The default 2.5s delay between page loads exists for a reason.
  Leave it alone or raise it. A handful of pages a day is very different from thousands.

LinkedIn publishes job listings through official channels — the
[Talent Solutions / Jobs APIs](https://learn.microsoft.com/en-us/linkedin/talent/) and job
alert emails/RSS. If you need reliable or ongoing access, those are the sanctioned routes and
they will not get your account flagged.

**Use at your own risk.** You accept full responsibility for how you use this.

---

## Install

Requires Python 3.9+.

```bash
# from the project directory
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install .                    # or: pip install -e .  for development
playwright install chromium      # one-time browser download (~150 MB)
```

This installs the `internship-radar` command.

## First run: log in

```bash
internship-radar login
```

A Chromium window opens. Sign into LinkedIn there (including 2FA), wait for your feed to
load, then return to the terminal and press Enter. The session is written to
`./.linkedin-profile/` and reused from then on.

`search` also performs this check automatically — if you are not logged in, it opens the
window and pauses with the same prompt.

## Usage

```bash
internship-radar search "<query>" --location "<location>" --out results.json
```

### Examples

```bash
# Basic search, JSON out
internship-radar search "software engineering intern" --location "Bengaluru, India" --out results.json

# CSV, three pages of results (~75 jobs)
internship-radar search "data science internship" -l "Remote" -o jobs.csv --pages 3

# Markdown summary, remote only, posted in the last week, capped at 20 rows
internship-radar search "backend intern" --remote --posted-within 7 --limit 20 -o jobs.md

# Explicit format regardless of the file extension
internship-radar search "ml intern" -o out.txt --format csv

# Reuse an existing session without a visible window
internship-radar search "product intern" --headless -o results.json
```

### Options for `search`

| Option | Default | Description |
| --- | --- | --- |
| `--location`, `-l` | *(any)* | Location filter, e.g. `"Bengaluru, India"` or `"Remote"` |
| `--out`, `-o` | `results.json` | Output file path |
| `--format`, `-f` | *(from extension)* | `json`, `csv`, or `md` |
| `--pages`, `-p` | `1` | Result pages to walk (25 jobs per page) |
| `--limit`, `-n` | *(none)* | Stop after N jobs |
| `--delay` | `2.5` | Seconds between page loads — please do not lower this |
| `--remote` | off | Remote roles only |
| `--posted-within` | *(any)* | Only jobs posted in the last N days |
| `--headless` | off | No visible window; needs an existing logged-in profile |
| `--user-data-dir` | `.linkedin-profile` | Where the browser profile lives |

## Output formats

**JSON** — metadata envelope plus a `results` array:

```json
{
  "query": "software engineering intern",
  "location": "Bengaluru, India",
  "scraped_at": "2026-08-02T14:44:01+00:00",
  "count": 2,
  "results": [
    {
      "company": "Example Corp",
      "title": "Software Engineering Intern",
      "location": "Bengaluru, Karnataka, India (Hybrid)",
      "link": "https://www.linkedin.com/jobs/view/1234567890/"
    }
  ]
}
```

**CSV** — one header row, columns `company,title,location,link`.

**Markdown** — a titled table with the query, timestamp, and clickable links.

## Project layout

```
internship_cli/
  cli.py       # click commands: search, login
  browser.py   # persistent Chromium context + login detection
  scraper.py   # search URL building, pagination, field extraction
  output.py    # json / csv / md writers
```

## Troubleshooting

**"No job cards matched the known selectors"** — LinkedIn changed their markup. The candidate
selector lists at the top of `internship_cli/scraper.py` (`_CARD_SELECTORS`, `_TITLE_SELECTORS`,
and friends) are the place to fix it. Run without `--headless` to inspect the page.

**Redirected to a login wall mid-run** — the session expired. Run `internship-radar login` again.

**"Could not launch Chromium"** — either `playwright install chromium` has not been run, or
another instance is already using the profile directory. A profile allows only one live session.

**Empty or duplicated field text** — LinkedIn renders titles twice for screen readers; the
`_dedupe_text` helper handles the common cases, but odd layouts may slip through.

## Security notes

- `.linkedin-profile/` contains **live session cookies for your LinkedIn account**. It is
  gitignored. Do not commit it, sync it, or share it — it is equivalent to handing over
  logged-in access.
- No credentials pass through this tool. You type them into LinkedIn's real login page.

## License

MIT.
