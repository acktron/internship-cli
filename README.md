# internship-cli

A small command line tool that drives **your own** logged-in Chromium browser to collect
LinkedIn hiring leads into a JSON, CSV, or Markdown file. Three modes:

- `posts` — one keyword search over LinkedIn **posts**, then filters to recent, genuinely-hiring
  posts. This is the main command.
- `company-posts` — company-first: for each name in a list, search `"<Company>" intern` and keep
  only STRONG matches (company mention + hiring + internship + AI/ML/software).
- `search` — searches the **jobs** tab for structured listings.

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

## Usage: `posts` (hiring posts)

```bash
internship-radar posts "AI intern hiring" --max-age-days 14 --out posts.json
```

Scrapes `linkedin.com/search/results/content/`, expands every "…see more" toggle, and keeps
only posts that pass both filters below. Content search is an infinite-scroll feed, so
`--scrolls N` controls how much loads (there is no page number).

Each kept row records **why** it was kept in `why_matched`, e.g.
`recency:2d ago; hiring:we're hiring; role:intern; domain:ai/llm; funding-in-post:seed funded`.

### Recency filter

Default: drop anything older than 14 days. Timestamps come from the `<time datetime=…>`
attribute when present, otherwise from the relative label, handling both LinkedIn's compact
form (`2d`, `3w`, `1mo`) and verbose form (`2 days ago`). Note that LinkedIn writes `m` for
minutes and `mo` for months.

LinkedIn's own `datePosted` filter is set in the URL too, but it only offers 24h / week /
month. Since 14 days maps to neither, the tool sends `past-month` and applies the exact cut
locally — sending `past-week` would silently throw away days 8–14 that you asked to keep.

Posts whose date cannot be parsed are dropped; pass `--keep-undated` to keep them.

### Hiring filter

A post must contain a **hiring** term, a **role** term, and a **domain** term
(`--match loose` relaxes this to hiring plus either one). Matching is word-boundary anchored,
which matters more than it sounds: a plain substring test makes `ml` match "html" and `ai`
match "said" or "trained". Override any group with a comma-separated list:

```bash
internship-radar posts "intern" \
  --role-terms "intern,internship,new grad" \
  --domain-terms "ai,ml,llm,computer vision"
```

### Examples

```bash
# Default: last 14 days, AI/ML internship hiring posts
internship-radar posts "AI intern hiring" -o posts.json

# Wider net: more scrolling, looser matching, CSV out
internship-radar posts "ML internship" --scrolls 6 --match loose -o posts.csv

# Only companies you trust, as Markdown
internship-radar posts "AI intern" --allowlist companies.txt -o posts.md

# Add an unverified Gemini label without filtering on it
export GEMINI_API_KEY=...
internship-radar posts "AI intern" --classify-llm -o posts.json
```

### Options for `posts`

| Option | Default | Description |
| --- | --- | --- |
| `--max-age-days` | `14` | Drop posts older than this |
| `--scrolls`, `-s` | `3` | Infinite-scroll rounds; each loads more posts |
| `--match` | `all` | `all` = hiring+role+domain; `loose` = hiring + either |
| `--hiring-terms` / `--role-terms` / `--domain-terms` | built-in | Comma-separated overrides |
| `--keep-undated` | off | Keep posts with an unparseable date |
| `--allowlist` | *(none)* | Only keep companies on this list |
| `--classify-llm` | off | Add an unverified Gemini label (needs `GEMINI_API_KEY`) |
| `--llm-filter` | off | Also drop companies Gemini doesn't call a startup |
| `--include-text` | off | Add the full post text as a column |
| `--limit`, `-n` | *(none)* | Stop after N kept posts |
| `--delay` | `2.0` | Seconds between scrolls |
| `--network-idle-ms` | `15000` | Wait for network idle after the page loads |
| `--wait-for-posts-ms` | `20000` | Wait for post containers before scraping |
| `--debug` | off | Dump the page and log per-selector match counts |

`company` is not always available. When a post has an attached job card it comes from there;
when a company page authored the post the poster is the company; otherwise it is guessed from
the poster's headline ("Architect @ TransUnion"), which is their employer and not necessarily
the hiring company. Whichever applies is recorded in `why_matched` as `company:job-card`,
`company:company-page` or `company:poster-headline`, so you can tell the difference.

Output columns: `poster`, `company`, `role`, `location`, `post_date`, `job_link`,
`why_matched`, `funding_signal`, `company_class`, `post_url`.

---

## The funding / "seeded startup" filter

You probably want to keep only funded startups. **LinkedIn posts do not contain funding data**,
so any such filter has to get that information somewhere else. This tool offers three layers,
and only one of them is authoritative.

**1. Funding mentions quoted from the post (always on, free).** Posts often say it outright:
"we're a seed-funded startup", "Series A", "YC W24". A regex lifts that phrase verbatim into
`funding_signal`. This is quoting your source rather than inferring, so it is the most
trustworthy signal here and costs nothing. It is also silent when the post says nothing.

**2. `--allowlist companies.txt` (recommended filter).** One company per line, `#` comments
allowed; a CSV works too, using the first column. Names are matched loosely, so `Acme` on the
list still matches a scraped `Acme Technologies Pvt Ltd`. This is the only deterministic,
defensible option. The curation burden is smaller than it sounds — paste a YC company export
or a Crunchbase/Tracxn CSV once. See `companies.example.txt`.

**3. `--classify-llm` (opt-in, annotation only).** Asks Gemini to bucket each company as
`funded_startup` / `big_tech_or_enterprise` / `agency_or_consultancy` / `education_or_training`
/ `unknown`. Adds a `company_class` column and **does not filter** unless you also pass
`--llm-filter`.

Be clear-eyed about what layer 3 is worth. A language model has no live view of a funding
database; it works from a fixed training cutoff, so a round closed last month is invisible to
it while a company that has since died may still look funded. Its accuracy is worst exactly
where you need help most — the small, obscure, name-colliding firms that dominate these search
results. It will be right about Razorpay, but you did not need help with Razorpay.

So the implementation is deliberately constrained: `unknown` is an allowed and encouraged
answer rather than a forced guess, every verdict is stamped `gemini-guess` with a confidence
and `verified: false`, and the model is **never** asked for a round, amount, investor, or date,
because a fabricated "Series A" string would sit in your spreadsheet looking exactly like a
fact. Use it to triage a long list. Use `--allowlist` when the answer has to be right.

---

## Usage: `company-posts` (company-first)

```bash
internship-radar company-posts --companies companies.txt --max-age-days 7 -o results.txt
```

Instead of one broad keyword search, walks a list of target companies (~20) and runs a
LinkedIn **content** search for each: `"<Company>" intern`. Scrolls until loaded posts age
past `--max-age-days` (so you get the full week, not only the newest hour). Keeps only
**STRONG** matches — bag-of-words keyword hits are not enough:

1. mentions the target company
2. a **hiring** term and an **internship** term in the **same sentence** or within ~15 words
   (an open-role phrase, not scattered incidental hits)
3. a **job-offer cue** (`we're hiring`, `apply`, `role:`, `position`, `join our team`, `DM`,
   `send your CV`, …) near that internship mention, **and** an AI/ML/data/software **domain**
   term near it too

Product posts, newsletters, and course ads that merely contain “hiring” or the company name
somewhere are dropped. `why_matched` tags each criterion plus a `proximity:…` note.

Results are grouped by company. Each row is `company`, `snippet`, `post_url`, `why_matched`.
Search-fallback URLs are fine when an exact permalink is not available.

A pause (`--company-delay`, default 5s) sits between companies so ~20 LinkedIn searches are not
fired back-to-back. Prefer raising it over lowering it.

### Companies file format

One company per line. Blank lines and `#` comments are ignored. CSV exports work too (first
column). See `companies.example.txt`.

```text
# companies.txt — one target per line
Sarvam AI
Razorpay
Groww
Postman
```

Copy the example and edit:

```bash
cp companies.example.txt companies.txt
```

(`companies.txt` is gitignored so your personal list stays local.)

### Options for `company-posts`

| Option | Default | Description |
| --- | --- | --- |
| `--companies` | *(required)* | Path to the companies file |
| `--max-age-days` | `7` | Drop posts older than this; scrolling stops once a loaded post exceeds this |
| `--scrolls`, `-s` | `15` | Max scroll rounds per company (ceiling; stops earlier at age boundary) |
| `--company-delay` | `5.0` | Seconds between companies — please keep ≥ 3 |
| `--delay` | `2.0` | Seconds between scrolls within a company search |
| `--limit`, `-n` | *(none)* | Max kept posts per company |
| `--hiring-terms` / `--role-terms` / `--domain-terms` | built-in | Comma-separated overrides |
| `--keep-undated` | off | Keep posts with an unparseable date |
| `--out`, `-o` | `results.txt` | Output file path |
| `--headless` | off | No visible window; needs an existing logged-in profile |

---

## Usage: `search` (jobs tab)

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
  cli.py       # click commands: posts, company-posts, search, login
  browser.py   # persistent Chromium context + login detection
  posts.py     # content search: infinite scroll, see-more, post extraction
  scraper.py   # jobs tab: search URL building, pagination, field extraction
  filters.py   # recency parsing, hiring match, company-strong match, allowlist
  classify.py  # optional Gemini company labelling (annotation only)
  output.py    # json / csv / md / txt writers
companies.example.txt  # format example for --companies / --allowlist
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

### A note on LinkedIn's SDUI rewrite

Content search now renders through server-driven UI
(`data-sdui-screen="com.linkedin.sdui.flagshipnav.search.SearchResultsContent"`). On that page
**every CSS class is a hashed build artefact** like `_1e5f23a7`, regenerated on each deploy, so
class-based selectors are permanently unusable there. Posts also carry no `urn:li:activity`.

The scraper therefore targets the hooks that page does expose:

| Field | Selector |
| --- | --- |
| Post container | `[role="listitem"][componentkey*="FLAGSHIP_SEARCH"]` |
| Post body | `[data-testid="expandable-text-box"]` |
| "…see more" | `[data-testid="expandable-text-button"]` |
| Poster name | `aria-label="Open control menu for post by <name>"` |
| Timestamp | the span beside `svg[aria-label^="Visibility"]` |
| Results list | `[data-testid="lazy-column"]` |

Two consequences worth knowing. The timestamp is read from the post *header* only, never the
body, because post bodies routinely contain date-like phrases ("5 Day Training Program",
"3rd August") that a whole-card scan would happily parse as the post's age. And outbound links
are wrapped as `/safety/go/?url=<encoded>`, so they are unwrapped back to the real target.

Legacy class selectors are kept as fallbacks for older LinkedIn surfaces.

**`posts` says "No post containers matched" / Seen 0** — LinkedIn changed its markup again.
Re-run with `--debug`:

```bash
internship-radar posts "AI intern hiring" --debug
```

That logs a match count for every candidate selector and writes three files:

| File | Contents |
| --- | --- |
| `debug_report.json` | Structured summary — **start here** |
| `debug_page.html` | The full rendered DOM |
| `debug_page.png` | Full-page screenshot of what the browser saw |

The report does not just say which guesses failed. It scans for elements carrying an activity
URN — the one anchor LinkedIn has kept across redesigns — and reports their tags, classes,
attributes and a truncated `outer_html` sample, which normally names the current container
selector outright. Update `_POST_SELECTORS` and friends at the top of `internship_cli/posts.py`
from that.

The scraper also self-heals to a degree: if no known selector matches, it finds posts by
activity URN and reads fields from the card's own text, logging
`no known selector matched; found N post(s) by activity URN instead`. That keeps runs working
through a redesign, though updating the selectors gives cleaner field separation.

Note `debug_page.html` is a rendered page from **your logged-in session** — it contains your
name and feed content. It is gitignored; treat it as personal data before sharing it.

**`posts` returns nothing but Seen > 0** — the run prints a tally like
`Seen 40 • kept 2 • too old 12 • not hiring 26`, which tells you which filter to loosen. If
almost everything is "not hiring", your terms do not match how people word these posts; try
`--match loose` or override `--hiring-terms`. If almost everything is "too old", raise
`--max-age-days`. If `seen` itself is low, raise `--scrolls`.

**Everything dropped as undated** — LinkedIn changed the timestamp markup. Check
`_SUBDESC_SELECTORS` in `internship_cli/posts.py`, or pass `--keep-undated` as a stopgap.

## Security notes

- `.linkedin-profile/` contains **live session cookies for your LinkedIn account**. It is
  gitignored. Do not commit it, sync it, or share it — it is equivalent to handing over
  logged-in access.
- No credentials pass through this tool. You type them into LinkedIn's real login page.

## License

MIT.
