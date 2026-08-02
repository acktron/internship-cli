"""Serialise scraped jobs to json, csv, or markdown."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

FORMATS = ("json", "csv", "md")
FIELDS = ("company", "title", "location", "link")

_EXTENSION_MAP = {
    ".json": "json",
    ".csv": "csv",
    ".md": "md",
    ".markdown": "md",
}


def infer_format(path: str | Path, explicit: str | None = None) -> str:
    """Pick the output format from an explicit flag, else the file extension."""
    if explicit:
        return explicit.lower()
    return _EXTENSION_MAP.get(Path(path).suffix.lower(), "json")


def _rows(jobs: Sequence) -> list[dict]:
    return [
        {field: getattr(job, field, "") if not isinstance(job, dict) else job.get(field, "")
         for field in FIELDS}
        for job in jobs
    ]


def _escape_md(value: str) -> str:
    return (value or "").replace("|", "\\|").replace("\n", " ").strip()


def write_results(
    jobs: Sequence,
    path: str | Path,
    fmt: str = "json",
    query: str = "",
    location: str = "",
) -> Path:
    """Write `jobs` to `path` in `fmt` and return the resolved path."""
    fmt = (fmt or "json").lower()
    if fmt not in FORMATS:
        raise ValueError(f"Unsupported format {fmt!r}. Choose one of: {', '.join(FORMATS)}")

    out_path = Path(path).expanduser()
    if out_path.parent and not out_path.parent.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = _rows(jobs)
    scraped_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    if fmt == "json":
        payload = {
            "query": query,
            "location": location,
            "scraped_at": scraped_at,
            "count": len(rows),
            "results": rows,
        }
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    elif fmt == "csv":
        with out_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(FIELDS))
            writer.writeheader()
            writer.writerows(rows)

    else:  # md
        header = [
            f"# LinkedIn job results — {query or 'search'}",
            "",
            f"- Location: {location or 'any'}",
            f"- Scraped: {scraped_at}",
            f"- Results: {len(rows)}",
            "",
            "| # | Company | Title | Location | Link |",
            "| --- | --- | --- | --- | --- |",
        ]
        body = [
            "| {idx} | {company} | {title} | {loc} | {link} |".format(
                idx=index,
                company=_escape_md(row["company"]),
                title=_escape_md(row["title"]),
                loc=_escape_md(row["location"]),
                link=f"[open]({row['link']})" if row["link"] else "",
            )
            for index, row in enumerate(rows, start=1)
        ]
        out_path.write_text("\n".join(header + body) + "\n", encoding="utf-8")

    return out_path.resolve()
