"""Serialise scraped records to json, csv, markdown, or plain text."""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

FORMATS = ("json", "csv", "md", "txt")

JOB_FIELDS = ("company", "title", "location", "link")
POST_FIELDS = (
    "poster", "company", "role", "location", "post_date",
    "job_link", "why_matched", "funding_signal", "company_class", "post_url",
    "source_type", "link_class",
)
# Compact listing for --format txt: one block per post.
TXT_FIELDS = ("company", "role", "post_url", "why_matched", "source_type")
# company-posts: target company + snippet + link + why.
COMPANY_POST_FIELDS = ("company", "poster", "snippet", "post_url", "why_matched", "source_type")


def make_snippet(text: str, max_chars: int = 240) -> str:
    """Collapse whitespace and truncate for a one-line listing."""
    cleaned = re.sub(r"\s+", " ", (text or "")).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 1].rstrip() + "…"

_LINK_FIELDS = {"link", "job_link", "post_url"}

_EXTENSION_MAP = {
    ".json": "json",
    ".csv": "csv",
    ".md": "md",
    ".markdown": "md",
    ".txt": "txt",
}


def infer_format(path: str | Path, explicit: str | None = None) -> str:
    """Pick the output format from an explicit flag, else the file extension."""
    if explicit:
        return explicit.lower()
    return _EXTENSION_MAP.get(Path(path).suffix.lower(), "json")


def _value(record: Any, field: str) -> Any:
    if isinstance(record, Mapping):
        return record.get(field, "")
    return getattr(record, field, "")


def _rows(records: Sequence, fields: Sequence[str]) -> list[dict]:
    return [{f: _value(record, f) for f in fields} for record in records]


def _escape_md(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ").strip()


def _header_label(field: str) -> str:
    return field.replace("_", " ").title()


def _write_txt(
    rows: list[dict],
    fields: Sequence[str],
    title: str,
    meta: Mapping[str, Any],
    scraped_at: str,
    group_by: str | None = None,
) -> str:
    lines = [title, ""]
    for key, value in meta.items():
        if key == "group_by":
            continue
        lines.append(f"{_header_label(key)}: {value}")
    lines.append(f"Scraped: {scraped_at}")
    lines.append(f"Results: {len(rows)}")
    lines.append("")

    if group_by:
        groups: dict[str, list[dict]] = {}
        order: list[str] = []
        for row in rows:
            key = str(row.get(group_by, "") or "(unknown)")
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(row)

        global_index = 0
        for company in order:
            bucket = groups[company]
            lines.append(f"=== {company} ({len(bucket)} post{'s' if len(bucket) != 1 else ''}) ===")
            lines.append("")
            for row in bucket:
                global_index += 1
                lines.append(f"--- {global_index} ---")
                for field in fields:
                    lines.append(f"{field}: {row.get(field, '') or ''}")
                lines.append("")
    else:
        for index, row in enumerate(rows, start=1):
            lines.append(f"--- {index} ---")
            for field in fields:
                lines.append(f"{field}: {row.get(field, '') or ''}")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_results(
    records: Sequence,
    path: str | Path,
    fmt: str = "json",
    fields: Sequence[str] = JOB_FIELDS,
    title: str = "LinkedIn results",
    meta: Mapping[str, Any] | None = None,
) -> Path:
    """Write `records` to `path` in `fmt` and return the resolved path."""
    fmt = (fmt or "json").lower()
    if fmt not in FORMATS:
        raise ValueError(f"Unsupported format {fmt!r}. Choose one of: {', '.join(FORMATS)}")

    out_path = Path(path).expanduser()
    if out_path.parent and not out_path.parent.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)

    # txt defaults to the compact field set unless the caller overrode fields.
    if fmt == "txt" and list(fields) == list(POST_FIELDS):
        fields = list(TXT_FIELDS)
    else:
        fields = list(fields)

    rows = _rows(records, fields)
    meta = dict(meta or {})
    group_by = meta.get("group_by") if isinstance(meta.get("group_by"), str) else None
    scraped_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    if fmt == "json":
        payload = {**meta, "scraped_at": scraped_at, "count": len(rows), "results": rows}
        out_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    elif fmt == "csv":
        with out_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    elif fmt == "txt":
        out_path.write_text(
            _write_txt(rows, fields, title, meta, scraped_at, group_by=group_by),
            encoding="utf-8",
        )

    else:  # md
        lines = [f"# {title}", ""]
        for key, value in meta.items():
            if key == "group_by":
                continue
            lines.append(f"- {_header_label(key)}: {value}")
        lines.append(f"- Scraped: {scraped_at}")
        lines.append(f"- Results: {len(rows)}")
        lines.append("")

        if group_by:
            groups: dict[str, list[dict]] = {}
            order: list[str] = []
            for row in rows:
                key = str(row.get(group_by, "") or "(unknown)")
                if key not in groups:
                    groups[key] = []
                    order.append(key)
                groups[key].append(row)

            global_index = 0
            for company in order:
                bucket = groups[company]
                lines.append(f"## {company} ({len(bucket)})")
                lines.append("")
                lines.append("| # | " + " | ".join(_header_label(f) for f in fields) + " |")
                lines.append("| --- | " + " | ".join("---" for _ in fields) + " |")
                for row in bucket:
                    global_index += 1
                    cells = []
                    for field in fields:
                        value = row.get(field, "")
                        if field in _LINK_FIELDS and value:
                            cells.append(f"[open]({value})")
                        else:
                            cells.append(_escape_md(value))
                    lines.append(f"| {global_index} | " + " | ".join(cells) + " |")
                lines.append("")
        else:
            lines.append("| # | " + " | ".join(_header_label(f) for f in fields) + " |")
            lines.append("| --- | " + " | ".join("---" for _ in fields) + " |")

            for index, row in enumerate(rows, start=1):
                cells = []
                for field in fields:
                    value = row.get(field, "")
                    if field in _LINK_FIELDS and value:
                        cells.append(f"[open]({value})")
                    else:
                        cells.append(_escape_md(value))
                lines.append(f"| {index} | " + " | ".join(cells) + " |")

        out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    return out_path.resolve()
