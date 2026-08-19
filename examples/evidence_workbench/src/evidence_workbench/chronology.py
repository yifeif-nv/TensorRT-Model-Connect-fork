# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic chronology extraction with exact page citations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Iterator

from .schema import canonical_json, citation_label, sha256_text
from .store import Workspace


_MONTHS = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}
_MONTH_PATTERN = "|".join(sorted(_MONTHS, key=len, reverse=True))
_DATE_PATTERNS = (
    (
        "iso",
        re.compile(
            r"(?<!\d)(?P<year>19\d{2}|20\d{2})-(?P<month>0?[1-9]|1[0-2])-(?P<day>0?[1-9]|[12]\d|3[01])(?!\d)"
        ),
    ),
    (
        "month_first",
        re.compile(
            rf"\b(?P<month_name>{_MONTH_PATTERN})\.?\s+"
            r"(?P<day>0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?[,]?\s+"
            r"(?P<year>19\d{2}|20\d{2})\b",
            re.IGNORECASE,
        ),
    ),
    (
        "day_first",
        re.compile(
            rf"\b(?P<day>0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?\s+"
            rf"(?P<month_name>{_MONTH_PATTERN})\.?[,]?\s+"
            r"(?P<year>19\d{2}|20\d{2})\b",
            re.IGNORECASE,
        ),
    ),
    (
        "numeric",
        re.compile(
            r"(?<!\d)(?P<first>0?[1-9]|[12]\d|3[01])[/.]"
            r"(?P<second>0?[1-9]|[12]\d|3[01])[/.]"
            r"(?P<year>19\d{2}|20\d{2})(?!\d)"
        ),
    ),
)


@dataclass(frozen=True)
class ParsedDate:
    raw: str
    normalized: str
    ambiguous: bool
    reason: str


class ChronologyBuilder:
    def __init__(self, workspace: Workspace):
        self.workspace = workspace

    def build(
        self,
        case_id: str,
        snapshot_id: str | None = None,
        *,
        record_audit: bool = True,
    ) -> dict[str, Any]:
        chosen = snapshot_id or self.workspace.head_snapshot_id(case_id)
        verification = self.workspace.verify(case_id, chosen)
        if not verification["ok"]:
            raise RuntimeError(
                "case integrity verification failed before chronology: "
                + "; ".join(verification["failures"])
            )
        manifest = self.workspace.load_manifest(case_id, chosen)
        events: list[dict[str, Any]] = []
        seen: set[tuple[str, int, int, str]] = set()
        pages_scanned = 0
        for source in manifest["sources"]:
            for page_entry in source["pages"]:
                if page_entry["status"] != "readable":
                    continue
                page = self.workspace.page_record(case_id, page_entry["record_sha256"])
                pages_scanned += 1
                text = str(page["text"])
                for start, end, parsed in iter_dates(text):
                    key = (
                        str(source["document_id"]),
                        int(page["page_number"]),
                        start,
                        parsed.raw,
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    quote_start, quote_end = _context_bounds(text, start, end)
                    quote = text[quote_start:quote_end].strip()
                    adjusted_start = text.find(quote, quote_start)
                    quote_start = adjusted_start if adjusted_start >= 0 else quote_start
                    quote_end = quote_start + len(quote)
                    event = {
                        "raw_date": parsed.raw,
                        "normalized_date": parsed.normalized,
                        "ambiguous": parsed.ambiguous,
                        "normalization_note": parsed.reason,
                        "citation_id": (f"{source['document_id']}:p{page['page_number']}"),
                        "label": citation_label(
                            str(source["filename"]),
                            str(source["document_id"]),
                            int(page["page_number"]),
                        ),
                        "filename": source["filename"],
                        "document_id": source["document_id"],
                        "source_sha256": source["source_sha256"],
                        "page_number": page["page_number"],
                        "quote": quote,
                        "start_offset": quote_start,
                        "end_offset": quote_end,
                        "extraction_method": page["extraction_method"],
                        "needs_review": bool(page["needs_review"] or parsed.ambiguous),
                        "evidence_image": page.get("evidence_image", ""),
                    }
                    event["event_id"] = sha256_text(
                        canonical_json(
                            {
                                key: event[key]
                                for key in (
                                    "citation_id",
                                    "source_sha256",
                                    "page_number",
                                    "start_offset",
                                    "end_offset",
                                    "raw_date",
                                    "quote",
                                )
                            }
                        )
                    )
                    events.append(event)
        events.sort(
            key=lambda event: (
                event["normalized_date"] or "9999-99-99",
                event["filename"].casefold(),
                event["page_number"],
                event["start_offset"],
            )
        )
        reviews = self.workspace.reviews(case_id, manifest["snapshot_id"], "chronology_event")
        for event in events:
            review = reviews.get(event["event_id"], {})
            event["review_status"] = review.get("status", "unreviewed")
            event["review_notes"] = review.get("notes", "")
            event["reviewer"] = review.get("reviewer", "")
            event["reviewed_at"] = review.get("reviewed_at", "")
        result = {
            "snapshot_id": manifest["snapshot_id"],
            "integrity_verified": True,
            "coverage": manifest["coverage"],
            "pages_scanned": pages_scanned,
            "events": events,
            "boundary": (
                "Dates are found deterministically in indexed text. Ambiguous numeric dates "
                "are never normalized by guessing, and event meaning remains human-reviewed."
            ),
        }
        if record_audit:
            self.workspace.record_event(
                case_id,
                "chronology_built",
                {
                    "snapshot_id": manifest["snapshot_id"],
                    "event_count": len(events),
                    "ambiguous_count": sum(1 for event in events if event["ambiguous"]),
                    "result_sha256": sha256_text(canonical_json(result)),
                },
            )
        return result


def iter_dates(text: str) -> Iterator[tuple[int, int, ParsedDate]]:
    matches: list[tuple[int, int, int, ParsedDate]] = []
    for precedence, (kind, pattern) in enumerate(_DATE_PATTERNS):
        for match in pattern.finditer(text):
            parsed = _parse_match(kind, match)
            matches.append((match.start(), match.end(), precedence, parsed))
    # Longer/specific spellings win when patterns overlap.
    matches.sort(key=lambda item: (item[0], item[2], -(item[1] - item[0])))
    occupied_until = -1
    for start, end, _precedence, parsed in matches:
        if start < occupied_until:
            continue
        occupied_until = end
        yield start, end, parsed


def _parse_match(kind: str, match: re.Match[str]) -> ParsedDate:
    raw = match.group(0)
    year = int(match.group("year"))
    ambiguous = False
    reason = ""
    if kind in {"month_first", "day_first"}:
        month = _MONTHS[match.group("month_name").lower().rstrip(".")]
        day = int(match.group("day"))
    elif kind == "iso":
        month = int(match.group("month"))
        day = int(match.group("day"))
    else:
        first = int(match.group("first"))
        second = int(match.group("second"))
        if first > 12 and second <= 12:
            day, month = first, second
            reason = "numeric order inferred because the first field exceeds 12"
        elif second > 12 and first <= 12:
            month, day = first, second
            reason = "numeric order inferred because the second field exceeds 12"
        else:
            return ParsedDate(
                raw=raw,
                normalized="",
                ambiguous=True,
                reason="numeric day/month order is ambiguous",
            )
    try:
        normalized = date(year, month, day).isoformat()
    except ValueError:
        normalized = ""
        ambiguous = True
        reason = "date fields do not form a valid calendar date"
    return ParsedDate(raw=raw, normalized=normalized, ambiguous=ambiguous, reason=reason)


def _context_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    # Line boundaries are safer than punctuation: abbreviations such as
    # "Dr." must not truncate the evidence immediately before a date.
    left = text.rfind("\n", 0, start)
    quote_start = left + 1 if left >= 0 else max(0, start - 180)

    right = text.find("\n", end)
    quote_end = right if right >= 0 else min(len(text), end + 260)
    if quote_end - quote_start > 600:
        quote_start = max(0, start - 180)
        quote_end = min(len(text), end + 300)
    return quote_start, quote_end
