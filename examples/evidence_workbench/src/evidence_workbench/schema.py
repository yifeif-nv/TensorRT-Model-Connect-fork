# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared schemas and deterministic helpers for Evidence Workbench."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
APP_VERSION = "0.1.0"
DEFAULT_CHUNK_CHARS = 3_000
DEFAULT_CHUNK_OVERLAP = 300
CASE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")
_WORD_PATTERN = re.compile(r"[^\W_][\w'-]*", re.UNICODE)


class EvidenceError(RuntimeError):
    """Base error surfaced to CLI and API users."""


class OptionalDependencyError(EvidenceError):
    """A document format requires an application-local optional dependency."""


class IntegrityError(EvidenceError):
    """Stored evidence failed a content or audit-chain integrity check."""


class ModelConnectError(EvidenceError):
    """A Model Connect subprocess failed or returned unusable output."""


@dataclass(frozen=True)
class PageInput:
    """One page ready to be committed to the evidence index."""

    page_number: int
    text: str
    extraction_method: str
    status: str
    quality_score: float
    needs_review: bool
    error: str = ""
    evidence_image: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Citation:
    """A stable citation to one exact indexed page and character range."""

    citation_id: str
    document_id: str
    filename: str
    source_sha256: str
    page_number: int
    quote: str
    start_offset: int
    end_offset: int
    extraction_method: str
    needs_review: bool
    evidence_image: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def utc_now() -> str:
    """Return a stable UTC timestamp spelling."""

    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def canonical_json(value: Any) -> str:
    """Serialize JSON for hashing and durable artifacts."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path, *, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", normalized.lower()).strip("-")
    return (slug or "case")[:48]


def validate_case_id(case_id: str) -> str:
    if not CASE_ID_PATTERN.fullmatch(case_id):
        raise EvidenceError(
            "case id must contain only lowercase letters, digits, and hyphens "
            "and be at most 63 characters"
        )
    return case_id


def safe_filename(filename: str) -> str:
    """Return a display/storage filename without accepting path traversal."""

    name = Path(filename).name.replace("\x00", "").strip()
    if not name or name in {".", ".."}:
        raise EvidenceError("filename is empty or unsafe")
    cleaned = re.sub(r"[\x00-\x1f\x7f/\\<>:\"|?*]+", "_", name)
    return cleaned[:180]


def evidence_alias(filename: str, document_id: str) -> str:
    """Return an unambiguous human label for one source instance."""

    return f"{filename} · {document_id[:8]}"


def citation_label(filename: str, document_id: str, page_number: int) -> str:
    """Return a stable, human-readable page citation."""

    return f"[{evidence_alias(filename, document_id)} p.{page_number}]"


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text).replace("\x00", "")


def search_terms(query: str) -> list[str]:
    """Extract safe FTS terms without exposing the FTS query language."""

    terms: list[str] = []
    seen: set[str] = set()
    for match in _WORD_PATTERN.finditer(normalize_text(query).casefold()):
        term = match.group(0).strip("'-")
        if len(term) < 2 or term in seen:
            continue
        seen.add(term)
        terms.append(term[:80])
    return terms[:24]


def chunk_text(
    text: str,
    *,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[tuple[int, int, str]]:
    """Split text deterministically while retaining source offsets."""

    if chunk_chars <= 0 or overlap < 0 or overlap >= chunk_chars:
        raise ValueError("chunk_chars must be positive and overlap smaller than chunk_chars")
    normalized = normalize_text(text)
    if not normalized:
        return []

    chunks: list[tuple[int, int, str]] = []
    start = 0
    while start < len(normalized):
        hard_end = min(len(normalized), start + chunk_chars)
        end = hard_end
        if hard_end < len(normalized):
            candidates = [
                normalized.rfind("\n\n", start + chunk_chars // 2, hard_end),
                normalized.rfind("\n", start + chunk_chars // 2, hard_end),
                normalized.rfind(". ", start + chunk_chars // 2, hard_end),
                normalized.rfind(" ", start + chunk_chars // 2, hard_end),
            ]
            boundary = max(candidates)
            if boundary > start:
                end = boundary + (2 if normalized[boundary : boundary + 2] == ". " else 1)
        chunk = normalized[start:end]
        if chunk.strip():
            chunks.append((start, end, chunk))
        if end >= len(normalized):
            break
        start = max(start + 1, end - overlap)
    return chunks


def _collapsed_with_map(text: str) -> tuple[str, list[int]]:
    """Collapse whitespace and retain a normalized-index to source-index map."""

    normalized = normalize_text(text)
    output: list[str] = []
    index_map: list[int] = []
    in_whitespace = False
    for index, char in enumerate(normalized):
        if char.isspace():
            if not in_whitespace and output:
                output.append(" ")
                index_map.append(index)
            in_whitespace = True
            continue
        in_whitespace = False
        output.append(char)
        index_map.append(index)
    if output and output[-1] == " ":
        output.pop()
        index_map.pop()
    return "".join(output), index_map


def locate_quote(page_text: str, quote: str) -> tuple[int, int, str] | None:
    """Locate a quote allowing whitespace-only differences, never fuzzy edits."""

    collapsed_page, index_map = _collapsed_with_map(page_text)
    collapsed_quote, _ = _collapsed_with_map(quote)
    if not collapsed_quote:
        return None
    position = collapsed_page.find(collapsed_quote)
    if position < 0:
        return None
    start = index_map[position]
    end_index = position + len(collapsed_quote) - 1
    end = index_map[end_index] + 1
    while end < len(page_text) and page_text[end].isspace():
        end += 1
    return start, end, page_text[start:end]


def locate_normalized_phrase(page_text: str, phrase: str) -> tuple[int, int] | None:
    """Locate a case-insensitive NFKC phrase with whitespace collapsing."""

    folded_page, index_map = _folded_collapsed_with_map(page_text)
    folded_phrase, _ = _folded_collapsed_with_map(phrase)
    if not folded_phrase:
        return None
    position = folded_page.find(folded_phrase)
    if position < 0:
        return None
    start = index_map[position]
    end = index_map[position + len(folded_phrase) - 1] + 1
    return start, end


def token_positions(text: str) -> list[tuple[str, int, int]]:
    """Return normalized whole-token values with original offsets."""

    normalized = normalize_text(text)
    return [
        (match.group(0).strip("'-").casefold(), match.start(), match.end())
        for match in _WORD_PATTERN.finditer(normalized)
        if match.group(0).strip("'-")
    ]


def _folded_collapsed_with_map(text: str) -> tuple[str, list[int]]:
    output: list[str] = []
    index_map: list[int] = []
    in_whitespace = False
    for source_index, source_char in enumerate(text):
        for char in unicodedata.normalize("NFKC", source_char).casefold():
            if char.isspace():
                if not in_whitespace and output:
                    output.append(" ")
                    index_map.append(source_index)
                in_whitespace = True
                continue
            in_whitespace = False
            output.append(char)
            index_map.append(source_index)
    if output and output[-1] == " ":
        output.pop()
        index_map.pop()
    return "".join(output), index_map


def csv_safe(value: Any) -> Any:
    """Neutralize spreadsheet formula prefixes in exported untrusted text."""

    if not isinstance(value, str):
        return value
    if value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def unique_preserving_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
