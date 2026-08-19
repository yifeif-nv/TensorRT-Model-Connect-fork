# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic lexical retrieval and coverage-aware negative results."""

from __future__ import annotations

import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

from .schema import (
    EvidenceError,
    canonical_json,
    citation_label,
    locate_normalized_phrase,
    normalize_text,
    search_terms,
    sha256_text,
    token_positions,
)
from .store import Workspace


SEARCH_MODES = {"phrase", "all", "any"}


class EvidenceSearch:
    def __init__(self, workspace: Workspace):
        self.workspace = workspace

    def search(
        self,
        case_id: str,
        query: str,
        *,
        mode: str = "all",
        limit: int = 20,
    ) -> dict[str, Any]:
        clean_query = normalize_text(query).strip()
        if not clean_query:
            raise EvidenceError("search query must not be empty")
        if mode not in SEARCH_MODES:
            raise EvidenceError(f"search mode must be one of: {', '.join(sorted(SEARCH_MODES))}")
        if limit < 1 or limit > 100:
            raise EvidenceError("search limit must be between 1 and 100")

        snapshot_id = self.workspace.head_snapshot_id(case_id)
        verification = self.workspace.verify(case_id, snapshot_id)
        if not verification["ok"]:
            raise EvidenceError(
                "case integrity verification failed before search: "
                + "; ".join(verification["failures"])
            )
        manifest = self.workspace.load_manifest(case_id, snapshot_id)
        terms = search_terms(clean_query)
        if not terms:
            raise EvidenceError("search query contains no indexable terms")

        if mode == "phrase":
            ranked = self._phrase_matches(case_id, manifest, clean_query, terms)
        else:
            ranked = self._term_matches(
                self.workspace.index_path(case_id, snapshot_id), terms, mode
            )
        ranked.sort(key=_result_sort_key)
        matches = ranked[:limit]

        def finalize() -> tuple[dict[str, Any], dict[str, Any]]:
            coverage = self.workspace.effective_coverage(case_id, manifest)
            if matches:
                status = (
                    "MATCHES_FOUND"
                    if coverage["complete_for_negative_assertions"]
                    else "MATCHES_FOUND_COVERAGE_INCOMPLETE"
                )
            elif not coverage["complete_for_negative_assertions"]:
                status = "COVERAGE_INCOMPLETE"
            elif mode == "phrase":
                status = "NOT_PRESENT_IN_INDEXED_TEXT"
            else:
                status = "NO_VERIFIED_MATCH"
            result = {
                "status": status,
                "query": clean_query,
                "mode": mode,
                "snapshot_id": manifest["snapshot_id"],
                "integrity_verified": True,
                "coverage": coverage,
                "matches": matches,
                "excluded_sources": coverage.get("excluded_documents", []),
                "boundary": (
                    "Negative results apply only to the verified indexed text. "
                    "Semantic absence is never inferred from embedding similarity or generation. "
                    f"{coverage.get('documents_excluded', 0)} excluded source(s) are not searched "
                    "and remain listed in this result."
                ),
            }
            payload = {
                "query_sha256": sha256_text(clean_query),
                "mode": mode,
                "snapshot_id": manifest["snapshot_id"],
                "status": status,
                "citation_ids": [match["citation_id"] for match in matches],
                "result_sha256": sha256_text(canonical_json(result)),
            }
            return payload, result

        result, _event_hash = self.workspace.atomic_audit_event(
            case_id,
            "search_completed",
            finalize,
        )
        return result

    def _phrase_matches(
        self,
        case_id: str,
        manifest: dict[str, Any],
        query: str,
        terms: list[str],
    ) -> list[dict[str, Any]]:
        """Exhaustively scan authenticated pages before asserting absence."""

        results: list[dict[str, Any]] = []
        for source in manifest["sources"]:
            for page in source["pages"]:
                record = self.workspace.page_record(case_id, page["record_sha256"])
                text = str(record["text"])
                location = locate_normalized_phrase(text, query)
                if location is None:
                    continue
                match_start, match_end = location
                quote_start, quote_end = _quote_bounds(text, match_start, match_end)
                results.append(
                    _result(
                        source=source,
                        page=page,
                        quote=text[quote_start:quote_end],
                        start_offset=quote_start,
                        end_offset=quote_end,
                        score=10_000 + len(terms) * 1_000,
                        matched_terms=terms,
                        phrase_match=True,
                    )
                )
        return results

    @staticmethod
    def _term_matches(index_path: Path, terms: list[str], mode: str) -> list[dict[str, Any]]:
        quoted = [f'"{term}"' for term in terms]
        expression = (" AND " if mode == "all" else " OR ").join(quoted)
        connection = sqlite3.connect(f"file:{index_path.as_posix()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        best_by_citation: dict[str, dict[str, Any]] = {}
        try:
            cursor = connection.execute(
                """
                SELECT chunks.*
                FROM chunk_fts
                JOIN chunks ON chunks.chunk_id = chunk_fts.chunk_id
                WHERE chunk_fts MATCH ?
                ORDER BY chunks.chunk_id
                """,
                (expression,),
            )
            for row in cursor:
                text = str(row["text"])
                positions = token_positions(text)
                counts = Counter(token for token, _start, _end in positions)
                matched_terms = [term for term in terms if counts[term.casefold()] > 0]
                if mode == "all" and len(matched_terms) != len(terms):
                    continue
                matched_positions = [
                    start for token, start, _end in positions if token in matched_terms
                ]
                if not matched_positions:
                    continue
                first_match = min(matched_positions)
                quote_start, quote_end = _quote_bounds(text, first_match, first_match + 1)
                page_start = int(row["start_offset"]) + quote_start
                page_end = int(row["start_offset"]) + quote_end
                result = {
                    "score": len(matched_terms) * 1_000
                    + sum(counts[term.casefold()] for term in matched_terms),
                    "citation_id": row["citation_id"],
                    "label": citation_label(
                        str(row["filename"]),
                        str(row["document_id"]),
                        int(row["page_number"]),
                    ),
                    "document_id": row["document_id"],
                    "filename": row["filename"],
                    "source_sha256": row["source_sha256"],
                    "page_number": row["page_number"],
                    "quote": text[quote_start:quote_end],
                    "start_offset": page_start,
                    "end_offset": page_end,
                    "extraction_method": row["extraction_method"],
                    "needs_review": bool(row["needs_review"]),
                    "evidence_image": row["evidence_image"],
                    "matched_terms": matched_terms,
                    "phrase_match": False,
                    "citation_integrity_verified": True,
                }
                previous = best_by_citation.get(str(row["citation_id"]))
                if previous is None or _result_sort_key(result) < _result_sort_key(previous):
                    best_by_citation[str(row["citation_id"])] = result
        except sqlite3.Error as exc:
            raise EvidenceError(f"search index query failed: {exc}") from exc
        finally:
            connection.close()
        return list(best_by_citation.values())


def _result(
    *,
    source: dict[str, Any],
    page: dict[str, Any],
    quote: str,
    start_offset: int,
    end_offset: int,
    score: int,
    matched_terms: list[str],
    phrase_match: bool,
) -> dict[str, Any]:
    return {
        "score": score,
        "citation_id": f"{source['document_id']}:p{page['page_number']}",
        "label": citation_label(
            str(source["filename"]),
            str(source["document_id"]),
            int(page["page_number"]),
        ),
        "document_id": source["document_id"],
        "filename": source["filename"],
        "source_sha256": source["source_sha256"],
        "page_number": page["page_number"],
        "quote": quote,
        "start_offset": start_offset,
        "end_offset": end_offset,
        "extraction_method": page["extraction_method"],
        "needs_review": bool(page["needs_review"]),
        "evidence_image": page.get("evidence_image", ""),
        "matched_terms": matched_terms,
        "phrase_match": phrase_match,
        "citation_integrity_verified": True,
    }


def _quote_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    left = max(text.rfind("\n", 0, start), start - 180)
    quote_start = left + 1 if left >= 0 and text[left : left + 1] == "\n" else max(0, left)
    right_newline = text.find("\n", end)
    quote_end = min(len(text), end + 420)
    if right_newline >= 0:
        quote_end = min(quote_end, right_newline)
    return quote_start, quote_end


def _result_sort_key(item: dict[str, Any]) -> tuple[int, str, int, int]:
    return (
        -int(item["score"]),
        str(item["filename"]).casefold(),
        int(item["page_number"]),
        int(item["start_offset"]),
    )
