# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic chronology contract tests.

Intent: Validate unambiguous normalization, ambiguity preservation, ordering,
and exact page-context citations without model-authored facts.
Preconditions: One verified text snapshot containing representative dates.
Postconditions: Every event points to exact source text and ambiguous numeric
dates remain unnormalized.
"""

from pathlib import Path

from evidence_workbench.chronology import ChronologyBuilder, iter_dates
from evidence_workbench.ingest import Ingestor
from evidence_workbench.store import Workspace


def test_iter_dates_normalizes_only_unambiguous_values() -> None:
    text = (
        "ISO 2026-08-18. Month August 19, 2026. Day 20 August 2026. "
        "Numeric 13/08/2026. Ambiguous 04/05/2026. Invalid 2026-02-31."
    )

    parsed = [value for _start, _end, value in iter_dates(text)]

    assert [item.normalized for item in parsed[:4]] == [
        "2026-08-18",
        "2026-08-19",
        "2026-08-20",
        "2026-08-13",
    ]
    assert parsed[4].raw == "04/05/2026"
    assert parsed[4].normalized == ""
    assert parsed[4].ambiguous is True
    assert parsed[5].raw == "2026-02-31"
    assert parsed[5].ambiguous is True


def test_chronology_quotes_are_exact_and_ordered(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "workspace")
    workspace.create_case("Chronology", "chronology")
    source = tmp_path / "notes.txt"
    source.write_text(
        "A late visit happened on September 10, 2026. Follow-up was arranged.\n"
        "An earlier call occurred on 2026-01-02 and was documented.\n"
        "An ambiguous entry says 03/04/2026 and needs review.",
        encoding="utf-8",
    )
    Ingestor(workspace).ingest("chronology", source)

    result = ChronologyBuilder(workspace).build("chronology")

    assert [event["normalized_date"] for event in result["events"][:2]] == [
        "2026-01-02",
        "2026-09-10",
    ]
    ambiguous = next(event for event in result["events"] if event["raw_date"] == "03/04/2026")
    assert ambiguous["normalized_date"] == ""
    assert ambiguous["needs_review"] is True
    page_record_hash = result["events"][0]["source_sha256"]
    assert page_record_hash
    original = source.read_text(encoding="utf-8")
    for event in result["events"]:
        assert event["quote"] == original[event["start_offset"] : event["end_offset"]]
        assert event["citation_id"].endswith(":p1")
        assert len(event["event_id"]) == 64


def test_aliases_have_distinct_events_and_review_targets(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "workspace")
    workspace.create_case("Aliases", "aliases")
    first = tmp_path / "Exhibit-A.txt"
    second = tmp_path / "Exhibit-B.txt"
    first.write_text("Visit on August 18, 2026.", encoding="utf-8")
    second.write_bytes(first.read_bytes())
    Ingestor(workspace).ingest("aliases", first)
    result = Ingestor(workspace).ingest("aliases", second)

    chronology = ChronologyBuilder(workspace).build("aliases")

    assert len(chronology["events"]) == 2
    assert len({event["citation_id"] for event in chronology["events"]}) == 2
    assert len({event["event_id"] for event in chronology["events"]}) == 2
    accepted = chronology["events"][0]
    workspace.record_review(
        "aliases",
        snapshot_id=result["snapshot"]["snapshot_id"],
        target_type="chronology_event",
        target_id=accepted["event_id"],
        status="accepted",
        reviewer="Reviewer",
    )
    reviewed = ChronologyBuilder(workspace).build("aliases")
    statuses = {event["event_id"]: event["review_status"] for event in reviewed["events"]}
    assert statuses[accepted["event_id"]] == "accepted"
    assert list(statuses.values()).count("unreviewed") == 1
