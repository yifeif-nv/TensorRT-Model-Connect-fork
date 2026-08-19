# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evidence storage, integrity, search, and negative-answer tests.

Intent: Prove that indexed evidence is content-addressed and search cannot make
negative assertions when coverage or integrity is incomplete.
Preconditions: Temporary local workspaces and deterministic text fixtures.
Postconditions: Snapshot IDs are stable, exact search is cited, and tampering or
coverage gaps fail closed.
"""

from __future__ import annotations

import json
import multiprocessing
import shutil
import sqlite3
import threading
from pathlib import Path

import pytest

from evidence_workbench.ingest import Ingestor
from evidence_workbench.schema import (
    EvidenceError,
    IntegrityError,
    PageInput,
    canonical_json,
    sha256_file,
    sha256_text,
)
from evidence_workbench.search import EvidenceSearch
from evidence_workbench.store import Workspace


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _process_ingest(workspace_root: str, source: str, start: object) -> None:
    start.wait()  # type: ignore[attr-defined]
    Ingestor(Workspace(workspace_root)).ingest("concurrent", source)


def test_text_ingest_search_and_verified_negative(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "workspace")
    workspace.create_case("Acme chronology", "acme")
    source = _write(
        tmp_path / "report.txt",
        "On March 4, 2026, the patient visited Clinic Alpha.\n"
        "The follow-up plan required a CT scan.",
    )

    result = Ingestor(workspace).ingest("acme", source)

    assert result["document"]["status"] == "indexed"
    assert result["snapshot"]["coverage"]["complete_for_negative_assertions"] is True
    found = EvidenceSearch(workspace).search("acme", "follow-up plan", mode="phrase")
    assert found["status"] == "MATCHES_FOUND"
    assert found["matches"][0]["label"].startswith("[report.txt · ")
    assert "follow-up plan" in found["matches"][0]["quote"]
    assert found["matches"][0]["citation_integrity_verified"] is True

    absent = EvidenceSearch(workspace).search("acme", "cardiac surgery", mode="phrase")
    assert absent["status"] == "NOT_PRESENT_IN_INDEXED_TEXT"
    assert absent["matches"] == []


def test_incomplete_coverage_blocks_negative_assertion(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "workspace")
    workspace.create_case("Scanned evidence", "scan")
    image = tmp_path / "scan.png"
    image.write_bytes(b"not-a-real-image-but-archivable")

    result = Ingestor(workspace).ingest("scan", image)

    assert result["snapshot"]["coverage"]["documents_failed"] == 1
    assert result["snapshot"]["coverage"]["pages_total"] == 0
    search = EvidenceSearch(workspace).search("scan", "missing phrase", mode="phrase")
    assert search["status"] == "COVERAGE_INCOMPLETE"
    assert search["coverage"]["complete_for_negative_assertions"] is False


def test_failed_zero_page_document_blocks_absence_with_readable_neighbor(
    tmp_path: Path,
) -> None:
    workspace = Workspace(tmp_path / "workspace")
    workspace.create_case("Mixed coverage", "mixed")
    readable = _write(tmp_path / "readable.txt", "Known evidence is present.")
    broken_pdf = tmp_path / "broken.pdf"
    broken_pdf.write_bytes(b"not a PDF")
    ingestor = Ingestor(workspace)
    ingestor.ingest("mixed", readable)
    result = ingestor.ingest("mixed", broken_pdf)

    assert result["snapshot"]["coverage"]["documents_failed"] == 1
    assert result["snapshot"]["coverage"]["complete_for_negative_assertions"] is False
    absent = EvidenceSearch(workspace).search("mixed", "never appears", mode="phrase")
    assert absent["status"] == "COVERAGE_INCOMPLETE"


def test_snapshot_id_is_independent_of_ingest_enumeration_order(tmp_path: Path) -> None:
    first_source = _write(tmp_path / "alpha.txt", "Alpha fact dated January 2, 2026.")
    second_source = _write(tmp_path / "beta.txt", "Beta fact dated January 3, 2026.")

    workspace_a = Workspace(tmp_path / "workspace-a")
    workspace_a.create_case("Order test", "order")
    results_a = Ingestor(workspace_a).ingest_many("order", [first_source, second_source])

    workspace_b = Workspace(tmp_path / "workspace-b")
    workspace_b.create_case("Order test", "order")
    results_b = Ingestor(workspace_b).ingest_many("order", [second_source, first_source])

    assert results_a[-1]["snapshot"]["snapshot_id"] == results_b[-1]["snapshot"]["snapshot_id"]


def test_committing_identical_document_reuses_snapshot(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "workspace")
    workspace.create_case("Duplicate", "duplicate")
    source = _write(tmp_path / "same.txt", "The same exact evidence.")
    first = Ingestor(workspace).ingest("duplicate", source)
    snapshot_directories_before = sorted(
        (tmp_path / "workspace/cases/duplicate/snapshots").iterdir()
    )

    second = Ingestor(workspace).ingest("duplicate", source)
    snapshot_directories_after = sorted(
        (tmp_path / "workspace/cases/duplicate/snapshots").iterdir()
    )

    assert first["snapshot"]["snapshot_id"] == second["snapshot"]["snapshot_id"]
    assert snapshot_directories_before == snapshot_directories_after


def test_identical_bytes_with_distinct_filenames_preserve_both_aliases(
    tmp_path: Path,
) -> None:
    workspace = Workspace(tmp_path / "workspace")
    workspace.create_case("Aliases", "aliases")
    first = _write(tmp_path / "Exhibit-A.txt", "Shared exhibit bytes dated 2026-08-18.")
    second = _write(tmp_path / "Exhibit-B.txt", first.read_text(encoding="utf-8"))

    Ingestor(workspace).ingest("aliases", first)
    result = Ingestor(workspace).ingest("aliases", second)

    sources = result["snapshot"]["sources"]
    assert {source["filename"] for source in sources} == {
        "Exhibit-A.txt",
        "Exhibit-B.txt",
    }
    assert len({source["document_id"] for source in sources}) == 2
    assert len({source["source_sha256"] for source in sources}) == 1


def test_sanitized_filename_collisions_have_distinct_human_citations(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "workspace")
    workspace.create_case("Colliding aliases", "colliding-aliases")
    source = _write(tmp_path / "shared.txt", "Distinct citation label evidence.")

    Ingestor(workspace).ingest("colliding-aliases", source, display_filename="a:b.txt")
    Ingestor(workspace).ingest("colliding-aliases", source, display_filename="a?b.txt")

    result = EvidenceSearch(workspace).search("colliding-aliases", "citation label", mode="phrase")
    assert len(result["matches"]) == 2
    assert len({match["label"] for match in result["matches"]}) == 2
    assert all("a_b.txt · " in match["label"] for match in result["matches"])


def test_symlinked_input_is_rejected(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "workspace")
    workspace.create_case("Symlink", "symlink")
    source = _write(tmp_path / "source.txt", "evidence")
    link = tmp_path / "link.txt"
    link.symlink_to(source)

    with pytest.raises(EvidenceError, match="symlinked"):
        Ingestor(workspace).ingest("symlink", link)


@pytest.mark.parametrize("target", ["source", "page", "index", "audit"])
def test_tampering_is_detected(tmp_path: Path, target: str) -> None:
    workspace = Workspace(tmp_path / f"workspace-{target}")
    workspace.create_case("Tamper", "tamper")
    source = _write(tmp_path / f"{target}.txt", "Evidence dated 2026-08-18.")
    result = Ingestor(workspace).ingest("tamper", source)
    manifest = result["snapshot"]
    case_root = workspace.root / "cases" / "tamper"
    if target == "source":
        path = case_root / "objects" / result["document"]["source_sha256"]
    elif target == "page":
        record_hash = manifest["sources"][0]["pages"][0]["record_sha256"]
        path = case_root / "page_records" / f"{record_hash}.json"
    elif target == "index":
        path = case_root / "snapshots" / manifest["snapshot_id"] / "index.sqlite"
    else:
        path = case_root / "audit.jsonl"
    path.chmod(0o600)
    path.write_bytes(path.read_bytes() + b"tampered")

    verification = workspace.verify("tamper")

    assert verification["ok"] is False
    assert verification["failures"]


def test_corrupt_existing_page_record_is_not_silently_reused(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "workspace")
    workspace.create_case("Page record", "page-record")
    source = _write(tmp_path / "record.txt", "record text")
    first = Ingestor(workspace).ingest("page-record", source)
    record_hash = first["snapshot"]["sources"][0]["pages"][0]["record_sha256"]
    record_path = workspace.root / "cases/page-record/page_records" / f"{record_hash}.json"
    record_path.chmod(0o600)
    record_path.write_text(json.dumps({"text": "wrong"}), encoding="utf-8")

    with pytest.raises(IntegrityError, match="page record"):
        Ingestor(workspace).ingest("page-record", source)


def test_page_record_contract_accepts_explicit_failed_page(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "workspace")
    workspace.create_case("Failed page", "failed-page")
    source = _write(tmp_path / "failure.txt", "raw source")
    result = workspace.commit_document(
        "failed-page",
        source_path=source,
        filename=source.name,
        pages=[
            PageInput(
                page_number=1,
                text="",
                extraction_method="test_failure",
                status="failed",
                quality_score=0.0,
                needs_review=True,
                error="deliberate fixture failure",
            )
        ],
        extraction={"provider": "fixture"},
        document_status="failed",
        document_error="fixture",
    )

    assert result["snapshot"]["coverage"]["pages_failed"] == 1
    assert result["snapshot"]["coverage"]["complete_for_negative_assertions"] is False


def test_manifest_coverage_cannot_be_self_recertified(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "workspace")
    workspace.create_case("Coverage tamper", "coverage-tamper")
    image = tmp_path / "evidence.png"
    image.write_bytes(b"invalid image")
    result = Ingestor(workspace).ingest("coverage-tamper", image)
    snapshot_root = (
        workspace.root / "cases/coverage-tamper/snapshots" / result["snapshot"]["snapshot_id"]
    )
    manifest_path = snapshot_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["coverage"]["complete_for_negative_assertions"] = True
    manifest_path.chmod(0o600)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    (snapshot_root / "manifest.sha256").chmod(0o600)
    (snapshot_root / "manifest.sha256").write_text(
        f"{sha256_file(manifest_path)}  manifest.json\n", encoding="ascii"
    )

    verification = workspace.verify("coverage-tamper")

    assert verification["ok"] is False
    assert any("coverage" in failure for failure in verification["failures"])
    with pytest.raises(EvidenceError, match="integrity verification failed"):
        EvidenceSearch(workspace).search("coverage-tamper", "absent phrase", mode="phrase")


def test_search_index_rows_and_fts_are_rebuilt_during_verification(
    tmp_path: Path,
) -> None:
    workspace = Workspace(tmp_path / "workspace")
    workspace.create_case("Index tamper", "index-tamper")
    source = _write(tmp_path / "index.txt", "Authenticated alpha beta phrase.")
    result = Ingestor(workspace).ingest("index-tamper", source)
    snapshot_root = (
        workspace.root / "cases/index-tamper/snapshots" / result["snapshot"]["snapshot_id"]
    )
    index_path = snapshot_root / "index.sqlite"
    connection = sqlite3.connect(index_path)
    connection.execute("DELETE FROM chunk_fts")
    connection.execute("DELETE FROM chunks")
    connection.commit()
    connection.close()
    manifest_path = snapshot_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["index_sha256"] = sha256_file(index_path)
    manifest_path.chmod(0o600)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    hash_path = snapshot_root / "manifest.sha256"
    hash_path.chmod(0o600)
    hash_path.write_text(f"{sha256_file(manifest_path)}  manifest.json\n", encoding="ascii")

    verification = workspace.verify("index-tamper")

    assert verification["ok"] is False
    assert any("index content" in failure for failure in verification["failures"])


def test_phrase_search_exhausts_authenticated_page_not_candidate_cap(
    tmp_path: Path,
) -> None:
    workspace = Workspace(tmp_path / "workspace")
    workspace.create_case("Long phrase", "long-phrase")
    filler = ("alpha " * 500 + "\n") * 620
    source = _write(tmp_path / "long.txt", filler + "alpha beta exact phrase at the end")
    Ingestor(workspace, max_source_bytes=10 * 1024 * 1024).ingest("long-phrase", source)

    result = EvidenceSearch(workspace).search(
        "long-phrase", "alpha beta exact phrase", mode="phrase"
    )

    assert result["status"] == "MATCHES_FOUND"
    assert "alpha beta exact phrase" in result["matches"][0]["quote"]


def test_all_terms_use_token_boundaries_and_any_quote_contains_match(
    tmp_path: Path,
) -> None:
    workspace = Workspace(tmp_path / "workspace")
    workspace.create_case("Tokens", "tokens")
    source = _write(tmp_path / "tokens.txt", "cat the opening. Later needle appears here.")
    Ingestor(workspace).ingest("tokens", source)

    all_result = EvidenceSearch(workspace).search("tokens", "cat he", mode="all")
    any_result = EvidenceSearch(workspace).search("tokens", "missing needle", mode="any")

    assert all_result["status"] == "NO_VERIFIED_MATCH"
    assert any_result["status"] == "MATCHES_FOUND"
    assert "needle" in any_result["matches"][0]["quote"]


def test_audit_deletion_or_valid_prefix_truncation_is_detected(tmp_path: Path) -> None:
    for variant in ("delete", "prefix"):
        workspace = Workspace(tmp_path / variant)
        workspace.create_case("Audit", "audit")
        source = _write(tmp_path / f"{variant}.txt", "audit evidence")
        Ingestor(workspace).ingest("audit", source)
        audit_path = workspace.root / "cases/audit/audit.jsonl"
        if variant == "delete":
            audit_path.unlink()
        else:
            first = audit_path.read_text(encoding="utf-8").splitlines()[0]
            audit_path.write_text(first + "\n", encoding="utf-8")

        verification = workspace.verify("audit")

        assert verification["ok"] is False
        assert any("audit" in failure for failure in verification["failures"])


def test_concurrent_workspace_instances_merge_head_updates(tmp_path: Path) -> None:
    workspace_a = Workspace(tmp_path / "workspace")
    workspace_a.create_case("Concurrent", "concurrent")
    workspace_b = Workspace(tmp_path / "workspace")
    first = _write(tmp_path / "first.txt", "first evidence")
    second = _write(tmp_path / "second.txt", "second evidence")
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def ingest(workspace: Workspace, source: Path) -> None:
        try:
            barrier.wait()
            Ingestor(workspace).ingest("concurrent", source)
        except BaseException as exc:  # pragma: no cover - assertion reports content.
            errors.append(exc)

    threads = [
        threading.Thread(target=ingest, args=(workspace_a, first)),
        threading.Thread(target=ingest, args=(workspace_b, second)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert errors == []
    assert workspace_a.load_manifest("concurrent")["coverage"]["documents_total"] == 2


def test_concurrent_processes_merge_head_updates(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace-process"
    workspace = Workspace(workspace_root)
    workspace.create_case("Concurrent", "concurrent")
    first = _write(tmp_path / "process-first.txt", "first process evidence")
    second = _write(tmp_path / "process-second.txt", "second process evidence")
    context = multiprocessing.get_context("fork")
    start = context.Event()
    processes = [
        context.Process(target=_process_ingest, args=(str(workspace_root), str(first), start)),
        context.Process(target=_process_ingest, args=(str(workspace_root), str(second), start)),
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(timeout=15)

    assert [process.exitcode for process in processes] == [0, 0]
    assert workspace.load_manifest("concurrent")["coverage"]["documents_total"] == 2


def test_prior_writer_version_remains_verifiable(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "workspace-version")
    workspace.create_case("Writer version", "writer-version")
    source = _write(tmp_path / "writer.txt", "durable evidence")
    current = Ingestor(workspace).ingest("writer-version", source)["snapshot"]
    current_root = workspace.root / "cases/writer-version/snapshots" / current["snapshot_id"]
    manifest = workspace.load_manifest("writer-version")
    manifest["app_version"] = "0.0.9"
    inventory = workspace._snapshot_inventory(
        "writer-version",
        manifest["sources"],
        manifest.get("excluded_sources", []),
        writer_version=manifest["app_version"],
    )
    snapshot_id = sha256_text(canonical_json(inventory))
    manifest["snapshot_id"] = snapshot_id
    manifest["inventory_sha256"] = snapshot_id
    snapshot_root = current_root.parent / snapshot_id
    shutil.copytree(current_root, snapshot_root)
    manifest_path = snapshot_root / "manifest.json"
    manifest_path.chmod(0o600)
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    manifest_hash = sha256_file(manifest_path)
    hash_path = snapshot_root / "manifest.sha256"
    hash_path.chmod(0o600)
    hash_path.write_text(f"{manifest_hash}  manifest.json\n", encoding="ascii")
    workspace.record_event(
        "writer-version",
        "snapshot_committed",
        workspace._snapshot_commit_payload(manifest, manifest_hash, set()),
    )
    workspace._write_head("writer-version", snapshot_id)

    verification = workspace.verify("writer-version")

    assert verification["ok"] is True
    assert workspace.load_manifest("writer-version")["app_version"] == "0.0.9"


def test_orphan_snapshot_is_recovered_only_with_intact_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = Workspace(tmp_path / "workspace-orphan")
    workspace.create_case("Orphan", "orphan")
    source = _write(tmp_path / "orphan.txt", "orphan recovery evidence")
    ingestor = Ingestor(workspace)
    original_record_event = workspace.record_event

    def interrupt_snapshot_event(case_id: str, event_type: str, payload: dict[str, object]) -> str:
        if event_type == "snapshot_committed":
            raise RuntimeError("simulated crash before audit binding")
        return original_record_event(case_id, event_type, payload)

    monkeypatch.setattr(workspace, "record_event", interrupt_snapshot_event)
    with pytest.raises(RuntimeError, match="simulated crash"):
        ingestor.ingest("orphan", source)
    assert workspace.head_snapshot_id("orphan", required=False) == ""
    assert len(list((workspace.root / "cases/orphan/snapshots").iterdir())) == 1

    monkeypatch.setattr(workspace, "record_event", original_record_event)
    recovered = ingestor.ingest("orphan", source)

    assert workspace.verify("orphan")["ok"] is True
    assert workspace.head_snapshot_id("orphan") == recovered["snapshot"]["snapshot_id"]

    audit_path = workspace.root / "cases/orphan/audit.jsonl"
    audit_head_path = workspace.root / "cases/orphan/AUDIT_HEAD.json"
    audit_path.unlink()
    audit_head_path.unlink()
    with pytest.raises(IntegrityError, match="existing snapshot is corrupt"):
        ingestor.ingest("orphan", source)


def test_page_review_carries_forward_when_target_record_is_unchanged(
    tmp_path: Path,
) -> None:
    workspace = Workspace(tmp_path / "workspace-review-carry")
    workspace.create_case("Carry review", "carry-review")
    source = _write(tmp_path / "ocr-source.txt", "reviewed OCR text")
    first = workspace.commit_document(
        "carry-review",
        source_path=source,
        filename=source.name,
        pages=[
            PageInput(
                page_number=1,
                text="reviewed OCR text",
                extraction_method="model_connect_ocr",
                status="readable",
                quality_score=0.8,
                needs_review=True,
            )
        ],
        extraction={"provider": "fixture"},
    )
    page = first["snapshot"]["sources"][0]["pages"][0]
    document_id = first["snapshot"]["sources"][0]["document_id"]
    target_id = f"{document_id}:p1"
    workspace.record_review(
        "carry-review",
        snapshot_id=first["snapshot"]["snapshot_id"],
        target_type="page",
        target_id=target_id,
        status="accepted",
        reviewer="Reviewer",
        expected_target_sha256=page["record_sha256"],
    )
    unrelated = _write(tmp_path / "unrelated.txt", "unrelated native evidence")

    Ingestor(workspace).ingest("carry-review", unrelated)

    manifest = workspace.load_manifest("carry-review")
    assert (
        workspace.effective_coverage("carry-review", manifest)["complete_for_negative_assertions"]
        is True
    )
    review = workspace.reviews("carry-review", manifest["snapshot_id"], "page")[target_id]
    assert review["status"] == "accepted"
    assert review["target_sha256"] == page["record_sha256"]
