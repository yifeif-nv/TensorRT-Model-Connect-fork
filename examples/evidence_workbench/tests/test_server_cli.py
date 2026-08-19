# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local HTTP security and command-line workflow tests.

Intent: Validate bearer authentication, original upload naming, end-to-end API
search, static asset delivery, and the machine-readable CLI surface.
Preconditions: A loopback ephemeral port and temporary local workspace.
Postconditions: Unauthenticated API calls fail, authenticated evidence flows
complete, and no network service is required outside the test process.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pytest

import evidence_workbench.server as server_module
from evidence_workbench.cli import main
from evidence_workbench.ingest import Ingestor
from evidence_workbench.schema import PageInput
from evidence_workbench.search import EvidenceSearch
from evidence_workbench.server import WorkbenchServer
from evidence_workbench.store import Workspace


@contextmanager
def _server(
    tmp_path: Path, *, max_upload_bytes: int = 250 * 1024 * 1024
) -> Iterator[tuple[str, str, Workspace]]:
    workspace = Workspace(tmp_path / "workspace")
    server = WorkbenchServer(
        workspace,
        Ingestor(workspace),
        host="127.0.0.1",
        port=0,
        token="unit-test-token",
        max_upload_bytes=max_upload_bytes,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.address
    try:
        yield f"http://{host}:{port}", server.token, workspace
    finally:
        server.shutdown()
        thread.join(timeout=3)
        assert not thread.is_alive()


def _request(
    url: str,
    token: str = "",
    *,
    method: str = "GET",
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> urllib.response.addinfourl:
    request_headers = dict(headers or {})
    if token:
        request_headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, data=body, headers=request_headers, method=method)
    return urllib.request.urlopen(request, timeout=5)


def test_server_requires_token_and_preserves_upload_filename(tmp_path: Path) -> None:
    with _server(tmp_path) as (base, token, workspace):
        with _request(f"{base}/") as response:
            assert response.status == 200
            assert b"Evidence Workbench" in response.read()
        try:
            _request(f"{base}/api/health")
            raise AssertionError("unauthenticated request unexpectedly succeeded")
        except urllib.error.HTTPError as exc:
            assert exc.code == 401

        case_body = json.dumps({"name": "API Case", "id": "api-case"}).encode()
        with _request(
            f"{base}/api/cases",
            token,
            method="POST",
            body=case_body,
            headers={"Content-Type": "application/json"},
        ) as response:
            assert response.status == 201
        source_body = b"The hearing occurred on August 18, 2026."
        with _request(
            f"{base}/api/cases/api-case/documents",
            token,
            method="POST",
            body=source_body,
            headers={"X-Filename": "hearing-notes.txt"},
        ) as response:
            uploaded = json.loads(response.read())
        assert uploaded["document"]["filename"] == "hearing-notes.txt"

        with _request(
            f"{base}/api/cases/api-case/search",
            token,
            method="POST",
            body=json.dumps({"query": "hearing occurred", "mode": "phrase"}).encode(),
            headers={"Content-Type": "application/json"},
        ) as response:
            search = json.loads(response.read())
        assert search["status"] == "MATCHES_FOUND"
        assert search["matches"][0]["filename"] == "hearing-notes.txt"

        document_id = uploaded["document"]["document_id"]
        with _request(f"{base}/api/cases/api-case/source/{document_id}", token) as response:
            assert response.read() == source_body
        export_urls = []
        for _ in range(2):
            with _request(
                f"{base}/api/cases/api-case/export",
                token,
                method="POST",
                body=b'{"include_originals":true,"draft":true}',
                headers={"Content-Type": "application/json"},
            ) as response:
                exported = json.loads(response.read())
            export_urls.append(exported["download_url"])
            with _request(f"{base}{exported['download_url']}", token) as response:
                assert response.headers.get_content_type() == "application/zip"
                assert response.read().startswith(b"PK")
        assert export_urls[0] != export_urls[1]
        assert workspace.verify("api-case")["ok"] is True


def test_source_download_streams_the_same_verified_file_after_path_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _server(tmp_path) as (base, token, workspace):
        workspace.create_case("Download", "download")
        source = tmp_path / "source.txt"
        original_bytes = b"ORIGINAL_EVIDENCE"
        source.write_bytes(original_bytes)
        ingested = Ingestor(workspace).ingest("download", source)
        document = ingested["document"]
        object_path = workspace.source_path("download", document["source_sha256"])
        replacement = tmp_path / "replacement.txt"
        replacement.write_bytes(b"ATTACKER_CONTROLLED")
        original_open = server_module._open_verified_file
        swapped = False

        def swap_after_verification(path: Path, expected_sha256: str):
            nonlocal swapped
            stream, size = original_open(path, expected_sha256)
            if path == object_path:
                object_path.chmod(0o600)
                replacement.replace(object_path)
                swapped = True
            return stream, size

        monkeypatch.setattr(server_module, "_open_verified_file", swap_after_verification)
        with _request(
            f"{base}/api/cases/download/source/{document['document_id']}", token
        ) as response:
            downloaded = response.read()

        assert swapped is True
        assert downloaded == original_bytes
        assert object_path.read_bytes() == b"ATTACKER_CONTROLLED"


def test_source_download_refuses_symlinked_object(tmp_path: Path) -> None:
    with _server(tmp_path) as (base, token, workspace):
        workspace.create_case("Symlink download", "symlink-download")
        source = tmp_path / "source.txt"
        source.write_bytes(b"ORIGINAL_EVIDENCE")
        ingested = Ingestor(workspace).ingest("symlink-download", source)
        document = ingested["document"]
        object_path = workspace.source_path("symlink-download", document["source_sha256"])
        target = tmp_path / "target.txt"
        target.write_bytes(object_path.read_bytes())
        object_path.unlink()
        object_path.symlink_to(target)

        with pytest.raises(urllib.error.HTTPError) as error:
            _request(f"{base}/api/cases/symlink-download/source/{document['document_id']}", token)

        assert error.value.code == 400


def test_page_review_rejects_a_record_replaced_after_inspection(tmp_path: Path) -> None:
    with _server(tmp_path) as (base, token, workspace):
        workspace.create_case("Review race", "review-race")
        source = tmp_path / "scan.png"
        source.write_bytes(b"stable-source-bytes")

        def commit(text: str) -> dict[str, object]:
            return workspace.commit_document(
                "review-race",
                source_path=source,
                filename=source.name,
                pages=[
                    PageInput(
                        page_number=1,
                        text=text,
                        extraction_method="model_connect_ocr",
                        status="readable",
                        quality_score=0.9,
                        needs_review=True,
                    )
                ],
                extraction={"provider": "fixture"},
            )

        first = commit("Inspected OCR record A")
        first_document = first["document"]  # type: ignore[index]
        document_id = first_document["document_id"]  # type: ignore[index]
        with _request(
            f"{base}/api/cases/review-race/page-record/{document_id}/p1", token
        ) as response:
            inspected = json.loads(response.read())

        second = commit("Replacement OCR record B")
        stale_review = json.dumps(
            {
                "snapshot_id": inspected["snapshot_id"],
                "target_id": inspected["citation_id"],
                "record_sha256": inspected["record_sha256"],
                "status": "accepted",
                "reviewer": "Reviewer",
            }
        ).encode()
        with pytest.raises(urllib.error.HTTPError) as error:
            _request(
                f"{base}/api/cases/review-race/reviews/page",
                token,
                method="POST",
                body=stale_review,
                headers={"Content-Type": "application/json"},
            )
        assert error.value.code == 400
        assert (
            workspace.effective_coverage("review-race")["complete_for_negative_assertions"] is False
        )

        second_document = second["document"]  # type: ignore[index]
        second_page = second_document["pages"][0]  # type: ignore[index]
        current_review = json.dumps(
            {
                "snapshot_id": second["snapshot"]["snapshot_id"],  # type: ignore[index]
                "target_id": f"{document_id}:p1",
                "record_sha256": second_page["record_sha256"],  # type: ignore[index]
                "status": "accepted",
                "reviewer": "Reviewer",
            }
        ).encode()
        with _request(
            f"{base}/api/cases/review-race/reviews/page",
            token,
            method="POST",
            body=current_review,
            headers={"Content-Type": "application/json"},
        ) as response:
            accepted = json.loads(response.read())
        assert accepted["target_sha256"] == second_page["record_sha256"]  # type: ignore[index]
        assert (
            workspace.effective_coverage("review-race")["complete_for_negative_assertions"] is True
        )


def test_server_rejects_unrecognized_host_and_cross_origin_write(tmp_path: Path) -> None:
    with _server(tmp_path) as (base, token, _workspace):
        request = urllib.request.Request(
            f"{base}/api/health",
            headers={"Authorization": f"Bearer {token}", "Host": "attacker.example"},
        )
        with pytest.raises(urllib.error.HTTPError) as host_error:
            urllib.request.urlopen(request, timeout=5)
        assert host_error.value.code == 421

        request = urllib.request.Request(
            f"{base}/api/cases",
            data=b'{"name":"bad"}',
            method="POST",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Origin": "http://attacker.example",
            },
        )
        with pytest.raises(urllib.error.HTTPError) as origin_error:
            urllib.request.urlopen(request, timeout=5)
        assert origin_error.value.code == 403


def test_server_preserves_distinct_aliases_that_share_a_safe_filename(tmp_path: Path) -> None:
    with _server(tmp_path) as (base, token, workspace):
        with _request(
            f"{base}/api/cases",
            token,
            method="POST",
            body=b'{"name":"Aliases","id":"aliases"}',
            headers={"Content-Type": "application/json"},
        ):
            pass
        for filename in ("a:b.txt", "a?b.txt"):
            with _request(
                f"{base}/api/cases/aliases/documents",
                token,
                method="POST",
                body=b"Shared searchable alias evidence.",
                headers={"X-Filename": filename},
            ):
                pass

        manifest = workspace.load_manifest("aliases")
        assert len(manifest["sources"]) == 2
        assert {item["filename"] for item in manifest["sources"]} == {"a_b.txt"}
        assert len({item["document_id"] for item in manifest["sources"]}) == 2
        result = EvidenceSearch(workspace).search("aliases", "searchable alias", mode="phrase")
        assert len({match["label"] for match in result["matches"]}) == 2


def test_server_rejects_oversized_upload_before_ingest(tmp_path: Path) -> None:
    with _server(tmp_path, max_upload_bytes=8) as (base, token, _workspace):
        with _request(
            f"{base}/api/cases",
            token,
            method="POST",
            body=b'{"name":"Limit","id":"limit"}',
            headers={"Content-Type": "application/json"},
        ):
            pass
        with pytest.raises(urllib.error.HTTPError) as error:
            _request(
                f"{base}/api/cases/limit/documents",
                token,
                method="POST",
                body=b"123456789",
                headers={"X-Filename": "too-large.txt"},
            )
        assert error.value.code == 400


def test_cli_create_ingest_search_verify(tmp_path: Path, capsys: object) -> None:
    workspace = tmp_path / "workspace"
    source = tmp_path / "cli.txt"
    source.write_text("Signed on 2026-08-18 by Example Person.", encoding="utf-8")

    assert main(["--workspace", str(workspace), "create-case", "CLI", "--id", "cli"]) == 0
    assert main(["--workspace", str(workspace), "ingest", "cli", str(source)]) == 0
    assert (
        main(
            [
                "--workspace",
                str(workspace),
                "search",
                "cli",
                "Example Person",
                "--mode",
                "phrase",
            ]
        )
        == 0
    )
    assert main(["--workspace", str(workspace), "verify", "cli"]) == 0
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert '"ok": true' in captured.out
    assert "MATCHES_FOUND" in captured.out


def test_cli_verify_returns_nonzero_after_tamper(tmp_path: Path) -> None:
    workspace_path = tmp_path / "workspace"
    source = tmp_path / "tamper.txt"
    source.write_text("evidence", encoding="utf-8")
    assert main(["--workspace", str(workspace_path), "create-case", "T", "--id", "t"]) == 0
    assert main(["--workspace", str(workspace_path), "ingest", "t", str(source)]) == 0
    workspace = Workspace(workspace_path)
    manifest = workspace.load_manifest("t")
    object_path = workspace.source_path("t", manifest["sources"][0]["source_sha256"])
    object_path.chmod(0o600)
    object_path.write_text("changed", encoding="utf-8")

    assert main(["--workspace", str(workspace_path), "verify", "t"]) == 1
