# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model Connect invocation and OCR failure-retention tests.

Intent: Prove exact shell-free argv, timeout/error handling, runtime receipts,
and fail-closed OCR ingestion without requiring a GPU in unit CI.
Preconditions: A fake executable emulates public trtmc stdout/stderr contracts.
Postconditions: Commands are auditable and failed OCR remains an indexed
coverage gap instead of disappearing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

from evidence_workbench.ingest import Ingestor
from evidence_workbench.schema import ModelConnectError
from evidence_workbench.store import Workspace
from evidence_workbench.trtmc import TrtmcRunner


def _write_png(path: Path) -> Path:
    image = Image.new("RGB", (32, 24), "white")
    image.save(path, "PNG")
    image.close()
    return path


@pytest.fixture
def fake_trtmc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    binary = tmp_path / "trtmc"
    bundle = tmp_path / "ocr.bundle"
    log = tmp_path / "argv.jsonl"
    bundle.write_bytes(b"BUNDLE\x00fixture")
    binary.write_text(
        """#!/usr/bin/env python3
import json, os, sys, time
with open(os.environ['FAKE_TRTMC_LOG'], 'a', encoding='utf-8') as stream:
    stream.write(json.dumps(sys.argv[1:]) + '\\n')
command = sys.argv[1]
if command == 'inspect':
    print('family: deepseek_ocr\\nruntime_strategy: deepseek_ocr_vision_language')
elif command == 'run':
    mode = os.environ.get('FAKE_TRTMC_MODE', 'ok')
    if mode == 'sleep':
        time.sleep(2)
    if mode == 'fail':
        print('deliberate runtime failure', file=sys.stderr)
        raise SystemExit(7)
    print(os.environ.get('FAKE_OCR_TEXT', 'OCR fixture text dated August 18, 2026.'))
elif command == 'embed':
    print(json.dumps({'embedding': [0.25, -0.5, 1.0]}))
else:
    raise SystemExit(3)
""",
        encoding="utf-8",
    )
    binary.chmod(0o755)
    monkeypatch.setenv("FAKE_TRTMC_LOG", str(log))
    return binary, bundle, log


def test_ocr_uses_exact_public_cli_contract(
    fake_trtmc: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binary, bundle, log = fake_trtmc
    image = _write_png(tmp_path / "page.png")
    monkeypatch.setenv("FAKE_OCR_TEXT", "Exact OCR output.")
    runner = TrtmcRunner(bundle, binary=binary, hf_python=sys.executable, timeout=2)

    result = runner.ocr(image, max_new_tokens=321)

    assert result.text == "Exact OCR output."
    assert result.status == "readable"
    assert result.needs_review is True
    calls = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert calls == [
        ["inspect", str(bundle.resolve())],
        [
            "run",
            str(bundle.resolve()),
            "--prompt",
            "Extract the text from this image.",
            "--image",
            str(image.resolve()),
            "--max-new-tokens",
            "321",
            "--greedy",
            "--chat-template",
            "--hf-python",
            str(Path(sys.executable).resolve()),
        ],
    ]


def test_identity_and_embedding_are_strict_json(
    fake_trtmc: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    binary, bundle, _log = fake_trtmc
    runner = TrtmcRunner(bundle, binary=binary)

    identity = runner.identity()
    embedding, runtime = runner.embed("test document")

    assert len(identity["bundle_sha256"]) == 64
    assert "deepseek_ocr" in identity["inspect_output"]
    assert embedding == [0.25, -0.5, 1.0]
    assert runtime.argv[1] == "embed"


def test_runtime_failure_and_timeout_surface_as_errors(
    fake_trtmc: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binary, bundle, _log = fake_trtmc
    image = _write_png(tmp_path / "page.png")
    runner = TrtmcRunner(bundle, binary=binary, timeout=0.05)

    monkeypatch.setenv("FAKE_TRTMC_MODE", "fail")
    with pytest.raises(ModelConnectError, match="exit 7"):
        runner.ocr(image)

    monkeypatch.setenv("FAKE_TRTMC_MODE", "sleep")
    with pytest.raises(ModelConnectError, match="timed out"):
        runner.ocr(image)


def test_ocr_ingest_records_model_receipt_and_review_boundary(
    fake_trtmc: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    binary, bundle, _log = fake_trtmc
    image = _write_png(tmp_path / "record.png")
    workspace = Workspace(tmp_path / "workspace")
    workspace.create_case("OCR", "ocr")

    result = Ingestor(workspace, ocr_runner=TrtmcRunner(bundle, binary=binary)).ingest("ocr", image)

    source = result["snapshot"]["sources"][0]
    page = source["pages"][0]
    assert page["status"] == "readable"
    assert page["extraction_method"] == "model_connect_ocr"
    assert page["needs_review"] is True
    assert result["snapshot"]["coverage"]["complete_for_negative_assertions"] is False
    assert source["extraction"]["provider"] == "TensorRT-Model-Connect"
    assert len(source["extraction"]["model"]["bundle_sha256"]) == 64


def test_ocr_runtime_failure_is_retained_as_failed_page(
    fake_trtmc: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binary, bundle, _log = fake_trtmc
    monkeypatch.setenv("FAKE_TRTMC_MODE", "fail")
    image = _write_png(tmp_path / "bad.png")
    workspace = Workspace(tmp_path / "workspace")
    workspace.create_case("OCR failure", "ocr-failure")

    result = Ingestor(workspace, ocr_runner=TrtmcRunner(bundle, binary=binary)).ingest(
        "ocr-failure", image
    )

    page = result["snapshot"]["sources"][0]["pages"][0]
    assert page["status"] == "failed"
    assert "exit 7" in page["error"]
    assert result["snapshot"]["coverage"]["pages_failed"] == 1


def test_runner_rejects_artifact_mutation_after_identity(
    fake_trtmc: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    binary, bundle, _log = fake_trtmc
    image = _write_png(tmp_path / "page.png")
    runner = TrtmcRunner(bundle, binary=binary)
    runner.identity()
    bundle.write_bytes(b"changed bundle contents")

    with pytest.raises(ModelConnectError, match="changed after runner initialization"):
        runner.ocr(image)


def test_ocr_snapshot_identity_excludes_temporary_paths(
    fake_trtmc: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    binary, bundle, _log = fake_trtmc
    image = _write_png(tmp_path / "same.png")
    snapshot_ids: list[str] = []
    page_record_ids: list[str] = []
    for name in ("first", "second"):
        workspace = Workspace(tmp_path / name)
        workspace.create_case("Stable OCR", "stable")
        result = Ingestor(workspace, ocr_runner=TrtmcRunner(bundle, binary=binary)).ingest(
            "stable", image
        )
        snapshot_ids.append(result["snapshot"]["snapshot_id"])
        page_record_ids.append(result["snapshot"]["sources"][0]["pages"][0]["record_sha256"])

    assert snapshot_ids[0] == snapshot_ids[1]
    assert page_record_ids[0] == page_record_ids[1]
